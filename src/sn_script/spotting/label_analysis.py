import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from sn_script.csv_utils import gametime_to_seconds
from tqdm import tqdm


def coverage_mask(interval_starts, interval_ends, points):
    """
    複数の区間 [start, end) と任意の時刻リスト points を与えたとき、
    points 各要素が少なくとも1つの区間に含まれているか(True/False)を返す。

    ここでは，ウィンドウ付きアクションが[start, end)に割り当てられ，scbiの発話開始時間がpointsに割り当てられる．

    O(M * log M + N * log M) (M=区間数, N=コメント行数)

    """
    assert len(interval_starts) == len(interval_ends)

    # アクション区間を元に座標圧縮してるので、O(M * log M) になる
    # 座標圧縮がない場合は O(S*logS) になる (S=ビデオの秒数 or フレーム数)
    boundaries = []
    for s, e in zip(interval_starts, interval_ends):
        boundaries.append((s, +1))  # 区間開始
        boundaries.append((e, -1))  # 区間終了

    # 同じ値なら +1 (開始) を先に sort する
    boundaries.sort(key=lambda x: (x[0], -x[1]))

    # いわゆる いもす法
    coverage_vals = []
    coverage_points = []
    coverage = 0
    for x, delta in boundaries:
        coverage += delta
        coverage_vals.append(coverage)
        coverage_points.append(x)

    # 各 point が labelウィンドウ の内か否か判定 (二分探索)
    result = np.zeros(len(points), dtype=bool)
    idxs = np.searchsorted(coverage_points, points, side='right')
    for i, idx in enumerate(idxs):
        if idx == 0:
            # 範囲外だからとりあえず False
            result[i] = False
        else:
            result[i] = (coverage_vals[idx - 1] > 0)

    return result


def assign_reference_flags(
    comment_df: pd.DataFrame,
    label_df_subset: pd.DataFrame,
    df_grouped: DataFrameGroupBy,
    label_col_name: str
):
    """
    (game, half) ごとに comment_df と label_df_subset を照合して、
    df の rows が [start, end) 区間に含まれるかどうかを「label_col_name」列で True/False にセットする。
    return: None
    """
    grouped_intervals = label_df_subset.groupby(["game", "half"])

    for (g, h), intervals in grouped_intervals:
        if (g, h) not in df_grouped.indices:
            continue

        idxs_df = df_grouped.indices[(g, h)]
        df_sub_gh = comment_df.iloc[idxs_df]

        interval_starts = intervals["start"].values
        interval_ends   = intervals["end"].values
        points          = df_sub_gh["start"].values.astype(float)

        in_any_interval = coverage_mask(interval_starts, interval_ends, points)
        comment_df.loc[idxs_df, label_col_name] = in_any_interval


def summarize_reference_flags(
    comment_df: pd.DataFrame,
    labe_names: pd.Index,
    binary_category_name: str
) -> pd.DataFrame:
    """
    comment_df に付加した「refer_{label_name}」列を集計し、
    結果を DataFrame (label_name, refer_count, additional_info_refer_count) で返す．
    """
    result_list = []
    for label_name in labe_names:
        colname = f"refer_{label_name}"

        if (colname not in comment_df.columns):
            # ここでは単にスキップ
            continue

        refer_count = comment_df[colname].sum()
        additional_info_refer_count = comment_df.loc[
            (comment_df[colname] == True) & (comment_df[binary_category_name] == 1), # noqa
            colname
        ].sum()

        result_list.append({
            "label": label_name,
            "refer_count": refer_count,
            "additional_info_refer_count": additional_info_refer_count,
            "ratio": additional_info_refer_count / refer_count if refer_count > 0 else None
        })

    result_df = pd.DataFrame(result_list)
    return result_df


def show_and_save_results(
    result_df: pd.DataFrame,
    window_former: int,
    window_latter: int,
    label_type="action"
):
    """
    集計結果を表示し、CSV 保存し、
    各ラベルごとの付加情報コメント割合などをコンソールに出力．
    """
    result_df["ratio"] = result_df["ratio"].map(lambda x: f"{x:.3%}" if x is not None else "-")

    # CSV 保存
    fname = f"database/misc/label_ratio_around_{label_type}_{window_former}-{window_latter}.csv"
    result_df.to_csv(fname, index=False) # 保存と同時にセル表示


def prepare_data(
    comment_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window_former: int,
    window_latter: int,
    binary_category_name: str,
):
    """
    前処理: label_df に start/end 列を作成しソート、df もソート＋orig_index管理。
    必要なラベル列を df に追加（初期 False）。
    グループ化結果やラベル一覧も返す。
    """
    # カラムチェック
    required_cols = {"game", "half", "time", "label"}
    if not required_cols.issubset(label_df.columns):
        raise ValueError(f"label_df must have columns {required_cols}")

    if label_df["time"].dtype == "O":
        label_df["time"] = label_df["time"].apply(gametime_to_seconds)

    # start/end
    label_df["start"] = label_df["time"] - window_former
    label_df["end"]   = label_df["time"] + window_latter

    # ソート
    label_df = label_df.sort_values(by=["game", "half", "start"]).reset_index(drop=True)

    comment_df = comment_df.reset_index(drop=False).rename(columns={"index": "orig_index"})
    comment_df = comment_df.sort_values(["game","half","start"]).reset_index(drop=True)

    label_grouped = label_df.groupby("label")
    label_names = label_grouped.size().index

    # 必要なフラグ列を追加
    for lbl in label_names:
        colname = f"refer_{lbl}"
        if colname not in comment_df.columns:
            comment_df[colname] = False

    comment_df_grouped = comment_df.groupby(["game", "half"])

    return comment_df, label_df, label_grouped, comment_df_grouped, label_names


def label_ratio_around_event(
    comment_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window_former=5,
    window_latter=5,
    binary_category_name="付加的情報か",
    label_type="action",
    return_changed_comment_df: bool = False
):
    """
    メイン関数: 前処理 → フラグ付け → 集計 → 表示／CSV保存

    全体の計算量:
    O(A * V * {M * log M + N * log M}) (A=アクション数, V=ビデオ(game,halfごと)数, M=アクション区間数, N=コメント行数)
    """
    # ---- 前処理 ----
    comment_df, label_df, label_grouped, comment_grouped, label_names = prepare_data(
        comment_df, label_df, window_former, window_latter, binary_category_name
    )

    # ---- フラグ付け ----
    for label_name, label_df_subset in tqdm(label_grouped, desc=f"{label_type} label loop"):
        colname = f"refer_{label_name}"
        assign_reference_flags(
            comment_df, label_df_subset, comment_grouped, colname
        )

    # df を元の順序に戻す
    comment_df = comment_df.sort_values("orig_index").reset_index(drop=True)

    # ---- 集計 ----
    result_df = summarize_reference_flags(comment_df, label_names, binary_category_name)

    # ---- 表示・出力 ----
    show_and_save_results(result_df, window_former, window_latter, label_type)

    if return_changed_comment_df:
        return result_df, comment_df

    return result_df
