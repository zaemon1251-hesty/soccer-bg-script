from collections import Counter

import pandas as pd
from sn_script.spotting.label_analysis import label_ratio_around_event
from tqdm import tqdm


def label_ratio_around_action_camera(
    comment_df: pd.DataFrame,
    action_df: pd.DataFrame,
    camera_df: pd.DataFrame,
    window_former=5,
    window_latter=5,
    binary_category_name="付加的情報か"
) -> pd.DataFrame:
    """
    (action_df, camera_df) それぞれに含まれるラベルを，
    周辺 ±(window_former, window_latter) 秒でコメントにフラグ付けし，
    最終的に (action_label, camera_label) ペアごとの
    付加的情報 (binary_category=1) 割合を集計して返す。

    ※ label_analysis.py の関数 "label_ratio_around_event" を2回呼び、
       comment_df に refer_{action_label}, refer_{camera_label} を両方作成した上で，
       行単位で (action, camera) ペアをカウントする簡易実装。
    """

    # 1) アクション側ラベル付与 → CSV 出力（副作用）
    #    comment_df に refer_{action_label} 列が作られる
    _, comment_df = label_ratio_around_event(
        comment_df=comment_df,
        label_df=action_df,
        window_former=window_former,
        window_latter=window_latter,
        binary_category_name=binary_category_name,
        label_type="action",  # 出力CSVの接尾語にも関わる
        return_changed_comment_df=True
    )
    comment_df.drop(columns=["orig_index"], inplace=True)

    # 2) カメラ側ラベル付与 → CSV 出力（副作用）
    #    comment_df に refer_{camera_label} 列が追加される
    _, comment_df = label_ratio_around_event(
        comment_df=comment_df,
        label_df=camera_df,
        window_former=window_former,
        window_latter=window_latter,
        binary_category_name=binary_category_name,
        label_type="camera",
        return_changed_comment_df=True
    )

    assert any("refer_" in col for col in comment_df.columns), f"No refer_ columns found !: {comment_df.columns}"

    # 3) (action_label, camera_label) のペアごとに
    #    出現数 / 付加的情報数 をカウントする
    action_labels = action_df["label"].unique()
    camera_labels = camera_df["label"].unique()

    pair_count = Counter()
    pair_additional = Counter()

    # 各行ごとに、True になっているアクションとカメラを全探索
    for _, row in tqdm(comment_df.iterrows()):
        is_additional = (row[binary_category_name] == 1)

        # Trueになっているアクション一覧
        triggered_actions = []
        for a_lbl in action_labels:
            col_a = f"refer_{a_lbl}"
            # もし参照列が無い(=実際には該当ラベルが無かった)ならスキップ
            if col_a in comment_df.columns and row[col_a] is True:
                triggered_actions.append(a_lbl)

        # Trueになっているカメラ一覧
        triggered_cameras = []
        for c_lbl in camera_labels:
            col_c = f"refer_{c_lbl}"
            if col_c in comment_df.columns and row[col_c] is True:
                triggered_cameras.append(c_lbl)

        # ペアをカウント
        for a in triggered_actions:
            for c in triggered_cameras:
                pair_count[(a, c)] += 1
                if is_additional:
                    pair_additional[(a, c)] += 1

    # 4) DataFrame化
    records = []
    for (a, c), cnt in pair_count.items():
        cnt_add = pair_additional[(a, c)]
        ratio = cnt_add / cnt if cnt > 0 else None
        records.append({
            "action_label": a,
            "camera_label": c,
            "count": cnt,
            "count_additional": cnt_add,
            "ratio": ratio
        })
    if records == []:
        print("No pairs !!!!!!!!")
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    # ratio をパーセント表示にするなど好みに応じて加工
    result_df["ratio(%)"] = result_df["ratio"].apply(lambda x: f"{x:.3%}" if x is not None else "-")

    # 5) 結果をCSV保存 (例: label_ratio_around_action-camera_5-5.csv)
    csv_path = f"database/misc/label_ratio_around_action-camera_{window_former}-{window_latter}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Pair ratio saved to: {csv_path}")

    return result_df, comment_df


# ============= 使い方の例 =============
if __name__ == "__main__":
    # 例: 実況コメント
    comment_df = pd.read_csv("path/to/comment.csv")
    # ここには (game, half, start, 付加的情報など) がある前提

    # アクションラベル
    action_df = pd.read_csv("path/to/action.csv")
    # ここには (game, half, time, label) がある前提
    # もし無ければ gametime_to_seconds 等で time を作る

    # カメララベル
    camera_df = pd.read_csv("path/to/camera.csv")
    # 同様に (game, half, time, label) がある前提

    # 実行
    pair_result_df, comment_df = label_ratio_around_action_camera(
        comment_df=comment_df,
        action_df=action_df,
        camera_df=camera_df,
        window_former=5,
        window_latter=5,
        binary_category_name="付加的情報か"
    )

    print(pair_result_df.head(20))
