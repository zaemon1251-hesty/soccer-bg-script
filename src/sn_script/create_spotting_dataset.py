"""
@deprecated(scbi-v2のような、0.0n秒のといった精度の高いタイムスタンプを持つデータセットに対しては、この関数は使えない)

要約
- SCBI-v1 (Soccer Commentary Background Information) の dataset splitを行う
- sn-caption のベースラインの入出力に合わせる。
    - 必要なカラム: frame_id, target_labe (1-indexed)
入出力
- csv to csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from tap import Tap

try:
    from SoccerNet.Downloader import getListGames

    from sn_script.config import (
        Config,
        binary_category_name,
    )
    from sn_script.csv_utils import gametime_to_seconds
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config, binary_category_name

class CreateDatasetArguments(Tap):
    version: str = "v1"
    stable_csv: str
    output_dir: str


def split_gamelist(gamelist: list[str]) -> dict[str, list[str]]:
    """
    ゲームの開始時間と終了時間のマップを返す
    """

    split_names = ["train", "valid", "test"]

    # どの試合がどのデータセットに含まれるか
    gamelist_dict = {split: [] for split in split_names}

    sncaption_gamelists = {
        split: getListGames(split, task="caption") for split in split_names
    }

    for game in gamelist:
        for split in split_names:
            if game in sncaption_gamelists[split]:
                gamelist_dict[split].append(game)
                break
        else:
            warnings
            gamelist_dict["train"].append(game)

    return gamelist_dict


def get_game_split_map() -> dict[str, str]:
    game_split_map: dict[str, str] = {}
    for split, gamelist in split_gamelist(Config.targets).items():
        for game in gamelist:
            game_split_map[game] = split
    return game_split_map


def map_category_id(label):
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    if isinstance(label, float) and label.is_integer():
        label = int(label)
    mapping = {
        0: 1, # 映像の説明
        1: 2  # 付加的情報を含むコメント
    }
    return mapping.get(label, None)


def preprocess_dataframe(commentary_df: pd.DataFrame) -> pd.DataFrame:
    assert "start" in commentary_df.columns, "start column is required"
    assert "game" in commentary_df.columns, "game column is required"
    assert binary_category_name in commentary_df.columns, f"{binary_category_name} column is required"

    # ゲームがどのデータセットに含まれるか
    game_split_map = get_game_split_map()

    commentary_df["split"] = commentary_df["game"].map(game_split_map).dropna()
    try:
        commentary_df["target_frameid"] = commentary_df["start"].apply(gametime_to_seconds).astype("int32") # 発話開始がターゲット。1fpsだからフレーム数と一致する
    except ValueError as e:
        for _, row in commentary_df.iterrows():
            try:
                gametime_to_seconds(row["start"])
            except ValueError:
                raise RuntimeError(row) from e

    commentary_df["target_label"] = commentary_df[binary_category_name].apply(map_category_id).dropna().astype("int32")

    return commentary_df


def preprocess_dataframe_v2(commentary_df: pd.DataFrame) -> pd.DataFrame:
    assert set(commentary_df.columns) >= {"game", "half", "start", "end", "text", binary_category_name}
    game_split_map = get_game_split_map()
    commentary_df["split"] = commentary_df["game"].map(game_split_map).dropna()
    # とりあえず、target_labelだけ統一しておく
    commentary_df["target_label"] = commentary_df[binary_category_name].apply(map_category_id).dropna().astype("int32")
    return commentary_df


def split_df_train_valid_test(commentary_df: pd.DataFrame):
    assert "split" in commentary_df.columns, "start column is required"
    assert "target_label" in commentary_df.columns, "target_label column is required"

    train_df = commentary_df[commentary_df["split"] == "train"]

    valid_df = commentary_df[commentary_df["split"] == "valid"]

    test_df = commentary_df[commentary_df["split"] == "test"]

    return train_df, valid_df, test_df

def main(args: CreateDatasetArguments):
    commentary_df = pd.read_csv(args.stable_csv)
    if args.version == "v1":
        commentary_df = preprocess_dataframe(commentary_df)
    elif args.version == "v2":
        commentary_df = preprocess_dataframe_v2(commentary_df)

    train_df, valid_df, test_df = split_df_train_valid_test(commentary_df)

    print("statistics:")
    print(f"{len(train_df)} train samples")
    print(f"{len(valid_df)} valid samples")
    print(f"{len(test_df)} test samples")

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    train_df.to_csv(Path(args.output_dir) / "train.csv", index=False)
    valid_df.to_csv(Path(args.output_dir) / "valid.csv", index=False)
    test_df.to_csv(Path(args.output_dir) / "test.csv", index=False)


if __name__ == "__main__":
    args = CreateDatasetArguments().parse_args()
    main(args)
