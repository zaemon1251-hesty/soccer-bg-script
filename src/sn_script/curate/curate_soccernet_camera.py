"""
要約
- SoccerNet-v2 のCameraデータをCSVに変換する
入出力
- jsons to csv
"""
import csv
import json
from pathlib import Path

import pandas as pd
from tap import Tap

from SoccerNet.Downloader import getListGames

try:
    from sn_script.config import (
        Config,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
    )


class Args(Tap):
    SoccerNet_path: str
    output_dir: str

def labels_cameras_json_to_csv(game_split_dict: dict[str, str], soccernet_path: str, csv_file_path):
    row_list = []
    for game, split in game_split_dict.items():
        json_path = Path(soccernet_path) / game / "Labels-cameras.json"

        if not json_path.exists():
            print(f"{json_path} does not exist")
            continue

        with open(json_path, encoding='utf-8') as json_file:
            data = json.load(json_file)

            for annotation in data['annotations']:
                row = {}
                row["game"] = game
                row["split"] = split
                for key, value in annotation.items():
                    if isinstance(value, dict):
                        for sub_k, sub_v in value.items():
                            row[f"{key}_{sub_k}"] = sub_v
                    else:
                        row[key] = value
                row_list.append(row)

    result_df = pd.DataFrame(row_list)
    result_df.to_csv(csv_file_path, index=False)


def get_game_split_map():
    game_split_map = {}
    for split in ["train", "valid", "test"]:
        games = getListGames(split)
        game_split_map.update({game: split for game in games})
    return game_split_map


if __name__ == "__main__":
    args = Args().parse_args()
    game_split_map = get_game_split_map()
    output_path = Path(args.output_dir) / "soccernet_cameras.csv"
    labels_cameras_json_to_csv(
        game_split_map,
        args.SoccerNet_path,
        output_path
    )
