"""
要約
- sn-caption 速報テキストcsvを作成
入出力
- jsons to csv
"""
import csv
import json
from dataclasses import dataclass

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

SOCCERNET_LABEL_CSV_PATH = Config.target_base_dir / "soccernet_labels.csv"

@dataclass(frozen=True)
class AnnotationArgments:
    game: str
    half: str
    time: str
    important: bool
    label: str
    split: str
    description: str


def get_game_split_map():
    game_split_map = {}
    for split in ["train", "valid", "test", "challenge"]:
        games = getListGames(split, "caption")
        game_split_map.update({game: split for game in games})
    return game_split_map


all_annotations: list[AnnotationArgments] = []

game_split_map = get_game_split_map()


for game, split in game_split_map.items():
    json_path = Config.base_dir / "SoccerNet" / game / "Labels-caption.json"

    # JSONファイルを読み込む
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)

    # 'annotations' : 速報コメントのリスト
    annotations = data.get("annotations", [])

    for annotation in annotations:
        important = annotation.get("important", "")
        game_time = annotation.get("gameTime", "")
        label = annotation.get("label", "")
        description = annotation.get("description", "")

        assert type(important) is bool
        half, time = game_time.split(" - ")
        args = AnnotationArgments(game, half, time, important, label, split, description)
        all_annotations.append(args)


# CSVファイルに保存
with open(SOCCERNET_LABEL_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["game", "important", "half", "time", "label", "split", "description"])
    for args in all_annotations:
        writer.writerow(
            [
                args.game,
                args.important,
                args.half,
                args.time,
                args.label,
                args.split,
                args.description,
            ]
        )

    print("CSVファイルの保存が完了しました。")
