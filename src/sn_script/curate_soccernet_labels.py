import json
import csv
from collections import namedtuple

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )

SOCCERNET_LABEL_CSV_PATH = Config.target_base_dir / "soccernet_labels.csv"

AnnotationArgments = namedtuple(
    "AnnotationArgments", ["game", "half", "time", "important", "label", "description"]
)

all_annotations = []

for target in Config.targets:
    json_path = Config.base_dir / target / "Labels-caption.json"

    # JSONファイルを読み込む
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # ゲームの名前を取得
    # raid_elmo内 SoccerNetの構造上SoccerNetというPath
    game = target.replace("SoccerNet/", "")

    # 'annotations' キーのデータを取得
    annotations = data.get("annotations", [])

    for annotation in annotations:
        important = annotation.get("important", "")
        game_time = annotation.get("gameTime", "")
        label = annotation.get("label", "")
        description = annotation.get("description", "")

        assert type(important) is bool
        half, time = game_time.split(" - ")
        args = AnnotationArgments(game, half, time, important, label, description)
        all_annotations.append(args)


# CSVファイルに保存
with open(SOCCERNET_LABEL_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # CSVのヘッダーを書き込む
    writer.writerow(["game", "important", "half", "time", "label", "description"])
    for args in all_annotations:
        writer.writerow(
            [
                args.game,
                args.important,
                args.half,
                args.time,
                args.label,
                args.description,
            ]
        )

    print("CSVファイルの保存が完了しました。")
