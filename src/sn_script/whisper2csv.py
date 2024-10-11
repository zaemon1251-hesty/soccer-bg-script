import csv
import json
from collections import defaultdict
from typing import Literal

import pandas as pd
from tap import Tap

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys
    sys.path.append(".")
    from src.sn_script.config import Config


# コマンドライン引数を設定
class Whisper2CsvArguments(Tap):
    input_json: str = None
    output_csv: str = None
    game: str = None
    half: str = "1"

    task: Literal["to_csv", "to_csv_stable", "sample"] = "to_csv"
    suffix: str = None

def run_to_csv_local(args: Whisper2CsvArguments):
    # JSONファイルを読み込む
    with open(args.input_json) as f:
        data = json.load(f)

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text"])
        segments = data["segments"]
        for i, item in enumerate(segments):
            if "id" not in item:
                item["id"] = i
            writer.writerow(
                [
                    item["id"],
                    args.game,
                    args.half,
                    item["start"],
                    item["end"],
                    item["text"],
                ]
            )

def run_to_csv_stable(args: Whisper2CsvArguments):
    assert args.suffix is not None

    game_list = Config.targets

    items = []

    id_offset = 0

    for game in game_list:
        for half in [1, 2]:
            json_basename = f"{half}_224p{args.suffix}.json"

            json_path = Config.base_dir / game / json_basename

            try:
                data = json.load(open(json_path))
            except Exception:
                continue

            for i, segment in enumerate(data["segments"]):
                row = {
                    "id": id_offset + i,
                    "game": game,
                    "half": half,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
                items.append(row)
            id_offset += len(data["segments"])

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text"])
        for item in items:
            writer.writerow(
                [
                    item["id"],
                    item["game"],
                    item["half"],
                    item["start"],
                    item["end"],
                    item["text"],
                ]
            )

def run_sample(args: Whisper2CsvArguments):
    sapmle_dict = sample_game_by_language(args, 2)
    target_games = [f'"{game}"' for games in sapmle_dict.values() for game in games]
    print(*target_games, sep=' ')


def sample_game_by_language(args: Whisper2CsvArguments, n: int):
    """
    各言語ごとに最大n個ずつサンプルゲームを取得
    """
    game_list = Config.targets
    json_basename = f"{args.half}_224p{args.suffix}.json"
    game_audio_info = defaultdict(list)

    for game in game_list:
        json_path = Config.base_dir / game / json_basename
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)
            game_audio_info[data["language"]].append(game)

    result_dict = {}
    for lang, games in game_audio_info.items():
        target_len = min(n, len(games))
        result_dict[lang] = games[:target_len]

    return result_dict


if __name__ == "__main__":
    args = Whisper2CsvArguments().parse_args()
    if args.task == "to_csv":
        run_to_csv_local(args)
    elif args.task == "to_csv_stable":
        run_to_csv_stable(args)
    elif args.task == "sample":
        run_sample(args)
