import csv
import json
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Literal

import pandas as pd
from tap import Tap
from tqdm import tqdm

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

    task: Literal["to_csv", "to_csv_stable", "sample", "add_rich_info"] = "to_csv"

    suffix: str = None
    suffix2: str = None

    # add_column用
    csv_path: str = None

def run_to_csv_local(args: Whisper2CsvArguments):
    # JSONファイルを読み込む
    with open(args.input_json) as f:
        data = json.load(f)

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text", "language", "speaker"])
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
                    data["language"],
                    item.get("speaker", None),
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
                    "language": data["language"],
                    "speaker": segment.get("speaker", None),
                }
                items.append(row)
            id_offset += len(data["segments"])

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text", "language", "speaker"])
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


def add_rich_info(args: Whisper2CsvArguments):
    """
    1. 実況音声の言語(en,esなど)をつける
    2. 話者分離の結果をつける
    3. X -> English 元のテキストをアライメントする
    args:
        suffix -> translate の suffix
        suffix2 -> transcribe の suffix
        csv_path -> scbi-v2 csvファイルのパス

    """
    assert args.csv_path is not None

    assert args.suffix is not None
    assert args.suffix2 is not None


    stable_df = pd.read_csv(args.csv_path)

    if args.game is not None:
        game_list = [args.game]
    else:
        game_list = Config.targets

    # 実況音声の言語, 話者分離の結果をつける（構築前にやっておくべきだった）
    for game, half in tqdm(list(product(game_list, [1, 2]))):
        json_basename = f"{half}_224p{args.suffix}.json"
        data = json.load(open(Config.base_dir / game / json_basename))
        language = data["language"]
        stable_df.loc[
            (stable_df["game"] == game) & (stable_df["half"] == half), "language"
        ] = language

        for segment in data["segments"]:
            start = segment["start"]
            end = segment["end"]
            speaker = segment.get("speaker", None)

            stable_df.loc[
                (stable_df["game"] == game) &
                (stable_df["half"] == half) &
                (stable_df["start"] >= start - 0.5) &
                (stable_df["end"] <= end + 0.5),
                "speaker"
            ] = speaker

    # 一時的に保存
    stable_df.to_csv(args.csv_path, index=False)

    # X -> English 元のテキストをアライメントする
    # Dynamic time warping (スコア:Jaccard係数) でアライメントする
    for game, half in tqdm(list(product(game_list, [1, 2]))):
        src_json_basename = f"{half}_224p{args.suffix2}.json"
        src_data = json.load(open(Config.base_dir / game / src_json_basename))

        for src_segment in enumerate(src_data["segments"]):
            src_start = src_segment["start"]
            src_end = src_segment["end"]
            partial_func = partial(
                jacard_coefficient, s2=[src_start, src_end]
            )
            idx = stable_df.loc[
                (stable_df["game"] == game) &
                (stable_df["half"] == half),
                ["start", "end"]
            ].apply(partial_func, axis=1).argmax(skipna=True)
            stable_df.loc[idx, "src_text"] = src_segment["text"]


def jacard_coefficient(s1, s2):
    # 区間 s1とs2 の Jaccard係数を計算
    # s1 = [start1, end1]
    # s2 = [start2, end2]
    start1, end1 = s1
    start2, end2 = s2
    union = max(end1, end2) - min(start1, start2)
    intersection = max(min(end1, end2) - max(start1, start2), 0)
    return intersection / union


if __name__ == "__main__":
    args = Whisper2CsvArguments().parse_args()
    if args.task == "to_csv":
        run_to_csv_local(args)
    elif args.task == "to_csv_stable":
        run_to_csv_stable(args)
    elif args.task == "sample":
        run_sample(args)
    elif args.task == "add_rich_info":
        add_rich_info(args)
