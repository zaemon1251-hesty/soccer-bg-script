import csv
import json
from collections import defaultdict
from functools import partial
from itertools import product
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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
    assert args.csv_path is not None
    assert args.suffix is not None
    assert args.suffix2 is not None

    stable_df = pd.read_csv(args.csv_path)

    if args.game is not None:
        game_list = [args.game]
    else:
        game_list = Config.targets

    # 実況音声の言語, 話者分離の結果をつける
    if ("language" not in stable_df.columns) or ("speaker" not in stable_df.columns):
        print("開始: 実況音声の言語, 話者分離結果の付与")
        add_language_and_speaker(stable_df, game_list, args.suffix)
        stable_df.to_csv(args.csv_path, index=False)
    else:
        print("すでに付与済み")

    # X -> English 元のテキストを、dynamic time warping でアライメントする
    print("開始: X -> English 対応付け")
    add_src_text(stable_df, game_list, args.suffix2)

    # 保存
    stable_df.to_csv(args.csv_path, index=False)
    print("終了")


def add_language_and_speaker(df: pd.DataFrame, game_list: List[str], suffix: str): # noqa: UP006
    for game, half in tqdm(list(product(game_list, [1, 2]))):
        json_basename = f"{half}_224p{suffix}.json"
        data = json.load(open(Config.base_dir / game / json_basename))
        language = data["language"]

        mask = (df["game"] == game) & (df["half"] == half)
        df.loc[mask, "language"] = language

        for segment in data["segments"]:
            start, end = segment["start"], segment["end"]
            speaker = segment.get("speaker", None)
            segment_mask = mask & (df["start"] >= start - 0.5) & (df["end"] <= end + 0.5)
            df.loc[segment_mask, "speaker"] = speaker


def add_src_text(df: pd.DataFrame, game_list: List[str], suffix2: str): # noqa: UP006
    for game, half in tqdm(list(product(game_list, [1, 2]))):
        src_json_basename = f"{half}_224p{suffix2}.json"
        src_data = json.load(open(Config.base_dir / game / src_json_basename))

        game_half_mask = (df["game"] == game) & (df["half"] == half)
        game_half_df = df[game_half_mask]

        if game_half_df.empty:
            continue

        src_segments = np.array([(seg["start"], seg["end"]) for seg in src_data["segments"]])
        df_segments = game_half_df[["start", "end"]].values

        # 全ての組み合わせに対してJaccard係数を計算
        jaccard_matrix = calculate_jaccard_matrix(df_segments, src_segments)

        # 最もJaccard係数が高いものを選択
        best_matches = jaccard_matrix.argmax(axis=1)

        # 対応するテキストを付与
        df.loc[game_half_mask, "src_text"] = [
            src_data["segments"][i]["text"]
            if jaccard_matrix[j, i] > 0 else None
            for j, i in enumerate(best_matches)
        ]


def calculate_jaccard_matrix(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    assert s1.shape[1] == 2, f"Expected shape (n, 2), got {s1.shape}"
    assert s2.shape[1] == 2, f"Expected shape (m, 2), got {s2.shape}"

    # strat1, end1: (n, 1)
    start1, end1 = s1[:, 0][:, None], s1[:, 1][:, None]

    # strat2, end2: (m, 1)
    start2, end2 = s2[:, 0], s2[:, 1]

    # union, intersection: (n, m)
    union = np.maximum(end1, end2) - np.minimum(start1, start2)
    intersection = np.maximum(np.minimum(end1, end2) - np.maximum(start1, start2), 0)

    return intersection / union


def jacard_coefficient(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:  # noqa: UP006
    start1, end1 = s1
    start2, end2 = s2
    union = max(end1, end2) - min(start1, start2)
    intersection = max(min(end1, end2) - max(start1, start2), 0)
    return intersection / union if union != 0 else 0

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
