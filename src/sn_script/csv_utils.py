from __future__ import annotations

import csv
import json
import time
from functools import wraps
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        half_number,
        model_type,
        random_seed,
        subcategory_name,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        half_number,
        model_type,
        random_seed,
        subcategory_name,
    )

HUMAN_ANOTATION_CSV_PATH = (
    Config.target_base_dir
    / f"{random_seed}_{half_number}_moriy_annotation_preprocessed.csv"
)

ALL_CSV_PATH = Config.target_base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
# ALL_CSV_PATH = (
#     Config.target_base_dir / f"500_denoised_{half_number}_tokenized_224p_all.csv"
# )

DENISED_TOKENIZED_CSV_TEMPLATE = f"denoised_{half_number}_tokenized_224p.csv"

ANNOTATION_CSV_PATH = (
    Config.target_base_dir
    / f"{random_seed}_denoised_{half_number}_tokenized_224p_annotation.csv"
)

# LLM_ANOTATION_CSV_PATH = (
#     Config.base_dir / f"{model_type}_{random_seed}_{half_number}_llm_annotation.csv"
# )

LLM_ANOTATION_CSV_PATH = (
    Config.target_base_dir / f"{model_type}_500game_{half_number}_llm_annotation.csv"
)

LLM_ANNOTATION_JSONL_PATH = (
    Config.target_base_dir / f"{model_type}_500game_{half_number}_llm_annotation.jsonl"
)  # ストリームで保存するためのjsonlファイル


def write_csv(data: dict | list, output_csv_path: str | Path):
    """CSVファイルに変換"""

    # JSONデータをPythonの辞書として読み込んだ場合、segmentsの中身だけを抽出する
    if isinstance(data, dict):
        data = data["segments"]

    if not isinstance(data, list):
        raise ValueError(f"data must be list or dict, but got {type(data)}")
    if output_csv_path.exists():
        logger.info(f"CSVファイルが既に存在します。:{output_csv_path}")
        return
    with open(output_csv_path, "w", newline="", encoding="utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダを書き込む
        writer.writerow(
            [
                "id",
                "start",
                "end",
                "text",
                binary_category_name,
                category_name,
                subcategory_name,
                "備考",
            ]
        )
        # 各segmentから必要なデータを抽出してCSVに書き込む
        for segment in data:
            writer.writerow(
                [
                    segment["id"],
                    seconds_to_gametime(segment["start"]),
                    seconds_to_gametime(segment["end"]),
                    segment["text"],
                    "",
                    "",
                    "",
                    "",
                ]
            )
    logger.info(f"CSVファイルが生成されました。:{output_csv_path}")


def dump_filled_comments(half_number: int):
    DUMP_FILE_PATH = f"filled_big_class_{half_number}.csv"

    df_list = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        tmp_df["game"] = target.replace("SoccerNet/", "")
        df_list.append(tmp_df)

    all_game_df = pd.concat(df_list)
    filled_big_class_df = all_game_df.loc[
        ~all_game_df[category_name].isnull()
    ].reset_index(drop=True)
    filled_big_class_df.to_csv(DUMP_FILE_PATH, index=False)


def clean():
    half_number = 1
    DUMP_FILE_PATH = f"filled_big_class_{half_number}.csv"
    df = pd.read_csv(DUMP_FILE_PATH)
    print(df.columns)
    df.drop(columns=["Unnamed: 6", "Unnamed: 7"], inplace=True)
    df.to_csv(DUMP_FILE_PATH, index=False)


def add_column_to_csv():
    # all_game_df = pd.read_csv(ALL_CSV_PATH)
    annotation_df = pd.read_csv(ANNOTATION_CSV_PATH)

    # all_game_df[binary_category_name] = pd.NA
    annotation_df[binary_category_name] = pd.NA
    column_order = [
        "id",
        "game",
        "start",
        "end",
        "text",
        binary_category_name,
        category_name,
        subcategory_name,
        "備考",
    ]
    # all_game_df = all_game_df.reindex(columns=column_order)
    annotation_df = annotation_df.reindex(columns=column_order)
    # all_game_df.to_csv(ALL_CSV_PATH, index=False, encoding="utf-8_sig")
    annotation_df.to_csv(ANNOTATION_CSV_PATH, index=False, encoding="utf-8_sig")


def create_tokenized_annotation_csv(number_of_comments: int = 100):
    all_game_df = pd.read_csv(ALL_CSV_PATH)

    annotation_df = all_game_df.sample(n=number_of_comments, random_state=random_seed)
    annotation_df.to_csv(ANNOTATION_CSV_PATH, index=False, encoding="utf-8_sig")


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        logger.info(f"{func.__name__}は{elapsed_time}秒かかりました")
        return result

    return wrapper


def seconds_to_gametime(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02}:{int(s):02}"


def gametime_to_seconds(gametime):
    m, s = gametime.split(":")
    return int(m) * 60 + int(s)


def fill_csv_from_json():
    annotation_df = pd.read_csv(LLM_ANOTATION_CSV_PATH)
    annotation_df[binary_category_name] = pd.NA
    annotation_df["備考"] = pd.NA

    with open(LLM_ANNOTATION_JSONL_PATH) as f:
        for line in f:
            result: dict = json.loads(line)
            comment_id = result.get("comment_id")
            category = result.get("category")
            reason = result.get("reason")
            annotation_df.loc[
                annotation_df["id"] == comment_id, binary_category_name
            ] = category
            annotation_df.loc[annotation_df["id"] == comment_id, "備考"] = reason
    annotation_df.to_csv(LLM_ANOTATION_CSV_PATH, index=False, encoding="utf-8_sig")


def split_dataset_csv():
    subcategory_annotation_csv_path = (
        Config.target_base_dir / "1_10_target_prompt_付加的情報のみ抽出.csv"
    )
    df = pd.read_csv(subcategory_annotation_csv_path)
    # split samples into fewshot : val = 20 : 50 from 70 samples
    fewshot_df = df.sample(n=20, random_state=random_seed)
    val_df = df.drop(fewshot_df.index).sample(n=50, random_state=random_seed)
    fewshot_df.to_csv(
        Config.target_base_dir
        / f"{half_number}_{random_seed}_fewshot_subcategory_annotation.csv",
        index=False,
    )
    val_df.to_csv(
        Config.target_base_dir
        / f"{half_number}_{random_seed}_val_subcategory_annotation.csv",
        index=False,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("type", type=str, help="type of function to run")
    args = parser.parse_args()

    if args.type == "create":
        create_tokenized_annotation_csv()
    elif args.type == "add":
        add_column_to_csv()
    elif args.type == "dump":
        fill_csv_from_json()
    elif args.type == "split":
        split_dataset_csv()
    else:
        raise ValueError(f"Invalid type: {args.type}")
