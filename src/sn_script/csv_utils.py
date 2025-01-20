from __future__ import annotations

import csv
import json
import os
import time
from functools import wraps
from pathlib import Path

import pandas as pd
from loguru import logger
from tap import Tap

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


class CsvUtilsArguments(Tap):
    type: str
    input_csv: str = None
    output_csv: str = None
    batch_result_dir: str = None

    def configure(self) -> None:
        self.add_argument("type", type=str, help="type of function to run")
        self.add_argument("--batch_result_dir", type=str, default=None, help="directory of batch results (jsonl files)")


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


def dump_filled_comments(dump_file_path: str, half_number: int):
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
    filled_big_class_df.to_csv(dump_file_path, index=False)


def clean(dump_file_path: str):
    df = pd.read_csv(dump_file_path)
    print(df.columns)
    df.drop(columns=["Unnamed: 6", "Unnamed: 7"], inplace=True)
    df.to_csv(dump_file_path, index=False)


def add_column_to_csv(annotation_csv_path: str):
    annotation_df = pd.read_csv(annotation_csv_path)

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
    annotation_df = annotation_df.reindex(columns=column_order)
    annotation_df.to_csv(annotation_csv_path, index=False, encoding="utf-8_sig")


def create_tokenized_annotation_csv(all_csv_path: str, annotation_csv_path: str, number_of_comments: int = 100):
    all_game_df = pd.read_csv(all_csv_path)

    annotation_df = all_game_df.sample(n=number_of_comments, random_state=random_seed)
    annotation_df.to_csv(annotation_csv_path, index=False, encoding="utf-8_sig")


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
    if not isinstance(gametime, str):
        return gametime
    if gametime.count(":") == 2:
        gametime = ":".join(gametime.split(":")[:2])
    m, s = gametime.split(":")
    return int(m) * 60 + int(s)


def fill_csv_from_json(
    llm_annotation_csv_path: str,
    llm_annotation_jsonl_path: str
):
    annotation_df = pd.read_csv(llm_annotation_csv_path)
    annotation_df[binary_category_name] = pd.NA
    annotation_df["備考"] = pd.NA

    with open(llm_annotation_jsonl_path) as f:
        for line in f:
            result: dict = json.loads(line)
            comment_id = result.get("comment_id")
            category = result.get("category")
            reason = result.get("reason")
            annotation_df.loc[
                annotation_df["id"] == comment_id, binary_category_name
            ] = category
            annotation_df.loc[annotation_df["id"] == comment_id, "備考"] = reason
    annotation_df.to_csv(llm_annotation_csv_path, index=False, encoding="utf-8_sig")


def split_dataset_csv(
    subcategory_annotated_csv_path,
    val_csv_path,
    fewshot_csv_path
):
    df = pd.read_csv(subcategory_annotated_csv_path)
    # split samples into fewshot : val = 20 : 50 from 70 samples
    fewshot_df = df.sample(n=20, random_state=random_seed)
    val_df = df.drop(fewshot_df.index).sample(n=50, random_state=random_seed)
    fewshot_df.to_csv(
        fewshot_csv_path,
        index=False,
    )
    val_df.to_csv(
        val_csv_path,
        index=False,
    )


def extract_info_from_jsonl(jsonl_path: str):
    informations = []
    with open(jsonl_path) as file:
        for line in file:
            # 各行をJSONとして読み込む
            data = json.loads(line)

            # "custom_id"を取得
            custom_id = data.get('custom_id')
            custom_id = int(custom_id)

            # "content"フィールドの文字列をデコードしてJSONに変換
            content_str = data['response']['body']['choices'][0]['message']['content']
            content_json = json.loads(content_str)
            reason = content_json.get('reason')
            category = content_json.get('category')
            informations.append({'custom_id': custom_id, '備考': reason, binary_category_name: category})
    return informations


if __name__ == "__main__":
    args: CsvUtilsArguments = CsvUtilsArguments().parse_args()

    if args.type == "create":
        create_tokenized_annotation_csv(args.input_csv, args.output_csv)
    elif args.type == "add":
        add_column_to_csv(args.input_csv)
    elif args.type == "dump":
        fill_csv_from_json(args.input_csv, args.output_csv)
    elif args.type == "split":
        subcategory_annotated_csv_path = Config.target_base_dir / f"{half_number}_{random_seed}_subcategory_annotation.csv"
        val_csv_path = Config.target_base_dir / f"{half_number}_{random_seed}_fewshot_subcategory_annotation.csv"
        fewshot_csv_path = Config.target_base_dir / f"{half_number}_{random_seed}_val_subcategory_annotation.csv"

        split_dataset_csv(subcategory_annotated_csv_path, fewshot_csv_path, val_csv_path)
    elif args.type == "denoised_to_llm_ready":
        input_df = pd.read_csv(args.input_csv)
        input_df = input_df.astype({"id": int})
        input_df["備考"] = pd.NA
        input_df[binary_category_name] = pd.NA
        input_df.to_csv(args.output_csv, index=False)
    elif args.type == "clean":
        # TODO notebookから移植
        raise NotImplementedError()
    elif args.type == "marge_result":
        input_df = pd.read_csv(args.input_csv)
        results = []
        for file_name in os.listdir(args.batch_result_dir):
            jsonl_path = os.path.join(args.batch_result_dir, file_name)
            results.extend(extract_info_from_jsonl(jsonl_path))

        result_df = pd.DataFrame(results)
        input_df.drop(columns=["備考", binary_category_name], inplace=True)
        input_df = input_df.astype({"id": int})
        output_df = pd.merge(input_df, result_df, left_on="id", right_on="custom_id") # idとcustom_idは一対一対応
        output_df.drop(columns=["custom_id"], inplace=True)
        output_df.to_csv(args.output_csv, index=False)
    elif args.type == "sample":
        # trasnlate の評価用にサンプルデータを取得
        # en,es,de,fr を 40ずつ取得
        num_samples = 160
        input_df = pd.read_csv(args.input_csv)
        sample_comment_ids = []
        for lang in ["en", "es", "de", "fr"]:
            sample_num = num_samples // 4
            sample_comment_ids.extend(
                input_df[(input_df["language"] == lang) & (input_df["src_text"].notnull())]
                .sample(n=sample_num, random_state=random_seed)["id"]
                .tolist()
            )
        mask = input_df["id"].isin(sample_comment_ids)
        sample_df = input_df[mask]
        sample_df.to_csv(args.output_csv, index=False)
    else:
        raise ValueError(f"Invalid type: {args.type}")
