from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import transformers
import yaml
from loguru import logger
from openai import OpenAI
from tap import Tap
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        subcategory_name,
    )
    from sn_script.csv_utils import gametime_to_seconds
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
        binary_category_name,
        subcategory_name,
    )
    from src.sn_script.csv_utils import gametime_to_seconds

# pandasのprogress_applyを使うために必要
tqdm.pandas()

# コマンドライン引数をパースするクラス
class LlmAnotatorArgs(Tap):
    type: Literal["main", "output_target_prompt"] = "main"

    # mainの場合
    target: Literal["binary", "subcategory"] = "binary"
    llm_ready_path: str = None # 入力 path (all_csv_pathの部分集合)
    llm_annotation_path: str = None # 出力 path
    all_csv_path: str = None # すべてのコメントが含まれるCSV
    llm_log_jsonl_path: str = None # ログを保存するjsonlファイルのパス (batchの場合はbathc用のjsonlファイルを保存する)
    prompt_yaml_path: str = None
    model_type: str = "gpt-3.5-turbo-1106"
    batch: bool = False

    # output_target_prompt の場合
    human_annotation_csv_path: str = None
    target_prompt_path: str = None
    filter: bool = False
    csv_mode: bool = False


# プロンプト作成用の引数
@dataclass
class PromptArgments:
    id: int
    comment: str
    game: str
    previous_comments: list[str] = field(default_factory=lambda: [])
    gap: int | None = None
    start: str | None = None
    end: str | None = None


def annotate_category_binary(row, model_type, log_jsonl_path):
    """

    Args:
        row (row object): dataframe single row

    Returns:
        str: binary category
        str: reason for category
    """
    comment_id = row["id"]
    if not pd.isnull(row[binary_category_name]):
        return row[binary_category_name], row["備考"]
    try:
        result = classify_comment(model_type, comment_id, "binary")
        category = result.get("category")
        reason = result.get("reason")

        # jsonl形式で保存する
        with open(log_jsonl_path, "a") as f:
            result["comment_id"] = comment_id
            json.dump(result, f)
            f.write("\n")

        return category, reason
    except Exception as e:
        logger.error(f"comment_id={comment_id} couldn't be annotated.")
        logger.error(e)
        return None, None

def annotate_subcategory(row, model_type, log_jsonl_path):
    """
    処理フロー
    1. まず、付加的情報が含まれてるかどうかを　PROMPT_YAML_PATH のプロンプトを通して確認する
    2. もし付加的情報が含まれていたら、SUBCATEGORY_YAML_PATH　のプロンプトを通してサブカテゴリを予測する
    3. 付加的情報が含まれていなかったら、0を返す

    Args:
        row (row object): dataframe single row

    Returns:
        str: category
        str: reason for category
    """
    comment_id = row["id"]
    try:
        # 付加的情報が含まれている場合
        subcategory_result = classify_comment(model_type, comment_id, "subcategory")

        # jsonl形式でアノテーション結果を保存する
        with open(log_jsonl_path, "a") as f:
            subcategory_result["comment_id"] = comment_id
            json.dump(subcategory_result, f)
            f.write("\n")

        subcategory = subcategory_result.get("subcategory")
        logger.info(f"subcategory:{subcategory}")
        subcategory_reason = subcategory_result.get("reason")
        return subcategory, subcategory_reason

    except Exception as e:
        logger.error(f"comment_id={comment_id} is not annotated.")
        logger.error(e)
        return None, None


def main(args: LlmAnotatorArgs):
    """LLMによるアノテーションを行う

    """
    annotation_df = pd.read_csv(args.llm_ready_path)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.add(
        f"logs/llm_anotator_{time_str}.log",
    )
    logger.info(f"{args.model_type=}")

    if args.target == "binary":
        if args.batch:
            generate_jsonl_for_batch_api(
                annotation_df, args.llm_log_jsonl_path, args.model_type
            )
            logger.info(f"Done generating jsonl for batch api. Saved to {args.llm_log_jsonl_path}")
            return
        logger.info("Start binary classification.")
        annotation_df[[binary_category_name, "備考"]] = annotation_df.progress_apply(
            lambda r: annotate_category_binary(r, args.model_type, args.llm_log_jsonl_path),
            axis=1,
            result_type="expand",
        )
        annotation_df.to_csv(args.llm_annotation_path, index=False)
        logger.info(f"Done binary classification. Saved to {args.llm_annotation_path}")
    elif args.target == "subcategory":
        logger.info("Start subcategory classification.")
        annotation_df[[subcategory_name, "備考_2"]] = annotation_df.progress_apply(
            lambda r: annotate_subcategory(r, args.model_type, args.llm_log_jsonl_path),
            axis=1,
            result_type="expand",
        )
        annotation_df.to_csv(args.llm_annotation_path, index=False)
        logger.info(
            f"Done subcategory classification. Saved to {args.llm_annotation_path}"
        )
    else:
        raise ValueError(f"Invalid target:{args.target}")


def classify_comment(model_type: str, comment_id: int, target="binary") -> dict:
    if target == "binary":
        global binary_prompt_config
        prompt_config = binary_prompt_config
    elif target == "subcategory":
        global subcategory_prompt_config
        prompt_config = subcategory_prompt_config
    else:
        raise ValueError(f"Invalid target:{target}")

    messages = get_messages(comment_id, prompt_config, target)

    if model_type == "meta-llama/Llama-2-70b-chat-hf":
        return _classify_comment_with_llama(messages)

    return _classify_comment_with_openai(messages, model_type)


def _classify_comment_with_openai(messages: list[str], model_type) -> dict:
    completion_params = {
        "model": model_type,
        "messages": messages,
        "n": 1,
        "stop": None,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**completion_params)
        content = response.choices[0].message.content
        if content is None:
            return {}
        else:
            return json.loads(content)
    except Exception as e:
        logger.error(e)
        return {}


def generate_jsonl_for_batch_api(annotation_df: pd.DataFrame, log_jsonl_path: str, model_type) -> str:
    """バッチAPI用のjsonlファイルを生成する"""
    request_jsons = []
    for _, row in annotation_df.iterrows():
        comment_id = row["id"]
        messages = get_messages(comment_id, binary_prompt_config, "binary")

        completion_body = {
            "model": model_type,
            "messages": messages,
            "n": 1,
            "stop": None,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        request_params = {
            "custom_id": f"{comment_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": completion_body,
        }
        request_jsons.append(request_params)

    with open(log_jsonl_path, "w") as f:
        for request_json in request_jsons:
            json.dump(request_json, f)
            f.write("\n")


def _classify_comment_with_llama(messages: list[str]) -> dict:
    # Concatenate all messages into a single string
    input_text = " ".join([message["content"] for message in messages])

    # Generate a response using the pipeline
    response = pipeline(
        input_text,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
    )

    response_text = response[0]["generated_text"] if response else None
    logger.info(f"response_text:{response_text}")

    # Process the response as needed
    if response_text is None:
        return {}
    else:
        return json.loads(response_text)


def get_messages(comment_id: int, prompt_config: dict, target: str) -> list[str]:
    messages = []

    if comment_id not in all_comment_df["id"].values:
        raise ValueError(f"comment_id={comment_id} is not found.")

    description = prompt_config["description"]
    messages.append(
        {
            "role": "system",
            "content": description,
        }
    )

    shots = prompt_config["shots"]
    for shot in shots:
        messages.append(
            {
                "role": "user",
                "content": shot["user"],
            },
        )
        messages.append(
            {
                "role": "assistant",
                "content": shot["assistant"],
            }
        )

    # max_history = 5 & game == game

    target_prompt = create_target_prompt(comment_id, target=target)
    messages.append(
        {
            "role": "user",
            "content": target_prompt,
        }
    )
    return messages


def create_target_prompt(
    comment_id: int,
    csv_mode: bool = False,
    include_debug_info: bool = False,
    target: str = "binary",
) -> str:

    if csv_mode:
        func = get_formatted_comment_csv
    else:
        func = get_formatted_comment

    target_comment_data = all_comment_df.iloc[comment_id]

    context_length = Config.context_length

    previous_comments_data = all_comment_df[
        (all_comment_df["game"] == target_comment_data["game"])
        & (all_comment_df["id"] < comment_id)
    ].tail(context_length)

    previous_comments = previous_comments_data["text"].tolist()

    if len(previous_comments_data) > 0:
        target_comment_start = gametime_to_seconds(target_comment_data["start"])
        previous_comments_start = gametime_to_seconds(
            previous_comments_data.iloc[-1]["start"]
        )
        start_gap_from_previous = max(target_comment_start - previous_comments_start, 0)
    else:
        start_gap_from_previous = None

    target_prompt_args = PromptArgments(
        comment_id,
        target_comment_data["text"],
        target_comment_data["game"],
        previous_comments,
        start_gap_from_previous,
        start=target_comment_data["start"],
        end=target_comment_data["end"],
    )

    message = func(target_prompt_args)
    return message


def get_formatted_comment(prompt_args: PromptArgments) -> str:
    """分類対象のコメントに関するプロンプトを作成する"""

    message = f"""
- game => {prompt_args.game}
- previous comments => {" ".join(prompt_args.previous_comments)}
- comment => {prompt_args.comment}
"""

    return message


def get_formatted_comment_csv(prompt_args: PromptArgments) -> str:
    message = f'''{prompt_args.id},{prompt_args.game},{prompt_args.start},{prompt_args.end},{prompt_args.gap},"{" ".join(prompt_args.previous_comments)}","{prompt_args.comment}"'''  # noqa
    return message


def output_target_prompt(
    annotation_csv_path: Path,
    target_path: Path,
    filter: bool = False,
    csv_mode: bool = False,
    include_debug_info: bool = False,
):
    if csv_mode and target_path.suffix != ".csv":
        raise ValueError(f"Invalid file extension:{target_path.suffix}")

    annotation_df = pd.read_csv(annotation_csv_path)

    # 付加的情報が含まれているコメントのみを抽出する
    if filter:
        annotation_df[binary_category_name] = annotation_df[binary_category_name].astype(int)
        annotation_df = annotation_df[annotation_df[binary_category_name] == 1]

    target_list = []

    # ヘッダ csvモードの場合のみ
    if csv_mode:
        target_list.append(
            "id,game,start,end,gap(sec),prev,target,subcategory"
        )

    for comment_id in annotation_df["id"]:
        target_list.append(
            create_target_prompt(comment_id, csv_mode, include_debug_info)
        )

    with open(target_path, "w") as f:
        f.write("\n".join(target_list))

    return 0


if __name__ == "__main__":

    args = LlmAnotatorArgs().parse_args()

    # load yaml
    if args.target == "binary":
        binary_prompt_config = yaml.safe_load(open(args.prompt_yaml_path))
    elif args.target == "subcategory":
        subcategory_prompt_config = yaml.safe_load(open(args.prompt_yaml_path))
    else:
        pass

    # load all comments
    all_comment_df = pd.read_csv(args.all_csv_path)

    if args.model_type == "meta-llama/Llama-2-70b-chat-hf":
        # use local llama model
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_type,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        client = None
    else:
        tokenizer = None
        pipeline = None

        # use openai api
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
    if args.type == "main":
        main(args)

    elif args.type == "output_target_prompt":
        # ChatGPT用のプロンプトを作成する
        output_target_prompt(
            args.human_annotation_csv_path,
            args.target_prompt_path,
            args.filter,
            args.csv_mode,
            False,
        )
    else:
        raise ValueError(f"Invalid type: {args.type}")
