from __future__ import annotations
from openai import OpenAI
import json
import os
import pandas as pd
from collections import namedtuple
import yaml
from loguru import logger
from datetime import datetime
from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm


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

# pandasのprogress_applyを使うために必要
tqdm.pandas()


# プロンプト作成用の引数
PromptArgments = namedtuple(
    "PromptArgments", ["id", "comment", "game", "previous_comments"]
)


# ALL_CSV_PATH = Config.target_base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
ALL_CSV_PATH = (
    Config.target_base_dir / f"500game_denoised_{half_number}_tokenized_224p_all.csv"
)

PROMPT_YAML_PATH = Config.target_base_dir.parent / "resources" / "classify_comment.yaml"
SUBCATEGORY_YAML_PATH = (
    Config.target_base_dir.parent / "resources" / "classify_comment-subcategory.yaml"
)

all_comment_df = pd.read_csv(ALL_CSV_PATH)
# load yaml
binary_prompt_config = yaml.safe_load(open(PROMPT_YAML_PATH, "r"))
subcategory_prompt_config = yaml.safe_load(open(SUBCATEGORY_YAML_PATH, "r"))

if model_type == "meta-llama/Llama-2-70b-chat-hf":
    # use local llama model
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_type,
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


def main(target: str = "binary"):
    """LLMによるアノテーションを行う

    以下の2つがグローバル変数として定義されていることが前提
    - LLM_ANOTATION_CSV_PATH
    - LLM_ANNOTATION_JSONL_PATH  # ストリームで保存するためのjsonlファイル
    """
    annotation_df = pd.read_csv(LLM_ANOTATION_CSV_PATH)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.add(
        "logs/llm_anotator_{time}.log".format(time=time_str),
    )
    logger.info(f"{model_type=}")
    logger.info(f"{random_seed=}")
    logger.info(f"{half_number=}")

    def annotate_category_binary(row):
        """_summary_

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
            with open(LLM_ANNOTATION_JSONL_PATH, "a") as f:
                result["comment_id"] = comment_id
                json.dump(result, f)
                f.write("\n")

            return category, reason
        except Exception as e:
            logger.error(f"comment_id={comment_id} couldn't be annotated.")
            logger.error(e)
            return None, None

    def annotate_subcategory(row):
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
        if not pd.isnull(row[category_name]):
            return row[subcategory_name]
        try:
            binary_cateogry, binary_reason = annotate_category_binary(row)
            binary_category = int(binary_cateogry)
            if binary_category == 1:
                # 付加的情報が含まれている場合
                subcategory_result = classify_comment(
                    model_type, comment_id, "subcategory"
                )

                # jsonl形式でアノテーション結果を保存する
                with open(LLM_ANNOTATION_JSONL_PATH, "a") as f:
                    subcategory_result["comment_id"] = comment_id
                    json.dump(subcategory_result, f)
                    f.write("\n")

                subcategory = subcategory_result.get("category")
                logger.info(f"subcategory:{subcategory}")
                subcategory_reason = subcategory_result.get("reason")
                return subcategory, subcategory_reason
            elif binary_category == 0:
                # 付加的情報が含まれていない場合
                return 0, binary_reason
            else:
                raise ValueError(f"Invalid binary_category:{binary_category}")

        except Exception as e:
            logger.error(f"comment_id={comment_id} is not annotated.")
            logger.error(e)
            return None, None

    if target == "binary":
        logger.info("Start binary classification.")
        annotation_df[[binary_category_name, "備考"]] = annotation_df.progress_apply(
            lambda r: annotate_category_binary(r),
            axis=1,
            result_type="expand",
        )
        annotation_df.to_csv(LLM_ANOTATION_CSV_PATH, index=False)
        logger.info(f"Done binary classification. Saved to {LLM_ANOTATION_CSV_PATH}")
    elif target == "subcategory":
        logger.info("Start subcategory classification.")
        annotation_df[[subcategory_name, "備考_2"]] = annotation_df.progress_apply(
            lambda r: annotate_subcategory(r),
            axis=1,
            result_type="expand",
        )
        annotation_df.to_csv(LLM_ANOTATION_CSV_PATH, index=False)
        logger.info(
            f"Done subcategory classification. Saved to {LLM_ANOTATION_CSV_PATH}"
        )
    else:
        raise ValueError(f"Invalid target:{target}")
    return None


def classify_comment(model_type: str, comment_id: int, target="binary") -> dict:
    if target == "binary":
        prompt_config = binary_prompt_config
    elif target == "subcategory":
        prompt_config = subcategory_prompt_config
    else:
        raise ValueError(f"Invalid target:{target}")

    messages = get_messages(comment_id, prompt_config)

    if model_type == "meta-llama/Llama-2-70b-chat-hf":
        return _classify_comment_with_llama(messages)

    return _classify_comment_with_openai(messages)


def _classify_comment_with_openai(messages: list[str]) -> dict:
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


def get_messages(comment_id: int, prompt_config: dict) -> list[str]:
    messages = []

    if comment_id not in all_comment_df.index:
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

    target_prompt = create_target_prompt(comment_id)
    messages.append(
        {
            "role": "user",
            "content": target_prompt,
        }
    )
    return messages


def create_target_prompt(comment_id: int) -> str:
    target_comment = all_comment_df.iloc[comment_id]
    context_length = 2

    previous_comments = (
        all_comment_df[
            (all_comment_df["game"] == target_comment["game"])
            & (all_comment_df.index < comment_id)
        ]
        .tail(context_length)["text"]
        .tolist()
    )

    target_prompt_args = PromptArgments(
        comment_id, target_comment["text"], target_comment["game"], previous_comments
    )

    message = _create_target_prompt(target_prompt_args)
    return message


def _create_target_prompt(prompt_args: PromptArgments) -> str:
    """分類対象のコメントに関するプロンプトを作成する"""

    message = f"""
- id => {prompt_args.id}
- game => {prompt_args.game}
- previous comments => {" ".join(prompt_args.previous_comments)}
- comment => {prompt_args.comment}
"""

    return message


def output_target_prompt(ANNOTATION_CSV_PATH, TARGET_PROMPT_CSV_PATH, filter=False):
    annotation_df = pd.read_csv(ANNOTATION_CSV_PATH)

    # 付加的情報が含まれているコメントのみを抽出する
    if filter:
        annotation_df = annotation_df[annotation_df[binary_category_name] == 1]

    targe_prompt_list = []
    for comment_id in annotation_df["id"]:
        targe_prompt_list.append(create_target_prompt(comment_id))

    with open(TARGET_PROMPT_CSV_PATH, "w") as f:
        f.write("\n".join(targe_prompt_list))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "type",
        type=str,
        help="type of function to run",
        choices=["main", "output_target_prompt"],
    )
    parser.add_argument(
        "--target",
        type=str,
        help="target for llm annotation category",
        default="binary",
        choices=["binary", "subcategory"],
    )
    parser.add_argument(
        "--prefix", type=str, help="file name prefix for llm_annotation_df", default=""
    )
    parser.add_argument(
        "--filter",
        help="filter comments that contain additional information",
        action="store_true",
    )
    args = parser.parse_args()

    if args.type == "main":
        if args.target == "binary":
            LLM_ANOTATION_CSV_PATH = (
                Config.target_base_dir
                / f"{model_type}_{random_seed}_{half_number}_llm_annotation-20240104a.csv"
                # f"{model_type}_500game_{half_number}_llm_annotation.csv"
            )
            LLM_ANNOTATION_JSONL_PATH = (
                Config.target_base_dir
                / f"{model_type}_{random_seed}_{half_number}_llm_annotation.jsonl"
            )
        elif args.target == "subcategory":
            LLM_ANOTATION_CSV_PATH = (
                Config.target_base_dir / f"{args.prefix}_subcategory_llm_annotation.csv"
            )
            LLM_ANNOTATION_JSONL_PATH = (
                Config.target_base_dir
                / f"{args.prefix}_{model_type}_{random_seed}_{half_number}_subcategory_llm_annotation.jsonl"
            )
        else:
            raise ValueError(f"Invalid target:{args.target}")
        main(args.target)
    elif args.type == "output_target_prompt":
        # ChatGPT用のプロンプトを作成する
        ANNOTATION_CSV_PATH = (
            Config.target_base_dir
            / "20240306_1_10_supplementary_comments_annotation.csv"
        )
        TARGET_PROMPT_CSV_PATH = (
            Config.target_base_dir.parent
            / "resources"
            / f"{random_seed}_{half_number}_target_prompt_{args.filter}.txt"
        )
        output_target_prompt(ANNOTATION_CSV_PATH, TARGET_PROMPT_CSV_PATH, args.filter)
    else:
        raise ValueError(f"Invalid type:{args.type}")
