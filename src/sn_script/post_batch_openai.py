import os
from datetime import datetime

from loguru import logger
from openai import OpenAI
from openai.types import Batch
from tap import Tap

# ログの設定
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

logger.add(f"logs/post_batch_openai_{time_str}.log")


class PostBatchOpenAIArguments(Tap):
    type: str

    # sendの場合
    llm_batch_jsonl_dir: str = None # 分割されたJSONLファイルのディレクトリ

    # retrieveの場合
    batch_id: str = None # バッチID

    # save_allの場合
    batch_ids: list[str] = [] # バッチIDのリスト
    output_dir: str = "output" # 出力ディレクトリ

    def configure(self) -> None:
        self.add_argument("type", type=str)
        self.add_argument("--batch_ids", nargs="*", type=str)


def send_file_to_openai(file_path):
    """JSONLファイルをOpenAIのバッチAPIに送信"""
    with open(file_path, 'rb') as file:
        try:
            # ファイルをアップロード
            logger.info(f"Uploading {file_path}...")
            batch_input_file = client.files.create(
                file=file,
                purpose='batch'
            )
            logger.info(f"File {file_path} uploaded successfully: {batch_input_file}")

            # バッチジョブを作成
            batch_input_file_id = batch_input_file.id
            logger.info(f"Creating batch job... {batch_input_file_id=}")
            batch_job = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            logger.info(f"Batch job created: {batch_job}")

        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")


def send(args: PostBatchOpenAIArguments):
    # JSONLファイルをすべて取得
    jsonl_files = [
        os.path.join(args.llm_batch_jsonl_dir, f)
        for f in os.listdir(args.llm_batch_jsonl_dir)
        if f.endswith('.jsonl')
    ]

    assert len(jsonl_files) > 0

    #ファイルを送信
    for file in jsonl_files:
        send_file_to_openai(file)


def retrieve(args: PostBatchOpenAIArguments):
    batch_job = client.batches.retrieve(args.batch_id)
    logger.info(batch_job)


def save_all(args: PostBatchOpenAIArguments):
    for batch_id in args.batch_ids:
        batch_job = client.batches.retrieve(batch_id)
        save_output_file(batch_job, save_dir=args.output_dir)


def save_output_file(batch_job: Batch, save_dir: str = "."):
    """バッチジョブの出力ファイルを保存"""
    content = client.files.content(batch_job.output_file_id).content
    output_file_path = os.path.join(save_dir, f"{batch_job.id}.jsonl")
    with open(output_file_path, 'wb') as file:
        file.write(content)
    logger.info(f"Output file saved: {output_file_path}")


if __name__ == "__main__":
    args = PostBatchOpenAIArguments().parse_args()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    if args.type == "send":
        send(args)
    elif args.type == "retrieve":
        retrieve(args)
    elif args.type == "save_all":
        save_all(args)
    else:
        raise ValueError(f"Invalid type: {args.type}")

