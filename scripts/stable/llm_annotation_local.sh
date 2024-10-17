#!/bin/bash

DATA_DIR=.

SUFFIX=_stable_en

# データのクリーニング
src="$DATA_DIR/comments/sample_games$SUFFIX.csv"
dst="$DATA_DIR/database/denoised/500game$SUFFIX.csv"
uv run python src/sn_script/csv_utils.py \
    clean \
    --input_csv $src \
    --output_csv $dst

# データクリーニング後のcsvを、llm_readyディレクトリにコピー
src=$dst
dst="$DATA_DIR/database/llm_ready/500game$SUFFIX.csv"
uv run python src/sn_script/csv_utils.py \
    denoised_to_llm_ready \
    --input_csv $src \
    --output_csv $dst

# # # llmアノテーション実行
llm_log_jsonl="$DATA_DIR/database/log_jsonline/500game$SUFFIX.jsonl"
src=$dst
all_csv=$src
prompt_yaml="$DATA_DIR/resources/classify_comment.yaml"
uv run python src/sn_script/llm_anotator.py \
    --llm_ready_path $src \
    --llm_log_jsonl_path $llm_log_jsonl \
    --all_csv_path $all_csv \
    --prompt_yaml_path $prompt_yaml \
    --model_type "gpt-3.5-turbo-1106" \
    --batch

# # jsonlファイルを移動
llm_batch_jsonl="$DATA_DIR/database/batch_jsonline/500game$SUFFIX.jsonl"
mv $llm_log_jsonl $llm_batch_jsonl

# # jsonl 分割 (3万)　linuxコマンドでsplit 拡張子をjsonlにする
llm_batch_jsonl_dir="$DATA_DIR/database/batch_jsonline/500game$SUFFIX"

split -l 30000 $llm_batch_jsonl $llm_batch_jsonl_dir/500game$SUFFIX-
# 拡張子をjsonlに変更
e$SUFFIX-*; do
    mv "$f" "${f}.jsonl"
done

# # LLM自動ラベル付けbatch実行
uv run python src/sn_script/post_batch_openai.py \
    send \
    --llm_batch_jsonl_dir $llm_batch_jsonl_dir

# batch_idsをログから取って、OpenAIから実行結果ファイルを取得
output_dir="$DATA_DIR/database/batch_jsonline/output"
mkdir -p $output_dir
batch_ids=(
    batch_6710e65ecaa48190884ee87f60e61c9e
    batch_6710e6c81d38819087f421d092ad645d
    batch_6710e6e57d408190ad490bcb02554e2a
    batch_6710e74f6d3481909a8df66fe44e869c
    batch_6710e7b8eb748190a06a9d70a54db543
    batch_6710e8220a6081909a49c6346fde29ff
    batch_6710e88c79848190bc26310250c52c07
    batch_6710e8f580788190baca2578a41fa94c
    batch_6710e95d3ac88190be18fcc15037715c
    batch_6710e9c64eb08190a3c1127db034b515
    batch_6710ea2de21c81908d5c5bd5df5135e4
    batch_6710ea97c8f48190b2c6364725d8b6ec
)
uv run python src/sn_script/post_batch_openai.py \
    save_all \
    --batch_ids ${batch_ids[@]} \
    --output_dir $output_dir

# LLM自動ラベル付け結果を、元のデータにマージ
src=$src
batch_jsonl_dir=$output_dir
dst="$DATA_DIR/database/llm_annotation/500game$SUFFIX.csv"
uv run python src/sn_script/csv_utils.py \
    marge_result \
    --input_csv $src \
    --batch_result_dir $batch_jsonl_dir \
    --output_csv $dst

# 名前を変更してコピー
scbi_csv="$DATA_DIR/database/stable/scbi-v2.csv"
cp $dst $scbi_csv
