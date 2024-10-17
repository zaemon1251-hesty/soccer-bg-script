#!/bin/bash

DATA_DIR=.

SUFFIX=_stable_en

# データクリーニング後のcsvを、llm_readyディレクトリにコピー
src="$DATA_DIR/database/denoised/500game$SUFFIX.csv"
dst="$DATA_DIR/database/llm_ready/500game$SUFFIX.csv"
uv run python src/sn_script/csv_utils.py \
    denoised_to_llm_ready \
    --input_csv $src \
    --output_csv $dst

# # llmアノテーション実行
llm_log_jsonl="$DATA_DIR/database/log_jsonline/500game$SUFFIX.jsonl"
src=$dst
dst="$DATA_DIR/database/llm_annotation/500game$SUFFIX.csv"
all_csv=$src
prompt_yaml="$DATA_DIR/resources/classify_comment.yaml"
uv run python src/sn_script/llm_anotator.py \
    --llm_ready_path $src \
    --llm_annotation_path $dst \
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
mkdir -p $llm_batch_jsonl_dir
split -l 30000 $llm_batch_jsonl $llm_batch_jsonl_dir/500game$SUFFIX-
# 拡張子をjsonlに変更
for f in $llm_batch_jsonl_dir/500game$SUFFIX-*; do
    mv "$f" "${f}.jsonl"
done

# LLM自動ラベル付け実行
uv run python src/sn_script/post_batch_openai.py \
    send \
    --llm_batch_jsonl_dir $llm_batch_jsonl_dir
