#!/bin/bash

# プロンプト調査
# 事例は固定 ラベルは 5:2 の比率
# 文言は変えず、Include, Not Include だけを変える


suffix=20241016-exp004

# # llmアノテーション用のデータcsvを準備
src="database/llm_annotation/gpt-3.5-turbo-1106_42_1_llm_annotation-20231217.csv"
dst="database/llm_ready/llm_annotation-$suffix.csv"
uv run python src/sn_script/csv_utils.py \
    denoised_to_llm_ready \
    --input_csv $src \
    --output_csv $dst

# llmアノテーション実行
llm_log_jsonl="database/log_jsonline/llm_annotation-$suffix.jsonl"
touch $llm_log_jsonl

src=$dst
dst="database/llm_annotation/llm_annotation-$suffix.csv"
all_csv="database/denoised/denoised_1_tokenized_224p_all.csv"
prompt_yaml="resources/classify_comment-include-only.yaml"
uv run python src/sn_script/llm_anotator.py \
    --llm_ready_path $src \
    --llm_annotation_path $dst \
    --llm_log_jsonl_path $llm_log_jsonl \
    --all_csv_path $all_csv \
    --prompt_yaml_path $prompt_yaml \
    --model_type "gpt-3.5-turbo-1106"

# 評価
human_csv="database/annotations/42_1_moriy_annotation_preprocessed.csv"
llm_csv="$dst"
uv run python src/sn_script/evaluate_llm_annotation.py \
    --target binary \
    --human_csv $human_csv \
    --llm_csv $llm_csv

# 結果
# 'accuracy': {'accuracy': 0.7}, 'precision': {'precision': 0.2631578947368421}, 'recall': {'recall': 0.8333333333333334}, 'f1': {'f1': 0.39999999999999997}},
# 低い。Include, Not Include は、yes no よりも適していない?