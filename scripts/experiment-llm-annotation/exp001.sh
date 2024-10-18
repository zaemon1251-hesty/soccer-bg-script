#!/bin/bash

# プロンプト変更の影響を調査
# 事例は固定　ラベルは5:2

# (1) 0,1で答える指示から、Yes or Noで答える指示にした
# (2) supplementary informationから 指示の文言を変更した


suffix=20241016-exp001

# # llmアノテーション用のデータcsvを準備
src="database/llm_annotation/gpt-3.5-turbo-1106_42_1_llm_annotation-20231217.csv"
dst="database/llm_ready/llm_annotation-$suffix.csv"
uv run python src/sn_script/csv_utils.py \
    denoised_to_llm_ready \
    --input_csv $src \
    --output_csv $dst

# # llmアノテーション実行
llm_log_jsonl="database/log_jsonline/llm_annotation-$suffix.jsonl"
touch $llm_log_jsonl

# src=$dst
dst="database/llm_annotation/llm_annotation-$suffix.csv"
all_csv="database/denoised/denoised_1_tokenized_224p_all.csv"
prompt_yaml="resources/classify_comment-yesno.yaml"
uv run python src/sn_script/llm_anotator.py \
    --llm_ready_path $src \
    --llm_annotation_path $dst \
    --llm_log_jsonl_path $llm_log_jsonl \
    --all_csv_path $all_csv \
    --prompt_yaml_path $prompt_yaml \
    --model_type "gpt-3.5-turbo-1106"

# 評価
human_csv="database/annotations/42_1_moriy_annotation_preprocessed.csv"
llm_csv=$dst
uv run python src/sn_script/evaluate_llm_annotation.py \
    --target binary \
    --human_csv $human_csv \
    --llm_csv $llm_csv

# ターゲットのハイフンを忘れていた場合
# 'category': {'exact_match': None, 'accuracy': {'accuracy': 0.8}, 'precision': {'precision': 0.35714285714285715}, 'recall': {'recall': 0.8333333333333334}, 'f1': {'f1': 0.5}}, 'subcategory': None}
# 'category': {'exact_match': None, 'accuracy': {'accuracy': 0.81}, 'precision': {'precision': 0.36}, 'recall': {'recall': 0.75}, 'f1': {'f1': 0.48648648648648657}}, 'subcategory': None}
# ターゲットのハイフンを入れた場合

# 低い
