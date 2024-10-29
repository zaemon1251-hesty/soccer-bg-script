#!/bin/bash

uv run python src/sn_script/spotting/generate_spotting_results.py \
    --input_csv /Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/test.csv \
    --output_json_dir database/result_spotting/

