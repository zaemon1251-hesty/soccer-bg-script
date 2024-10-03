#!/bin/bash

uv run python src/sn_script/whisper2csv.py \
    --game "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"  \
    --half 1 \
    --input_json database/sn-caption-data/1_224p_faster_whisper.json \
    --output_csv 1example_faster_whisper.csv
