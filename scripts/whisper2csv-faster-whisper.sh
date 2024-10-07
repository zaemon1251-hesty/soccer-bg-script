#!/bin/bash

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
JSON_PATH="database/sn-caption-data/1_224p_faster_whisper.json"

uv run python src/sn_script/whisper2csv.py \
    --game $GAME  \
    --half 1 \
    --input_json $JSON_PATH \
    --output_csv 1example_faster_whisper.csv
