#!/bin/bash

DATA_DIR="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"
GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1
SIZE_SUF="_224p"

# v2
SUFFIX="_whisperx_large_v2"
JSON_PATH="$DATA_DIR/$GAME/$HALF$SIZE_SUF$SUFFIX.json"
uv run python src/sn_script/whisper2csv.py \
    --game "$GAME"  \
    --half "$HALF" \
    --input_json "$JSON_PATH" \
    --output_csv "database/comments/1example$SUFFIX.csv"

# v3
SUFFIX="_whisperx_large_v3"
JSON_PATH="$DATA_DIR/$GAME/$HALF$SIZE_SUF$SUFFIX.json"
uv run python src/sn_script/whisper2csv.py \
    --game "$GAME"  \
    --half "$HALF" \
    --input_json "$JSON_PATH" \
    --output_csv "database/comments/1example$SUFFIX.csv"

# v2 paper vad param
SUFFIX="_whisperx_large_v2_paper_vad_param"
JSON_PATH="$DATA_DIR/$GAME/$HALF$SIZE_SUF$SUFFIX.json"
uv run python src/sn_script/whisper2csv.py \
    --game "$GAME"  \
    --half "$HALF" \
    --input_json "$JSON_PATH" \
    --output_csv "database/comments/1example$SUFFIX.csv"
