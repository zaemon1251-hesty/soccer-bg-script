#!/bin/bash

SoccerNet_path="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"
SIZE_SUF="_224p"

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1

# json_to_csv 関数を定義
json_to_csv() {
    SUFFIX=$1
    JSON_PATH="$SoccerNet_path/$GAME/$HALF$SIZE_SUF$SUFFIX.json"
    uv run python src/sn_script/whisper2csv.py \
        --game "$GAME"  \
        --half "$HALF" \
        --input_json "$JSON_PATH" \
        --output_csv "database/comments/1example$SUFFIX.csv"
}

### large-v2
MODEL="faster-whisper-large-v2"

# デフォルト
SUFFIX="_$MODEL"
json_to_csv "$SUFFIX"

# v2 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
json_to_csv "$SUFFIX"

# v2 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
json_to_csv "$SUFFIX"

# v2 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
json_to_csv "$SUFFIX"

# v2 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
json_to_csv "$SUFFIX"


### large-v3
MODEL="faster-whisper-large-v3"

# v3 デフォルトパラメータ
SUFFIX="_$MODEL"
json_to_csv "$SUFFIX"

# v3 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
json_to_csv "$SUFFIX"

# v3 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
json_to_csv "$SUFFIX"

# v3 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
json_to_csv "$SUFFIX"

# v3 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
json_to_csv "$SUFFIX"
