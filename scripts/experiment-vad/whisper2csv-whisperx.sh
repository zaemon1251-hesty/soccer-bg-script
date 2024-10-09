#!/bin/bash

SoccerNet_path="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"
GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1
SIZE_SUF="_224p"

json_to_csv() {
    SUFFIX=$1
    JSON_PATH="$SoccerNet_path/$GAME/$HALF$SIZE_SUF$SUFFIX.json"
    uv run python src/sn_script/whisper2csv.py \
        --game "$GAME"  \
        --half "$HALF" \
        --input_json "$JSON_PATH" \
        --output_csv "database/comments/1example$SUFFIX.csv"
}

## large-v2
MODEL="whisperx-large-v2"
# v2 デフォルトパラメータ
SUFFIX="_$MODEL"
json_to_csv "$SUFFIX"

# v2 paper vad param
# SUFFIX="_$MODEL-paper_vad_param"
# json_to_csv "$SUFFIX"

# v2 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    json_to_csv "$SUFFIX"
done

## large-v3
MODEL="whisperx-large-v3"

# v3 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    json_to_csv "$SUFFIX"
done

