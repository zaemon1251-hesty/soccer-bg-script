#!/bin/bash

SoccerNet_path="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"
GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1
SIZE_SUF="_224p"

json_to_csv() {
    SUFFIX=$1
    uv run python src/sn_script/whisper2csv.py \
        --task to_csv_stable \
        --suffix "$SUFFIX" \
        --output_csv "database/comments/sample_games$SUFFIX.csv"
}
SUFFIX="_exp_translate"
json_to_csv "$SUFFIX"