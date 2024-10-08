#!/bin/bash

SoccerNet_path=/Users/heste/workspace/soccernet/SoccerNet_in_lrlab

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1

# csv to plot 関数を定義
csv_to_plot() {
    SUFFIX=$1;
    echo "Plotting: $SUFFIX"
    uv run python src/sn_script/plot_voice_detection.py \
        --game "$GAME"  \
        --SoccerNet_path "$SoccerNet_path" \
        --half $HALF \
        --csv_file database/comments/1example$SUFFIX.csv \
        --plot_output_dir docs \
        --seglen 60 \
        --prefix "seglen60/$SUFFIX/"
}

### large-v2
MODEL="faster-whisper-large-v2"

# デフォルト
SUFFIX="_$MODEL"
csv_to_plot "$SUFFIX"

# v2 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
csv_to_plot "$SUFFIX"

# v2 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
csv_to_plot "$SUFFIX"

# v2 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
csv_to_plot "$SUFFIX"

# v2 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
csv_to_plot "$SUFFIX"


### large-v3
MODEL="faster-whisper-large-v3"

# v3 デフォルトパラメータ
SUFFIX="_$MODEL"
csv_to_plot "$SUFFIX"

# v3 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
csv_to_plot "$SUFFIX"

# v3 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
csv_to_plot "$SUFFIX"

# v3 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
csv_to_plot "$SUFFIX"

# v3 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
csv_to_plot "$SUFFIX"
