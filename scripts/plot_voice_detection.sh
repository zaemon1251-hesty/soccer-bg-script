#!/bin/bash

uv run python src/sn_script/plot_voice_detection.py \
    --game "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"  \
    --SoccerNet_path /Users/heste/workspace/soccernet/SoccerNet_in_lrlab \
    --half 1 \
    --csv_file database/comments/500game_whisper_spans_denoised.csv \
    --plot_output_dir docs \
    --seglen 10 \
    --prefix "seglen10/"
