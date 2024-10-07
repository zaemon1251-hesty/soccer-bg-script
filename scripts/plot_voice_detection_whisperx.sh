#!/bin/bash

SoccerNet_path="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1
SUFFIX="_whisperx_large_v2"
SUFFIX="_whisperx_large_v2_paper_vad_param"

# large-v2
uv run python src/sn_script/plot_voice_detection.py \
    --game "$GAME"  \
    --SoccerNet_path "$SoccerNet_path" \
    --half $HALF \
    --csv_file database/comments/1example$SUFFIX.csv \
    --plot_output_dir docs \
    --seglen 60 \
    --prefix "seglen60/$SUFFIX/"

# large-v3
# SUFFIX="_whisperx_large_v3"
# uv run python src/sn_script/plot_voice_detection.py \
#     --game "$GAME"  \
#     --SoccerNet_path "$SoccerNet_path" \
#     --half $HALF \
#     --csv_file database/comments/1example$SUFFIX.csv \
#     --plot_output_dir docs \
#     --seglen 60 \
#     --prefix "seglen60/$SUFFIX/"

