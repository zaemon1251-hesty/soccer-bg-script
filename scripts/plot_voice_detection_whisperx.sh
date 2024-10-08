#!/bin/bash

SoccerNet_path="/Users/heste/workspace/soccernet/SoccerNet_in_lrlab"

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1
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


## large-v2
MODEL="whisperx-large-v2"

# v2 paper vad param
# SUFFIX="_$MODEL_paper_vad_param"
# csv_to_plot "$SUFFIX"

# v2 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    csv_to_plot "$SUFFIX"
done



## large-v3
MODEL="whisperx-large-v3"

# v3 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    csv_to_plot "$SUFFIX"
done

