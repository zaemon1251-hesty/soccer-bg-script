#!/bin/bash

uv run python src/sn_script/vad2csv.py \
    --game "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"  \
    --half 1 \
    --input_vad annotation_tool/audio/vad.txt \
    --output_csv database/comments/1example_vad_annotation.csv
