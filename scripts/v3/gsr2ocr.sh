#!/bin/bash

gamestate_path="/local/moriy/SoccerNetGS/v3-720p"
output_base_path=/local/moriy/data/SoccerNet""

python src/sn_script/v3/gsr_to_ocr.py \
    --gamestate_path $gamestate_path \
    --output_base_path $output_base_path