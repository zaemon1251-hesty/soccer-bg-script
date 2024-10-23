#!/bin/bash

python src/sn_script/v3/v3_to_gsr.py \
    --SoccerNet_path="/raid_elmo/home/lr/moriy/SoccerNet" \
    --output_base_path "/raid_elmo/home/lr/moriy/SoccerNetGS/v3"

# 計算機サーバーのローカルにコピー
rsync -avz /raid_elmo/home/lr/moriy/SoccerNetGS/v3/ /local/moriy/SoccerNetGS/v3/

