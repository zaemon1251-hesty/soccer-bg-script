#!/bin/bash

external_dir=/Users/heste/workspace/soccernet/SoccerNet_in_lrlab

input_v3_dir=$external_dir
input_player_master_csv=database/misc/sncaption_players.csv
output_csv_path=database/misc/players_in_frames.csv

uv run python src/sn_script/v3/v3_to_players.py \
    --input_v3_dir $input_v3_dir \
    --input_player_master_csv $input_player_master_csv \
    --output_csv_path $output_csv_path
