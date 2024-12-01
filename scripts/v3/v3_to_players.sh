#!/bin/bash

external_dir=/raid_elmo/home/lr/moriy/SoccerNet

metadata_dir=$external_dir/commentary_analysis

input_v3_dir=$external_dir
input_player_master_csv=$metadata_dir/misc/sncaption_players.csv
side_team_map_csv=$metadata_dir/misc/side_to_team.csv
output_csv_path=$metadata_dir/misc/players_in_frames.csv

python src/sn_script/v3/v3_to_players.py \
    --input_v3_dir $input_v3_dir \
    --input_player_master_csv $input_player_master_csv \
    --output_csv_path $output_csv_path \
    --side_team_map_csv $side_team_map_csv
