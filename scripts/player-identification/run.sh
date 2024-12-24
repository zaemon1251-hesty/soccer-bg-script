#!/bin/bash

csv_path="database/misc/RAGモジュール出力サンプル-13090437a14481f485ffdf605d3408cd.csv"

python src/sn_script/video2images.py \
    --SoccerNet_path /local/moriy/SoccerNet  \
    --output_base_path /local/moriy/SoccerNetGS/rag-eval  \
    --target_game "" --resolution 720p --fps 25 --threads 4 \
    --input_csv_path $csv_path


python src/sn_script/video2images.py \
    --SoccerNet_path /local/moriy/SoccerNet  \
    --output_base_path /local/moriy/SoccerNetGS/rag-eval  \
    --target_game "" --resolution 720p --fps 25 --threads 4 \
    --output video \
    --input_csv_path $csv_path
