#!/bin/bash

uv run python src/sn_script/create_spotting_dataset.py \
    --version v2 \
    --stable_csv database/stable/scbi-v2.csv \
    --output_dir commentary_dataset/
