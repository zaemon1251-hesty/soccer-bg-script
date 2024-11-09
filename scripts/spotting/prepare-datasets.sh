#!/bin/bash

# uv run python src/sn_script/create_spotting_dataset.py \
#     --version v2 \
#     --stable_csv database/stable/scbi-v2.csv \
#     --output_dir commentary_dataset/

uv run python src/sn_script/spotting/generate_hf_dataset.py \
    --csv_dir Benchmarks/TemporallyAwarePooling/data/ \
    --hf_dataset_dir database/hf_dataset \
    --push \
    --dataset_name zaemon/scbi