#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

SUFFIX=_stable_en


################ sppech2text ################
## whisperx large-v3
MODEL=whisperx-large-v3
# パラメータ
CHUNK_SIZE=20
# 書き起こし実行
python src/sn_script/speech2text.py \
    --half 1 \
    --model $MODEL \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN \
    --chunk_size $CHUNK_SIZE \
    --device_index 0 \
    --task translate

python src/sn_script/speech2text.py \
    --half 2 \
    --model $MODEL \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN \
    --chunk_size $CHUNK_SIZE \
    --device_index 2 \
    --task translate


################ whisper2csv ################
DATA_DIR="/raid_elmo/home/lr/moriy/SoccerNet/commentary_analysis"
python src/sn_script/whisper2csv.py \
    --task to_csv_stable \
    --suffix "$SUFFIX" \
    --output_csv "$DATA_DIR/comments/sample_games$SUFFIX.csv"


################ llm_annotator ################
./scripts/stable/llm_annotation_local.sh
