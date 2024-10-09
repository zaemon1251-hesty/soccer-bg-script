#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

## large-v3
MODEL=whisperx-large-v3

# パラメータ
CHUNK_SIZE=20

# 実行
SUFFIX=_stable_version2

python src/sn_script/speech2text.py \
    --target_game "all" \
    --half 1 \
    --model $MODEL \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN \
    --chunk_size $CHUNK_SIZE

python src/sn_script/speech2text.py \
    --target_game "all" \
    --half 2 \
    --model $MODEL \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN \
    --chunk_size $CHUNK_SIZE