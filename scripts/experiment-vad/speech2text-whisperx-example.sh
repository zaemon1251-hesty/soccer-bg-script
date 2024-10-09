#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
HALF=1

## large-v2
MODEL=whisperx-large-v2

# v2 vad_onset = 0.767, vad_offset = 0.377 論文掲載パラメータ
# SUFFIX=_whisperx_large_v2_paper_vad_param
# python src/sn_script/speech2text.py \
#     --target_game "$GAME" \
#     --half $HALF \
#     --model whisperx-large-v2 \
#     --suffix $SUFFIX \
#     --hf_token $HF_TOKEN \
#     --vad_onset 0.767 \
#     --vad_offset 0.377

# v2 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX=_$MODEL-chunk_size-$CHUNK_SIZE
    python src/sn_script/speech2text.py \
        --target_game "$GAME" \
        --half $HALF \
        --model $MODEL \
        --suffix $SUFFIX \
        --hf_token $HF_TOKEN \
        --chunk_size $CHUNK_SIZE
done

## large-v3
MODEL=whisperx-large-v3

# v3 chunk_size = 10, 20, 30
for CHUNK_SIZE in 10 20 30
do
    echo "chunk_size: $CHUNK_SIZE"
    SUFFIX=_$MODEL-chunk_size-$CHUNK_SIZE
    python src/sn_script/speech2text.py \
        --target_game "$GAME" \
        --half $HALF \
        --model $MODEL \
        --suffix $SUFFIX \
        --hf_token $HF_TOKEN \
        --chunk_size $CHUNK_SIZE
done
