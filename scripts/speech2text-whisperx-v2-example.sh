#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"
SUFFIX=_whisperx_large_v2

python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --half 1 \
    --model whisperx-large-v2 \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN

# 出力されたファイルを確認
ls "/raid_elmo/home/lr/moriy/SoccerNet/$GAME"
