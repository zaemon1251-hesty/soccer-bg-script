#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"

python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model faster-whisper \
    --suffix "_faster_whisper"
