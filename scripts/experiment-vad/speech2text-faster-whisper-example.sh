#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`

GAME="england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"

### large-v2
MODEL="faster-whisper-large-v2"

# v2 デフォルトパラメータ
SUFFIX="_$MODEL"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX"

# v2 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --threshold 0.3

# v2 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --min_silence_duration_ms 500

# v2 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --min_silence_duration_ms 250

# v2 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --speech_pad_ms 200

### large-v3
MODEL="faster-whisper-large-v3"

# v3 デフォルトパラメータ
SUFFIX="_$MODEL"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX"

# v3 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --threshold 0.3

# v3 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --min_silence_duration_ms 500

# v3 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --min_silence_duration_ms 250

# v3 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
python src/sn_script/speech2text.py \
    --target_game "$GAME" \
    --model $MODEL \
    --suffix "$SUFFIX" \
    --speech_pad_ms 200
