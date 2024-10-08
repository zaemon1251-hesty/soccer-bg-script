#!/bin/bash

eval() {
    # $1がwhisperなら、predictionファイル名が database/comments/500game_whisper_spans_denoised
    # であることに注意
    SUFFIX=$1;

    if [ "$SUFFIX" = "whisper" ]; then
        input_prediction_csv="database/comments/500game_whisper_spans_denoised.csv"
    else
        input_prediction_csv="database/comments/1example$SUFFIX.csv"
    fi

    echo "Evaluating: $SUFFIX"
    uv run python src/sn_script/evaluate_vad.py \
        --game "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace"  \
        --half 1 \
        --input_annotation_csv database/comments/1example_vad-long_annotation.csv \
        --input_prediction_csv $input_prediction_csv
}


### whisper
SUFFIX="whisper"
eval "$SUFFIX"

### faster-whisper
## large-v2
MODEL="faster-whisper-large-v2"

# デフォルト
SUFFIX="_$MODEL"
eval "$SUFFIX"

# v2 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
eval "$SUFFIX"

# v2 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
eval "$SUFFIX"

# v2 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
eval "$SUFFIX"

# v2 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
eval "$SUFFIX"


## large-v3
MODEL="faster-whisper-large-v3"

# v3 デフォルトパラメータ
SUFFIX="_$MODEL"
eval "$SUFFIX"

# v3 threshold = 0.3
SUFFIX="_$MODEL-threshold-0.3"
eval "$SUFFIX"

# v3 min_silence_duration_ms = 500
SUFFIX="_$MODEL-min_silence_duration_ms-500"
eval "$SUFFIX"

# v3 min_silence_duration_ms = 250
SUFFIX="_$MODEL-min_silence_duration_ms-250"
eval "$SUFFIX"

# v3 speech_pad_ms = 200
SUFFIX="_$MODEL-speech_pad_ms-200"
eval "$SUFFIX"


### whisperx
## large-v2
MODEL="whisperx-large-v2"
for CHUNK_SIZE in 10 20 30 40 50 60
do
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    eval "$SUFFIX"
done

## large-v3
MODEL="whisperx-large-v3"
for CHUNK_SIZE in 10 20 30 40 50 60
do
    SUFFIX="_$MODEL-chunk_size-$CHUNK_SIZE"
    eval "$SUFFIX"
done
