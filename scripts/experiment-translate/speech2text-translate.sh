#!/bin/bash

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")')

# 配列として定義
target_games=(
    "spain_laliga/2016-2017/2017-05-14 - 21-00 Las Palmas 1 - 4 Barcelona"
    "spain_laliga/2015-2016/2016-04-02 - 21-30 Barcelona 1 - 2 Real Madrid"
    "spain_laliga/2015-2016/2015-10-04 - 21-30 Atl. Madrid 1 - 1 Real Madrid"
    "italy_serie-a/2014-2015/2015-04-25 - 21-45 Inter 2 - 1 AS Roma"
    "germany_bundesliga/2016-2017/2016-12-03 - 17-30 Dortmund 4 - 1 B. Monchengladbach"
    "germany_bundesliga/2016-2017/2016-11-05 - 17-30 Hamburger SV 2 - 5 Dortmund"
    "italy_serie-a/2016-2017/2016-10-30 - 17-00 AC Milan 1 - 0 Pescara"
    "italy_serie-a/2016-2017/2017-01-21 - 22-45 AC Milan 1 - 2 Napoli"
    "italy_serie-a/2015-2016/2015-08-29 - 21-45 AC Milan 2 - 1 Empoli"
    "italy_serie-a/2015-2016/2015-11-22 - 22-45 Inter 4 - 0 Frosinone"
    "france_ligue-1/2016-2017/2017-01-21 - 19-00 Nantes 0 - 2 Paris SG"
    "europe_uefa-champions-league/2016-2017/2016-09-28 - 21-45 Ludogorets 1 - 3 Paris SG"
    "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United"
    "europe_uefa-champions-league/2016-2017/2016-10-19 - 21-45 Barcelona 4 - 0 Manchester City"
    "europe_uefa-champions-league/2015-2016/2015-11-24 - 22-45 FC Porto 0 - 2 Dyn. Kiev"
    "europe_uefa-champions-league/2015-2016/2015-09-29 - 21-45 Arsenal 2 - 3 Olympiakos Piraeus"
    "europe_uefa-champions-league/2015-2016/2015-09-16 - 21-45 Chelsea 4 - 0 Maccabi Tel Aviv"
    "europe_uefa-champions-league/2015-2016/2015-11-03 - 22-45 Benfica 2 - 1 Galatasaray"
)

# `python src/sn_script/whisper2csv.py --task sample --suffix _stable_version2`

echo "Target games:"
for game in "${target_games[@]}"; do
    echo "$game"
done

# large-v3
MODEL=whisperx-large-v3

# パラメータ
CHUNK_SIZE=20

# 実行
SUFFIX=_exp_translate

# 配列をスペース区切りで展開する際にクォートを使用
python src/sn_script/speech2text.py \
    --target_games "${target_games[@]}" \
    --half 1 \
    --model $MODEL \
    --suffix $SUFFIX \
    --hf_token $HF_TOKEN \
    --chunk_size $CHUNK_SIZE \
    --task translate \
    --device_index 3
