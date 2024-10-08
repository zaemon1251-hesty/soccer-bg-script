# 使い方

## 書き起こし区間性能調査

1. speech2text-$MODEL_NAME.sh で音声から書き起こしjsonを出力
2. whisper2csv-$MODEL_NAME.sh でjsonからcsvを出力
3. plot_voice_detection-$MODEL_NAME.sh でcsvから音声検出のグラフを出力
4. evaluate-vad.sh でcsvから評価結果を出力
