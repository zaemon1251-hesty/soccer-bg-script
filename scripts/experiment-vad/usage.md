# 使い方

## 書き起こし区間性能調査

(前提)プロジェクトルートで以下のコマンドを実行する

1. `./scripts/experiment-vad/speech2text-$MODEL_NAME.sh` で音声から書き起こしjsonを出力
2. `./scripts/experiment-vad/whisper2csv-$MODEL_NAME.sh` でjsonからcsvを出力
3. `./scripts/experiment-vad/plot_voice_detection-$MODEL_NAME.sh` でcsvから音声検出のグラフを出力
4. `./scripts/experiment-vad/evaluate-vad.sh` でcsvから評価結果を出力
