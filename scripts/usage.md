# 使用法

(前提)プロジェクトルートで以下のコマンドを実行する

## SCBI構築

SCBI (Soccer Commentary Background Information): ラベル付き実況コメントデータ

1. `./scripts/stable/speech2text-whisperx-stable.sh` で音声から書き起こしjsonを出力
2. `./scripts/stable/whisper2csv-whisperx-stable.sh` でjsonからcsvを出力

## SoccerNet各種データのダウンロード

`TASK`: v3, caption, spotting, jersey, など

```sh
python src/sn_script/download_soccernet_data.py --task `TASK`
```

## 各種CSVの作成

- sn-captionに含まれる選手名の抽出

```sh
python src/sn_script/curate_sncaption_players.py
```
