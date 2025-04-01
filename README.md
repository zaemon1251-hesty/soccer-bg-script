# soccer-bg-script

[実況生成のデモ](https://github.com/zaemon1251-hesty/soccer-bg-commentary)の事前準備を含めた研究用のスクリプト

## 前提

pythonの環境について、anaconda で管理していますが、一部スクリプトは[uv](https://github.com/astral-sh/uv)に依存しています
TODO どっちかに統一する

## 何をするのか

- SoccerNetのデータのダウンロードと加工（映像・ラベル含む）
- 自動書き起こし (とその評価)
- 自動ラベル付けの実行(データセットSCBIの構築)
- SCBIの分析
- 映像解析: ffmpegを使った動画の加工
- 映像解析: gsrの結果とメタデータを使った選手名の名寄せ
- 映像解析:ボールの追跡

[実況生成のデモ](https://github.com/zaemon1251-hesty/soccer-bg-commentary)動作に関わるのは映像解析だけです。
なお、いくつかのサンプル(20個の30秒クリップ)の映像解析結果は、実況生成のデモのリポジトリ下のzipファイル中に配置されています。
新しい映像に対して実況生成を行う場合は、このリポジトリのスクリプトを使って映像解析を行い、その結果を実況生成のデモに渡してください。

## このリポジトリの使い方

注意:（実況生成デモ動作に関わるのは映像解析だけなので）映像解析のみ記載。

### 事前準備

- このリポジトリを `pip install -e .`　する。あと、sn-gamestateとtracklabも環境構築しておく。
- <https://drive.google.com/file/d/1eKNQ5l5H19Wxat0N46ubjjhPDWrEUMOs/view?usp=drive_link> からzipをダウンロードし、 database というディレクトリ名で解凍
- SoccerNetの映像をダウンロードして、適当なディレクトリに (例: `/local/moriy/SoccerNet`) に配置

### 映像解析

```bash
# 0. 映像解析対象のビデオのメタデータcsv (例: database/misc/RAGモジュール出力サンプル-13090437a14481f485ffdf605d3408cd.csv) を用意
# csvは次の項目を含めば良い id, game, half, time
# メモ: webサービスとしては、処理したい映像とメタデータを事前登録してもらって、システム側でいい感じにこのcsvの形式を出力できたら良さそう

# 1. 映像から画像を切り出す
python src/sn_script/video2images.py \
    --SoccerNet_path /local/moriy/SoccerNet  \
    --output_base_path "画像を置いておくディレクトリ (例: /local/moriy/SoccerNetGS/rag-eval)"  \
    --target_game "" --resolution 720p --fps 25 --threads 4 \
    --input_csv_path "映像解析対象のビデオのメタデータcsvのパス"

# 2. gsr実行
cd (tracklab)
python -m tracklab.main -cn soccernet-v2-exp005 \
    modules.reid.training_enabled=False \
    modules.team._target_=sn_gamestate.team.TrackletTeamClustering \
    dataset.dataset_path=/local/moriy/SoccerNetGS/rag-eval

# 3. 選手名への対応付け
cd (sn-script)
python src/sn_script/video_analysis/result2player.py \
    --gsr_result_pklz "path/to/gsr_result.pklz" \ # (例 2024-12-17/10-57-24/states/sn-gamestate-v2.pklz)
    --evaluatoin_sample_path "映像解析対象のビデオのメタデータcsvのパス" \
    --output_csv_path "選手名のcsvのパス"

# 4. ボールの追跡
cd (sn-script)
python src/sn_script/video_analysis/ball_tracking_yolo.py \
    --image_dir "画像を置いておくディレクトリ"
    --output_csv "ボールの追跡結果を置くcsvのパス"

# 5. ボールの追跡結果を使って、ボールの位置とgsrを紐付け
cd (sn-script)
python src/sn_script/video_analysis/process_tracked_ball.py \
    --src_csv_file "ボールの追跡結果を置くcsvのパス"
    --input_player_csv  "選手名のcsvのパス" \
    --sample_metadata_file "映像解析対象のビデオのメタデータcsvのパス" \
    --output_csv_file "ボールの位置とgsrを紐付けたcsvのパス"

```
