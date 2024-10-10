"""
要約
- sn-echos 実況コメントスパンcsvを作成
入出力
- jsons to csv
"""
import csv
import glob
import json
import os

try:

    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config

base_dir = "/Users/heste/workspace/soccernet/sn-echoes/Dataset/whisper_v2_en"

# whisper_v2_en 下のgame名を収集
game_list = glob.glob("*/*/*", root_dir=base_dir, recursive=True)

json_basename = "{half_number}_asr.json"

json_files = [
    (
        os.path.join(base_dir, game, json_basename.format(half_number=half_number)),
        game,
        half_number,
    )
    for game in game_list
    for half_number in [1, 2]
]

# 出力するCSVファイル
csv_output_path = Config.target_base_dir / "comments" / "sn_echoes_whisper_v2_en.csv"

# CSVに書き込むためのヘッダー
csv_header = ['id', 'game', 'half', 'comment_id', 'start', 'end', 'text']

# CSVファイルの作成 (エクセル対応のため utf-8-sig )
with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()

    # 各jsonlファイルを読み込んで処理
    idx = 1
    for file_path, game, half in json_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        with open(file_path) as f:
            asr_data = json.load(f)
            for comment_id, seg_data in asr_data["segments"].items():
                start, end, text = seg_data
                # CSVに書き込むため、ファイルごとのデータを整理
                csv_row = {
                    'id': idx,
                    'game': game,
                    'half': half,
                    'comment_id': comment_id,
                    'start': start,
                    'end': end,
                    'text': text,
                }
                # CSVに書き込む
                writer.writerow(csv_row)
                # インデックスを更新
                idx += 1
        print(f"Done! {file_path=}")
    print(f"last idx: {idx}")
print(f"All Done! CSV file saved: {csv_output_path}")
