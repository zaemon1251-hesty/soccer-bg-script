"""
要約
- whisperの出力した実況コメントスパンを加工してできたjsonlファイルをcsvに変換する
入出力
- jsonlines to csv
"""
import csv
import json
import os

try:

    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


# jsonlファイルのパスを指定
game_list = Config.targets

jsonl_basename = "denoised_{half_number}_224p.jsonl"

jsonl_files = [
    (
        os.path.join(Config.base_dir, game, jsonl_basename.format(half_number=half_number)),
        game,
        half_number,
    )
    for game in game_list
    for half_number in [1, 2]
]

# 出力するCSVファイル
csv_output_path = Config.target_base_dir / "500game_whisper_spans_denoised.csv"

# CSVに書き込むためのヘッダー
csv_header = ['id', 'game', 'half', 'seek', 'start', 'end', 'text', 'tokens', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob',]

# CSVファイルの作成 (エクセル対応のため utf-8-sig )
with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()

    # 各jsonlファイルを読み込んで処理
    for file_path, game, half in jsonl_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        with open(file_path) as f:
            for line in f:
                try:
                    # jsonlファイルの行をパース
                    data = json.loads(line.strip())

                    # textにエスケープされた文字がある場合、デコードする
                    if 'text' in data:
                        data['text'] = data['text'].encode().decode('unicode_escape')

                    # ファイルのパスを追加
                    data['game'] = game

                    # half
                    data["half"] = half

                    # CSVに書き込むため、ファイルごとのデータを整理
                    csv_row = {
                        'id': data.get('id'),
                        'game': data.get('game'),
                        'half': data.get('half'),
                        'seek': data.get('seek'),
                        'start': data.get('start'),
                        'end': data.get('end'),
                        'text': data.get('text'),
                        'tokens': data.get('tokens'),
                        'temperature': data.get('temperature'),
                        'avg_logprob': data.get('avg_logprob'),
                        'compression_ratio': data.get('compression_ratio'),
                        'no_speech_prob': data.get('no_speech_prob'),
                    }

                    # CSVに書き込む
                    writer.writerow(csv_row)

                except json.JSONDecodeError:
                    print(f"Error parsing line in file {file_path}: {line}")
                except UnicodeDecodeError:
                    print(f"Unicode decode error in file {file_path}: {line}")
