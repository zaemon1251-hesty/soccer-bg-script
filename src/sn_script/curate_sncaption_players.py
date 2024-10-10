"""
要約
- sn-caption のプレイヤー情報を抽出
入出力
- jsons to csv
"""
import json
from datetime import datetime

import pandas as pd

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        half_number,
        model_type,
        random_seed,
        subcategory_name,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
    )


PLAYERS_CSV_PATH = Config.target_base_dir / "sncaption_players.csv"

# プレイヤー情報を含むテーブルを作成
players_data = []


for target in Config.targets:
    json_path = Config.base_dir / target / "Labels-caption.json"

    # JSONファイルを読み込む
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)

    # 試合のメタ情報
    timestamp = data["timestamp"]
    date = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
    score = data["score"]
    teams = data["teams"]
    round_number = data["round"]
    game = target.replace("SoccerNet/", "")

    # homeとawayそれぞれのプレイヤーを処理
    for side in ["home", "away"]:
        team_name = teams[0] if side == "home" else teams[1]
        for player in data["lineup"][side]["players"]:
            player_info = {
                "game": game,
                "team": team_name,
                "side": side,
                "date": date,
                "round": round_number,
                "score": score,
                **player,
            }
            players_data.append(player_info)


# write csv
players_df = pd.DataFrame(players_data)
players_df.to_csv(PLAYERS_CSV_PATH)
