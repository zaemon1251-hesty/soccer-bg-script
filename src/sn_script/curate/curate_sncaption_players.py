"""
要約
- sn-caption のプレイヤー情報を抽出
入出力
- jsons to csv
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tap import Tap

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

class CurateSncaptionPlayersArguments(Tap):
    SoccerNet_path: str
    output_csv: str = str(Config.target_base_dir / "sncaption_players.csv")

args = CurateSncaptionPlayersArguments().parse_args()

# プレイヤー情報を含むテーブルを作成
players_data = []


for target in Config.targets:
    json_path = Path(args.SoccerNet_path) / target / "Labels-caption.json"

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
        team_name = data["gameHomeTeam"] if side == "home" else data["gameAwayTeam"] #  ホームのチームが先頭というわけではない
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
players_df.to_csv(args.output_csv, index=False)
