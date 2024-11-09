"""
入力: Labels-v3.json, player_master.csv
出力: csv (ヘッダーは game, half, time, points, team, name, short_name, country, points)

"""
import json
import os
from datetime import datetime

import pandas as pd
from loguru import logger
from sn_script.csv_utils import gametime_to_seconds
from sn_script.v3.v3_to_gsr import convert_to_attributes
from SoccerNet.Downloader import getListGames
from tap import Tap
from tqdm import tqdm

# key: (game, half)
# value: {side: team}
side_team_map = {
    ("england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham", 1): {
        "left": "West Ham",
        "right": "Liverpool",
    },

}

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
logger.add(f"logs/{time_str}.log")

v3_basename = "Labels-v3.json"

game_list = getListGames("all")


class ConvertToPlayersArguments(Tap):
    input_v3_dir: str
    input_player_master_csv: str
    output_csv_path: str


def _convert_bboxes(bboxes_data, player_df, list_of_dicts, game, half, time):
    for bbox in bboxes_data:
        role, team = convert_to_attributes(bbox['class'])
        if role not in ["player", "goalkeeper"]:
            continue

        jersey_number = bbox["ID"]
        if jersey_number is not None and jersey_number.isnumeric():
            jersey_number = int(jersey_number)
        else:
            continue

        # debug
        if (game, half) == ("england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea", "1"):
            print(f"{role=}, {team=}, {jersey_number=}")
            team_name = side_team_map.get((game, half))[team]
            print(f"{team_name=}")

        side_team = side_team_map.get((game, half))
        if side_team is None:
            continue
        team = side_team[team]

        player_row = player_df[
            (player_df["game"] == game) &
            (player_df["team"] == team) &
            (player_df["jersey_number"] == jersey_number)
        ]
        if player_row.empty:
            continue

        player_data = {
            "game": game,
            "half": half,
            "time": time,
            "team": team,
            "name": player_row["name"],
            "short_name": player_row["short_name"],
            "country": player_row["country"],
            "points": bbox["points"],
        }
        list_of_dicts.append(player_data)


def convert(v3_data, player_df, output_path):
    list_of_dicts = []
    for scene in ["actions", "replays"]:
        for _, action_data in v3_data[scene].items():
            image_metadata = action_data["imageMetadata"]
            bboxes_data = action_data["bboxes"]
            game = image_metadata["localpath"]
            half, time_str = image_metadata["gameTime"].split(" - ")
            time = gametime_to_seconds(time_str)
            _convert_bboxes(bboxes_data, player_df, list_of_dicts, game, half, time)

    result_df = pd.DataFrame(list_of_dicts)
    result_df.to_csv(output_path, index=False)


def convert_to_players(args: ConvertToPlayersArguments):

    player_df = pd.read_csv(args.input_player_master_csv)

    for game in tqdm(game_list):
        game_path = os.path.join(args.input_v3_dir, game, v3_basename)
        if not os.path.exists(game_path):
            continue
        game_data = json.load(open(game_path))

        convert(game_data, player_df, args.output_csv_path)



if __name__ == "__main__":
    args = ConvertToPlayersArguments().parse_args()
    convert_to_players(args)
