import csv
import json

from SoccerNet.Downloader import getListGames

try:
    from sn_script.config import (
        Config,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
    )


def labels_v2_json_to_csv(game_split_dict: dict[str, str], csv_file_path):
    # CSVファイルを開く
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # ヘッダーを定義
        header = ['game', 'gameAwayTeam', 'gameDate', 'gameHomeTeam', 'gameScore', 'gameTime', 'label', 'position', 'team', 'visibility', 'split']
        csv_writer.writerow(header)

        for game, split in game_split_dict.items():
            json_path = Config.base_dir / game / "Labels-v2.json"

            if not json_path.exists():
                print(f"{json_path} does not exist")
                continue

            # JSONファイルを開く
            with open(json_path, encoding='utf-8') as json_file:
                data = json.load(json_file)

                # 各種情報を取得
                game_info = data.get("UrlLocal", "")
                game_away_team = data.get("gameAwayTeam", "")
                game_date = data.get("gameDate", "")
                game_home_team = data.get("gameHomeTeam", "")
                game_score = data.get("gameScore", "")

                # annotationsに基づいてCSV行を書き出す
                for annotation in data['annotations']:
                    row = [
                        game_info,                # game列にはUrlLocalの内容
                        game_away_team,           # gameAwayTeam
                        game_date,                # gameDate
                        game_home_team,           # gameHomeTeam
                        game_score,               # gameScore
                        annotation.get('gameTime', ''),
                        annotation.get('label', ''),
                        annotation.get('position', ''),
                        annotation.get('team', ''),
                        annotation.get('visibility', ''),
                        split
                    ]
                    csv_writer.writerow(row)


def get_game_split_map():
    game_split_map = {}
    for split in ["train", "valid", "test"]:
        games = getListGames(split, "spotting")
        game_split_map.update({game: split for game in games})
    return game_split_map


if __name__ == "__main__":
    game_split_map = get_game_split_map()
    output_path = Config.target_base_dir / "soccernet_spotting_labels.csv"
    labels_v2_json_to_csv(game_split_map, output_path)
