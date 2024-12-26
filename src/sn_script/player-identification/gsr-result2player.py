import warnings
import zipfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd
from tap import Tap


class Args(Tap):
    # 選手同定モジュールの出力結果
    gsr_result_pklz: str = "/Users/heste/workspace/soccernet/tracklab/outputs/sn-gamestate-v2/2024-12-26/23-51-02/states/sn-gamestate-v2.pklz"

    # left,rigth からチームを取得するためのCSV
    side_team_map_csv: str = "/Users/heste/workspace/soccernet/sn-script/database/misc/side_to_team.csv"

    # sn-captionのメタデータで作ったCSV
    player_master_csv: str = "/Users/heste/workspace/soccernet/sn-script/database/misc/sncaption_players.csv"

    # 評価用の映像のメタデータを管理するCSV: sample_id,game,half,time
    evaluatoin_sample_path = "/Users/heste/workspace/soccernet/sn-script/database/misc/RAGモジュール出力サンプル-13090437a14481f485ffdf605d3408cd.csv"

    output_csv_path: str = "/Users/heste/workspace/soccernet/sn-script/database/misc/players_in_frames_sn_gamestate_29-33.csv"


@dataclass
class GSRStates:
    gamestate_zip_file: Path
    gamestate_summary_file: Path
    gamestate_file: Path
    gamestate_images_file: Path
    gamestate_df: pd.DataFrame
    image_df: pd.DataFrame

    @classmethod
    def initialize(cls, gamestate_zip_file: Path):
        parent_dir = gamestate_zip_file.parent

        if not (parent_dir / "summary.json").exists():
            zipfile.ZipFile(gamestate_zip_file).extractall(parent_dir)

        # 見つける the summary.json file
        gamestate_summary_file = parent_dir / "summary.json"

        # 見つける the / {data_id}.pkl file
        gamestate_files = []
        gamestate_images_files = []
        for path in glob(str(parent_dir / "*.pkl")):
            if "_image.pkl" in path:
                gamestate_images_files.append(Path(path))
            else:
                # {data_id} はどこにも記載されていないので、とりあえず_image.pklではないものを gamestate_file とする
                gamestate_files.append(Path(path))

        gamestate_df_list = []
        image_df_list = []

        gamestate_files = sorted(gamestate_files)
        gamestate_images_files = sorted(gamestate_images_files)
        for gamestate_file, gamestate_images_file in zip(gamestate_files, gamestate_images_files):
            gamestate_df = pd.read_pickle(gamestate_file)
            image_df = pd.read_pickle(gamestate_images_file)

            gamestate_df_list.append(gamestate_df)
            image_df_list.append(image_df)

        gamestate_df = pd.concat(gamestate_df_list, ignore_index=True)
        image_df = pd.concat(image_df_list, ignore_index=True)

        return GSRStates(
            gamestate_zip_file=gamestate_zip_file,
            gamestate_summary_file=gamestate_summary_file,
            gamestate_file=gamestate_file,
            gamestate_images_file=gamestate_images_file,
            gamestate_df=gamestate_df,
            image_df=image_df
        )


def _convert_detections(
    detection_df: pd.DataFrame,
    player_df: pd.DataFrame,
    game: str,
    half: int,
    time: int,
):
    assert {"role", "team", "jersey_number", "bbox_ltwh"}.issubset(detection_df.columns)

    list_of_dicts = []

    any_role_valid_flag = False
    for _, row in detection_df.iterrows():
        role, team = row['role'], row['team']
        if role in ["player", "goalkeeper"]:
            any_role_valid_flag = True
            pass
        else:
            continue

        jersey_number = row["jersey_number"]
        if isinstance(jersey_number, str) and jersey_number.isnumeric():
            jersey_number = int(jersey_number)
        elif isinstance(jersey_number, float) and jersey_number.is_integer():
            jersey_number = int(jersey_number)
        elif isinstance(jersey_number, int):
            pass
        else:
            warnings.warn(f"jersey_number is None: {game=}, {half=}, {time=}, {team=}, {jersey_number=}", stacklevel=3)
            continue

        side_team = side_team_map.get((game, half))
        if side_team is not None:
            team = side_team[team]
        else:
            warnings.warn(f"side_team is None: {game=}, {half=}, {time=}, {team=}, {jersey_number=}", stacklevel=3)
            continue

        player_df["shirt_number"] = player_df["shirt_number"].astype(int)
        player_row = player_df[
            (player_df["game"] == game) &
            (player_df["team"] == team) &
            (player_df["shirt_number"] == jersey_number)
        ]

        if not player_row.empty:
            player_row = player_row.iloc[0] # 高々ひとつしか取れないはず
            # row["bbox_ltwh"] は ndarray
            x1, y1, x2, y2 = row["bbox_ltwh"]
            player_data = {
                "game": game,
                "half": half,
                "time": time,
                "team": team,
                "name": player_row["name"],
                "short_name": player_row["short_name"],
                "jersey_number": jersey_number,
                "country": player_row["country"],
            }
            player_data["x1_720p"] = int(x1)
            player_data["y1_720p"] = int(y1)
            player_data["x2_720p"] = int(x2)
            player_data["y2_720p"] = int(y2)
            list_of_dicts.append(player_data)
        else:
            warnings.warn(f"player_row not found: {game=}, {half=}, {time=}, {team=}, {jersey_number=}", stacklevel=2)
            continue
    if not any_role_valid_flag:
        warnings.warn(f"any_role_valid_flag is False: {game=}, {half=}, {time=}", stacklevel=2)
        pass

    return list_of_dicts

def gametime_to_seconds(gametime):
    if not isinstance(gametime, str):
        return gametime
    if gametime.count(":") == 2:
        gametime = ":".join(gametime.split(":")[:2])
    m, s = gametime.split(":")
    return int(m) * 60 + int(s)



if __name__ == "__main__":
    args = Args().parse_args()

    gsr_result_path = args.gsr_result_pklz
    side_team_map_csv = args.side_team_map_csv
    player_master_csv = args.player_master_csv
    output_csv_path = args.output_csv_path
    evaluatoin_sample_path = args.evaluatoin_sample_path

    gsr_result = GSRStates.initialize(Path(gsr_result_path))
    gamestate_df = gsr_result.gamestate_df

    side_team_map_df = pd.read_csv(side_team_map_csv)
    player_df = pd.read_csv(player_master_csv)

    evaluation_sample_df = pd.read_csv(evaluatoin_sample_path)

    # left, right からチームを取得するための辞書
    side_team_map = {
        (row["game"], row["half"]): {
            "left": row["left"],
            "right": row["right"],
        }
        for _, row in side_team_map_df.iterrows()
    }

    # video_idの昇順と、評価サンプルIDの昇順でソートして対応付ける
    gamestate_df["mapping_id"] = pd.factorize(
        gamestate_df["video_id"].astype(int)
    )[0]

    evaluation_sample_df["mapping_id"] = pd.factorize(
        evaluation_sample_df["id"].astype(int)
    )[0]

    # sample() は 1つだけ取れる想定
    game_df = evaluation_sample_df.groupby(['game', 'half', 'time']).sample().reset_index()

    # mapping_id で結合
    merged_df = pd.merge(
        gamestate_df,
        evaluation_sample_df,
        on="mapping_id",
        how="left"
    )

    list_of_dicts = []
    merged_df.image_id = merged_df.image_id.astype(int)
    for (game, half), group in merged_df.groupby(["game", "half"]):
        print(f"{game=}, {half=}")
        for time in group["time"].unique():
            target_detections = merged_df[
                (merged_df["game"] == game) &
                (merged_df["half"] == half) &
                (merged_df["time"] == time)
            ]

            # 区間の真ん中が発話タイミング
            # 2秒前後を対象とする、2*25fps=50枚程度
            mean_image_id = target_detections.image_id.mean()
            target_detections = target_detections[
                (target_detections["image_id"] <= mean_image_id + 50) &
                (target_detections["image_id"] >= mean_image_id - 50)
            ]

            time_int = gametime_to_seconds(time)
            list_of_dicts_per_scene = _convert_detections(
                target_detections,
                player_df,
                game,
                half,
                time_int,
            )
        print(f"{len(list_of_dicts_per_scene)=}")
        list_of_dicts.extend(list_of_dicts_per_scene)

    if list_of_dicts:
        # remove duplicates
        output_df = pd.DataFrame(list_of_dicts)
        output_df = output_df.drop_duplicates(subset=["game", "half", "time", "team", "jersey_number"])
        output_df.to_csv(output_csv_path, index=False)
