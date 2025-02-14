import sys
import warnings
from pathlib import Path

import pandas as pd
from sn_script.csv_utils import gametime_to_seconds
from sn_script.player_identification.gsr_data import GSRStates
from tap import Tap

fps = 25

class Args(Tap):
    # 選手同定モジュールの出力結果
    gsr_result_pklz: str = "/Users/heste/workspace/soccernet/tracklab/outputs/sn-gamestate-v2/2024-12-17/10-57-24/states/sn-gamestate-v2.pklz"
    # 下は追加サンプル分
    # gsr_result_pklz: str = "/Users/heste/workspace/soccernet/tracklab/outputs/sn-gamestate-v2/2024-12-26/23-51-02/states/sn-gamestate-v2.pklz"

    # left,rigth からチームを取得するためのCSV
    side_team_map_csv: str = "/Users/heste/workspace/soccernet/sn-script/database/misc/side_to_team.csv"

    # sn-captionのメタデータで作ったCSV
    player_master_csv: str = "/Users/heste/workspace/soccernet/sn-script/database/misc/sncaption_players.csv"

    # 評価用の映像のメタデータを管理するCSV: sample_id,game,half,time
    evaluatoin_sample_path = "/Users/heste/workspace/soccernet/sn-script/database/misc/RAGモジュール出力サンプル-13090437a14481f485ffdf605d3408cd.csv"

    output_csv_path: str = "/Users/heste/workspace/soccernet/sn-script/database/demo/players_in_frames_sn_gamestate.csv"
    output_jsonl: str = None


def _convert_detections(
    detection_df: pd.DataFrame,
    player_df: pd.DataFrame,
    game: str,
    half: int,
    mean_time: int,
):
    assert {"role", "team", "jersey_number", "bbox_ltwh"}.issubset(detection_df.columns)
    player_df["shirt_number"] = player_df["shirt_number"].astype(int)

    list_of_dicts = []

    max_image_id = detection_df["image_id"].max()
    min_image_id = detection_df["image_id"].min()
    mean_image_id = detection_df["image_id"].mean()
    num_images = max_image_id - min_image_id + 1

    # time から 前後15秒の合計30秒を切り取ったから，大体 num_images / 30 = fps が出るはず
    assert abs(num_images / 30 - fps) <= 1., f"{num_images=}, {fps=}"

    def frame_to_time(image_id, mean_time):
        """image_id から time を計算する"""
        return mean_time + (image_id - mean_image_id) / fps

    any_role_valid_flag = False
    for _, row in detection_df.iterrows():
        role, team = row["role"], row["team"]
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
            jersey_number = None

        side_team = side_team_map.get((game, half))
        if side_team is not None:
            team = side_team[team]
        else:
            team = None

        name = None
        short_name = None
        long_name = None
        country = None
        if jersey_number is not None and team is not None:
            player_row = player_df[(player_df["game"] == game) & (player_df["team"] == team) & (player_df["shirt_number"] == jersey_number)]
            if not player_row.empty:
                player_row = player_row.iloc[0]  # 高々ひとつしか取れないはず
                name = player_row["name"]
                short_name = player_row["short_name"]
                long_name = player_row["long_name"]
                country = player_row["country"]
            else:
                warnings.warn(f"player_row not found: {game=}, {half=}, {mean_time=}, {team=}, {jersey_number=}", stacklevel=2)

        x,y,l,w = row["bbox_ltwh"]
        x1, y1, x2, y2 = x, y, x+l, y+w

        player_data = {
            "game": game,
            "half": half,
            "time": frame_to_time(row["image_id"], mean_time),
            "team": team,
            "name": name,
            "short_name": short_name,
            "jersey_number": jersey_number,
            "country": country,
            "long_name": long_name,
            "image_id": row["image_id"] - min_image_id + 1,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        }
        list_of_dicts.append(player_data)
    if not any_role_valid_flag:
        warnings.warn(f"any_role_valid_flag is False: {game=}, {half=}, {mean_time=}", stacklevel=2)
        pass

    return list_of_dicts


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
    gamestate_df["mapping_id"] = pd.factorize(gamestate_df["video_id"].astype(int))[0]

    evaluation_sample_df["mapping_id"] = pd.factorize(evaluation_sample_df["id"].astype(int))[0]

    # sample() は 1つだけ取れる想定
    game_df = evaluation_sample_df.groupby(["game", "half", "time"]).sample().reset_index()

    # mapping_id で結合
    merged_df = pd.merge(gamestate_df, evaluation_sample_df, on="mapping_id", how="left")

    list_of_dicts = []
    merged_df.image_id = merged_df.image_id.astype(int)
    for (game, half), group in merged_df.groupby(["game", "half"]):
        print(f"{game=}, {half=}")
        for utterance_start_time in group["time"].unique():
            target_detections = merged_df[(merged_df["game"] == game) & (merged_df["half"] == half) & (merged_df["time"] == utterance_start_time)]

            # 区間の真ん中が発話タイミング
            # 2秒前までを対象とする、2*25fps=50枚程度
            # mean_image_id = target_detections.image_id.mean()
            # target_detections = target_detections[(target_detections["image_id"] <= mean_image_id) & (target_detections["image_id"] >= mean_image_id - 50)]

            # 全てのフレームを対象とする

            mean_time_int = gametime_to_seconds(utterance_start_time)
            list_of_dicts_per_scene = _convert_detections(
                target_detections,
                player_df,
                game,
                half,
                mean_time_int,
            )
        print(f"{len(list_of_dicts_per_scene)=}")
        list_of_dicts.extend(list_of_dicts_per_scene)

    if not list_of_dicts:
        warnings.warn("list_of_dicts is empty", stacklevel=2)
        sys.exit(1)

    # remove duplicates
    output_df = pd.DataFrame(list_of_dicts)
    output_df = output_df.drop_duplicates(subset=["game", "half", "time", "team", "jersey_number"])
    if args.output_csv_path:
        output_df.to_csv(output_csv_path, index=False)
    if args.output_jsonl:
        output_df.to_json(args.output_jsonl, orient="records", lines=True)
