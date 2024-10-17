"""
Labels-v3.json から GSR 形式のjson に変換するスクリプト
"""
import json
import os
import zipfile
from pathlib import Path
from typing import List

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
from tap import Tap

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys
    sys.path.append(".")
    from src.sn_script.config import Config

class V3Json2GsrArguments(Tap):
    SoccerNet_path: str
    output_base_path: str

GAMES: List[str] = getListGames("all")

# Labels-GameState.json のファイル末尾に存在するカテゴリ情報
MASTER_CATRGORIES = [
    {
        "supercategory": "object",
        "id": 1,
        "name": "player"
    },
    {
        "supercategory": "object",
        "id": 2,
        "name": "goalkeeper"
    },
    {
        "supercategory": "object",
        "id": 3,
        "name": "referee"
    },
    {
        "supercategory": "object",
        "id": 4,
        "name": "ball"
    },
    {
        "supercategory": "pitch",
        "id": 5,
        "name": "pitch",
        "lines": [
            "Big rect. left bottom",
            "Big rect. left main",
            "Big rect. left top",
            "Big rect. right bottom",
            "Big rect. right main",
            "Big rect. right top",
            "Circle central",
            "Circle left",
            "Circle right",
            "Goal left crossbar",
            "Goal left post left",
            "Goal left post right",
            "Goal right crossbar",
            "Goal right post left",
            "Goal right post right",
            "Middle line",
            "Side line bottom",
            "Side line left",
            "Side line right",
            "Side line top",
            "Small rect. left bottom",
            "Small rect. left main",
            "Small rect. left top",
            "Small rect. right bottom",
            "Small rect. right main",
            "Small rect. right top"
        ]
        },
    {
        "supercategory": "camera",
        "id": 6,
        "name": "camera"
    },
    {
        "supercategory": "object",
        "id": 7,
        "name": "other"
    }
]


def game_to_id(game: str, half: int):
    """001_1, 002_2, 003_1, ..."""
    assert game in GAMES, "invalid game"
    game_index = GAMES.index(game)
    video_id = f"{game_index:03d}_{half}"
    return video_id

def convert_to_attributes(bbox_clazz):
    clazz_id = FRAME_CLASS_DICTIONARY.get(bbox_clazz)
    role = None
    team = None
    if clazz_id == 0:
        role = "ball"
    elif clazz_id == 1:
        role = "player"
        team = "left"
    elif clazz_id == 2:
        role = "player"
        team = "right"
    elif clazz_id == 3:
        role = "goalkeeper"
        team = "left"
    elif clazz_id == 4:
        role = "goalkeeper"
        team = "right"
    elif clazz_id == 5:
        role = "referee"
    elif clazz_id == 6:
        role = "ball"
    elif clazz_id == 7:
        role = "player"
    elif clazz_id == 8:
        role = "player"
    elif clazz_id == 9:
        role = "goalkeeper"

    return role, team


# from Labels-v3.json to Labels-GameState.json format
def convert_to_gamestate(game_path, gamestate_base_dir):
    v3_json_path = os.path.join(game_path, "Labels-v3.json")
    with open(v3_json_path) as f:
        v3_data = json.load(f)

    metadata = v3_data["GameMetadata"]

    list_actions = metadata["list_actions"]

    first_action = v3_data["actions"][list_actions[0]]
    last_action = v3_data["actions"][list_actions[-1]]

    split = first_action["imageMetadata"]["set"]

    half = first_action["imageMetadata"]["half"]

    super_id = game_to_id(metadata["UrlLocal"], half)

    game_id = v3_data["actions"][list_actions[0]]["imageMetadata"]["gameID"]

    # gamestateのフォーマット
    gamestate_data = {
        "info": {
            "version": "0.1",
            "game_id": game_id,
            "id": super_id,
            "num_tracklets": "11", # TODO
            "action_position": None,
            "action_class": None,
            "visibility": True,
            "game_time_start": first_action["imageMetadata"]["gameTime"],
            "game_time_stop": last_action["imageMetadata"]["gameTime"],
            "clip_start": first_action["imageMetadata"]["position"],
            "clip_stop": last_action["imageMetadata"]["position"],
            "name": f"SNGS-{super_id}",
            "im_dir": "img1",
            "frame_rate": 5, # TODO
            "seq_length": len(list_actions),
            "im_ext": ".png"
        },
        "images": [],
        "annotations": [],
        "categories": MASTER_CATRGORIES
    }

    # actionのみ対象にする replayはデータが足りない場合追加する
    for image_name, action_data in v3_data['actions'].items():
        # image情報
        image_info = {
            "is_labeled": True,
            "image_id": f"{super_id}-{Path(image_name).stem}",  # Example, adjust with unique ID
            "file_name": image_name,
            "height": action_data["imageMetadata"]["height"],
            "width": action_data["imageMetadata"]["width"],
            "has_labeled_person": True,
            "has_labeled_pitch": False,
            "has_labeled_camera": False,
            "ignore_regions_y": [],
            "ignore_regions_x": []
        }
        gamestate_data['images'].append(image_info)

        # 画像をコピーする準備
        v3_image_path = os.path.join(game_path, "Frames-v3", image_name)
        gamestate_image_path = os.path.join(gamestate_base_dir, split, gamestate_data["info"]["name"], gamestate_data["info"]["im_dir"], image_name)
        Path(v3_image_path).parent.mkdir(parents=True, exist_ok=True)
        Path(gamestate_image_path).parent.mkdir(parents=True, exist_ok=True)

        # なければ解凍
        try:
            zipfilepath = ""
            if not os.path.exists(v3_image_path):
                # unzip
                zipfilepath = os.path.join(game_path, "Frames-v3.zip")
                with zipfile.ZipFile(zipfilepath, 'r') as zippedFrames:  # noqa: N806
                    with zippedFrames.open(image_name) as imginfo:
                        with open(gamestate_image_path, 'wb') as f:
                            f.write(imginfo.read())
        except FileExistsError:
            # すでに解凍済み
            pass
        except Exception as e:
            print(f"{zipfilepath=}")
            print(f"{e=}")
            # 画像が使えないから、らべるまるごとスキップする
            continue
        # コピー
        if not os.path.exists(gamestate_image_path):
            with open(v3_image_path, 'rb') as f:
                with open(gamestate_image_path, 'wb') as g:
                    g.write(f.read())

        # bbox の変換
        for idx, bbox in enumerate(action_data['bboxes']):
            role, team = convert_to_attributes(bbox['class'])
            annotation = {
                "id": f"{super_id}-{Path(image_name).stem}-{idx + 1}",  # Example, adjust as needed
                "image_id": f"{super_id}-{Path(image_name).stem}",
                "track_id": idx + 1,
                "supercategory": "object", # 物体を表す識別子
                "category_id": 1,  # 人物のカテゴリID
                "attributes": {
                    "role": role,
                    "jersey": bbox['ID'] if (isinstance(bbox['ID'], str) and bbox['ID'].isnumeric()) else None,  # Assuming ID is numeric
                    "team": team  # Assuming class contains 'left' or 'right'
                },
                "bbox_image": {
                    "x": bbox["points"]["x1"],
                    "y": bbox["points"]["y1"],
                    "x_center": (bbox["points"]["x1"] + bbox["points"]["x2"]) / 2,
                    "y_center": (bbox["points"]["y1"] + bbox["points"]["y2"]) / 2,
                    "w": bbox["points"]["x2"] - bbox["points"]["x1"],
                    "h": bbox["points"]["y2"] - bbox["points"]["y1"]
                },
                "bbox_pitch": None,
                "bbox_pitch_raw": None,
            }
            gamestate_data['annotations'].append(annotation)

    # Write to the new Labels-GameState.json file
    gamestate_json_path = os.path.join(gamestate_base_dir, split, gamestate_data["info"]["name"], "Labels-GameState.json")
    with open(gamestate_json_path, 'w') as f:
        json.dump(gamestate_data, f, indent=4)

if __name__ == "__main__":
    args = V3Json2GsrArguments().parse_args()
    # Example usage
    games = getListGames("all", task="frames")

    for game in games:
        game_path = os.path.join(args.SoccerNet_path, game)
        convert_to_gamestate(game_path, args.output_base_path)
