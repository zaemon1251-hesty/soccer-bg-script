"""
Labels-v3.json から GSR 形式のjson に変換するスクリプト
"""
import json
import os
import zipfile
from pathlib import Path
from typing import List  # noqa: UP035

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
from tap import Tap
from tqdm import tqdm


class V3Json2GsrArguments(Tap):
    SoccerNet_path: str
    output_base_path: str


GAMES: List[str] = getListGames("all")  # noqa: UP006


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

class PersonIdentifier:
    """
    video ごとに、(role, team, jersey_number) から person_id を返す

    """
    def __init__(self, v3_data):
        self.v3_data = v3_data
        self.cur_person_num = 0
        self.done_preprocess = False
        self.preprocess()

    def preprocess(self):
        assert 'actions' in self.v3_data or 'replays' in self.v3_data
        assert self.cur_person_num == 0
        assert not self.done_preprocess

        self.person_inv_dict = {} # key: role, team, jersey_number_or_literal , value: person_id
        for scene in ['actions', 'replays']:
            for _, action_data in self.v3_data[scene].items():
                for bbox in action_data['bboxes']:
                    jersey_number_or_literal = bbox['ID']
                    role, team = convert_to_attributes(bbox['class'])
                    team = normalize_team(team, action_data['imageMetadata']['gameTime'])
                    # すでに person_id が割り当てられている場合はスキップ
                    if (role, team, jersey_number_or_literal) in self.person_inv_dict:
                        continue
                    # person_id を割り当てる
                    self.person_inv_dict[role, team, jersey_number_or_literal] = self.get_new_id()
        self.done_preprocess = True

    def get_new_id(self):
        self.cur_person_num += 1
        return self.cur_person_num

    def get_person_id(self, role, team, jersey_number):
        if not self.done_preprocess:
            raise RuntimeError("先に preprocess() を実行してください")
        person_id = self.person_inv_dict.get(
            (role, team, jersey_number),
            None
        )
        return person_id


def normalize_team(team: str, game_time: str):
    """前半のサイドに統一する
    person_id 構築時は利用するが、Game State Reconstruction 用の json 作成時は利用しない
    Args:
        team (str): "left" or "rigth"
        game_time (str): "1 - MM:SS"
    """
    assert team in ["left", "right", None]
    assert game_time[0] in ["1", "2"]

    if team is None:
        return None

    half = int(game_time[0])

    if half == 1:
        return team
    else:
        return "left" if team == "right" else "right"


def game_to_id(game: str, half: int):
    """001_1, 002_2, 003_1, ..."""
    assert game in GAMES, "invalid game"
    game_index = GAMES.index(game)
    video_id = f"{game_index:03d}_{half}"
    return video_id


def convert_to_attributes(bbox_clazz):
    clazz_id = FRAME_CLASS_DICTIONARY.get(bbox_clazz, -1)
    role = "other"
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


def get_gamestate_dict_and_metadatas(
    v3_data: dict,
    resol720p: bool = False
):
    metadata = v3_data["GameMetadata"]
    list_actions = metadata["list_actions"]
    list_replays = metadata["list_replays"]
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
            "num_tracklets": "11", # TODO ちゃんと計算する
            "action_position": 0,
            "action_class": "action",
            "visibility": True,
            "game_time_start": first_action["imageMetadata"]["gameTime"],
            "game_time_stop": last_action["imageMetadata"]["gameTime"],
            "clip_start": first_action["imageMetadata"]["position"],
            "clip_stop": last_action["imageMetadata"]["position"],
            "name": f"SNGS-{super_id}",
            "im_dir": "img1" if not resol720p else "img720",
            "frame_rate": 5, # TODO ちゃんと計算する
            "seq_length": len(list_actions) + len(list_replays),
            "im_ext": ".png"
        },
        "images": [],
        "annotations": [],
        "categories": MASTER_CATRGORIES
    }
    return (
        gamestate_data,
        super_id,
        game_id,
        split,
        metadata,
        list_actions,
        list_replays,
        half
    )


def get_image_dict(
    image_name: str,
    action_data: dict,
    super_id: str
):
    # image情報
    image_info = {
        "is_labeled": True,
        "image_id": f"{super_id}-{Path(image_name).stem}",
        "file_name": image_name,
        "height": action_data["imageMetadata"]["height"],
        "width": action_data["imageMetadata"]["width"],
        "has_labeled_person": True,
        "has_labeled_pitch": False,
        "has_labeled_camera": False,
        "ignore_regions_y": [],
        "ignore_regions_x": []
    }
    return image_info


def process_copying_image(
    image_name: str,
    action_data: dict,
    gamestate_data: dict,
    game_path: str,
    gamestate_base_dir: str,
    split: str
):
    # 画像をコピーする準備
    v3_image_path = os.path.join(game_path, "Frames-v3", image_name)
    gamestate_image_path = os.path.join(gamestate_base_dir, split, gamestate_data["info"]["name"], gamestate_data["info"]["im_dir"], image_name)
    Path(v3_image_path).parent.mkdir(parents=True, exist_ok=True)
    Path(gamestate_image_path).parent.mkdir(parents=True, exist_ok=True)

    # なければ解凍
    try:
        zipfilepath = ""
        if not os.path.exists(v3_image_path):
            tqdm.write(f"unzipping: {v3_image_path=}")
            # unzip
            zipfilepath = os.path.join(game_path, "Frames-v3.zip")
            with zipfile.ZipFile(zipfilepath, 'r') as zippedFrames:  # noqa: N806
                with zippedFrames.open(image_name) as imginfo:
                    with open(v3_image_path, 'wb') as f:
                        f.write(imginfo.read())
    except FileExistsError:
        # すでに解凍済み
        pass
    except Exception as e:
        tqdm.write(f"{zipfilepath=}")
        tqdm.write(f"{e=}")
        # 画像が使えないから、ラベルまるごとスキップする
        return

    # コピー
    if not os.path.exists(gamestate_image_path):
        tqdm.write(f"Image copying: {v3_image_path=}")
        with open(v3_image_path, 'rb') as f:
            with open(gamestate_image_path, 'wb') as g:
                g.write(f.read())


def process_annotations(
    image_name: str,
    action_data: dict,
    gamestate_data: dict,
    super_id: str,
    person_identifier: PersonIdentifier
):
    tqdm.write(f"Annotation: {image_name}")

    for idx, bbox in enumerate(action_data['bboxes']):
        role, team = convert_to_attributes(bbox['class'])
        person_id = person_identifier.get_person_id(
            role,
            normalize_team(team, action_data['imageMetadata']['gameTime']),
            bbox['ID']
        )
        annotation = {
            "id": f"{super_id}-{Path(image_name).stem}-{idx + 1}",
            "image_id": f"{super_id}-{Path(image_name).stem}",
            "track_id": person_id,
            "supercategory": "object", # 物体を表す識別子
            "category_id": 1,  # 人物のカテゴリID
            "attributes": {
                "role": role,
                "jersey": bbox['ID'] if (isinstance(bbox['ID'], str) and bbox['ID'].isnumeric()) else None,
                "team": team  # 'left' or 'right'
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

    tqdm.write(f"End annotation: {image_name}")


def process_scene(
    scene: str,
    gamestate_data: dict,
    v3_data: dict,
    game_path: str,
    gamestate_base_dir: str,
    split: str,
    super_id: str,
    person_identifier: PersonIdentifier,
):
    tqdm.write(f"scene: {scene}")
    for image_name, action_data in tqdm(v3_data[scene].items()):
        tqdm.write(f"image name: {image_name}")
        # image情報
        image_info = get_image_dict(image_name, action_data, super_id)
        gamestate_data['images'].append(image_info)
        # 画像をコピー
        process_copying_image(image_name, action_data, gamestate_data, game_path, gamestate_base_dir, split)
        # アノテーションを追加
        process_annotations(
            image_name,
            action_data,
            gamestate_data,
            super_id,
            person_identifier=person_identifier
        )
    tqdm.write(f"End scene: {scene}")


def convert_to_gamestate(game_path, gamestate_base_dir, resol720p=False):
    """Labels-v3.json から Labels-GameState.json 形式に変換する"""
    tqdm.write(f"Start game: {game_path}")

    v3_json_path = os.path.join(game_path, "Labels-v3.json")
    v3_data = json.load(open(v3_json_path))
    (
        gamestate_data,
        super_id,
        game_id,
        split,
        metadata,
        list_actions,
        list_replays,
        half
    ) = get_gamestate_dict_and_metadatas(
        v3_data,
        resol720p
    )
    person_identifier = PersonIdentifier(v3_data)
    for scene in tqdm(['actions', 'replays']):
        # シーンごとに処理
        process_scene(
            scene,
            gamestate_data,
            v3_data,
            game_path,
            gamestate_base_dir,
            split,
            super_id,
            person_identifier=person_identifier
        )

    # 書き出し
    gamestate_json_path = os.path.join(gamestate_base_dir, split, gamestate_data["info"]["name"], "Labels-GameState.json")
    with open(gamestate_json_path, 'w') as f:
        json.dump(gamestate_data, f, indent=4)

    tqdm.write(f"End game: {game_id}")

if __name__ == "__main__":
    args = V3Json2GsrArguments().parse_args()
    # Example usage
    games = getListGames("all", task="frames")

    for game in tqdm(games):
        game_path = os.path.join(args.SoccerNet_path, game)
        convert_to_gamestate(game_path, args.output_base_path)
