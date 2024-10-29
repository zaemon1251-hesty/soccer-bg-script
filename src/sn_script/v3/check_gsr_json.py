import json
import os

import numpy as np
import pandas as pd
from tap import Tap


class CheckGsrJsonArguments(Tap):
    input_dir: str


def sanitize_bbox_ltwh(bbox: np.array, image_shape=None, rounded=False):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        bbox (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, width, height]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the bounding box coordinates, type becomes int.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, width, height]`.
    """
    assert isinstance(
        bbox, np.ndarray
    ), f"Expected bbox to be of type np.ndarray, got {type(bbox)}"
    assert bbox.shape == (4,), f"Expected bbox to be of shape (4,), got {bbox.shape}"
    if image_shape is not None:
        bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))
        bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))
        bbox[2] = max(1, min(bbox[2], image_shape[0] - 1 - bbox[0]))
        bbox[3] = max(1, min(bbox[3], image_shape[1] - 1 - bbox[1]))
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox
def ltwh_to_xywh(bbox, image_shape=None, rounded=False):
    """
    Converts coordinates `[left, top, w, h]` to `[center_x, center_y, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltwh(bbox, image_shape)
    bbox = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]])
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox

def sanitize_bbox_xywh(bbox, images_shape=None, rounded=False):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        box (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[x_center, y_center, width, height]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the bounding box coordinates, type becomes int.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[x_center, y_center, width, height]`.
    """
    return ltwh_to_xywh(xywh_to_ltwh(bbox, images_shape), images_shape, rounded)


def xywh_to_ltwh(bbox, image_shape=None, rounded=False):
    """
    Converts coordinates `[center_x, center_y, w, h]` to `[left, top, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_xywh(bbox, image_shape)
    bbox = np.array([bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]])
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox

def transform_bbox_image(row):
    row = row.astype(float)
    return {"x": row[0], "y": row[1], "w": row[2], "h": row[3]}


def extract_category(attributes):
    if attributes['role'] == 'goalkeeper':
        team = attributes['team']
        role = "goalkeeper"
        jersey_number = None
        if attributes['jersey'] is not None:
            jersey_number = int(attributes['jersey']) if attributes['jersey'].isdigit() else None
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif attributes['role'] == "player":
        team = attributes['team']
        role = "player"
        jersey_number = None
        if attributes['jersey'] is not None:
            jersey_number = int(attributes['jersey']) if attributes['jersey'].isdigit() is not None else None
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif attributes['role'] == "referee":
        team = None
        role = "referee"
        jersey_number = None
        # position = additional_info  # TODO no position for referee in json file (referee's position is not specified in the dataset)
        category = f"{role}"
    elif attributes['role'] == "ball":
        team = None
        role = "ball"
        jersey_number = None
        category = f"{role}"
    else:
        assert attributes['role'] == "other" or attributes['role'] is None
        team = None
        role = "other"
        jersey_number = None
        category = f"{role}"
    return category


def dict_to_df_detections(annotation_dict, categories_list):
    df = pd.DataFrame.from_dict(annotation_dict)

    annotations_pitch_camera = df.loc[df['supercategory'] != 'object']   # remove the rows with non-human categories

    df = df.loc[df['supercategory'] == 'object']        # remove the rows with non-human categories

    df['bbox_ltwh'] = df.apply(lambda row: xywh_to_ltwh([row['bbox_image']['x_center'], row['bbox_image']['y_center'], row['bbox_image']['w'], row['bbox_image']['h']]), axis=1)
    df['team'] = df.apply(lambda row: row['attributes']['team'], axis=1)
    df['team_cluster'] = (df["team"] == "left").astype(float)
    df['role'] = df.apply(lambda row: row['attributes']['role'], axis=1)
    df['jersey_number'] = df.apply(lambda row: row['attributes']['jersey'], axis=1)
    df['position'] = None # df.apply(lambda row: row['attributes']['position'], axis=1)         for now there is no position in the json file
    df['category'] = df.apply(lambda row: extract_category(row['attributes']), axis=1)
    df['track_id'] = df['track_id']
    # df['id'] = df['id']

    columns = ['id', 'image_id', 'track_id', 'bbox_ltwh', 'bbox_pitch', 'team_cluster',
               'team', 'role', 'jersey_number', 'position', 'category']
    df = df[columns]

    video_level_categories = list(df['category'].unique())

    return df, annotations_pitch_camera, video_level_categories

def read_json_file(file_path):
    try:
        with open(file_path) as file:
            file_json = json.load(file)
        return file_json
    except Exception:
        return None

def main(args: CheckGsrJsonArguments):
    print("checking json files in", args.input_dir)

    detection_df_list = []
    for video_folder in os.listdir(args.input_dir):
        video_folder_path = os.path.join(args.input_dir, video_folder)
        gamestate_path = os.path.join(video_folder_path, 'Labels-GameState.json')
        gamestate_data = read_json_file(gamestate_path)
        if gamestate_data is None:
            continue

        info_data = gamestate_data['info']
        # v3 なら、 super_id == video_id
        video_id = info_data.get("id", str(video_folder.split('-')[-1]))

        annotations_data = gamestate_data['annotations']
        categories_data = gamestate_data['categories']

        detections_df, _, _ = dict_to_df_detections(annotations_data, categories_data)
        detections_df['video_id'] = video_id
        detections_df['person_id'] = detections_df['track_id'].astype(str) + detections_df['video_id'].astype(str)

        detection_df_list.append(detections_df)

    all_detections_df = pd.concat(detection_df_list)

    all_detections_df['person_id'] = pd.factorize(all_detections_df['person_id'])[0]
    all_detections_df['id'] = all_detections_df['video_id'].astype(str) + "_" + all_detections_df['image_id'].astype(str) + "_" + all_detections_df['track_id'].astype(str)
    all_detections_df.set_index("id", drop=False, inplace=True)

    # id の重複がないかチェック
    id_dup_flg = len(all_detections_df) == len(all_detections_df['id'].unique())
    print("id は重複すべきでない:", "重複あり" if not id_dup_flg else "重複なし")

    # track_idは重複すべき
    track_id_dup_flg = len(all_detections_df) != len(all_detections_df['track_id'].unique())
    print("track_id は重複すべき:", "重複あり" if track_id_dup_flg else "重複なし")

    # player,goalkeeper はteamが絶対あるはず
    player_team_null_cnt = all_detections_df.loc[all_detections_df['role'].isin(['player', "goalkeeper"]), 'team'].isnull().sum()
    print("player の team は null であってはならない:", f"null あり ({player_team_null_cnt}件)" if player_team_null_cnt > 0 else "null なし")

    # 理解した
    # 元の実装だと、1つの image に同じ track_id を持つobject が複数あるのが NG
    # でも、今回の要件だと、チームが分かればいいので、同じ track_id を持つobject が複数あってもOK
    # となれば、そもそも bbox["ID"]ごとに分ける必要なかったかも

    # player,goalkeeper,referee なら、 pid と video_id は必須
    player_pid_null_cnt = all_detections_df.loc[all_detections_df['role'].isin(['player', "goalkeeper", "referee"]), 'person_id'].isnull().sum()
    print("player, goalkeeper, referee の person_id は null であってはならない:", f"null あり ({player_pid_null_cnt}件)" if player_pid_null_cnt > 0 else "null なし")

    player_video_id_null_cnt = all_detections_df.loc[all_detections_df['role'].isin(['player', "goalkeeper", "referee"]), 'video_id'].isnull().sum()
    print("player, goalkeeper, referee の video_id は null であってはならない:", f"null あり ({player_video_id_null_cnt}件)" if player_video_id_null_cnt > 0 else "null なし")


if __name__ == "__main__":
    args = CheckGsrJsonArguments().parse_args()
    main(args)
