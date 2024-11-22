"""
gamestate_data = {
    "info": {
        "version": "0.1",
        "game_id": game_id,
        "id": super_id,
        "num_tracklets": "11",
        "action_position": 0,
        "action_class": "action",
        "visibility": True,
        "game_time_start": first_action["imageMetadata"]["gameTime"],
        "game_time_stop": last_action["imageMetadata"]["gameTime"],
        "clip_start": first_action["imageMetadata"]["position"],
        "clip_stop": last_action["imageMetadata"]["position"],
        "name": f"SNGS-{super_id}",
        "im_dir": "img1",
        "frame_rate": 5,
        "seq_length": len(list_actions) + len(list_replays),
        "im_ext": ".png"
    },
    "images": [image_info for image_info in image_infos],
    "annotations": [annotation for annotation in annotations],
    "categories": MASTER_CATRGORIES
}

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
"""
import json
import os
from glob import glob

from tap import Tap
from tqdm import tqdm


class GsrToOcrArguments(Tap):
    gamestate_path: str
    output_base_path: str


def convert_gamestate_to_ocr_format(gsr_datas, output_path):
    """
    Converts a list of game state data to OCR format and saves it to the specified path.
    """
    # Initialize the OCR dataset structure
    ocr_data = {
        "metainfo": {
            "dataset_type": "TextRecogDataset",
            "task_name": "textrecog"
        },
        "data_list": []
    }
    for gsr_data in gsr_datas:
        images = gsr_data.get("images", [])
        annotations = gsr_data.get("annotations", [])

        for annotation in annotations:
            image_id = annotation["image_id"]
            image_info = next(image for image in images if image["image_id"] == image_id)
            ocr_entry = {
                "instances": [{"text": annotation["attributes"]["jersey"]}],
                "img_path": image_info["file_name"],
            }
            ocr_data["data_list"].append(ocr_entry)

    # Save the OCR data to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as outfile:
        json.dump(ocr_data, outfile, indent=4)
    print(f"OCR data saved to {output_path}")

if __name__ == "__main__":
    args = GsrToOcrArguments().parse_args()
    gsr_data = []

    # args.gamestate_path / $SPLIT (train, valid, test) / $GAME_ID / Labels-GameState.json
    for gsr_path in glob(os.path.join(args.gamestate_path, "*", "*", "Labels-GameState.json")):
        gsr_data.append(json.load(open(gsr_path)))

    for split in tqdm(["train", "valid", "test"]):
        # Define output JSON path
        output_json_path = os.path.join(args.output_base_path, f"textrecog_{split}.json")

        # Filter data for the split (assuming split-specific logic)
        gsr_datas = [
            data for data in gsr_data
            if split in gsr_path.split(os.sep)  # Assumes directory names contain 'train', 'valid', 'test'
        ]
        print(f"Processing {len(gsr_datas)} games for split: {split}")
        # Convert and save
        convert_gamestate_to_ocr_format(gsr_datas, output_json_path)
