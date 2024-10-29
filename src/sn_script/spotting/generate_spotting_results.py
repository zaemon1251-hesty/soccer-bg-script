import json  # noqa: I001
import os
from pathlib import Path
import pandas as pd
from tap import Tap
from sn_script.config import binary_category_name


class CheckGsrJsonArguments(Tap):
    input_csv: str
    output_json_dir: str
    target_col: str = "start"


def main(args: CheckGsrJsonArguments):
    scbi_split_df = pd.read_csv(args.input_csv)
    assert set(scbi_split_df.columns) >= {"id", "game", "half", "start", "end", "text", binary_category_name, "split"}

    all_json_data = dict()

    game_list = scbi_split_df.groupby("game").groups.keys()
    for game in game_list:
        json_data = {
            "predictions": [],
        }
        video_df = scbi_split_df[scbi_split_df["game"] == game].sort_values(by=["half", "start"])
        for _, row in video_df.iterrows():
            class_i = row[binary_category_name]
            target = row[args.target_col]
            half = row["half"]
            split = row["split"]
            seconds = int(target) % 60
            minutes = int(target) // 60
            prediction_data = dict()
            prediction_data["gameTime"] = (
                f"{int(half)} - {int(minutes):02d}:{int(seconds):02d}"
            )
            prediction_data["label"] = "comments"
            prediction_data["category"] = class_i
            prediction_data["position"] = str(int((target) * 1000))
            prediction_data["half"] = int(half)
            prediction_data["confidence"] = "1.0"
            json_data["predictions"].append(prediction_data)

        json_data["predictions"] = sorted(
            json_data["predictions"],
            key=lambda x: (int(x["half"]), int(x["position"])),
        )
        json_data["game"] = game

        output_json = os.path.join(
            args.output_json_dir,
            f"outputs/{split}",
            game,
            "results_spotting.json"
        )
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)

        with open(output_json, "w") as output_file:
            json.dump(json_data, output_file, indent=4)

        print(game)
        # 長さ一緒か チェック
        print(f"{len(json_data['predictions'])=} {len(video_df)=}")

        all_json_data[game] = json_data

    return all_json_data


if __name__ == "__main__":
    main(CheckGsrJsonArguments().parse_args())
