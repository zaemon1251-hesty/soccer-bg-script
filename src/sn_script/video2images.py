from __future__ import annotations

import os
import subprocess
from itertools import product

from SoccerNet.Downloader import getListGames
from tap import Tap
from tqdm import tqdm

GAMES: list[str] = getListGames("all", task="caption")


class Video2ImangeArgments(Tap):
    SoccerNet_path: str
    output_base_path: str


def game_to_id(game: str, half: int):
    """001_1, 002_2, 003_1, ..."""
    assert game in GAMES, "invalid game"
    game_index = GAMES.index(game)
    video_id = f"{game_index:03d}_{half}"
    return video_id


def generate_images_from_video(input_path, output_dir):
    output_path_template = os.path.join(output_dir, "%06d.jpg")

    command = [
        "ffmpeg",
        f'-i "{input_path}"',
        "-threads 1",
        "-r 2",
        "-loglevel panic",
        "-f image2",
        f'"{output_path_template}"',
    ]

    command = " ".join(command)
    try:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False

    return True


def main(args: Video2ImangeArgments):
    debug_count = 0
    game_list = {}
    for split in ["train", "valid", "test", "challenge"]:
        game_list[split] = getListGames(split, task="caption")
        print(f"Generating images for {split} split")
        for game, half in tqdm(list(product(game_list[split], [1, 2]))):
            if debug_count > 10:
                break

            video_id = game_to_id(game, half)
            input_path = os.path.join(args.SoccerNet_path, game, f"{half}_224p.mkv")
            output_dir  = os.path.join(args.output_base_path, f"{split}/SNGS-{video_id}/img1/")

            os.makedirs(output_dir, exist_ok=True)
            generate_images_from_video(input_path, output_dir)

            debug_count += 1

        print(f"Done for {split} split")

if __name__ == "__main__":
    args = Video2ImangeArgments().parse_args()
    main(args)
