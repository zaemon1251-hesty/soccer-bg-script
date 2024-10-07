import os
import subprocess
from itertools import product
from typing import List

from loguru import logger
from SoccerNet.Downloader import getListGames
from tap import Tap
from tqdm import tqdm

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


GAMES: List[str] = getListGames("all", task="caption") # noqa


class Video2ImangeArgments(Tap):
    SoccerNet_path: str
    output_base_path: str
    resolution: str = "224p"
    fps: int = 2
    threads: int = 1
    target_game: str = "all"


def game_to_id(game: str, half: int):
    """001_1, 002_2, 003_1, ..."""
    assert game in GAMES, "invalid game"
    game_index = GAMES.index(game)
    video_id = f"{game_index:03d}_{half}"
    return video_id


def generate_images_from_video(input_path, output_dir, fps=2, threads=1):
    output_path_template = os.path.join(output_dir, "%06d.jpg")

    command = [
        "ffmpeg",
        f'-i "{input_path}"',
        f"-threads {threads}",
        f"-r {fps}",
        "-loglevel panic",
        "-q:v 1",
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
    game_list = {}

    for split in ["train", "valid", "test", "challenge"]:
        game_list[split] = getListGames(split, task="caption")

        #　対象ゲームだけをフィルタリング
        if args.target_game != "all":
            game_list[split] = [game for game in game_list[split] if game == args.target_game]

        logger.info(f"Generating images for {split} split")
        logger.info(f"Total {len(game_list[split])} games")

        for game, half in tqdm(list(product(game_list[split], [1, 2]))): # 全ての試合 x 前後半
            video_id = game_to_id(game, half)
            input_path = os.path.join(args.SoccerNet_path, game, f"{half}_{args.resolution}.mkv")
            output_dir  = os.path.join(args.output_base_path, f"{split}/SNGS-{video_id}/img1/")

            os.makedirs(output_dir, exist_ok=True)
            generate_images_from_video(input_path, output_dir, fps=args.fps, threads=args.threads)


        logger.info(f"Done for {split} split")

if __name__ == "__main__":
    args = Video2ImangeArgments().parse_args()
    main(args)
