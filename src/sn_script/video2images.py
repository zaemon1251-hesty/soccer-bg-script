import os
import subprocess
from itertools import product
from typing import List, Optional  # noqa

import pandas as pd
from loguru import logger
from SoccerNet.Downloader import getListGames
from tap import Tap
from tqdm import tqdm

from sn_script.csv_utils import gametime_to_seconds

GAMES: List[str] = getListGames("all", task="caption")  # noqa


class Video2ImangeArgments(Tap):
    SoccerNet_path: str
    output_base_path: str
    resolution: str = "224p"
    fps: int = 2
    threads: int = 1
    target_game: str = "all"
    input_csv_path: Optional[str] = None
    output: str = "image"  # image or video
    window: int = 15
    start: Optional[int] = None
    end: Optional[int] = None
    half: Optional[int] = None


def game_to_id(game: str, half: int):
    """001_1, 002_2, 003_1, ..."""
    assert game in GAMES, "invalid game"
    game_index = GAMES.index(game)
    video_id = f"{game_index:03d}{half:02d}"
    return video_id


def generate_from_video(input_path, output_dir, fps=2, threads=1, start=None, end=None, output="image", video_id=None):
    command = [
        "ffmpeg",
        f'-i "{input_path}"',
        f"-threads {threads}",
        f"-r {fps}",
        "-loglevel panic",
        "-q:v 1",
    ]

    if start is not None:
        command.insert(2, f"-ss {start}")

    if end is not None:
        command.insert(2, f"-to {end}")

    if output == "image":
        output_path_template = os.path.join(output_dir, "%06d.jpg")
        command.append("-f image2")
        command.append(f'"{output_path_template}"')
    elif output == "video":
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        command.append(f'"{output_path}"')

    try:
        command = " ".join(command)
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False

    return True


def main(args: Video2ImangeArgments):
    game_list = {}

    halfs = [1, 2]
    if args.half is not None:
        halfs = [args.half]

    for split in ["train", "valid", "test", "challenge"]:
        game_list[split] = getListGames(split, task="caption")

        # 対象ゲームだけをフィルタリング
        if args.target_game != "all":
            game_list[split] = [game for game in game_list[split] if game == args.target_game]

        logger.info(f"Generating images for {split} split")
        logger.info(f"Total {len(game_list[split])} games")
        for game, half in tqdm(list(product(game_list[split], halfs))):  # 全ての試合 x 前後半
            video_id = game_to_id(game, half)
            input_path = os.path.join(args.SoccerNet_path, game, f"{half}_{args.resolution}.mkv")
            output_dir = os.path.join(args.output_base_path, f"{split}/SNGS-{video_id}/img1/")

            os.makedirs(output_dir, exist_ok=True)
            generate_from_video(
                input_path, output_dir, fps=args.fps, threads=args.threads, output=args.output, video_id=video_id, start=args.start, end=args.end
            )

        logger.info(f"Done for {split} ID")


def run_from_csv(args: Video2ImangeArgments):
    from SoccerNet.Downloader import SoccerNetDownloader

    game_df = pd.read_csv(args.input_csv_path)
    assert {"id", "game", "half", "time"} <= set(game_df.columns), "必要なカラムがありません"

    downloader = SoccerNetDownloader(args.SoccerNet_path)
    downloader.password = os.getenv("SOCCERNET_PASSWORD")

    for row in game_df.itertuples():
        video_id = f"{row.id:04d}"
        game = row.game
        half = row.half
        split = "test"

        if ":" in row.time:
            time = gametime_to_seconds(row.time)
        else:
            time = int(row.time)

        start = time - args.window
        end = time + args.window

        input_path = os.path.join(args.SoccerNet_path, game, f"{half}_{args.resolution}.mkv")

        if args.output == "image":
            output_dir = os.path.join(args.output_base_path, f"{split}/SNGS-{video_id}/img1/")
        elif args.output == "video":
            output_dir = os.path.join(args.output_base_path, "video")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_path):
            logger.info(f"Downloading {input_path}")
            downloader.downloadGame(game, files=[os.path.basename(input_path)])

        if len(os.listdir(output_dir)) >= 30:  # 30はテキトウ
            logger.info(f"Skip {row.id} split")
            continue

        generate_from_video(input_path, output_dir, fps=args.fps, threads=args.threads, start=start, end=end, output=args.output, video_id=video_id)

        logger.info(f"Done for {row.id} split")


if __name__ == "__main__":
    args = Video2ImangeArgments().parse_args()
    run_from_csv(args)
