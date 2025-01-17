import os
from typing import List, Literal, Union  # noqa: UP035 python3.8で動くようにするため

from loguru import logger
from tap import Tap

from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from SoccerNet.utils import getListGames

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


PASSWORD = os.environ.get("SOCCERNET_PASSWORD")

LOCAL_DIRECTORY = Config.base_dir

# GAMES[game_id] = game_name
GAMES: List[str] = getListGames("all")  # noqa: UP006


class Arguments(Tap):
    task: Literal["caption", "gsr", "jersey", "spotting", "tracking", "video", "v3"] = "caption"
    type: Literal["challenge-label", "underbar", "default"] = "default"
    target_game: str = "all"  # noqa: UP006, UP007


def download_gsr_label(arg: Arguments):
    # LOCAL_DIRECTORY を無視して、label_dir にダウンロードする
    label_dir = "/raid_elmo/home/lr/moriy/SoccerNetGS"

    my_soccer_net_downloader = SNdl(LocalDirectory=label_dir)
    my_soccer_net_downloader.downloadDataTask(task="gamestate-2024", split=["challenge"])

def download_jersey_label(arg: Arguments):
    my_soccer_net_downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    my_soccer_net_downloader.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])

def download_spotting_label(arg: Arguments):
    my_soccer_net_downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    my_soccer_net_downloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])

def download_tacking_label(arg: Arguments):
    my_soccer_net_downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    my_soccer_net_downloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
    my_soccer_net_downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])

def download_v3_label(arg: Arguments):
    my_soccer_net_downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    my_soccer_net_downloader.downloadGames(files=["Labels-v3.json", "Frames-v3.zip"], split=["train","valid","test"], task="frames")


def download_caption_label(arg: Arguments):
    downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    # downloader.password = PASSWORD

    if arg.type == "challenge-label":
        downloader.downloadGames(
            ["Labels-caption.json"],
            split=["challenge"],
            verbose=True,
            task="caption",
        )

    elif arg.type == "underbar":
        #  これ、何がダウンロードできるの?
        downloader.downloadDataTask(
            split=["test_labels", "challenge_labels"],
            verbose=True,
            task="caption",
        )

    elif arg.type == "default":
        target_ids = get_target_ids(arg.target_game)

        logger.info(f"Target ids are {target_ids}.")

        for target_id in target_ids:
            downloader.LocalDirectory = LOCAL_DIRECTORY

            logger.info(f"Downloading {GAMES[target_id]}...")

            downloader.downloadGameIndex(
                target_id, files=["Labels-caption.json"], verbose=1,
            )

    else:
        raise ValueError("Invalid type.")


def download_video(arg: Arguments):
    downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    downloader.password = PASSWORD

    target_ids = [i for i, game in enumerate(GAMES) if game in arg.target_game]

    logger.info(f"Target ids are {target_ids}.")

    for target_id in target_ids:
        downloader.LocalDirectory = LOCAL_DIRECTORY

        logger.info(f"Downloading {GAMES[target_id]}...")

        downloader.downloadGameIndex(
            target_id,
            files=[
                "1_720p.mkv",
                "2_720p.mkv",
            ],
            verbose=1,
        )

    logger.info("Done!")

def get_target_ids(target_games: Union[List[str], str]) -> List[int]: # noqa: UP007, UP006
    if target_games == "all":
        target_games = []
        for target in Config.targets:
            target_games.append(target.strip().rstrip("/"))
        return  [i for i, game in enumerate(GAMES) if game in target_games]

    elif isinstance(target_games, list):
        return [i for i, game in enumerate(GAMES) if game in target_games]

    else:
        return [GAMES.index(target_games)]


def main(arg: Arguments):
    if arg.task == "caption":
        download_caption_label(arg)

    elif arg.task == "gsr":
        download_gsr_label(arg)

    elif arg.task == "jersey":
        download_jersey_label(arg)

    elif arg.task == "spotting":
        download_spotting_label(arg)

    elif arg.task == "tracking":
        download_tacking_label(arg)

    elif arg.task == "video":
        download_video(arg)

    elif arg.task == "v3":
        download_v3_label(arg)

    else:
        raise ValueError("Invalid task.")


if __name__ == "__main__":
    logger.info(f"{LOCAL_DIRECTORY} is used.")

    args = Arguments().parse_args()

    main(args)

    logger.info("Done!")
