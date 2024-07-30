import os

from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from SoccerNet.utils import getListGames

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config

PASSWORD = os.environ.get("SOCCERNET_PASSWORD")

LOCAL_DIRECTORY = Config.base_dir / "SoccerNet"


def main(choice: int = 0):
    downloader = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    # downloader.password = PASSWORD

    if choice == 0:
        downloader.downloadGames(
            ["Labels-caption.json"],
            split=["challenge"],
            verbose=True,
            task="caption",
        )

    elif choice == 1:
        downloader.downloadDataTask(
            split=["test_labels", "challenge_labels"],
            verbose=True,
            task="caption",
        )

    elif choice == 2:
        game_list = getListGames("all")

        target_games = []

        for target in Config.targets:
            target_games.append(target.strip().rstrip("/"))

        target_ids = [i for i, game in enumerate(game_list) if game in target_games]

        print(f"Target ids are {target_ids}.")

        for target_id in target_ids:
            downloader.LocalDirectory = LOCAL_DIRECTORY

            print(f"Downloading {game_list[target_id]}...")

            downloader.downloadGameIndex(
                target_id,
                files=[
                    # "1_720p.mkv",
                    # "2_720p.mkv",
                    "Labels-caption.json",
                    # "Labels.json",
                    # "Labels-v2.json",
                ],
                verbose=1,
            )

    else:
        raise ValueError("Invalid choice.")

if __name__ == "__main__":
    print(LOCAL_DIRECTORY, "is used.")
    print(f"Target games are {Config.targets}.")
    main(1)

    print("Done!")
