import zipfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd


@dataclass
class GSRStates:
    gamestate_zip_file: Path
    gamestate_summary_file: Path
    gamestate_file: Path
    gamestate_images_file: Path
    gamestate_df: pd.DataFrame
    image_df: pd.DataFrame

    @classmethod
    def initialize(cls, gamestate_zip_file: Path):
        parent_dir = gamestate_zip_file.parent

        if not (parent_dir / "summary.json").exists():
            zipfile.ZipFile(gamestate_zip_file).extractall(parent_dir)

        # 見つける the summary.json file
        gamestate_summary_file = parent_dir / "summary.json"

        # 見つける the / {data_id}.pkl file
        gamestate_files = []
        gamestate_images_files = []
        for path in glob(str(parent_dir / "*.pkl")):
            if "_image.pkl" in path:
                gamestate_images_files.append(Path(path))
            else:
                # {data_id} はどこにも記載されていないので、とりあえず_image.pklではないものを gamestate_file とする
                gamestate_files.append(Path(path))

        gamestate_df_list = []
        image_df_list = []

        gamestate_files = sorted(gamestate_files)
        gamestate_images_files = sorted(gamestate_images_files)
        for gamestate_file, gamestate_images_file in zip(gamestate_files, gamestate_images_files):
            gamestate_df = pd.read_pickle(gamestate_file)
            image_df = pd.read_pickle(gamestate_images_file)

            gamestate_df_list.append(gamestate_df)
            image_df_list.append(image_df)

        gamestate_df = pd.concat(gamestate_df_list, ignore_index=True)
        image_df = pd.concat(image_df_list, ignore_index=True)

        return GSRStates(
            gamestate_zip_file=gamestate_zip_file,
            gamestate_summary_file=gamestate_summary_file,
            gamestate_file=gamestate_file,
            gamestate_images_file=gamestate_images_file,
            gamestate_df=gamestate_df,
            image_df=image_df
        )
