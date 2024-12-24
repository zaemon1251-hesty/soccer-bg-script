# フレームごとのLaplace varianceと付加的情報の割合の相関
import os
import traceback
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

# pandas tqdm
tqdm.pandas()


class Args(Tap):
    soccernet_path: str = "/raid_elmo/home/lr/moriy/SoccerNet"
    resolution: str = "720p"
    scbi_csv: str = "/raid_elmo/home/lr/moriy/SoccerNet/commentary_analysis/stable/scbi-v2.csv"

def get_frame(file_path, time):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time * fps))
    ret, frame = cap.read()
    cap.release()
    return frame


def get_laplacian_variance(frame):
    """
    フレームのシャープさを表すLaplacian varianceを計算

    シャープであれば、Laplacian varianceは大きくなる

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    variance = np.var(laplacian_abs)
    return variance


def get_laplacian_sum(frame):
    """
    フレームのシャープさを表すLaplacian varianceを計算

    return スカラー値

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    return np.sum(laplacian_abs)


def calc_laplacian_variance(row, soccernet_path, resolution, method="variance", duration="frame"):
    if pd.isna(row['game']) or pd.isna(row['half']) or pd.isna(row['start']):
            return None
    if method == "variance":
        get_laplacian = get_laplacian_variance
    elif method == "sum":
        get_laplacian = get_laplacian_sum


    # フレームごとのLaplacian varianceを計算
    game = row['game']
    half = int(row['half'])
    time = float(row['start'])

    video_path = os.path.join(soccernet_path, game, f"{half}_{resolution}.mkv")

    if duration == "frame":
        try:
            frame = get_frame(video_path, time)
            return get_laplacian(frame)
        except Exception:
            return None
    elif duration == "mean_before_15sec":
        try:
            frames = [get_frame(video_path, max(time - i, 0)) for i in range(15)]
            with Pool(15) as p:
                return np.mean(p.map(get_laplacian, frames))
        except Exception:
            traceback.print_exc()
            return None
    else:
        raise ValueError(f"Invalid duration: {duration}")


if __name__ == "__main__":
    args = Args().parse_args()
    soccernet_path = args.soccernet_path
    resolution = args.resolution
    scbi_csv = args.scbi_csv

    scbi_df = pd.read_csv(scbi_csv)
    partial_calc_laplacian_variance = partial(
        calc_laplacian_variance, soccernet_path=soccernet_path, resolution=resolution,
        method="variance", duration="mean_before_15sec"
    )
    scbi_df['laplacian_variance'] = scbi_df.progress_apply(partial_calc_laplacian_variance, axis=1)
    scbi_df.to_csv("scbi_v2_laplacian_variance.csv", index=False)
