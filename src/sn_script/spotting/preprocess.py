# 前処理
import pandas as pd
from sn_script.csv_utils import gametime_to_seconds


def preprocess_gametime(df: pd.DataFrame):
    df["half"] = df["gameTime"].str.split(" - ").str[0].astype(float)
    df["time"] = df["gameTime"].str.split(" - ").str[1].map(gametime_to_seconds).astype(float)
    df["game"] = df["game"].str.rstrip("/")
    return df
