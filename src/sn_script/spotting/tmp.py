import os
from glob import glob

from SoccerNet.Downloader import getListGames

local_dir = (
    "/Users/heste/workspace/soccernet/sn-providing/data/spotting/commentary_gold"
)
local_games = glob(f"{local_dir}/**/results_spotting.json", recursive=True)
# 絶対パスから相対パスに変換 + 親フォルダの名前を取得
local_games = [
    os.path.relpath(os.path.dirname(path), local_dir)
    for path in local_games
]

v3_games = getListGames("all", task="frames")

intersection = set(local_games) & set(v3_games)
for game in intersection:
    if "england_epl" in game:
        print(game)
