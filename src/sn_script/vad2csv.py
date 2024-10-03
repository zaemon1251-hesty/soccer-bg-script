import csv

import pandas as pd
from tap import Tap


# コマンドライン引数を設定
class Vad2CsvArguments(Tap):
    input_vad: str
    output_csv: str
    game: str
    half: str


if __name__ == "__main__":
    args = Vad2CsvArguments().parse_args()

    # vad(tsv)ファイルを読み込む
    with open(args.input_vad) as f:
        data = pd.read_csv(f, delimiter="\t")
        data.columns = ["start", "end"]

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text"])
        for row_id, row in data.iterrows():
            writer.writerow([row_id, args.game, args.half, row["start"], row["end"], None])
