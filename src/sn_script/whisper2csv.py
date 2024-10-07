import csv
import json

from tap import Tap


# コマンドライン引数を設定
class Whisper2CsvArguments(Tap):
    input_json: str
    output_csv: str
    game: str
    half: str


if __name__ == "__main__":
    args = Whisper2CsvArguments().parse_args()

    # JSONファイルを読み込む
    with open(args.input_json) as f:
        data = json.load(f)

    # CSVファイルを書き込む
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "game", "half", "start", "end", "text"])
        segments = data["segments"]
        for i, item in enumerate(segments):
            if "id" not in item:
                item["id"] = i
            writer.writerow(
                [
                    item["id"],
                    args.game,
                    args.half,
                    item["start"],
                    item["end"],
                    item["text"],
                ]
            )
