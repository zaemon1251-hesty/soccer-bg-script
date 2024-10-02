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


GAMES: List[str] = getListGames("all", task="caption")


class Video2WavArguments(Tap):
    SoccerNet_path: str
    threads: int = 4
    target_game: str = "all"
    resolution: str = "224p"  # 動画の解像度 (デフォルトは "720p")


def convert_video_to_wav(input_path, output_wav_path, threads=1):
    command = [
        "ffmpeg",
        "-i", input_path,        # 入力ファイル
        "-vn",                   # ビデオストリームを無視
        "-acodec", "pcm_s16le",  # 音声コーデック (WAV形式)
        "-ar", "16000",          # サンプリングレート 16kHz
        "-ac", "2",              # チャンネル数 (ステレオ)
        "-threads", str(threads),
        "-loglevel", "panic",    # ログを非表示に
        output_wav_path          # 出力ファイル (.wav)
    ]

    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {input_path} to WAV: {e}")
        return False


def main(args: Video2WavArguments):
    game_list = {}

    for split in ["train", "valid", "test", "challenge"]:
        game_list[split] = getListGames(split, task="caption")

        # 対象ゲームだけをフィルタリング
        if args.target_game != "all":
            game_list[split] = [game for game in game_list[split] if game == args.target_game]

        logger.info(f"Converting videos to WAV for {split} split")
        logger.info(f"Total {len(game_list[split])} games")

        for game, half in tqdm(list(product(game_list[split], [1, 2]))):  # 全ての試合 x 前後半
            input_path = os.path.join(args.SoccerNet_path, game, f"{half}_{args.resolution}.mkv")
            output_wav_path = os.path.join(args.SoccerNet_path, game, f"{half}_{args.resolution}.wav")

            if not os.path.exists(input_path):
                logger.error(f"Video file not found: {input_path}")
                continue

            # WAVファイルが既に存在する場合はスキップ
            if os.path.exists(output_wav_path):
                logger.info(f"WAV already exists for {input_path}, skipping...")
                continue

            # WAVファイルの出力ディレクトリを作成
            os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

            success = convert_video_to_wav(input_path, output_wav_path, threads=args.threads)
            if not success:
                logger.error(f"Failed to convert {game}, half {half} to WAV")

        logger.info(f"Done converting videos to WAV for {split} split")


if __name__ == "__main__":
    args = Video2WavArguments().parse_args()
    main(args)
