import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import whisper
import whisperx
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from loguru import logger
from tap import Tap
from tqdm import tqdm
from whisperx.asr import FasterWhisperPipeline

try:
    from sn_script.config import Config, half_number
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config, half_number


SttModels = Literal[
    "whisper-large-v2",
    "conformer",
    "reason",
    "faster-whisper-large-v2",
    "faster-whisper-large-v3",
    "whisperx-large-v2",
    "whisperx-large-v3",
]

# コマンドライン引数を設定
class Speech2TextArguments(Tap):
    target_game: str = "all"
    suffix: str = ""
    model: SttModels = "whisper-large-v2"
    half: str = 1
    hf_token: str = ""

    # whisperx の VADの設定
    vad_onset: float = 0.500
    vad_offset: float = 0.377

    # whisperxのmergeの設定
    chunk_size: int = 30

    # faster-whisper の VADの設定
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400


def get_stt_model(model_name: SttModels, args: Speech2TextArguments=None):
    if model_name == "whisper-large-v2":
        return whisper.load_model("large")
    elif model_name == "faster-whisper-large-v2":
        return WhisperModel(
            "large-v2",
            device="cuda",
        )
    elif model_name == "faster-whisper-large-v3":
        return WhisperModel(
            "large-v3",
            device="cuda",
        )
    elif model_name == "whisperx-large-v2":
        return whisperx.load_model("large-v2", device="cuda", vad_options=args.vad_option)
    elif model_name == "whisperx-large-v3":
        return whisperx.load_model("large-v3", device="cuda", vad_options=args.vad_option)
    elif model_name == "conformer":
        raise NotImplementedError("Conformer model is not implemented yet")
    elif model_name == "reason":
        raise NotImplementedError("Reason model is not implemented yet")
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def main(args: Speech2TextArguments):
    target_games = []
    if args.target_game == "all":
        target_games = Config.targets
    else:
        target_games = [args.target_game]

    if args.model in ("faster-whisper-large-v2", "faster-whisper-large-v3"):
        vad_option = VadOptions(
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            max_speech_duration_s=args.max_speech_duration_s,
            min_silence_duration_ms=args.min_silence_duration_ms,
            window_size_samples=args.window_size_samples,
            speech_pad_ms=args.speech_pad_ms,
        )
        args.vad_option = vad_option._asdict()
    elif args.model in ("whisperx-large-v2", "whisperx-large-v3"):
        vad_option = {
            "vad_onset": args.vad_onset,
            "vad_offset": args.vad_offset,
        }
        args.vad_option = vad_option

    model = get_stt_model(args.model, args=args)

    for target in tqdm(target_games):
        target_dir_path = Config.base_dir / target

        if not os.path.exists(target_dir_path):
            logger.info(f"Video not found: {target_dir_path}")
            continue

        run_transcribe(model, target_dir_path, args=args)


def run_transcribe(model, game_dir: Path, args: Speech2TextArguments):
    video_path = game_dir / f"{args.half}_224p.mkv"
    output_text_path = game_dir / f"{args.half}_224p{args.suffix}.txt"
    output_json_path = game_dir / f"{args.half}_224p{args.suffix}.json"

    if args.model in ("faster-whisper-large-v2", "faster-whisper-large-v3"):
        segments, info = model.transcribe(
            str(video_path),
            vad_filter=True,
            vad_parameters=args.vad_option,
        )
        # iteratorからlistに変換
        segments = list(segments)

        with open(output_text_path, "w") as f:
            text = " ".join([segment.text for segment in segments])
            f.writelines(text)
        with open(output_json_path, "w") as f:
            transcription_output = {
                "segments": [
                    segment._asdict()
                    for segment in segments
                ],
                "info": info._asdict(),
            }
            json.dump(transcription_output, f, indent=2, ensure_ascii=False)
    elif args.model in ("whisperx-large-v2", "whisperx-large-v3"):
        assert isinstance(model, FasterWhisperPipeline), f"Model is not whisperx: {model}"

        # ASR
        audio = whisperx.load_audio(str(video_path))
        result = model.transcribe(
            audio,
            chunk_size=args.chunk_size,
        )
        language = result["language"]

        # forced-alignment
        model_a, metadata = whisperx.load_align_model(language_code=language, device="cuda")
        result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda", return_char_alignments=False)

        # 話者分離
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_token, device="cuda")
        diarize_segments = diarize_model(str(video_path))
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # メタ情報の追加
        result["language"] = language
        result["arguments"] = args.as_dict()

        with open(output_text_path, "w") as f:
            text = " ".join([segment["text"] for segment in result["segments"]])
            f.writelines(text)
        with open(output_json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 正しくファイルが作成されたか確認
        assert output_text_path.exists()
        assert output_json_path.exists()

    elif args.model == "whisper-large-v2":
        assert model.is_multilingual
        result = model.transcribe(
            str(video_path),
            verbose=True
        )
        with open(output_text_path, "w") as f:
            f.writelines(result["text"])
        with open(output_json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented yet")


if __name__ == "__main__":
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    args = Speech2TextArguments().parse_args()

    logger.add(
        f"logs/llm_anotator_{time_str}.log",
    )
    main(args)
