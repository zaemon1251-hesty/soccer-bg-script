import json
from itertools import product

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys
    sys.path.append(".")
    from src.sn_script.config import Config


def sec_to_hms(seconds: float):
    "s -> hh:mm:ss.fff"
    floating = float(seconds) - int(seconds)
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{int(floating * 1000):03d}"


def load_json(game, half):
    json_basename = f"{half}_224p_stable_en.json"
    json_path = Config.base_dir / game / json_basename
    try:
        return json.load(open(json_path))
    except Exception:
        return None


def write_vtt(data, output_vtt_path):
    with open(output_vtt_path, "w") as f:
        f.write("WEBVTT\n\n")
        for item in data["segments"]:
            f.write(
                f"{sec_to_hms(item['start'])} --> {sec_to_hms(item['end'])}\n"
            )
            f.write(f"{item['text'].strip()}\n")
            f.write("\n")
    return True


def to_vtt():
    game_list = Config.targets
    for game,half in product(game_list, [1, 2]):
        data = load_json(game, half)
        if data is None:
            continue
        output_vtt_path = Config.base_dir / game / f"{half}_asr.vtt"
        write_vtt(data, output_vtt_path)


if __name__ == "__main__":
    to_vtt()
