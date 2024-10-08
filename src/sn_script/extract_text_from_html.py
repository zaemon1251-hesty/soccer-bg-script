import sys
from pathlib import Path

import trafilatura
from loguru import logger
from tqdm import tqdm

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    sys.path.append(".")
    from src.sn_script.config import Config


def extract_text_from_html(input_path: Path, output_dir: Path):
    basename = input_path.stem + ".txt"
    output_path = output_dir / basename
    if output_path.exists():
        return
    html = open(input_path).read()
    text = trafilatura.extract(html)
    if text is not None:
        with open(output_path, "w") as f:
            f.write(text)
    else:
        raise RuntimeError("failed to extract text for some reason")


if __name__ == "__main__":
    input_dir = Config.target_base_dir / "knowledge_base_raw"
    output_dir = Config.target_base_dir / "knowledge_base_text"

    output_dir.mkdir(exist_ok=True, parents=True)

    for input_path in tqdm(list(input_dir.glob("*.html"))):
        try:
            extract_text_from_html(input_path, output_dir)
        except Exception as e:
            logger.debug(e)
            logger.info("failed to extract text from", input_path)
            continue
