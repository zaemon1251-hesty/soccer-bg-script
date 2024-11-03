import os
from pathlib import Path

from datasets import ClassLabel, Features, Value, load_dataset
from sn_script.config import binary_category_name
from tap import Tap


class GenerateHfDatasetArguments(Tap):
    csv_dir: Path
    hf_dataset_dir: Path
    push: bool = False
    dataset_name: str = "placeholder!!!!!!!"


def main(args: GenerateHfDatasetArguments):
    dataset_files = {
        "train": [(args.csv_dir / "train.csv").as_posix()],
        "validation": [(args.csv_dir / "valid.csv").as_posix()],
        "test": [(args.csv_dir / "test.csv").as_posix()],
    }
    features = Features(
        {
            "id": Value("int32", id=None),
            "game": Value("string", id=None),
            "half": Value("int32"),
            "start": Value("float32"),
            "end": Value("float32"),
            "text": Value("string"),
            binary_category_name: ClassLabel(num_classes=2, names=['1', '0']),
        }
    )
    dataset = load_dataset("csv", data_files=dataset_files, features=features)
    if args.push:
        dataset.push_to_hub(
            args.dataset_name,
            token=os.getenv("HF_TOKEN"),
        )

    dataset.save_to_disk(args.hf_dataset_dir)


if __name__ == "__main__":
    args = GenerateHfDatasetArguments().parse_args()
    main(args)
