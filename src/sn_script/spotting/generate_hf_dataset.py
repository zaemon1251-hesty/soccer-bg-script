import os
from pathlib import Path

from datasets import ClassLabel, DatasetDict, Features, Value, load_dataset
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


def describe_generated_hf_dataset(args: GenerateHfDatasetArguments):
    scbi_hf: DatasetDict = load_dataset(args.dataset_name)
    print(scbi_hf)
    # データのサイズなどを表示
    print(f"{scbi_hf['train'].num_rows=}")
    print(f"{scbi_hf['validation'].num_rows=}")
    print(f"{scbi_hf['test'].num_rows=}")

if __name__ == "__main__":
    args = GenerateHfDatasetArguments().parse_args()
    describe_generated_hf_dataset(args)
