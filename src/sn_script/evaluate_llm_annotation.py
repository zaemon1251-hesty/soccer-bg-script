from __future__ import annotations

import ast
from abc import ABC, abstractmethod

import evaluate
import pandas as pd
from loguru import logger
from typing_extensions import TypedDict

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        half_number,
        model_type,
        random_seed,
        subcategory_name,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        half_number,
        model_type,
        random_seed,
        subcategory_name,
    )


LLM_ANOTATION_CSV_PATH = (
    Config.target_base_dir
    / f"{model_type}_{random_seed}_{half_number}_llm_annotation.csv"
)
HUMAN_ANOTATION_CSV_PATH = (
    Config.target_base_dir
    / f"{random_seed}_{half_number}_moriy_annotation_preprocessed.csv"
)


class LlmAnnotationResult(TypedDict):
    """
    Meta information for LLM annotation evaluation
    """

    model_type: str
    random_seed: int
    half_number: int
    comment_ids: int
    category: EvalIndicator
    subcategory: EvalIndicator | None


class EvalIndicator(TypedDict):
    exact_match: float | None
    f1: float
    precision: float
    recall: float
    accuracy: float


class EvaluateAnnotationBase(ABC):

    @abstractmethod
    def evaluate(self) -> LlmAnnotationResult:
        raise NotImplementedError


class EvaluateAnnotationMultilabel(EvaluateAnnotationBase):
    """calculate evaluation metrics for annotation"""

    def __init__(self) -> None:
        human_df = pd.read_csv(HUMAN_ANOTATION_CSV_PATH)
        llm_df = pd.read_csv(LLM_ANOTATION_CSV_PATH)

        # 文字列として格納されている配列を配列化
        for col_name in [category_name, subcategory_name]:
            human_df[col_name] = human_df[col_name].apply(lambda x: ast.literal_eval(x))
            llm_df[col_name] = llm_df[col_name].apply(lambda x: ast.literal_eval(x))

        assert human_df.shape == llm_df.shape
        assert human_df["id"].equals(llm_df["id"])

        self.llm_human_df = pd.merge(
            llm_df,
            human_df,
            how="inner",
            on="id",
            suffixes=("_llm", "_human"),
        )

    def evaluate(self) -> LlmAnnotationResult:
        """
        Mohammad S Sorower,
        A Literature Survey on Algorithms for Multi-label Learning, 2010.

        Exact Match Ratio (EMR) = mean^{n}_{i=1} (Yi = Zi)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Precision = mean^{n}_{i=1} (Yi \bigcap Zi) / Zi
        Recall = mean^{n}_{i=1} (Yi \bigcap Zi) / Yi
        Accuracy = mean^{n}_{i=1} (Yi \bigcap Zi) / (Yi \bigcup Zi)
        """
        category_result = self._calculate(category_name)
        subcategory_result = self._calculate(subcategory_name)
        result = LlmAnnotationResult(
            model_type=model_type,
            random_seed=random_seed,
            half_number=half_number,
            comment_ids=self.llm_human_df["id"].tolist(),
            category=category_result,
            subcategory=subcategory_result,
        )
        return result

    def _calculate(self, col_name) -> EvalIndicator:
        er = self.llm_human_df.apply(
            lambda row: row[col_name + "_llm"] == row[col_name + "_human"],
            axis=1,
        ).mean()
        accuracy = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_llm"]) | set(row[col_name + "_human"])),
            axis=1,
        ).mean()
        precision = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_llm"])),
            axis=1,
        ).mean()
        recall = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_human"])),
            axis=1,
        ).mean()

        f1 = 2 * (precision * recall) / (precision + recall)

        return EvalIndicator(
            exact_match=er, accuracy=accuracy, precision=precision, recall=recall, f1=f1
        )


class EvaluateAnnotationSingle(EvaluateAnnotationBase):
    """calculate evaluation metrics for annotation"""

    def __init__(self) -> None:
        human_df = pd.read_csv(HUMAN_ANOTATION_CSV_PATH)
        llm_df = pd.read_csv(LLM_ANOTATION_CSV_PATH)

        assert human_df.shape == llm_df.shape
        assert human_df["id"].equals(llm_df["id"])

        self.llm_human_df = pd.merge(
            llm_df,
            human_df,
            how="inner",
            on="id",
            suffixes=("_llm", "_human"),
        )
        self.accuracy_runner = evaluate.load("accuracy")
        self.precision_runner = evaluate.load("precision")
        self.recall_runner = evaluate.load("recall")
        self.f1_runner = evaluate.load("f1")

    def evaluate(self) -> LlmAnnotationResult:
        """
        Precision =
                        true positives
                        true positives + false positives
        Recall =
                true positives
                true positives + false negatives
        Fβ =
                (β2 +1)PR
                β2P+R
        Accuracy =
                true positives + true negatives
                true positives + true negatives + false positives + false negatives
        """
        category_result = self._calculate(binary_category_name)
        result = LlmAnnotationResult(
            model_type=model_type,
            random_seed=random_seed,
            half_number=half_number,
            comment_ids=self.llm_human_df["id"].tolist(),
            category=category_result,
            subcategory=None,
        )
        return result

    def _calculate(self, col_name) -> EvalIndicator:
        references = self.llm_human_df[col_name + "_human"].tolist()
        predictioins = self.llm_human_df[col_name + "_llm"].tolist()
        fp = [
            self.llm_human_df.iloc[i]["id"]
            for i, (ref, pred) in enumerate(zip(references, predictioins))
            if ref == 0 and pred == 1
        ]

        fn = [
            self.llm_human_df.iloc[i]["id"]
            for i, (ref, pred) in enumerate(zip(references, predictioins))
            if ref == 1 and pred == 0
        ]

        logger.info(f"False Positive list:{fp}")
        logger.info(f"False Negative list:{fn}")

        accuracy = self.accuracy_runner.compute(
            references=references, predictions=predictioins
        )
        precision = self.precision_runner.compute(
            references=references, predictions=predictioins
        )
        recall = self.recall_runner.compute(
            references=references, predictions=predictioins
        )
        f1 = self.f1_runner.compute(references=references, predictions=predictioins)

        return EvalIndicator(
            exact_match=None,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )


def preprocess_human_annotation():
    human_df = pd.read_csv(HUMAN_ANOTATION_CSV_PATH)
    human_df[category_name] = human_df[category_name].str.split(" ")
    human_df[subcategory_name] = human_df[subcategory_name].astype(str).str.split(" ")

    # 文字列として格納されている配列を配列化
    human_df[category_name] = human_df[category_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 配列化されていないものを配列化
    human_df[category_name] = human_df[category_name].apply(
        lambda x: [x] if not isinstance(x, list) and not pd.isna(x) else x
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: [x] if not isinstance(x, list) and not pd.isna(x) else x
    )

    # 格納されている配列の中の文字列となっている数字を小数点に変換
    human_df[category_name] = human_df[category_name].apply(
        lambda x: (
            [int(i) if i not in ["None", "nan"] else None for i in x]
            if isinstance(x, list)
            else x
        )
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: (
            [float(i) if i not in ["None", "nan"] else None for i in x]
            if isinstance(x, list)
            else x
        )
    )

    # 大分類に格納されてる配列の要素数に小分類を合わせる.
    # その際、大分類が2のindexと小分類のNoneではないindexが対応するようにpaddingする
    for i, row in human_df.iterrows():
        assert len(row[category_name]) >= len(row[subcategory_name])

        if len(row[category_name]) != len(row[subcategory_name]):
            renewal_small_category = [None] * len(row[category_name])
            if 2 in row[category_name]:
                two_index = list(row[category_name]).index(2)
                renewal_small_category[two_index] = row[subcategory_name][0]
            human_df.at[i, subcategory_name] = renewal_small_category

    human_df.to_csv(
        str(HUMAN_ANOTATION_CSV_PATH).replace(".csv", "_preprocessed.csv"),
        index=False,
    )


def evaluate_subcategory():
    from sklearn.metrics import classification_report

    human_df = pd.read_csv(SUBCATEGORY_ANNOTATION_CSV_PATH).sort_values("id")
    llm_df = pd.read_csv(SUBCATEGORY_LLM_CSV_PATH).sort_values("id")

    assert set(human_df["id"]) == set(llm_df["id"])

    # Prepare the data for accuracy and F1 score calculation with the values as strings
    y_true_str = human_df[subcategory_name].astype(str)
    y_pred_str = llm_df[subcategory_name].astype(str)

    # prepross
    logger.info(set(y_true_str))
    logger.info(set(y_pred_str))

    # Recalculate classification report for detailed analysis (precision, recall, f1-score by class)
    class_report = classification_report(y_true_str, y_pred_str, output_dict=True)

    false_positive_ids = []
    false_negative_ids = []

    for i, (true_label, pred_label) in enumerate(zip(y_true_str, y_pred_str)):
        if true_label != pred_label:
            if true_label == "Irrelevant":
                false_positive_ids.append(human_df.iloc[i]["id"])
            elif true_label == "Relevant":
                false_negative_ids.append(human_df.iloc[i]["id"])

    logger.info("key, precision, recall, f1-score, support")
    for k, v in class_report.items():
        if isinstance(v, dict):
            logger.info(
                f"{k}, {v['precision']}, {v['recall']}, {v['f1-score']}, {v['support']}"
            )
        else:
            logger.info(k, v)

    logger.info(f"False Positive list:{false_positive_ids}")
    logger.info(f"False Negative list:{false_negative_ids}")

    logger.info("Done evaluation")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--target", type=str, help="type of function to run")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument(
        "--prefix", type=str, help="file name prefix for subcategory llm csv"
    )
    args = parser.parse_args()
    time_str = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    logger.add(
        f"logs/evaluate_llm_annotation_{time_str}.log"
    )
    if args.preprocess:
        preprocess_human_annotation()
    if args.target == "binary":
        evaluator = EvaluateAnnotationSingle()
        result = evaluator.evaluate()
        logger.info(result)
    if args.target == "subcategory":
        SUBCATEGORY_ANNOTATION_CSV_PATH = (
            Config.target_base_dir / "1_10_val_subcategory_annotation.csv"
        )
        SUBCATEGORY_LLM_CSV_PATH = (
            Config.target_base_dir
            / f"{args.prefix}_{model_type}_subcategory_llm_annotation.csv"
        )
        evaluate_subcategory()
    else:
        raise ValueError(f"Invalid type: {args.target}")
