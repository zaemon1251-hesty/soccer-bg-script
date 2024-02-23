import pandas as pd

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )
    from sn_script.llm_anotator import create_target_prompt
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (  # noqa
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )
    from src.sn_script.llm_anotator import create_target_prompt


LLM_ANOTATION_CSV_ALL_PATH = (
    Config.target_base_dir / f"{model_type}_500game_{half_number}_llm_annotation.csv"
)


def sample_supplementary_comments(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    与えられたデータフレームから、付加的情報か」が1のコメントをランダムにn個サンプリングする
    """
    return df[df[binary_category_name] == 1].sample(n=n, random_state=random_seed)


def write_supplementary_comments_csv(df: pd.DataFrame, path: str) -> None:
    """
    与えられたデータフレームをcsvファイルに書き出す
    """
    supplementary_comments_df = sample_supplementary_comments(df, 100)

    comment_ids = supplementary_comments_df.index.tolist()

    targe_prompt_list = []
    for comment_id in comment_ids:
        targe_prompt_list.append(create_target_prompt(comment_id))

    with open(path, "w") as f:
        f.write("\n".join(targe_prompt_list))


if __name__ == "__main__":
    all_comment_df = pd.read_csv(LLM_ANOTATION_CSV_ALL_PATH)
    SUP_SUBCATEGORY_COMMENTS_CSV_PATH = (
        Config.target_base_dir / "supplementary_comments.csv"
    )
    write_supplementary_comments_csv(all_comment_df, SUP_SUBCATEGORY_COMMENTS_CSV_PATH)
    print("Done writing supplementary_comments")
