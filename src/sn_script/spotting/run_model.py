# noqa: N806
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tap import Tap
from xgboost import XGBClassifier

from sn_script.config import binary_category_name
from SoccerNet.utils import getListGames


class TrainXGBoostWithEmbeddings(Tap):
    comment_csv: Path = Path("database/stable/scbi-v2_refer_label.csv")
    emb_csv: Path = Path("database/weights/prev_text_embeddings.npz")
    mode: Literal["text", "label", "both"] = "label"


def train_xgboost_with_embeddings(
    comment_df: pd.DataFrame, refer_prefix="refer_", embedding_col="embedding", label_col="binary_category", mode: Literal["text", "label", "both"] = "label"
):
    """
    comment_df:
      - embedding_col には 768次元程度のベクトル(list or np.ndarray)
      - refer_prefix から始まる複数列が True/False
      - label_col が 0/1 (付加的情報か否か)

    return: 学習済み xgbモデル と、(X_train, X_test, y_train, y_test, y_pred) などの学習結果
    """

    # 1) 特徴量X を作成
    # (A) embedding: shape (N, 768)
    embeddings_list = list(comment_df[embedding_col])
    # => embeddings_list[i] が (768,) の np.ndarray

    X_embed = np.stack(embeddings_list, axis=0)  # (N, 768) になるはず

    assert len(X_embed.shape) == 2, f"Embedding shape is not 2D: {X_embed.shape}"

    # (B) refer_◯◯ を探す
    refer_cols = [c for c in comment_df.columns if c.startswith(refer_prefix)]

    # True/False -> 1/0
    X_refer = comment_df[refer_cols].astype(int).to_numpy()  # (N, K)

    assert X_embed.shape[0] == X_refer.shape[0]

    # 最終的に concat
    if mode == "both":
        X = np.concatenate([X_embed, X_refer], axis=1)  # shape (N, 768+K)
    elif mode == "text":
        X = X_embed  # textのみ
    elif mode == "label":
        X = X_refer  # label 情報のみ
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # 2) 目的変数y
    y = comment_df[label_col].values  # shape (N,)

    # 3) データ分割は gameごとに行う
    train_games = getListGames("train")
    test_games = getListGames("valid")
    X_train, y_train = X[comment_df["game"].isin(train_games)], y[comment_df["game"].isin(train_games)]
    X_test, y_test = X[comment_df["game"].isin(test_games)], y[comment_df["game"].isin(test_games)]

    # 4) XGBoost モデル定義
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")

    # 5) 学習
    xgb_model.fit(X_train, y_train)

    # 6) 推論 & 評価
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print(f"Accuracy on test: {acc:.4f}")
    print("Classification Report:\n", report)

    xgb_model.save_model("database/weights/xgb_model.json")

    return xgb_model, (X_train, X_test, y_train, y_test, y_pred)


def load_embeddings_npz(path="embeddings.npz"):
    data = np.load(path, allow_pickle=False)
    comment_ids = data["comment_ids"]
    embeddings = data["embeddings"]
    return comment_ids, embeddings


if __name__ == "__main__":
    args = TrainXGBoostWithEmbeddings().parse_args()
    comment_df = pd.read_csv(args.comment_csv)

    # 事前に作成した Embedding を読み込む
    cids, embs = load_embeddings_npz(args.emb_csv)
    comment_df["embedding"] = comment_df["id"].map(dict(zip(cids, embs)))

    # NaN stats を削除
    print(f"{comment_df['embedding'].isna().sum()=}")
    comment_df = comment_df.dropna(subset=["embedding"])
    print(f"Dropeed emb NaN rows. {comment_df.shape=}")

    # 0 or 1 以外の行を削除
    comment_df = comment_df[comment_df[binary_category_name].isin([0, 1])]
    print(f"Dropped label non-binary rows. {comment_df.shape=}")

    xgb_model, data_tuple = train_xgboost_with_embeddings(comment_df, label_col=binary_category_name, mode=args.mode)
    print("Done !")
