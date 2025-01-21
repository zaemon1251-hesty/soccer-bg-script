from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def add_prev_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    各 (game, half) ごとにソートして1つ前のコメントを prev_text にする
    """
    grouped = df.groupby(["game", "half"])
    df_list = []
    for _, group_df in grouped:
        group_df = group_df.copy().sort_values("start")
        group_df["prev_text"] = group_df["text"].shift(1)
        df_list.append(group_df)
    new_df = pd.concat(df_list)
    return new_df



class CommentDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        df から必要な情報を取り出して Dataset に保持しておく
        """
        assert {"id", "prev_text"}.issubset(set(df.columns)), "Columns 'id' and 'prev_text' must be in the DataFrame"

        self.comment_ids = df["id"].tolist()
        self.texts = df["prev_text"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "id": self.comment_ids[idx],
            "text": self.texts[idx]
        }


def collate_fn(batch: list, tokenizer=None, max_length=512):
    """
    batch: Dataset.__getitem__ から返される dict のリスト
    tokenizer: transformers のトークナイザ
    """
    comment_ids = [item["id"] for item in batch]
    texts = [item["text"] for item in batch]
    # Tokenize
    batch_dict = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # 戻り値に comment_id も含める
    batch_dict["id"] = comment_ids
    return batch_dict


def cls_pool(last_hidden_states: Tensor) -> Tensor:
    return last_hidden_states[:, 0, :]


def main():
    base_dir = Path("/raid_elmo/home/lr/moriy/SoccerNet/commentary_analysis/stable")
    comment_df = pd.read_csv(base_dir / "scbi-v2.csv")

    # 前処理
    comment_df = add_prev_text(comment_df)
    comment_df["prev_text"] = comment_df["prev_text"].fillna("")  # shift 後に NaN になる場合を考慮
    comment_df["prev_text"] = "passage: " + comment_df["prev_text"]

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    model = model.cuda()

    dataset = CommentDataset(comment_df)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer)
    )

    emb_with_id = {}
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # バッチを GPU に載せる
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            comment_ids = batch["id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = cls_pool(outputs.last_hidden_state)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

            # ID と埋め込みを対応付け
            for cid, emb in zip(comment_ids, embeddings):
                emb_with_id[cid] = emb

        torch.cuda.empty_cache()

    # DataFrame に Embedding を反映させる
    comment_df["embedding"] = comment_df["id"].map(emb_with_id)

    # NPZ に保存
    emb_dict = comment_df.set_index("id")["embedding"].to_dict()
    save_embeddings_npz(emb_dict, save_path=base_dir / "prev_text_embeddings.npz")


def save_embeddings_npz(emb_dict, save_path="embeddings.npz"):
    """
    emb_dict: {comment_id: embedding} の形式を想定 (embedding: np.array or list)
    save_path: 保存先パス
    """
    # comment_id と埋め込みをリスト化
    comment_ids = list(emb_dict.keys())  # shape (N,)
    embedding_list = list(emb_dict.values())  # shape (N,) 要素は (D,) の配列

    # numpy化
    comment_ids_np = np.array(comment_ids, dtype=np.int64)   # あるいは str に合わせるなど
    embeddings_np = np.stack(embedding_list, axis=0)         # shape (N, D)

    # npzにまとめて保存
    np.savez(save_path, comment_ids=comment_ids_np, embeddings=embeddings_np)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
