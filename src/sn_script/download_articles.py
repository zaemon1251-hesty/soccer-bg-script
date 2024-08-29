import pandas as pd
import requests
from tqdm import tqdm

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


def download_article(article_url: str, output_dir: str):
    article_name = article_url.split("/")[-1]
    output_path = output_dir / article_name
    if output_path.exists():
        return
    response = requests.get(article_url)
    with open(output_path, "wb") as f:
        f.write(response.content)



if __name__ == "__main__":
    wiki_csv_path = Config.target_base_dir / "player_wikipedia_articles.csv"
    output_dir = Config.target_base_dir / "knowledge_base_raw"
    wiki_df = pd.read_csv(wiki_csv_path)

    assert "wikipedia_article" in wiki_df.columns, "wikipedia_article column not found in the csv file"

    for article_url in tqdm(wiki_df["wikipedia_article"]):
        try:
            download_article(article_url, output_dir)
        except Exception as e:
            print(f"Error downloading {article_url}: {e}")
            continue
