import requests
import pandas as pd
import time
from tqdm import tqdm

try:
    from sn_script.config import (
        Config,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
    )
tqdm.pandas()

# csv書き出し先
player_article_csv_path = Config.target_base_dir / "player_wikipedia_articles.csv"

# CSVファイルの読み込み
player_csv_path = Config.target_base_dir / "sncaption_players.csv"
player_df = pd.read_csv(player_csv_path)

# Wikipedia APIの検索URL
wiki_api_url = "https://en.wikipedia.org/w/api.php"

# 選手の記事を格納するリスト
player_articles = []


def get_wikipedia_article(query, retries=3):
    params = {
        "action": "opensearch",
        "search": query,
        "format": "json",
        "limit": 10,
    }

    for attempt in range(retries):
        try:
            response = requests.get(wiki_api_url, params=params, timeout=10)
            response.raise_for_status()  # HTTPエラーをチェック

            # data = [検索ワード, [候補1, 候補2, ...], [候補1の説明, 候補2の説明, ...], [候補1のURL, 候補2のURL, ...]]
            data = response.json()

            if len(data[1]) == 0:
                return None

            # サッカー選手のエンティティが含まれる候補を選ぶ
            cand_index = 0
            for i, entity_name in enumerate(data[1]):
                if "football" in entity_name.lower() or \
                    "soccer" in entity_name.lower():
                    cand_index = i
                    break

            # 候補1のURLを取り出す
            assert data and len(data) > 3 and data[3], f"Unexpected data: {data}"

            article_url = data[3][cand_index]
            return article_url

        except (
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError,
        ) as e:
            print(f"Error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(2)  # 待機してからリトライ
    return None


def run():
    # 各選手についてWikipedia記事を検索
    player_df_group = player_df.groupby(by="hash")  # flashscore hash により重複を除く

    # test
    player_df_group = player_df_group

    def hash_func(group):
        long_name = group["long_name"].iloc[0]
        short_name = group["short_name"].iloc[0]
        hash = group["hash"].iloc[0]
        # 選手名で検索
        article_url = get_wikipedia_article(long_name)
        if article_url is None:
            # フルネームがない場合、名前のみで検索
            article_url = get_wikipedia_article(short_name)

        player_articles.append(
            {
                "hash": hash,
                "wikipedia_article": article_url,
            }
        )
        # サーバーに負荷をかけないように待機
        time.sleep(1)

    player_df_group.progress_apply(hash_func)

    # 結果をデータフレームに変換してCSVに保存
    result_df = pd.DataFrame(player_articles)
    result_df.to_csv(player_article_csv_path, index=False)


if __name__ == "__main__":
    run()
