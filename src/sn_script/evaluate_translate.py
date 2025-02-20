import os
from os import environ

import deepl
import pandas as pd
from evaluate import load
from google.cloud import translate as translate_gcloud_cls
from googletrans import Translator
from tap import Tap


DEEPL_API_KEY = os.environ.get('DEEPL_API_KEY')

PROJECT_ID = environ.get("PROJECT_ID", "")

PARENT = f"projects/{PROJECT_ID}"


class EvaluateTranslateArguments(Tap):
    input_csv: str
    output_csv: str
    engine: str = "google"
    src_text_col: str = "text"
    en_text_col: str = "text_en"


def translate_gcloud(text: str, target_language_code: str) -> translate_gcloud_cls.Translation:
    client = translate_gcloud_cls.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0]


def translate(input_csv, output_csv, engine="google", src_text_col="text", en_text_col="text_en"):
    df = pd.read_csv(input_csv)

    # 前処理
    ## 前後の空白を削除
    df[src_text_col] = df[src_text_col].str.strip()
    # textまたはtext_enが空の行を削除
    df = df.dropna(subset=[src_text_col, en_text_col])

    if engine == "deepl":
        translator = deepl.Translator(DEEPL_API_KEY)
        df['text_en_deepl'] = [
            translator.translate_text(text, target_lang="EN-US").text
            for text in df[src_text_col]
        ]
    elif engine == "gcloud":
        df['text_en_gcloud'] = [
            translate_gcloud(text, "en").translated_text
            for text in df[src_text_col]
        ]
    elif engine == "google":
        df['text_en_google'] = [
            Translator().translate(text, dest='en').text
            for text in df[src_text_col]
        ]

    # 保存
    df = df[["id", src_text_col, en_text_col, f'text_en_{engine}', "language"]]
    df.to_csv(output_csv)

def main(args: EvaluateTranslateArguments):
    input_csv = args.input_csv
    output_csv = args.output_csv

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        translate(input_csv, output_csv, engine=args.engine, src_text_col=args.src_text_col, en_text_col=args.en_text_col)
        df = pd.read_csv(output_csv)


    # bleuスコアを計算
    bleu = load('bleu')
    bleu_score = bleu.compute(predictions=df[args.en_text_col].tolist(), references=df[f'text_en_{args.engine}'].tolist())["bleu"]

    # 言語ごとのBLEUを計算
    bleu_score_lang = {}
    for lang in df['language'].unique():
        mask = df['language'] == lang
        num_samples = mask.sum()
        bleu_score_lang[lang] = {
            "bleu": bleu.compute(
                        predictions=df.loc[mask, args.en_text_col].tolist(),
                        references=df.loc[mask, f'text_en_{args.engine}'].tolist()
                    )["bleu"],
            "num_samples": num_samples
        }

    # 結果を表示
    print("Overall BLEU Score:", bleu_score)
    print("Language-wise BLEU Score:")
    for lang, res_dic in bleu_score_lang.items():
        print(f"{lang}: {res_dic=}")

if __name__ == '__main__':
    args = EvaluateTranslateArguments().parse_args()
    main(args)
