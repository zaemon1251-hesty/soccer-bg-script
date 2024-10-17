import os
from os import environ

import deepl
import pandas as pd
from evaluate import load
from google.cloud import translate as translate_gcloud_cls
from googletrans import Translator
from tap import Tap


class EvaluateTranslateArguments(Tap):
    input_csv: str
    output_csv: str
    engine: str = "google"

DEEPL_API_KEY = os.environ.get('DEEPL_API_KEY')
PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


def translate_gcloud(text: str, target_language_code: str) -> translate_gcloud_cls.Translation:
    client = translate_gcloud_cls.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0]


def translate(input_csv, output_csv, engine="google"):
    df = pd.read_csv(input_csv)

    # 前処理
    ## 前後の空白を削除
    df['text'] = df['text'].str.strip()
    # textまたはtext_enが空の行を削除
    df = df.dropna(subset=['text', 'text_en'])

    if engine == "deepl":
        translator = deepl.Translator(DEEPL_API_KEY)
        df['text_en_deepl'] = [
            translator.translate_text(text, target_lang="EN-US").text
            for text in df['text']
        ]
    elif engine == "gcloud":
        df['text_en_gcloud'] = [
            translate_gcloud(text, "en").translated_text
            for text in df['text']
        ]
    elif engine == "google":
        df['text_en_google'] = [
            Translator().translate(text, dest='en').text
            for text in df['text']
        ]

    # 保存
    df.to_csv(output_csv)

def main(args: EvaluateTranslateArguments):
    input_csv = args.input_csv
    output_csv = args.output_csv

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        translate(input_csv, output_csv, engine=args.engine)
        df = pd.read_csv(output_csv)

    # Load the BLEU and ROUGE metrics from Hugging Face's evaluate library
    bleu = load('bleu')
    rouge = load('rouge')

    # Calculate BLEU and ROUGE scores
    bleu_score = bleu.compute(predictions=df['text_en'].tolist(), references=df[f'text_en_{args.engine}'].tolist())
    rouge_score = rouge.compute(predictions=df['text_en'].tolist(), references=df[f'text_en_{args.engine}'].tolist())

    # Print the results
    print("BLEU Score:", bleu_score)
    print("ROUGE Score:", rouge_score)


if __name__ == '__main__':
    args = EvaluateTranslateArguments().parse_args()
    main(args)
