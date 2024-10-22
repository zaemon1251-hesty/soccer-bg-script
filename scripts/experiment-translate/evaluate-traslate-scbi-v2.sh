#/bin/bash

# whisperのtranslateを評価する
scbi_csv=database/stable/scbi-v2.csv
sample_csv=database/comments/sample_scbi-v2_translate.csv
uv run python src/sn_script/csv_utils.py \
    sample \
    --input_csv $scbi_csv \
    --output_csv $sample_csv

# 正解データ(参照例)は 元の言語でtranscribeした結果を google翻訳 or deepl で翻訳したもの
uv run python src/sn_script/evaluate_translate.py \
    --input_csv "$sample_csv"  \
    --output_csv "database/comments/sample_scbi-v2_translate_gcloud.csv" \
    --engine gcloud \
    --src_text_col src_text \
    --en_text_col text

# 結果
# Overall BLEU Score: 0.3586844254141253
# Language-wise BLEU Score:
# es: res_dic={'bleu': 0.1506458702186206, 'num_samples': 40}
# en: res_dic={'bleu': 0.7904873145230659, 'num_samples': 40}
# fr: res_dic={'bleu': 0.23901667416542036, 'num_samples': 40}
# de: res_dic={'bleu': 0.31958759596100567, 'num_samples': 40}


# (enが高いのは当たり前として)esが低いが、他の言語はまあまあ
# whisper論文だと BLEUは 0.3~0.4 が目安だから、この結果は低いように見えるが、(1)と(2)を考慮すると、まあまあの結果と言える
#   (1)google翻訳はあくまで参照例であり、完全に信頼できるものではない
#   (2)分割が同一でない中の比較だから、本来はもっと高くなる