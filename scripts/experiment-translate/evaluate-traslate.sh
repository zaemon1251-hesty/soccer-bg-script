#/bin/bash

# whisperのtranslateを評価する
# 正解データ(参照例)は 元の言語での書き起こしを google or deepl で翻訳したもの
uv run python src/sn_script/evaluate_translate.py \
    --input_csv "database/comments/sample_compare_translate.csv"  \
    --output_csv "database/comments/sample_compare_translate_gcloud.csv" \
    --engine gcloud

# 結果
# BLEU Score: {'bleu': 0.1736468806397884, 'precisions': [0.42477256822953113, 0.21451846488052137, 0.13032064504031501, 0.08024091703905188], 'brevity_penalty': 0.9883469454094838, 'length_ratio': 0.9884143178281168, 'translation_length': 5716, 'reference_length': 5783}
# ROUGE Score: {'rouge1': 0.43535243663255907, 'rouge2': 0.26898359211001033, 'rougeL': 0.41793446490543884, 'rougeLsum': 0.41836912887722943}
# whisper論文だと BLEUは 0.3~0.4 が目安
# それと比較すると、そこまで高くない
# ただし、正解として使った機械翻訳完全ではないので、それを考慮すると、まあまあかもしれない