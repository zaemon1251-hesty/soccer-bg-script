import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tap import Tap

time_str = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
logger.add(
    f"logs/evaluate_vad_{time_str}.log"
)

# コマンドライン引数の定義
class EvalVadArgs(Tap):
    """
    id,start,end,text,game,half を持つ
    """
    game: str
    half: int
    input_annotation_csv: str  # アノテーションファイルのパス
    input_prediction_csv: str  # 予測結果ファイルのパス

# VADの評価指標計算
def evaluate_vad(annotation_df, prediction_df, game, half):
    # 31.25ミリ秒のチャンク長（サンプルレートが16kHzの場合）
    chunk_duration_ms = 31.25
    chunk_duration_s = chunk_duration_ms / 1000

    # 最大の終了時刻を取得
    max_end = annotation_df['end'].max()

    # predictionの区間のうち、関係ある行だけを取得
    prediction_df = prediction_df[
        (prediction_df["game"] == game) &
        (prediction_df["half"] == half) &
        (prediction_df['end'] <= max_end)
    ].sort_values('start')

    # 全体の時間範囲を31.25ミリ秒に区切る
    time_chunks = np.arange(0, max_end, chunk_duration_s)

    # 各チャンクについてアノテーションと予測の真偽値を計算
    annotation_labels = np.zeros(len(time_chunks), dtype=int)
    prediction_labels = np.zeros(len(time_chunks), dtype=int)

    # アノテーション区間を反映
    for _, row in annotation_df.iterrows():
        start_idx = np.searchsorted(time_chunks, row['start'])
        end_idx = np.searchsorted(time_chunks, row['end'])
        annotation_labels[start_idx:end_idx] = 1  # 発話区間は1とする

    # 予測区間を反映
    for _, row in prediction_df.iterrows():
        start_idx = np.searchsorted(time_chunks, row['start'])
        end_idx = np.searchsorted(time_chunks, row['end'])
        prediction_labels[start_idx:end_idx] = 1  # 発話予測区間は1とする TODO:予測スコアが欲しい

    # ROC-AUC、Precision、Recall、Accuracyを計算
    roc_auc = roc_auc_score(annotation_labels, prediction_labels)
    precision = precision_score(annotation_labels, prediction_labels)
    recall = recall_score(annotation_labels, prediction_labels)
    accuracy = accuracy_score(annotation_labels, prediction_labels)

    # 結果を表示
    logger.info(f'ROC-AUC: {roc_auc:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'Accuracy: {accuracy:.4f}')

# メイン関数
def main(args: EvalVadArgs):
    # CSVファイルの読み込み
    annotation_df = pd.read_csv(args.input_annotation_csv)
    prediction_df = pd.read_csv(args.input_prediction_csv)

    # VADの評価指標を計算
    evaluate_vad(annotation_df, prediction_df, args.game, args.half)

if __name__ == '__main__':
    eval_args = EvalVadArgs().parse_args()
    main(eval_args)
