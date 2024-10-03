from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tap import Tap

# 日本語フォントの設定
font = {"family": "IPAexGothic"}
matplotlib.rc("font", **font)


# コマンドライン引数を設定
class DrawVoiceDetectionArguments(Tap):
    SoccerNet_path: str
    game: str
    half: str = "1"
    csv_file: str
    plot_output_dir: str
    seglen: int = 10
    prefix: str = ""

args = DrawVoiceDetectionArguments().parse_args()
assert args.half in ["1", "2"], "Half must be either 1 or 2"

# CSVファイルを読み込む (game, start, end の形式)
vad_df = pd.read_csv(args.csv_file)
vad_df = vad_df[
    (vad_df["game"] == args.game)
    & (vad_df["half"] == int(args.half))
].sort_values("start", ascending=True) # 指定されたゲームハーフのVAD情報を取得

# wavファイルを読み込む
wavfile_path = f"{args.SoccerNet_path}/{args.game}/{args.half}_224p.wav"
wav, fs = sf.read(wavfile_path)  # fs = 16kHz
wav = wav[:, 0] # ステレオ音声をモノラルに変換

# デバッグ用
print(vad_df)
print(f"WAV data loaded: {len(wav)} samples at {fs} Hz")

# プロットのための定数
wl = len(wav)
wsec = int(wl / fs)
seglen = args.seglen  # 秒単位のセグメント長
colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown']  # 区間ごとの色

# プロット横断時、同じセグメントに同じ色を使うための変数
last_color_j = -1
last_row_id = -1

# プロットの作成
wx = np.linspace(0, wsec, wl)
for i in range(0, wsec // seglen): # seglen秒ごとにプロット
    if i > 6:  # プロットは7つまで
        break

    plt.figure()
    plt.xlim(i * seglen, (i + 1) * seglen)
    plt.plot(wx, wav, label='音声波形')

    # seglen秒区間に含まれるVAD情報を取得
    between_vad_df = vad_df.loc[
        vad_df['start'].between(i * seglen, (i + 1) * seglen) |
        vad_df['end'].between(i * seglen, (i + 1) * seglen)
    ]

    # VADの区間を矩形でプロット
    for j, row in between_vad_df.iterrows():
        row_id, start_time, end_time = row["id"], row['start'], row['end']

        color_j = j % len(colors)  # 色を循環して使用
        if row_id == last_row_id:
            color_j = last_color_j

        color = colors[color_j]
        plt.axvspan(start_time, end_time, color=color, alpha=0.3, label=f'音声活動 {j+1}')

    # 次のセグメント用に変数を更新
    last_color_j = color_j
    last_row_id = row_id

    plt.xlabel('時間 (秒)')
    plt.ylabel('振幅 / 音声活動')
    plt.title(f'{args.game}\nHalf {args.half} - {i * seglen}秒〜{(i + 1) * seglen}秒')
    plt.grid(axis="x", which="both")

    # プロットをファイルに保存
    plot_filename = Path(args.plot_output_dir) / f'{args.game}' / f'{args.half}' / f'{args.prefix}seg_{i:03d}.png'
    if not plot_filename.parent.exists():
        plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()
    print(f'プロットを保存しました: {plot_filename}')
