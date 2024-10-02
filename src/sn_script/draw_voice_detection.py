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

args = DrawVoiceDetectionArguments().parse_args()


# CSVファイルを読み込む (game, start, end の形式)
csv_file = Path(args.csv_file)
if csv_file.exists():
    vad_df = pd.read_csv(csv_file)
else:
    raise RuntimeError("CSV file not found:", csv_file)


# 指定されたゲームハーフのVAD情報を取得
vad_df = vad_df[
    (vad_df["game"] == args.game)
    & (vad_df["half"] == int(args.half))
].sort_values("start", ascending=True)
print(vad_df)


# wavファイルを読み込む
assert args.half in ["1", "2"], "Half must be either 1 or 2"
wavfile_path = f"{args.SoccerNet_path}/{args.game}/{args.half}_224p.wav"

wav, fs = sf.read(wavfile_path)  # fs = 16kHz
print(f"WAV data loaded: {len(wav)} samples at {fs} Hz")


# ステレオ音声をモノラルに変換
wav = wav[:, 0]

wl = len(wav)
wsec = int(wl / fs)
wx = np.linspace(0, wsec, wl)

# VAD情報を0/1の波形に変換
# しかし、やってみると区間幅が小さく重なってしまうケースが多かったので、区間ごとに背景色を塗るように変更

# プロットの作成と保存
seglen = 60  # 秒単位のセグメント長
colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown']  # 区間ごとの色
for i in range(0, wsec // seglen):
    plt.figure()
    plt.xlim(i * seglen, (i + 1) * seglen)
    plt.plot(wx, wav, label='音声波形')

    # 区間iに含まれるVAD情報を取得
    between_vad_df = vad_df.loc[vad_df['start'].between(i * seglen, (i + 1) * seglen)]

    # VADの区間を矩形でプロット（色を使って区別）
    for j, row in between_vad_df.iterrows():
        start_time, end_time = row['start'], row['end']
        color = colors[j % len(colors)]  # 色を循環して使用
        plt.axvspan(start_time, end_time, color=color, alpha=0.3, label=f'音声活動 {j+1}')

    plt.xlabel('時間 (秒)')
    plt.ylabel('振幅 / 音声活動')
    plt.title(f'{args.game}, Half {args.half}')
    plt.grid(axis="x", which="both")

    # プロットをファイルに保存
    plot_filename = Path(args.plot_output_dir) / f'{args.game}' / f'{args.half}/seg_{i:03d}.png'
    if not plot_filename.parent.exists():
        plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()
    print(f'プロットを保存しました: {plot_filename}')
