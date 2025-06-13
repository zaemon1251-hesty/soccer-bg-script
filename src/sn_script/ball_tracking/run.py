import os
import pandas as pd
import numpy as np
from math import sqrt
from typing import Dict, List, Tuple, Optional
from tap import Tap

class BallTrackingConfig(Tap):
    """
    コマンドライン引数の設定
    """
    src_csv_file: str = "database/demo/yolo_ball_tracking_results.csv"
    single_csv_file: str = "database/demo/ball_tracking_single.csv"
    interpolated_csv_file: str = "database/demo/ball_tracking_interpolated.csv"
    input_player_csv: str = "database/demo/players_in_frames_sn_gamestate.csv"
    sample_metadata_file: str = "database/demo/sample_metadata.csv"
    sub_output_player_csv: str = "database/demo/players_in_frames_sn_gamestate_with_image_id.csv"
    output_player_csv: str = "database/demo/players_in_frames_with_ball.csv"


class BallSelector:
    """
    1画像につき高々1つのボールを選択する（動的計画法による最適経路探索）
    """
    def __init__(self, alpha=100.0, missing_penalty=10.0):
        self.alpha = alpha
        self.missing_penalty = missing_penalty

    def select(self, src_csv_file: str, single_csv_file: str):
        df = pd.read_csv(src_csv_file)
        results_tracking = []
        for video_id, group in df.groupby("video_id"):
            frames = sorted(group["image_id"].unique())
            frame_candidates = {}
            for f in frames:
                candidates = []
                df_frame = group[group["image_id"] == f]
                for _, row in df_frame.iterrows():
                    if pd.notnull(row["x1"]):
                        cx = (row["x1"] + row["x2"]) / 2
                        cy = (row["y1"] + row["y2"]) / 2
                        conf_val = row["conf"]
                        candidates.append({
                            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
                            "center": (cx, cy), "conf": conf_val
                        })
                if len(candidates) == 0:
                    candidates.append(None)
                frame_candidates[f] = candidates
            dp: Dict[int, List[Tuple[float, int, Optional[int]]]] = {}
            frames_sorted = frames
            dp[frames_sorted[0]] = []
            for i, cand in enumerate(frame_candidates[frames_sorted[0]]):
                cost = self.missing_penalty if cand is None else (1 - cand["conf"])
                dp[frames_sorted[0]].append((cost, i, None))
            for f_idx in range(1, len(frames_sorted)):
                f = frames_sorted[f_idx]
                dp[f] = []
                for j, cand in enumerate(frame_candidates[f]):
                    cand_cost = self.missing_penalty if cand is None else (1 - cand["conf"])
                    best_cost = float('inf')
                    best_prev = None
                    for (prev_cost, prev_i, _) in dp[frames_sorted[f_idx-1]]:
                        prev_cand = frame_candidates[frames_sorted[f_idx-1]][prev_i]
                        if prev_cand is None or cand is None:
                            trans_cost = 0
                        else:
                            dx = cand["center"][0] - prev_cand["center"][0]
                            dy = cand["center"][1] - prev_cand["center"][1]
                            distance = sqrt(dx*dx + dy*dy)
                            trans_cost = self.alpha * distance
                        total_cost = prev_cost + trans_cost + cand_cost
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_prev = prev_i
                    dp[f].append((best_cost, j, best_prev))
            best_path = {}
            last_frame = frames_sorted[-1]
            best_final = min(dp[last_frame], key=lambda x: x[0])
            best_index = best_final[1]
            best_path[last_frame] = best_index
            for f_idx in range(len(frames_sorted)-1, 0, -1):
                f = frames_sorted[f_idx]
                _, _, best_prev = dp[f][best_path[f]]
                best_path[frames_sorted[f_idx-1]] = best_prev
            for f in frames_sorted:
                chosen_candidate = frame_candidates[f][best_path[f]]
                if chosen_candidate is None:
                    results_tracking.append({
                        "video_id": video_id, "image_id": f,
                        "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan, "conf": np.nan
                    })
                else:
                    results_tracking.append({
                        "video_id": video_id, "image_id": f,
                        "x1": chosen_candidate["x1"], "y1": chosen_candidate["y1"],
                        "x2": chosen_candidate["x2"], "y2": chosen_candidate["y2"],
                        "conf": chosen_candidate["conf"]
                    })
        df_track = pd.DataFrame(results_tracking)
        df_track.sort_values(["video_id", "image_id"], inplace=True)
        df_track.to_csv(single_csv_file, index=False)
        print(f"最適化されたボール追跡結果を {single_csv_file} に出力しました")


class BallInterpolator:
    """
    欠損フレームの線形補完
    """
    def __init__(self, max_gap_length=25):
        self.max_gap_length = max_gap_length

    def interpolate(self, single_csv_file: str, interpolated_csv_file: str):
        df = pd.read_csv(single_csv_file)
        df.sort_values(["video_id", "image_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_list = []
        for _, group_loop_var in df.groupby("video_id"):
            group = group_loop_var.sort_values("image_id").reset_index(drop=True)
            valid_idx = group.index[group["x1"].notna()].tolist()
            for i in range(len(valid_idx) - 1):
                start_idx = valid_idx[i]
                end_idx   = valid_idx[i+1]
                gap = end_idx - start_idx - 1
                if gap > 0 and gap <= self.max_gap_length:
                    start_row = group.loc[start_idx]
                    end_row   = group.loc[end_idx]
                    frame_start = start_row["image_id"]
                    frame_end   = end_row["image_id"]
                    total_gap = frame_end - frame_start
                    for j in range(1, gap + 1):
                        ratio = j / total_gap
                        group.loc[start_idx + j, "x1"] = start_row["x1"] + ratio * (end_row["x1"] - start_row["x1"])
                        group.loc[start_idx + j, "y1"] = start_row["y1"] + ratio * (end_row["y1"] - start_row["y1"])
                        group.loc[start_idx + j, "x2"] = start_row["x2"] + ratio * (end_row["x2"] - start_row["x2"])
                        group.loc[start_idx + j, "y2"] = start_row["y2"] + ratio * (end_row["y2"] - start_row["y2"])
                        group.loc[start_idx + j, "conf"] = start_row["conf"] + ratio * (end_row["conf"] - start_row["conf"])
            df_list.append(group)
        df_interp = pd.concat(df_list, ignore_index=True)
        df_interp.to_csv(interpolated_csv_file, index=False)
        print(f"補完結果を {interpolated_csv_file} に保存しました")


class GSRMerger:
    """
    gsrの出力csvとボール追跡結果を紐付ける
    """
    def __init__(self, tol=2):
        self.tol = tol

    def merge(self, input_player_csv: str, sample_metadata_file: str, interpolated_csv_file: str, sub_output_player_csv: str, output_player_csv: str):
        df_players = pd.read_csv(input_player_csv)
        df_sample = pd.read_csv(sample_metadata_file)
        df_ball = pd.read_csv(interpolated_csv_file)
        if not os.path.exists(sub_output_player_csv):
            df_players['time'] = pd.to_numeric(df_players['time'], errors='coerce')
            df_players.dropna(subset=['time'], inplace=True)
        def find_closest_sample(row, df_sample):
            candidates = df_sample[(df_sample['game'] == row['game']) & (df_sample['half'] == row['half'])]
            if candidates.empty:
                return np.nan
            if len(candidates) == 1:
                return candidates['id'].values[0]
            def distance(sample):
                t = row['time']
                start = sample['start']
                end = sample['end']
                if start <= t <= end:
                    return 0
                else:
                    return min(abs(t - start), abs(t - end))
            candidates = candidates.copy()
            candidates['dist'] = candidates.apply(distance, axis=1)
            best = candidates.loc[candidates['dist'].idxmin()]
            return best['id']
        if not os.path.exists(sub_output_player_csv):
            df_players['sample_id'] = df_players.apply(lambda row: find_closest_sample(row, df_sample), axis=1)
        if not os.path.exists(sub_output_player_csv):
            df_players = pd.merge(df_players, df_sample[['id', 'start', 'end']], left_on='sample_id', right_on='id', how='left')
            df_players.drop(columns='id', inplace=True)
            df_players.rename(columns={'start': 'sample_start', 'end': 'sample_end'}, inplace=True)
        if not os.path.exists(sub_output_player_csv):
            df_players.to_csv(sub_output_player_csv, index=False)
            print(f"{sub_output_player_csv} に sample_id と image_id を付与して保存しました")
        def find_ball_detection(row, df_ball, tol=2):
            try:
                sample_id_int = int(row['sample_id'])
            except:
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
            video_id = f"SNGS-{sample_id_int:04d}"
            candidates = df_ball[df_ball['video_id'] == video_id]
            if candidates.empty:
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
            candidates = candidates.copy()
            candidates['time_diff'] = (candidates['image_id'] - row['image_id']).abs()
            best = candidates.loc[candidates['time_diff'].idxmin()]
            if best['time_diff'] > tol:
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
            return pd.Series([best['x1'], best['y1'], best['x2'], best['y2'], best['conf']])
        if os.path.exists(sub_output_player_csv):
            df_players = pd.read_csv(sub_output_player_csv)
        df_players[['ball_x1', 'ball_y1', 'ball_x2', 'ball_y2', 'ball_conf']] = \
            df_players.apply(lambda row: find_ball_detection(row, df_ball, self.tol), axis=1)
        df_players.to_csv(output_player_csv, index=False)
        print(f"結合結果を '{output_player_csv}' として保存しました")


def main():
    args = BallTrackingConfig().parse_args()
    # 1画像1ボール選択
    BallSelector().select(args.src_csv_file, args.single_csv_file)
    # 線形補完
    BallInterpolator().interpolate(args.single_csv_file, args.interpolated_csv_file)
    # gsr出力csvと紐付け
    GSRMerger().merge(
        args.input_player_csv,
        args.sample_metadata_file,
        args.interpolated_csv_file,
        args.sub_output_player_csv,
        args.output_player_csv
    )

if __name__ == "__main__":
    main()
