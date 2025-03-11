import os
from math import sqrt

import numpy as np
import pandas as pd
from tap import Tap


class ProcessTrackedBallArgs(Tap):
    src_csv_file: str = "database/demo/yolo_ball_tracking_results.csv"
    output_csv_file: str = "database/demo/players_with_ball.csv"
    input_player_csv: str = "database/demo/players_in_frames_sn_gamestate.csv"
    sample_metadata_file: str = "database/demo/sample_metadata.csv"
    max_gap_length: int = 25
    alpha: float = 100.0
    missing_penalty: float = 10.0


def process_ball_tracking(args):
    df = pd.read_csv(args.src_csv_file)
    results_tracking = []

    for video_id, group in df.groupby("video_id"):
        frames = sorted(group["image_id"].unique())
        frame_candidates = get_frame_candidates(group, frames)
        dp = calculate_optimal_path(frames, frame_candidates, args.alpha, args.missing_penalty)
        best_path = backtrack_best_path(frames, dp)
        results_tracking.extend(select_best_candidates(video_id, frames, frame_candidates, best_path))

    df_track = pd.DataFrame(results_tracking)
    df_track.sort_values(["video_id", "image_id"], inplace=True)
    return df_track


def get_frame_candidates(group, frames):
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
                    "x1": row["x1"],
                    "y1": row["y1"],
                    "x2": row["x2"],
                    "y2": row["y2"],
                    "center": (cx, cy),
                    "conf": conf_val
                })
        if len(candidates) == 0:
            candidates.append(None)
        frame_candidates[f] = candidates
    return frame_candidates


def calculate_optimal_path(frames, frame_candidates, alpha, missing_penalty):
    dp = {frames[0]: []}
    for i, cand in enumerate(frame_candidates[frames[0]]):
        cost = missing_penalty if cand is None else (1 - cand["conf"])
        dp[frames[0]].append((cost, i, None))

    for f_idx in range(1, len(frames)):
        f = frames[f_idx]
        dp[f] = []
        for j, cand in enumerate(frame_candidates[f]):
            cand_cost = missing_penalty if cand is None else (1 - cand["conf"])
            best_cost, best_prev = float('inf'), None
            for (prev_cost, prev_i, _) in dp[frames[f_idx - 1]]:
                prev_cand = frame_candidates[frames[f_idx - 1]][prev_i]
                trans_cost = 0 if prev_cand is None or cand is None else alpha * sqrt((cand["center"][0] - prev_cand["center"][0])**2 + (cand["center"][1] - prev_cand["center"][1])**2)
                total_cost = prev_cost + trans_cost + cand_cost
                if total_cost < best_cost:
                    best_cost, best_prev = total_cost, prev_i
            dp[f].append((best_cost, j, best_prev))
    return dp


def backtrack_best_path(frames, dp):
    best_path = {}
    last_frame = frames[-1]
    best_final = min(dp[last_frame], key=lambda x: x[0])
    best_index = best_final[1]
    best_path[last_frame] = best_index

    for f_idx in range(len(frames) - 1, 0, -1):
        f = frames[f_idx]
        _, _, best_prev = dp[f][best_path[f]]
        best_path[frames[f_idx - 1]] = best_prev
    return best_path


def select_best_candidates(video_id, frames, frame_candidates, best_path):
    results_tracking = []
    for f in frames:
        chosen_candidate = frame_candidates[f][best_path[f]]
        if chosen_candidate is None:
            results_tracking.append({
                "video_id": video_id,
                "image_id": f,
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "conf": np.nan
            })
        else:
            results_tracking.append({
                "video_id": video_id,
                "image_id": f,
                "x1": chosen_candidate["x1"],
                "y1": chosen_candidate["y1"],
                "x2": chosen_candidate["x2"],
                "y2": chosen_candidate["y2"],
                "conf": chosen_candidate["conf"]
            })
    return results_tracking


def interpolate_missing_frames(df_track, max_gap_length):
    df_track.sort_values(["video_id", "image_id"], inplace=True)
    df_track.reset_index(drop=True, inplace=True)
    df_list = []

    for _, g in df_track.groupby("video_id"):
        group = g.sort_values("image_id").reset_index(drop=True)
        valid_idx = group.index[group["x1"].notna()].tolist()

        for i in range(len(valid_idx) - 1):
            start_idx, end_idx = valid_idx[i], valid_idx[i + 1]
            gap = end_idx - start_idx - 1

            if gap > 0 and gap <= max_gap_length:
                start_row, end_row = group.loc[start_idx], group.loc[end_idx]
                frame_start, frame_end = start_row["image_id"], end_row["image_id"]
                total_gap = frame_end - frame_start

                for j in range(1, gap + 1):
                    ratio = j / total_gap
                    group.loc[start_idx + j, "x1"] = start_row["x1"] + ratio * (end_row["x1"] - start_row["x1"])
                    group.loc[start_idx + j, "y1"] = start_row["y1"] + ratio * (end_row["y1"] - start_row["y1"])
                    group.loc[start_idx + j, "x2"] = start_row["x2"] + ratio * (end_row["x2"] - start_row["x2"])
                    group.loc[start_idx + j, "y2"] = start_row["y2"] + ratio * (end_row["y2"] - start_row["y2"])
                    group.loc[start_idx + j, "conf"] = start_row["conf"] + ratio * (end_row["conf"] - start_row["conf"])
        df_list.append(group)

    return pd.concat(df_list, ignore_index=True)


def merge_with_player_data(df_interp, args):
    df_players = pd.read_csv(args.input_player_csv)
    metadata_df = pd.read_csv(args.sample_metadata_file)

    if not os.path.exists(args.output_csv_file):
        df_players['time'] = pd.to_numeric(df_players['time'], errors='coerce')
        df_players.dropna(subset=['time'], inplace=True)
        df_players['sample_id'] = df_players.apply(lambda row: find_closest_sample(row, metadata_df), axis=1)
        df_players = pd.merge(df_players, metadata_df[['id', 'start', 'end']], left_on='sample_id', right_on='id', how='left')
        df_players.drop(columns='id', inplace=True)
        df_players.rename(columns={'start': 'sample_start', 'end': 'sample_end'}, inplace=True)

    df_players[['ball_x1', 'ball_y1', 'ball_x2', 'ball_y2', 'ball_conf']] = df_players.apply(lambda row: find_ball_detection(row, df_interp), axis=1)
    df_players.to_csv(args.output_csv_file, index=False)
    print(f"結合結果を '{args.output_csv_file}' として保存しました")


def find_closest_sample(row, metadata_df):
    candidates = metadata_df[(metadata_df['game'] == row['game']) & (metadata_df['half'] == row['half'])]
    if candidates.empty:
        return np.nan
    if len(candidates) == 1:
        return candidates['id'].values[0]

    def distance(sample):
        t = row['time']
        start, end = sample['start'], sample['end']
        return 0 if start <= t <= end else min(abs(t - start), abs(t - end))

    candidates['dist'] = candidates.apply(distance, axis=1)
    best = candidates.loc[candidates['dist'].idxmin()]
    return best['id']


def find_ball_detection(row, df_ball, tol=2):
    try:
        sample_id_int = int(row['sample_id'])
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    video_id = f"SNGS-{sample_id_int:04d}"
    candidates = df_ball[df_ball['video_id'] == video_id]
    if candidates.empty:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

    candidates['time_diff'] = (candidates['image_id'] - row['image_id']).abs()
    best = candidates.loc[candidates['time_diff'].idxmin()]
    if best['time_diff'] > tol:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    return pd.Series([best['x1'], best['y1'], best['x2'], best['y2'], best['conf']])


def main():
    args = ProcessTrackedBallArgs().parse_args()
    df_track = process_ball_tracking(args)
    df_interp = interpolate_missing_frames(df_track, args.max_gap_length)
    merge_with_player_data(df_interp, args)


if __name__ == "__main__":
    main()
