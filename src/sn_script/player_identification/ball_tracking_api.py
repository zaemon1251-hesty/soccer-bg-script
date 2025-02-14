import os
import cv2
import pandas as pd
import supervision as sv
from inference import get_model

# Roboflow APIキーとモデルIDを設定
model = get_model(
    model_id="football-ball-detection-rejhg/4",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

results = []  # CSVに出力するための行リスト
root_dir = "/local/moriy/SoccerNetGS/rag-eval/test"  # SNGS-XXXXディレクトリがあるルートディレクトリ

# SNGS-XXXX ディレクトリを走査
for video_dir in sorted(os.listdir(root_dir)):
    if not video_dir.startswith("SNGS-"):
        continue

    img_dir = os.path.join(root_dir, video_dir, "img1")
    if not os.path.exists(img_dir):
        continue

    # img1内の画像ファイルを image_id の昇順に処理（例："000001.jpg"～"000750.jpg"）
    for image_file in sorted(os.listdir(img_dir)):
        # ファイル名が "%06d.jpg" 形式であることを確認
        name, ext = os.path.splitext(image_file)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            image_id = int(name)
        except ValueError:
            continue

        image_path = os.path.join(img_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"画像の読み込みに失敗: {image_path}")
            continue

        # モデルによる推論（信頼度0.3を下回る検出は除外）
        inference_result = model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(inference_result)

        # 検出があれば各検出結果を1行として追加
        if len(detections.xyxy) > 0:
            for box in detections.xyxy:
                x1, y1, x2, y2 = box  # バウンディングボックス座標（例: 左上と右下）
                results.append({
                    "video_id": video_dir,
                    "image_id": image_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
        else:
            # 検出なしの場合は、NaNを入れておく（後処理しやすいように）
            results.append({
                "video_id": video_dir,
                "image_id": image_id,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None
            })

# DataFrameに変換してCSVに保存
assert results, "検出結果が空です"
df = pd.DataFrame(results)
df.sort_values(["video_id", "image_id"], inplace=True)
df.to_csv("ball_tracking_results.csv", index=False)
print("CSVの出力が完了しました： ball_tracking_results.csv")
