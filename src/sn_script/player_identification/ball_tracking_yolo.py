from ultralytics import YOLO
import cv2
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

os.environ["YOLO_VERBOSE"] = "False"

# 事前にYOLOv8のモデルを読み込み（モデルパスは適宜変更してください）
model = YOLO("/local/moriy/model/soccernet/sn-gamestate/yolo/yolov8x6.pt")

results = []  # CSVに出力するための行リスト
root_dir = "/local/moriy/SoccerNetGS/rag-eval/test"  # SNGS-XXXXディレクトリがあるルートディレクトリ

# SNGS-XXXX ディレクトリを走査
for video_dir in tqdm(sorted(os.listdir(root_dir))):
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

        # ========== ここからYOLOv8による推論部分 ==========
        # YOLOv8で推論を実行（信頼度0.1以下の検出は除外、verboseをFalseに設定）
        results_yolo = model(frame, conf=0.2, verbose=False)

        # 推論結果から、ボールクラスのみ抽出（bbox.cls == 32
        if results_yolo and results_yolo[0].boxes is not None and len(results_yolo[0].boxes) > 0:
            boxes_obj = results_yolo[0].boxes
            # 各検出のクラス、信頼度、バウンディングボックス座標を取得
            cls_array = boxes_obj.cls.cpu().numpy()
            conf_array = boxes_obj.conf.cpu().numpy()
            xyxy_array = boxes_obj.xyxy.cpu().numpy()

            # ボールクラスのみフィルタ：クラスIDが32 かつ
            condition = (cls_array == 32)
            indices = np.where(condition)[0]
            filtered_boxes = xyxy_array[indices]
            filtered_conf = conf_array[indices]
            filtered_cls = cls_array[indices]
        else:
            filtered_boxes = []
            filtered_conf = []
            filtered_cls = []

        # 検出結果をCSV用に保存（複数検出がある場合は各ボックスを1行として記録）
        if len(filtered_boxes) > 0:
            for i in range(len(filtered_boxes)):
                x1, y1, x2, y2 = filtered_boxes[i]
                conf_val = filtered_conf[i]
                cls_val = int(filtered_cls[i])
                # クラス名を取得（model.namesが定義されていれば利用）
                class_name = model.names[cls_val] if hasattr(model, "names") else str(cls_val)
                results.append({
                    "video_id": video_dir,
                    "image_id": image_id,
                    "class_id": cls_val,
                    "class_name": class_name,
                    "conf": conf_val,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
        else:
            # 検出なしの場合は、座標にNoneを記録
            results.append({
                "video_id": video_dir,
                "image_id": image_id,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "class_id": None,
                "class_name": None,
                "conf": None,
            })
        # ========== ここまでYOLOv8による推論部分 ==========

# DataFrameに変換してCSVに保存
assert results, "検出結果が空です"
df = pd.DataFrame(results)
df.sort_values(["video_id", "image_id"], inplace=True)
df.to_csv("yolo_ball_tracking_results.csv", index=False)
print("CSVの出力が完了しました： yolo_ball_tracking_results.csv")
