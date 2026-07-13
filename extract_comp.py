# -*- coding: utf-8 -*-
"""幫 templates.json 的 131 個模板補抽構圖欄位(comp)。

對每個模板的來源照片重跑 MediaPipe PoseLandmarker,新增:
  comp: {
    hip:   [x, y]           髖中點在畫框的相對座標(0-1)
    bbox:  [x0, y0, x1, y1] 人物邊界框(可見關鍵點,0-1)
    eye_y: float            眼睛線 y(0-1)
    img_wh:[w, h]           原始照片像素尺寸(換算長寬比用)
  }
原有 id/src/full/lm 欄位不動。輸出前自動備份 templates.json.bak。
"""
import json, os, shutil, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(BASE, "templates.json")
IMGS = os.path.join(BASE, "pose_imgs")

# MediaPipe C++ 層讀不了非 ASCII 路徑:模型複製到 ASCII 暫存路徑再載入
model_tmp = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
shutil.copy(os.path.join(BASE, "pose_landmarker_lite.task"), model_tmp)

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_tmp),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

with open(TPL, encoding="utf-8") as f:
    data = json.load(f)

ok, fail = 0, []
for scene, tpls in data.items():
    for t in tpls:
        path = os.path.join(IMGS, scene, t["src"])
        if not os.path.exists(path):
            fail.append((t["id"], "找不到檔案"))
            continue
        # 圖片同樣避開非 ASCII 路徑:np.fromfile 讀 bytes 再 imdecode
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            fail.append((t["id"], "圖片解碼失敗"))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(img)
        if not res.pose_landmarks:
            fail.append((t["id"], "偵測不到人"))
            continue
        lm = res.pose_landmarks[0]
        vis = [p for p in lm if p.visibility > 0.5]
        if len(vis) < 8:
            fail.append((t["id"], f"可見關鍵點過少({len(vis)})"))
            continue
        clamp = lambda v: max(0.0, min(1.0, v))
        xs = [clamp(p.x) for p in vis]
        ys = [clamp(p.y) for p in vis]
        hip_x = clamp((lm[23].x + lm[24].x) / 2)
        hip_y = clamp((lm[23].y + lm[24].y) / 2)
        eye_y = clamp((lm[2].y + lm[5].y) / 2)  # 左右眼中心
        t["comp"] = {
            "hip": [round(hip_x, 4), round(hip_y, 4)],
            "bbox": [round(min(xs), 4), round(min(ys), 4),
                     round(max(xs), 4), round(max(ys), 4)],
            "eye_y": round(eye_y, 4),
            "img_wh": [img.width, img.height],
        }
        ok += 1

print(f"成功 {ok} / 失敗 {len(fail)}")
for fid, reason in fail:
    print(f"  ✗ {fid}: {reason}")

shutil.copy(TPL, TPL + ".bak")
with open(TPL, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
print("已寫回 templates.json(備份:templates.json.bak)")
