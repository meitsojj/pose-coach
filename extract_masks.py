# -*- coding: utf-8 -*-
"""幫 131 個模板抽人物分割遮罩,存 masks/tNNN.png(RGBA:人形白色不透明,背景透明),
並在 templates.json 各模板加 "mask" 欄位。可重跑(覆蓋)。"""
import json, os, sys, io, shutil, tempfile
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
MASKS = os.path.join(BASE, "masks")
os.makedirs(MASKS, exist_ok=True)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

model_tmp = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
if not os.path.exists(model_tmp):
    shutil.copy(os.path.join(BASE, "pose_landmarker_lite.task"), model_tmp)
landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_tmp),
    running_mode=vision.RunningMode.IMAGE, num_poses=1,
    output_segmentation_masks=True))

TPL = os.path.join(BASE, "templates.json")
with open(TPL, encoding="utf-8") as f:
    data = json.load(f)

H_OUT = 480  # 遮罩輸出高度(畫布上放大繪製足夠)
ok, fail, idx = 0, [], 0
for scene, tpls in data.items():
    for t in tpls:
        idx += 1
        name = f"t{idx:03d}.png"
        path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            fail.append((t["id"], "圖片解碼失敗")); continue
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not res.segmentation_masks:
            fail.append((t["id"], "無分割遮罩")); continue
        m = res.segmentation_masks[0].numpy_view()
        h, w = m.shape[:2]
        s = H_OUT / h
        m = cv2.resize(m, (int(w * s), H_OUT))
        alpha = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        alpha = cv2.medianBlur(alpha, 5)  # 去鋸齒毛邊
        rgba = np.zeros((alpha.shape[0], alpha.shape[1], 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = alpha
        okk, buf = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        with open(os.path.join(MASKS, name), "wb") as fo:
            fo.write(buf.tobytes())
        t["mask"] = f"masks/{name}"
        ok += 1

print(f"成功 {ok} / 失敗 {len(fail)}")
for fid, r in fail:
    print(f"  ✗ {fid}: {r}")
total = sum(os.path.getsize(os.path.join(MASKS, f)) for f in os.listdir(MASKS))
print(f"masks/ 總大小: {total/1024:.0f} KB")

shutil.copy(TPL, TPL + ".bak")
with open(TPL, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
print("已寫回 templates.json")
