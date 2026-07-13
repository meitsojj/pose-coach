# -*- coding: utf-8 -*-
"""淡化透明片效果 demo:全圖淡化 / 人物去背淡化 / 現行剪影,疊在模擬相機畫面上比較。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
picks = sys.argv[2].split(",")

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

H = 540
def cam_bg(w):
    """模擬相機實景(漸層+雜訊),測試殘影在真實畫面上的可讀性"""
    g = np.linspace(70, 25, H, dtype=np.uint8)
    bg = np.dstack([np.tile(g[:, None], (1, w))] * 3)
    bg[..., 0] = np.clip(bg[..., 0].astype(int) + 15, 0, 255)  # 偏藍
    noise = np.random.default_rng(7).integers(-8, 8, bg.shape, dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

for scene, tpls in data.items():
    for t in tpls:
        if t["id"] not in picks:
            continue
        img_path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        s = H / h; iw = int(w * s)
        photo = cv2.resize(bgr, (iw, H)).astype(np.float32)

        mask_path = os.path.join(BASE, t["mask"])
        rgba = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        alpha = cv2.resize(rgba[..., 3], (iw, H)).astype(np.float32) / 255.0

        # a. 全圖淡化 35%
        a = cam_bg(iw).astype(np.float32)
        a = a * 0.65 + photo * 0.35

        # b. 人物去背淡化 55%(遮罩切出人物)
        b = cam_bg(iw).astype(np.float32)
        m3 = np.dstack([alpha * 0.55] * 3)
        b = b * (1 - m3) + photo * m3

        # c. 現行剪影(染色 + 輪廓光近似)
        c = cam_bg(iw).astype(np.float32)
        sil = np.dstack([alpha * 0.92] * 3)
        tint = np.zeros_like(c); tint[:] = (88, 79, 39)  # 深青 BGR
        c = c * (1 - sil) + tint * sil
        edge = cv2.Canny((alpha * 255).astype(np.uint8), 60, 160)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
        c[edge > 0] = (191, 212, 45)

        gap = np.full((H, 10, 3), 255, np.uint8)
        combo = np.hstack([x.astype(np.uint8) for x in
                           [photo, gap, a, gap, b, gap, c]])
        for i, lab in enumerate(["photo", "a", "b", "c"]):
            cv2.putText(combo, lab, (i * (iw + 10) + 12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        ok, buf = cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 84])
        with open(os.path.join(OUT, f"fade_{t['id']}.jpg"), "wb") as fo:
            fo.write(buf.tobytes())
        print("輸出", t["id"])
