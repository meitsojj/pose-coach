# -*- coding: utf-8 -*-
"""三聯對照圖:講義原照 | 骨架疊圖 | 純火柴人(與網頁同款簡化畫法)。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
picks = sys.argv[2].split(",")

# 與 anchor-preview.html 相同:只連軀幹四肢,頭另畫圓+脖子
CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
        (23,24),(23,25),(24,26),(25,27),(26,28),(27,31),(28,32)]
JOINTS = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]

def draw_skel(canvas, pts, color, lw):
    for a, b in CONN:
        cv2.line(canvas, pts[a], pts[b], color, lw)
    for i in JOINTS:
        cv2.circle(canvas, pts[i], lw + 2, color, -1)
    # 頭圓:雙耳中點 + 肩寬×0.36;脖子:頭圓底→肩中點
    cx, cy = (pts[7][0] + pts[8][0]) // 2, (pts[7][1] + pts[8][1]) // 2
    r = int(np.hypot(pts[11][0] - pts[12][0], pts[11][1] - pts[12][1]) * 0.36)
    cv2.circle(canvas, (cx, cy), max(r, 4), color, lw)
    sx, sy = (pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2
    cv2.line(canvas, (cx, cy + max(r, 4)), (sx, sy), color, lw)

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

H = 540  # 每格高度
for scene, tpls in data.items():
    for t in tpls:
        if t["id"] not in picks:
            continue
        path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        s = H / h
        img = cv2.resize(bgr, (int(w * s), H))
        iw = img.shape[1]
        pts = [(int(p[0] * iw), int(p[1] * H)) for p in t["lm"]]

        overlay = img.copy()
        draw_skel(overlay, pts, (80, 220, 60), 3)

        stick = np.full((H, iw, 3), 20, np.uint8)
        draw_skel(stick, pts, (191, 212, 45), 3)  # 網頁的 aqua 色(BGR)

        gap = np.full((H, 12, 3), 255, np.uint8)
        combo = np.hstack([img, gap, overlay, gap, stick])
        for i, label in enumerate(["1 講義原照", "2 骨架疊圖", "3 火柴人(網頁)"]):
            cv2.putText(combo, f"{i+1}", (i * (iw + 12) + 14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        ok, buf = cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with open(os.path.join(OUT, f"cmp_{t['id']}.jpg"), "wb") as fo:
            fo.write(buf.tobytes())
        print("輸出", t["id"])
