# -*- coding: utf-8 -*-
"""把 templates.json 存的 lm 骨架疊回原照片,驗證骨架是否忠於講義。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
picks = sys.argv[2].split(",")

CONN = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
        (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
        (23,25),(24,26),(25,27),(26,28),(27,31),(28,32)]

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

for scene, tpls in data.items():
    for t in tpls:
        if t["id"] not in picks:
            continue
        path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h)) for p in t["lm"]]
        for a, b in CONN:
            cv2.line(bgr, pts[a], pts[b], (80, 220, 60), 4)
        for p in pts:
            cv2.circle(bgr, p, 6, (0, 80, 255), -1)
        scale = 520 / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with open(os.path.join(OUT, f"lm_{t['id']}.jpg"), "wb") as fo:
            fo.write(buf.tobytes())
        print("輸出", t["id"])
