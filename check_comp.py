# -*- coding: utf-8 -*-
"""抽查 comp 欄位:把 bbox/髖中點/眼睛線畫回原照片,輸出到指定資料夾。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
picks = ["戶外人像-1", "咖啡廳-3", "全身pose-5", "半身pose-2", "雙人-1", "不露臉-4"]

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

for scene, tpls in data.items():
    for t in tpls:
        if t["id"] not in picks:
            continue
        path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        c = t["comp"]
        x0, y0, x1, y1 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(c["bbox"])]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 3)
        hx, hy = int(c["hip"][0] * w), int(c["hip"][1] * h)
        cv2.circle(bgr, (hx, hy), 12, (0, 0, 255), -1)
        ey = int(c["eye_y"] * h)
        cv2.line(bgr, (0, ey), (w, ey), (255, 200, 0), 3)
        scale = 480 / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with open(os.path.join(OUT, f"check_{t['id']}.jpg"), "wb") as fo:
            fo.write(buf.tobytes())
        print("輸出", t["id"], "| bbox 高", round(c["bbox"][3] - c["bbox"][1], 3),
              "| hip", c["hip"], "| eye_y", c["eye_y"])
