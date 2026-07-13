# -*- coding: utf-8 -*-
"""認親結果目視對照:每組 [AI 生成圖 | 認親模板原照]。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

H = 420
cells = []
for scene, tpls in data.items():
    for t in tpls:
        if "doll" not in t:
            continue
        # 生成圖(取 doll 資產直接畫在白底)
        doll_path = os.path.join(BASE, t["doll"].replace("/", os.sep))
        rgba = cv2.imdecode(np.fromfile(doll_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        a = rgba[..., 3:].astype(np.float32) / 255
        gen = (rgba[..., :3].astype(np.float32) * a + 245 * (1 - a)).astype(np.uint8)
        s = H / gen.shape[0]
        gen = cv2.resize(gen, (int(gen.shape[1] * s), H))
        # 模板原照
        ref = cv2.imdecode(np.fromfile(os.path.join(BASE, "pose_imgs", scene, t["src"]),
                                        dtype=np.uint8), cv2.IMREAD_COLOR)
        s = H / ref.shape[0]
        ref = cv2.resize(ref, (int(ref.shape[1] * s), H))
        gap = np.full((H, 6, 3), 0, np.uint8)
        pair = np.hstack([gen, gap, ref])
        bar = np.full((34, pair.shape[1], 3), 50, np.uint8)
        cv2.putText(bar, t["id"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .8, (60, 220, 255), 2)
        cells.append(np.vstack([bar, pair]))

W = max(c.shape[1] for c in cells)
cells = [np.hstack([c, np.full((c.shape[0], W - c.shape[1], 3), 20, np.uint8)]) for c in cells]
sheet = np.vstack(cells)
cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(
    os.path.join(OUT, "match_check.jpg"))
print("輸出 match_check.jpg,共", len(cells), "組")
