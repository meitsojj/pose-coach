# -*- coding: utf-8 -*-
"""用 comp 構圖資料自動篩選生成參考包候選模板,輸出縮覽表供目視複核。
規則:排除雙人場景;排除橫圖(多為左右對比拼圖);站位過偏/人物過小扣分。
每場景取前 5 名候選。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

cands = {}
for scene, tpls in data.items():
    if scene == "雙人":
        continue
    scored = []
    for t in tpls:
        c = t["comp"]
        w, h = c["img_wh"]
        if w >= h:            # 橫圖 → 疑似拼圖
            continue
        hip_x = c["hip"][0]
        bh = c["bbox"][3] - c["bbox"][1]
        score = -abs(hip_x - 0.5) * 3 - abs(bh - 0.72)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    cands[scene] = [t for _, t in scored[:5]]

# 縮覽表:每場景一列,每格原照縮圖+id
CELL_H, CELL_W = 300, 170
rows = []
for scene, tpls in cands.items():
    cells = []
    for t in tpls:
        p = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        hh, ww = bgr.shape[:2]
        s = min(CELL_W / ww, (CELL_H - 26) / hh)
        img = cv2.resize(bgr, (int(ww * s), int(hh * s)))
        cell = np.full((CELL_H, CELL_W, 3), 30, np.uint8)
        y0 = (CELL_H - 26 - img.shape[0]) // 2
        x0 = (CELL_W - img.shape[1]) // 2
        cell[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img
        num = t["id"].split("-")[-1]
        cv2.putText(cell, num, (6, CELL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, .7, (60, 220, 255), 2)
        cells.append(cell)
    while len(cells) < 5:
        cells.append(np.full((CELL_H, CELL_W, 3), 30, np.uint8))
    row = np.hstack(cells)
    label = np.full((36, row.shape[1], 3), 55, np.uint8)
    row = np.vstack([label, row])
    rows.append((scene, row))

half = (len(rows) + 1) // 2
for part, chunk in enumerate([rows[:half], rows[half:]], 1):
    sheet = np.vstack([r for _, r in chunk])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    with open(os.path.join(OUT, f"candidates_{part}.jpg"), "wb") as fo:
        fo.write(buf.tobytes())
    print(f"candidates_{part}.jpg:", ", ".join(s for s, _ in chunk))

with open(os.path.join(OUT, "candidates.json"), "w", encoding="utf-8") as f:
    json.dump({s: [t["id"] for t in ts] for s, ts in cands.items()}, f, ensure_ascii=False, indent=1)
print("候選總數:", sum(len(v) for v in cands.values()))
