# -*- coding: utf-8 -*-
"""三種骨架呈現風格 demo:A 細節火柴人 / B 素描人偶 / C 剪影+骨架。"""
import json, os, sys, io, shutil, tempfile
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
picks = sys.argv[2].split(",")

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

AQUA = (191, 212, 45)  # BGR
LIMBS = [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)]
TORSO = [11, 12, 24, 23]

def head_geo(pts):
    cx, cy = (pts[7][0]+pts[8][0])/2, (pts[7][1]+pts[8][1])/2
    r = np.hypot(pts[11][0]-pts[12][0], pts[11][1]-pts[12][1]) * 0.36
    return int(cx), int(cy), max(int(r), 4)

def hand_pts(pts, side):  # side 0=左 1=右
    w, p, i = (15,17,19) if side == 0 else (16,18,20)
    return pts[w], ((pts[p][0]+pts[i][0])//2, (pts[p][1]+pts[i][1])//2)

def foot_tri(pts, side):
    a, h, t = (27,29,31) if side == 0 else (28,30,32)
    return np.array([pts[a], pts[h], pts[t]], np.int32)

def style_A(canvas, pts, lw):  # 細節火柴人:含手掌方向與腳掌三角
    for a, b in LIMBS + [(11,12),(23,24),(11,23),(12,24)]:
        cv2.line(canvas, pts[a], pts[b], AQUA, lw)
    for s in (0, 1):
        wr, fg = hand_pts(pts, s)
        cv2.line(canvas, wr, fg, AQUA, lw)               # 手掌方向
        cv2.circle(canvas, fg, lw+1, AQUA, -1)
        cv2.polylines(canvas, [foot_tri(pts, s)], True, AQUA, max(lw-1,2))  # 腳掌三角
    for i in [11,12,13,14,23,24,25,26]:
        cv2.circle(canvas, pts[i], lw+2, AQUA, -1)
    cx, cy, r = head_geo(pts)
    cv2.circle(canvas, (cx, cy), r, AQUA, lw)
    sx, sy = (pts[11][0]+pts[12][0])//2, (pts[11][1]+pts[12][1])//2
    cv2.line(canvas, (cx, cy+r), (sx, sy), AQUA, lw)

def style_B(canvas, pts, hgt):  # 素描人偶:膠囊肢體+軀幹面+手腳形狀
    fill = np.zeros_like(canvas)
    u = max(int(hgt*0.012), 3)  # 粗細基準
    thick = {(11,13):3.0,(13,15):2.2,(12,14):3.0,(14,16):2.2,
             (23,25):3.6,(25,27):2.6,(24,26):3.6,(26,28):2.6}
    col = (170, 190, 90)
    cv2.fillPoly(fill, [np.array([pts[i] for i in TORSO], np.int32)], col)
    for (a, b), k in thick.items():
        cv2.line(fill, pts[a], pts[b], col, int(u*k))
        cv2.circle(fill, pts[b], int(u*k/2), col, -1)
    for s in (0, 1):
        wr, fg = hand_pts(pts, s)
        c = ((wr[0]+fg[0])//2, (wr[1]+fg[1])//2)
        ang = np.degrees(np.arctan2(fg[1]-wr[1], fg[0]-wr[0]))
        cv2.ellipse(fill, c, (int(u*1.6), int(u*0.9)), ang, 0, 360, col, -1)
        cv2.fillPoly(fill, [foot_tri(pts, s)], col)
    cx, cy, r = head_geo(pts)
    cv2.circle(fill, (cx, cy), r, col, -1)
    cv2.line(fill, (cx, cy), ((pts[11][0]+pts[12][0])//2, (pts[11][1]+pts[12][1])//2), col, u*2)
    cv2.addWeighted(fill, 0.85, canvas, 1, 0, canvas)   # 主體
    edge = cv2.Canny(cv2.cvtColor(fill, cv2.COLOR_BGR2GRAY), 40, 120)
    canvas[edge > 0] = (230, 240, 180)                   # 輪廓提亮
    # 關節鉚釘(素描人偶感)
    for i in [11,12,13,14,23,24,25,26]:
        cv2.circle(canvas, pts[i], max(u//2,2), (40,50,25), -1)

def style_C(canvas, mask, pts, lw):  # 剪影+骨架
    sil = (mask > 0.5).astype(np.uint8)
    overlay = canvas.copy()
    overlay[sil > 0] = (140, 110, 60)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, AQUA, 2)
    style_A(canvas, pts, max(lw-1, 2))

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

H = 540
for scene, tpls in data.items():
    for t in tpls:
        if t["id"] not in picks:
            continue
        path = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        h, w = bgr.shape[:2]
        s = H / h
        iw = int(w * s)
        img = cv2.resize(bgr, (iw, H))
        pts = [(int(p[0]*iw), int(p[1]*H)) for p in t["lm"]]
        dark = lambda: np.full((H, iw, 3), 18, np.uint8)

        cA, cB, cC = dark(), dark(), dark()
        style_A(cA, pts, 3)
        style_B(cB, pts, H)
        if res.segmentation_masks:
            m = cv2.resize(res.segmentation_masks[0].numpy_view(), (iw, H))
            style_C(cC, m, pts, 3)

        gap = np.full((H, 10, 3), 255, np.uint8)
        combo = np.hstack([img, gap, cA, gap, cB, gap, cC])
        for i, lab in enumerate(["photo", "A", "B", "C"]):
            cv2.putText(combo, lab, (i*(iw+10)+12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
        ok, buf = cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 84])
        with open(os.path.join(OUT, f"styles_{t['id']}.jpg"), "wb") as fo:
            fo.write(buf.tobytes())
        print("輸出", t["id"])
