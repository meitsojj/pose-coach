# -*- coding: utf-8 -*-
"""ingest AI 人偶圖:自動姿勢比對認親 → 去背 → 產 dolls/ 疊層資產 → 更新 templates.json。

用法:py ingest_dolls.py           # 處理 gen_imgs/ 全部圖檔
輸出:dolls/tNNN.webp(RGBA 去背人偶,與遮罩同編號)
     templates.json 各命中模板加 doll / doll_comp / doll_lm 欄位
"""
import json, os, sys, io, shutil, tempfile, math
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(BASE, "gen_imgs")
DOLLS = os.path.join(BASE, "dolls")
os.makedirs(DOLLS, exist_ok=True)

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

# 8 個關節角:左右肘(肩-肘-腕)、左右肩(肘-肩-髖)、左右髖(肩-髖-膝)、左右膝(髖-膝-踝)
ANGLES = [(11,13,15),(12,14,16),(13,11,23),(14,12,24),
          (11,23,25),(12,24,26),(23,25,27),(24,26,28)]

def angle(a, b, c):
    v1 = (a[0]-b[0], a[1]-b[1]); v2 = (c[0]-b[0], c[1]-b[1])
    d1 = math.hypot(*v1); d2 = math.hypot(*v2)
    if d1 < 1e-6 or d2 < 1e-6: return None
    cos = max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(d1*d2)))
    return math.degrees(math.acos(cos))

def pose_angles(lm):
    return [angle(lm[a], lm[b], lm[c]) for a, b, c in ANGLES]

def ang_diff(a1, a2):
    ds = [abs(x-y) for x, y in zip(a1, a2) if x is not None and y is not None]
    return sum(ds)/len(ds) if ds else 999

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)
all_tpls = [(scene, t) for scene, tpls in data.items() for t in tpls]

# 先偵測全部檔案,再做「一模板一圖」的貪婪分配(撞號時差值小者得,輸家取次佳未佔用)
detected = []
for fname in sorted(os.listdir(GEN)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue
    path = os.path.join(GEN, fname)
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not res.pose_landmarks:
        detected.append((fname, None, None, None)); continue
    lm = [(p.x, p.y) for p in res.pose_landmarks[0]]
    detected.append((fname, bgr, res, lm))

tpl_angles = {t["id"]: pose_angles(t["lm"]) for _, t in all_tpls}
tpl_by_id = {t["id"]: t for _, t in all_tpls}
pairs = []  # (diff, fname, tid)
for fname, bgr, res, lm in detected:
    if lm is None: continue
    stem = os.path.splitext(fname)[0]
    if stem in tpl_by_id:  # 檔名即模板 id → 鎖定
        pairs.append((-1, fname, stem)); continue
    ags = pose_angles(lm)
    for tid, ta in tpl_angles.items():
        pairs.append((ang_diff(ags, ta), fname, tid))
pairs.sort()
assigned_f, assigned_t = {}, set()
for d, fname, tid in pairs:
    if fname in assigned_f or tid in assigned_t: continue
    assigned_f[fname] = (tid, d if d >= 0 else ang_diff(
        pose_angles(dict((f, l) for f, _, _, l in detected)[fname]), tpl_angles[tid]))
    assigned_t.add(tid)

results = []
for fname, bgr, res, lm in detected:
    if lm is None:
        results.append((fname, None, None, "偵測不到人")); continue
    tid, best = assigned_f[fname]
    t = tpl_by_id[tid]

    # 去背疊層資產
    h, w = bgr.shape[:2]
    m = cv2.resize(res.segmentation_masks[0].numpy_view(), (w, h))
    alpha = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    alpha = cv2.medianBlur(alpha, 5)
    s = 720 / h
    small = cv2.resize(bgr, (int(w*s), 720))
    a_small = cv2.resize(alpha, (int(w*s), 720))
    rgba = np.dstack([small, a_small])
    idx = t["mask"].split("/")[-1].split(".")[0]  # tNNN
    out = f"dolls/{idx}.webp"
    ok, buf = cv2.imencode(".webp", rgba, [cv2.IMWRITE_WEBP_QUALITY, 82])
    if not ok:
        out = f"dolls/{idx}.png"
        ok, buf = cv2.imencode(".png", rgba)
    with open(os.path.join(BASE, out.replace("/", os.sep)), "wb") as fo:
        fo.write(buf.tobytes())

    # doll 自身構圖(疊層錨定用)
    vis = [p for p in res.pose_landmarks[0] if p.visibility > 0.5]
    clamp = lambda v: max(0.0, min(1.0, v))
    xs = [clamp(p.x) for p in vis]; ys = [clamp(p.y) for p in vis]
    t["doll"] = out
    t["doll_comp"] = {
        "hip": [round(clamp((lm[23][0]+lm[24][0])/2), 4), round(clamp((lm[23][1]+lm[24][1])/2), 4)],
        "bbox": [round(min(xs),4), round(min(ys),4), round(max(xs),4), round(max(ys),4)],
        "img_wh": [w, h]}
    t["doll_lm"] = [[round(x,4), round(y,4)] for x, y in lm]
    kb = len(buf)//1024
    results.append((fname, t["id"], round(best,1), f"{out}({kb}KB)"))

print(f"{'檔案':<44}{'認親模板':<14}{'平均關節角差':<10}輸出")
warn = []
for fname, tid, diff, note in results:
    print(f"{fname[:42]:<44}{str(tid):<14}{str(diff)+'°':<10}{note}")
    if diff is not None and diff > 20: warn.append((fname, tid, diff))

shutil.copy(os.path.join(BASE, "templates.json"), os.path.join(BASE, "templates.json.bak"))
with open(os.path.join(BASE, "templates.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
print("templates.json 已更新")
if warn:
    print("\n⚠ 關節角差 >20° 建議重生成:")
    for fname, tid, diff in warn:
        print(f"  {fname} → {tid}({diff}°)")
