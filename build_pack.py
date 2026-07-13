# -*- coding: utf-8 -*-
"""產出 AI 生成參考包:42 個精選模板 → OpenPose 骨架參考圖 + 原照參考 + prompts.md。"""
import json, os, sys, io
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(BASE, "生成參考包")
os.makedirs(PACK, exist_ok=True)

SELECT = {
    "戶外人像": [6, 9, 7, 4, 11], "半身pose": [10, 3, 1, 6, 5],
    "男生": [9, 3, 4, 12, 11], "咖啡廳": [5, 1, 10],
    "全身pose": [8, 11, 4, 12, 5], "獨旅自拍": [9, 1, 12, 11],
    "夜拍": [2, 8, 7, 4], "姿勢靈感": [2, 4, 6, 9],
    "花海": [8, 11, 9], "不露臉": [3, 1, 12, 4],
}
SCENE_HINT = {
    "戶外人像": "outdoor scene", "半身pose": "upper body emphasis",
    "男生": "masculine body proportions", "咖啡廳": "sitting or leaning, cafe context",
    "全身pose": "full body standing", "獨旅自拍": "casual traveler vibe",
    "夜拍": "dynamic playful pose", "姿勢靈感": "sitting on a stool where applicable",
    "花海": "standing in open field", "不露臉": "face turned away from camera",
}

# MediaPipe 33 → OpenPose COCO-18
def mp2op(lm, midfn):
    m = lambda i: lm[i]
    return [m(0), midfn(lm[11], lm[12]), m(12), m(14), m(16), m(11), m(13), m(15),
            m(24), m(26), m(28), m(23), m(25), m(27), m(5), m(2), m(8), m(7)]

LIMBS = [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
         (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
          (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
          (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170)]

with open(os.path.join(BASE, "templates.json"), encoding="utf-8") as f:
    data = json.load(f)

rows, count = [], 0
for scene, nums in SELECT.items():
    want = {f"{scene}-{n}" for n in nums}
    for t in data[scene]:
        if t["id"] not in want:
            continue
        count += 1
        w, h = t["comp"]["img_wh"]
        H = 1024; W = int(w / h * H)
        pts = [(int(p[0] * W), int(p[1] * H)) for p in t["lm"]]
        op = mp2op(pts, lambda a, b: ((a[0]+b[0])//2, (a[1]+b[1])//2))
        canvas = np.zeros((H, W, 3), np.uint8)
        for (a, b), col in zip(LIMBS, COLORS):
            bgr = (col[2], col[1], col[0])
            cv2.line(canvas, op[a], op[b], bgr, 8)
        for p, col in zip(op, COLORS + [(255,0,170)]):
            cv2.circle(canvas, p, 9, (col[2], col[1], col[0]), -1)
        cv2.imencode(".png", canvas)[1].tofile(os.path.join(PACK, f"{t['id']}_pose.png"))

        src = os.path.join(BASE, "pose_imgs", scene, t["src"])
        bgr = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        s = 768 / bgr.shape[0]
        small = cv2.resize(bgr, (int(bgr.shape[1] * s), 768))
        cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(
            os.path.join(PACK, f"{t['id']}_ref.jpg"))

        body = "full body" if t["full"] else "upper body"
        rows.append(f"| {t['id']} | {scene} | {body} | {SCENE_HINT[scene]} |")

BASE_PROMPT = ("gray 3D mannequin figure, smooth matte light-gray body with subtle "
               "contour topology lines, articulated art doll, plain white studio "
               "background, soft even lighting, clean 3D render, single figure, "
               "whole body in frame not cropped, pose exactly matching the reference")
NEG_PROMPT = "photo-realistic human, skin, face details, hair, clothes, text, watermark, cropped limbs, multiple people"

md = f"""# 生成參考包 — 人偶風模板圖(42 個)

每個模板兩個檔案:
- `<id>_pose.png` — OpenPose 標準骨架圖(有 ControlNet 就餵這張,pose 完全一致)
- `<id>_ref.jpg` — 課程原照參考(無 ControlNet 就把這張丟給 AI 工具,要求「同姿勢換成灰色人偶」)

## 基礎 Prompt(每張通用,再加下表的場景提示)

```
{BASE_PROMPT}
```

Negative prompt:

```
{NEG_PROMPT}
```

## 生成要求(三條)
1. 單人、全身入鏡、四肢不裁切
2. 背景素色簡潔(白/淺灰)
3. 直式,比例照原照(多數為 3:4 或 9:16)

## 完成後
生成圖存到 `pose-coach/gen_imgs/`,檔名 = 模板 id(如 `全身pose-8.png`)。
全部丟進去後跟 Claude 說一聲,會自動 ingest:重抽骨架/構圖/遮罩 → 比對關節角(差太多會列出требующие重生成)→ 產出淡化疊層上線。

## 模板清單

| id | 場景 | 構圖 | 場景提示(加進 prompt) |
|---|---|---|---|
""" + "\n".join(rows) + "\n"
md = md.replace("требующие", "需要")  # guard

with open(os.path.join(PACK, "README.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(f"完成:{count} 個模板 → 生成參考包/(pose.png + ref.jpg 各 {count} 張 + README.md)")
