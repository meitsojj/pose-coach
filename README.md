# pose-coach 拍照姿勢教練

純前端的拍照 pose + 構圖引導工具(MediaPipe Pose,全程裝置端運算,照片與相機畫面不上傳)。

## 頁面

| 檔案 | 用途 |
|---|---|
| `pose-guide.html` | POSE 引導鏡:選場景模板 → 相機疊幽靈骨架 → 關節角評分提示 |
| `templates.json` | 131 個場景 pose 模板(11 場景;僅骨架關鍵點與場景標籤,不含照片) |
| `demo.html` | (內部工具)AI 照片點評,需自備 Claude API key |
| `batch.html` | (內部工具)模板批次抽取 pipeline |

## 開發備註

- 模板來源照片在本機 `pose_imgs/`(.gitignore 排除,不上 repo)
- 相機功能需 HTTPS,本 repo 以 GitHub Pages 部署
