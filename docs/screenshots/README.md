# 截图与演示资源指引

本目录托管 README 顶部展示的演示 GIF 和功能截图。**这是项目"门面"，请优先准备好。**

## 必备文件清单

| 文件名 | 用途 | 尺寸建议 | 录制要点 |
|--------|------|---------|---------|
| `demo.gif` | README 顶部主演示 | 720px 宽 / ≤ 8 MB | OBS 录 30-60s · 1) 启动 2) Ctrl+1 选模式 3) 自动点击 4) 收入统计 |
| `main.png` | 主控制面板截图 | 1280×720 | 程序刚启动的界面 |
| `filter.png` | 筛选设置截图 | 1280×720 | 勾选了几个品质和分类的状态 |
| `calibrate.png` | 坐标校准截图 | 1280×720 | 显示坐标列表 + 截图拾取按钮 |
| `review.png` | ML 标注审核截图 | 1280×720 | 已加载样本的审核对话框 |

## 录制 GIF 的推荐流程

### 方案 A（最简单）— ScreenToGif
1. 下载 [ScreenToGif](https://www.screentogif.com/)（免费 / 开源 / Windows 原生）
2. 框选目标录制区域（建议 1280×720 ROI）
3. 录制 30-60 秒完整流程
4. 在 ScreenToGif 编辑器里：
   - 帧率降到 15 fps（减小体积）
   - 颜色深度调 8 bit
   - 导出 GIF
5. 用 [Squoosh.app](https://squoosh.app/) 或 [TinyPNG](https://tinypng.com/) 再压一遍，目标 < 8 MB

### 方案 B（更高质量）— OBS + ffmpeg
1. OBS 录 MP4（建议 720p / 30fps / CRF 23）
2. `ffmpeg -i demo.mp4 -vf "fps=15,scale=720:-1" -loop 0 demo.gif`
3. 配合 [gifski](https://gif.ski/) 二次优化色板

### 方案 C（最省事）— 直接传视频
GitHub README 支持直接嵌入 MP4（拖到 issue 区获取 user-images URL）。
不需要 GIF 体积压缩，画质更高。

## 截图建议

- 全部截图统一在 **2560×1440 / 150% 缩放** 下截取（与开发分辨率一致）
- 主控制面板要展示 1-2 次成功上架的日志（视觉效果好）
- 筛选设置要勾选 5+ 项（让用户看到能筛多细）
- 标注审核要加载真实样本（不要空对话框）
- 截图后用 [Compresspng.com](https://compresspng.com/) 压一遍

## 上传后无需改 README

README 引用路径已经写死为 `docs/screenshots/xxx.png`，文件名对上即可自动渲染。

> 截图准备好之前，README 顶部会显示 broken image 图标，是正常现象。
