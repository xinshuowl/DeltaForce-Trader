# 截图与演示资源指引

本目录托管 README 顶部展示的功能截图。**这是项目"门面"，请优先准备好。**

## 当前已就位的文件

| 文件名 | 用途 | 状态 |
|--------|------|------|
| `main.png` | 主控制面板截图 | ✅ |
| `filter.png` | 筛选设置截图 | ✅ |
| `calibrate.png` | 坐标校准截图 | ✅ |
| `review.png` | ML 标注审核截图 | ✅ |
| ~~`demo.gif`~~ | ~~README 顶部主演示~~ | 已改用 MP4 走 Release CDN, 见下文 |

## demo 演示视频走 Release CDN 方案

GitHub 出于安全考虑禁止 README 通过相对路径渲染 `<video>` 标签 — 必须用 `github.com/*/releases/download/*` 或 `github.com/user-attachments/*` 的 URL。本项目用 **Release attachment** 方案:

- demo.mp4 作为 `v1.1.0` Release 的额外附件上传
- README 通过 `<video src="https://github.com/xinshuowl/DeltaForce-Trader/releases/download/v1.1.0/demo.mp4">` 引用
- 优势: 体积小 (4.5 MB vs GIF 8.9 MB) / 画质好 / 可暂停 / GitHub CDN 加速
- 仓库不再托管视频文件 (瘦身 9 MB)

### 录制 / 替换 demo 视频的流程

下次想换 demo 视频时按以下步骤:

```powershell
# 1. 用 OBS 或 ScreenToGif 录 MP4 (720p / 30fps / 30-60s)
#    OBS 推荐: 设置 → 输出 → 录像 → 品质 "高质量, 中等文件大小", CRF 23

# 2. 如果体积过大, 用 ffmpeg 压缩 (建议压到 < 8MB):
ffmpeg -y -i demo_raw.mp4 -movflags +faststart -pix_fmt yuv420p `
       -vf "fps=20,scale=trunc(iw/2)*2:trunc(ih/2)*2" `
       -c:v libx264 -preset slow -crf 23 docs\screenshots\demo.mp4

# 3. 上传到 Release (作为 Release Asset, 不需要重新打 tag)
gh release upload v1.1.0 docs\screenshots\demo.mp4 --clobber

# 4. 上传完成后删除本地的 mp4 (仓库不托管视频文件)
Remove-Item docs\screenshots\demo.mp4

# 5. README 里的 URL 不变, 自动指向新版 demo
```

> 注意: `--clobber` 会覆盖同名旧 asset。如果想保留旧版, 改个版本号 / 文件名重新引用。

## 录制视频的工具推荐

| 工具 | 优点 | 适用场景 |
|------|------|---------|
| **OBS Studio** | 免费 / 开源 / 画质最佳 | 主推荐 |
| **ScreenToGif** | 同时支持录 GIF/MP4, 内置编辑器 | 想简单点 |
| **NVIDIA ShadowPlay** | N 卡用户 / 一键录制游戏窗口 | 显卡支持时 |
| **Windows 自带 Xbox Game Bar** | Win+G 一键开 | 临时录 |

## 截图建议 (静态 PNG)

- 全部截图统一在 **2560×1440 / 150% 缩放** 下截取 (与开发分辨率一致)
- 主控制面板要展示 1-2 次成功上架的日志 (视觉效果好)
- 筛选设置要勾选 5+ 项 (让用户看到能筛多细)
- 标注审核要加载真实样本 (不要空对话框)
- 截图后用 [Compresspng.com](https://compresspng.com/) 或 [Squoosh.app](https://squoosh.app/) 压一遍, 目标单张 < 300 KB

## 上传后无需改 README

README 引用路径已经写死为 `docs/screenshots/xxx.png`, 文件名对上即可自动渲染。
