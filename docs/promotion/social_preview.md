# GitHub Social Preview 图设计

## 规格

| 项 | 值 |
|----|-----|
| 尺寸 | **1280 × 640 px** (推荐) 或 1280 × 320 (最小) |
| 格式 | PNG / JPG / GIF (建议 PNG) |
| 文件大小 | ≤ 1 MB |
| 安全区 | 上下左右各留 80 px 边距 (避免被裁) |

## 上传位置

GitHub repo → Settings → 滚到 **Social preview** → Upload an image

## 设计稿（文字布局，你拿去画图）

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   [军绿色背景 + 半透明十字准星纹理]                              │
│                                                                │
│      🎯  DELTA FORCE TRADER                                     │
│      ╾──────────────────────╼                                   │
│                                                                │
│      三角洲行动 · 交易行自动上下架                                │
│      AUTO LISTING / DELISTING BOT                              │
│                                                                │
│      ─ 一键挂满 · 挂机循环 · ML 识别 · 30+ 坐标可视校准 ─        │
│                                                                │
│                                                  Python · PyQt5 │
│                                                  ⭐ MIT  📦 Win  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 颜色板

参考三角洲行动主视觉：

| 用途 | 颜色 | Hex |
|------|------|-----|
| 背景主色 | 军绿 | `#2F3E2A` ~ `#4A5A3F` 渐变 |
| 主标题文字 | 浅米色 | `#E8DDB5` |
| 副标题 | 灰白 | `#C8CDB7` |
| 强调色 (徽章) | 弹药橙 | `#E08533` |
| 装饰线 | 暗金 | `#A88532` |

## 装饰元素建议

- **左下角**：一个简化的箭头 / 准星 SVG icon
- **右下角**：Python / PyQt5 / MIT 三个 badge 形状 (可以是真实的徽章导出)
- **背景**：低对比度的"军用纸张/网格"纹理 (避免抢标题)
- **角落** (可选)：游戏内一些非品牌道具的剪影 (如医疗包/弹匣)，**不能用游戏官方 logo/角色**

---

## 设计工具推荐

| 工具 | 适合 | 链接 |
|------|------|------|
| **Figma** | 专业排版 (免费) | https://www.figma.com |
| **Canva** | 套模板最快 | https://www.canva.com 搜 "github banner" |
| **Photopea** | PS 替代品 (浏览器) | https://www.photopea.com |
| **GIMP** | 本地免费 PS | https://www.gimp.org |
| **Bannerbear** | 程序化生成 (有 API) | https://www.bannerbear.com |

### Canva 最快路径

1. 打开 Canva → 选 "自定义尺寸" → 1280 × 640
2. 搜索模板 "github banner" / "open source banner"
3. 改字、改色、加图标，5 分钟出图

---

## 自动生成方案 (如果你不想画图)

如果时间紧 / 不想画图，可以用 **GitHub 自带的 OG 图生成器**：

1. README 顶部加这一行：
   ```
   <p align="center"><img src="https://socialify.git.ci/xinshuowl/DeltaForce-Trader/image?description=1&font=Inter&issues=1&language=1&owner=1&pattern=Charlie%20Brown&stargazers=1&theme=Auto" /></p>
   ```
2. 不上传图，让 GitHub 默认 OG 图也能自动带描述 + 语言 + star 数

但**自定义图的转化率比自动图高 30-50%**，建议还是花时间做一张。

---

## 完成后效果

社交平台（Twitter / 微博 / QQ / 微信 / Telegram）转发 GitHub 链接时，会自动渲染：

```
┌──────────────────────────────────┐
│  [你的 1280×640 自定义图]          │
│                                  │
│  xinshuowl/DeltaForce-Trader     │
│  三角洲行动自动上下架工具...      │
│  ⭐ 2  🍴 0  Python              │
└──────────────────────────────────┘
```

视觉冲击 / 点击率比文字链接高 3-5 倍。
