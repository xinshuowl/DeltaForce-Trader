# 技术博文草稿 — 用 ML + OCR 写一个游戏交易行机器人

## 投放平台

- **掘金** — 适合技术深度文章，会被推到首页
- **V2EX** — 适合"作品分享"，节点选 `创造` 或 `分享发现`
- **思否 / CSDN** — 长尾流量
- **少数派** — 偏产品向，门槛高，慎投
- **GitHub Trending 候选** — 文章写好后顺便去 r/Python 发英文版

## 标题候选

1. ✅ **【作品分享】用 Python + ML + OCR 写了一个三角洲行动的交易行机器人，开源**
2. ✅ **如何识别游戏 UI 上的"非绑定"物品 — 一次 ML 二分类的工程实践**
3. ⚠ **从 0 到 1 给游戏写自动化工具 — PyQt5 + OpenCV + scikit-learn 实战**

## 文章框架（建议 1500-2500 字）

```
1. 引子：手动操作太累，决定写工具         (200 字)
2. 技术选型：为什么选 Python 全家桶       (200 字)
3. 难点 1：识别"非绑定"物品                (400 字)
4. 难点 2：多格大型物品防重复点击          (400 字)
5. 难点 3：跨页去重的三层机制              (300 字)
6. 难点 4：可视化坐标校准                  (300 字)
7. 性能优化：从 2s/件 压到 1s/件           (300 字)
8. 工程化：GitHub Actions 自动构建        (200 字)
9. 收尾：开源地址 + Star 求支持            (100 字)
```

---

## 正文草稿

### 1. 引子：手动操作太累

三角洲行动 (Delta Force) 玩家应该都知道，每次仓库满了要把非绑定的道具一件件挂上交易行 — 上架满 15 个位置，等成交，下架，整理仓库，循环。手动跑一轮 30 分钟，写脚本跑一轮 3 分钟。

于是有了这个项目：[xinshuowl/DeltaForce-Trader](https://github.com/xinshuowl/DeltaForce-Trader)。开源 MIT 协议，纯 Python，PyQt5 GUI，PyInstaller 打包，GitHub Actions 自动发版。

这篇文章聊聊里面几个有意思的工程问题。

### 2. 技术选型

| 模块 | 技术 | 替代方案 | 理由 |
|------|------|---------|------|
| GUI | PyQt5 | tkinter, customtkinter | 表现力强 + 主题化方便 |
| 截屏 | PIL.ImageGrab / mss | win32 API | 跨主流分辨率/DPI 兼容 |
| 图像处理 | OpenCV | scikit-image | 性能 + 生态 |
| OCR | RapidOCR + pytesseract | EasyOCR, PaddleOCR | RapidOCR 在中文+小字体上速度快 |
| ML | scikit-learn RandomForest | TF/PyTorch | 模型小 (< 5 MB), PyInstaller 不爆炸 |
| 自动化 | pyautogui + keyboard | win32 SendInput | 跨版本兼容 + 反作弊容易做随机化 |
| 打包 | PyInstaller | nuitka | spec 文件成熟, GH Actions 友好 |

### 3. 难点 1：识别"非绑定"物品

游戏里物品分"绑定"和"非绑定"，只有非绑定的能挂到交易行。区分这两类有几个办法：

**方案 A：固定颜色阈值**
绑定物品在 UI 左下角有一个"锁"图标。可以截图判定。
**问题**：新版本游戏 UI 改了图标位置 / 颜色，整个工具失效。

**方案 B：OCR 物品 tooltip**
鼠标悬浮读 "绑定" 二字。
**问题**：每个物品都要悬浮 + OCR，单格 0.5s，108 格 = 54s 不可接受。

**方案 C：ML 二分类（最终方案）**
对每格 cell 提取特征 (HSV 直方图 + LBP 纹理 + 边缘密度)，训练 RandomForest 做二分类。

关键代码：

```python
def extract_cell_features(cell_bgr):
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    edges = cv2.Canny(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
    edge_density = edges.mean()
    return np.concatenate([h_hist, s_hist, v_hist, [edge_density]])
```

**效果**：108 格 + 推理 ≈ 200ms，准确率 95%+。模型只有 5 MB，可以直接打进 exe。

**亮点**：内置「审核标注」UI，识别不准时用户可以**自己加样本一键重训**，不用回头找作者。

### 4. 难点 2：多格大型物品防重复点击

一把裸枪在仓库占 4 列 × 2 行 = 8 个 cell。ML 会把这 8 个 cell 都识别成 "非绑定" → 8 次点击 = 上架同一把枪 8 次（或者上架失败 8 次）。

**解决**：把 ML 输出的 cell 列表跑一遍连通域分析 (BFS 4-邻接)，相邻格合并成 group，每个 group 只点最左上角 cell。

```python
def merge_into_groups(items):
    cells = {(it["row"], it["col"]): it for it in items}
    visited = set()
    groups = []
    for pos in sorted(cells.keys()):
        if pos in visited:
            continue
        group = []
        queue = [pos]
        while queue:
            cur = queue.pop()
            if cur in visited or cur not in cells:
                continue
            visited.add(cur)
            group.append(cur)
            r, c = cur
            queue.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
        rep = min(group)
        item = cells[rep].copy()
        item["_group"] = group
        groups.append(item)
    return groups
```

代表 cell 取 `(row, col)` 字典序最小，自然保证"从左到右、从上到下"的处理顺序。

### 5. 难点 3：跨页去重的三层机制

翻页后，"已经被筛选过滤掉的道具"还会被 ML 重新检测到。如何防止重复处理？

**第一层：位置黑名单** `skip_positions`
当前页内屏蔽 (row, col)。翻页清空。

**问题**：翻页后位置含义变了，无法跨页继承。

**第二层：图像指纹** `skip_fingerprints`
对每个 cell 做 8×8 dHash 取 8 字节指纹。哪怕翻页后位置变了，**图标本身的指纹不变**。

```python
def cell_fingerprint(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]  # 9x8 - 8x8 - 64 bits
    return np.packbits(diff.flatten()).tobytes()  # 8 bytes
```

**问题**：堆叠物品翻页前后图标可能不同 (stack 数量变了)。

**第三层：物品名字** `skip_names`
点开物品后 OCR 读到名字，确认不符筛选则加入。最强兜底。

**三层合力**：位置 + 指纹 + 名字，任何一层失效都有下一层接住。

### 6. 难点 4：可视化坐标校准

不同用户的分辨率 / DPI / 游戏窗口位置不一样，坐标会偏移。传统做法："改代码 config 里的数字" — 用户体验地狱。

**最终方案**：30+ 个坐标点全在 GUI 里可视化校准。流程：
1. 用户按 Ctrl+Z
2. 软件自动最小化 + 截屏
3. 弹出截图窗口 + 下拉框 ("交易行 tab" / "整理仓库" / ...)
4. 用户点击图上对应位置 (单点) 或拖矩形 (区域)
5. 坐标自动回填到对应输入框
6. 永久保存到 `user_config.json`

**核心实现**: `QGraphicsView` + 鼠标事件 + 坐标系转换。

### 7. 性能优化

最初版本：单件上架 ~2.5s。最新版本：单件上架 ~1.0s。

**关键优化**：

| 改动 | 收益 |
|------|------|
| OCR 异步化 (上架等待期间并行跑) | -0.7s/件 |
| 跳转判定从"等 0.45s 再检测"改成"密集轮询从 0.18s 起" | -0.2s/件 |
| `save_debug_screenshot` 异步写盘 | -0.03s/件 |
| 单件等待序列全面 review (8 处) | -0.2s/件 |

**反例**：把"跳转判定首轮 wait"压到 0.18s 后，`has_page_changed` 在弹窗刚渐入就判定 True，导致后续 `maximize_quantity` 点击落空。教训：**性能优化不能只看 happy path，要看动画/异步状态机的边界情况**。

### 8. 工程化：GitHub Actions 自动构建

每次推 `vX.Y.Z` 标签自动：
1. 跑 PyInstaller 打包
2. 压缩 zip
3. 上传 Release Asset

完整 yml 在 `.github/workflows/build.yml`。关键是用 `softprops/action-gh-release@v2` 一行实现 release notes 自动生成 + 文件上传。

### 9. 开源地址 + 求 Star

GitHub: https://github.com/xinshuowl/DeltaForce-Trader

如果对你有帮助，麻烦给个 ⭐ Star，这是开源作者最大的动力。

---

## V2EX 投稿建议

V2EX 不喜欢 "求 star" 太直接的文章，标题改成 **"分享：我做了一个三角洲行动的交易行机器人 (开源)"**，节点选 `分享创造`，正文删掉"求 Star"那段，正文末尾加："欢迎讨论"。

V2EX 帖子寿命短，**上首页前 4 小时**是黄金期，发帖后回复每条评论保持帖子热度。

## 掘金投稿建议

掘金更欢迎技术深度，**侧重写 ML / OCR 部分的实现细节**，多贴代码。

文章末尾的"开源地址 + Star"放 GitHub 链接就行，掘金对此宽容。

发布到分类：**人工智能** 或 **Python**。

## 通用 SEO 关键词

文章正文里自然出现这些词，搜索引擎更容易抓取：

- 三角洲行动 自动上架
- Delta Force 交易行 脚本
- Python 游戏自动化
- ML 物品识别
- OpenCV 游戏脚本
- PyQt5 实战项目
