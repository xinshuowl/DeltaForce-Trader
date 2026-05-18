# 贡献指南

感谢愿意为 DeltaForce-Trader 出一份力！这里写了几条最关键的协作约定，节省你和我的来回时间。

## 提交问题前

1. **先搜 [现有 Issues](https://github.com/xinshuowl/DeltaForce-Trader/issues?q=is%3Aissue)** — 90% 的"我遇到了这个错"已经有答案
2. **看 README 的"常见问题"表** — 90% 的"我跑不起来"在那张表里
3. **看主程序的「使用说明」选项卡** — 有完整图文教程
4. 仍未解决再开 Issue / Discussion

## 提交 Issue 的最低门槛

- **附带版本号**（程序标题栏 / `git log -1`）
- **附带运行日志摘录**（`logs/autoshop_*.log` 末 200 行就够）
- **写复现步骤**（每行不超 1 个操作）

> 没有日志 / 不知道版本的 Issue 通常会被打回。不是嫌弃，是真的查不动。

## 提交 PR

### 风格约定

- **缩进**：4 空格（项目统一 Python 风格）
- **命名**：
  - 文件 / 模块：`snake_case.py`
  - 类：`CamelCase`
  - 函数 / 变量：`snake_case`
  - 私有方法 / 字段：`_underscore_prefix`
- **注释语言**：中文为主（项目目标用户群是中文玩家）
- **避免**：在文件顶部加无意义 docstring；过度抽象的工具函数

### 流程

```powershell
# 1. fork + clone
git clone https://github.com/<你的用户名>/DeltaForce-Trader
cd DeltaForce-Trader

# 2. 装依赖
pip install -r requirements.txt

# 3. 起新分支
git switch -c fix/some-bug

# 4. 改代码 + 自检
python -c "import ast; ast.parse(open('core/workflow.py', encoding='utf-8').read())"
python -c "from core.workflow import WorkflowEngine"

# 5. 在游戏内**实测**一遍 Ctrl+1 完整流程  (重要)
#    PR 描述里说明跑了多少件、有没有崩

# 6. 提交 + 推上去
git commit -F message.txt
git push -u origin fix/some-bug
gh pr create
```

### 改动范围建议

- ✅ **Bug 修复 / 性能优化** — 永远欢迎，直接来
- ✅ **支持新的分辨率** — 加 `config.py` 里的预设，附校准说明
- ✅ **文档 / 注释 / FAQ** — 永远缺
- ⚠️ **新增对话框 / 大模块** — 先在 [Discussions](https://github.com/xinshuowl/DeltaForce-Trader/discussions) 聊清楚定位，避免做了之后被拒
- ❌ **改默认坐标值** — 除非你给出的坐标在更多用户那里也对，否则不合并（默认值是当前主要用户群验证过的）
- ❌ **引入大型新依赖** — 比如 TensorFlow / PyTorch，本项目要 PyInstaller 打包，依赖膨胀会让 exe 体积爆炸

### Code Review 标准

我会优先看：

1. **改动有没有副作用** — 不能因为修一个 bug 引入另一个
2. **改动有没有在游戏里实测** — 没有"我看代码觉得对了所以提 PR"的待遇
3. **改动有没有解释清楚** — commit message 写清楚 "为什么" 比 "改了什么" 重要

## 开发提示

### 调试

按 **Ctrl+4** 触发一次诊断截图，会输出：
- `debug_4_annotated.png` — 道具检测结果可视化
- `debug_2_v_heatmap.png` — V 通道亮度热力图
- `debug_listing_count.png` — 数量 OCR 裁切区

### 关键文件

| 文件 | 改动频率 | 改动注意 |
|------|---------|---------|
| `core/workflow.py` | 高 | 上架核心流程，改前看现有四层 skip 机制 |
| `core/automation.py` | 中 | 鼠标键盘操作，改 wait 时长要考虑反作弊 |
| `core/detector.py` | 中 | OCR + 网格识别，配合 debug_ 图调试 |
| `core/ml_detector.py` | 低 | ML 检测器，重训模型用 review_dialog |
| `gui/main_window.py` | 中 | 主界面，PyQt5 stylesheet 注意 selector 特异性 |
| `config.py` | 低 | 坐标/时间常量，改默认值要慎重 |

### 提交规范

参考已有 commit：

```
perf+fix: 单件提速 ~0.4s + 修复滑动条点击落空
fix(gui): 暴露 CELL_GRID 坐标到 GUI
perf: 减少漏点 / 误翻页 — 重叠滚动 + 翻页前二次扫描验证
```

格式：`<类型>[(范围)]: <一句话总结>`，正文写"为什么"。

类型：`feat` / `fix` / `perf` / `refactor` / `docs` / `style` / `test` / `ci` / `chore`

---

再次感谢！哪怕只是修一个错别字，也是对项目的支持 ❤️
