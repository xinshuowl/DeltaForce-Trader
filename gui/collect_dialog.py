"""
道具数据采集标注对话框 — 实时截取游戏画面或从 Shop/ 文件夹加载截图,
OCR 提取道具信息, 用户手动标注品质 (颜色), 保存到 item_database.json.
"""
import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab

# Win32 API for robust mouse wheel scrolling in games.
# PyAutoGUI 的 scroll 在部分游戏窗口/高 DPI 下不可靠, 改用 mouse_event 直接发送.
try:
    import ctypes
    _user32 = ctypes.windll.user32
    # 开启进程 DPI 感知, 确保 SetCursorPos 坐标与 ImageGrab 物理像素一致.
    # (最好在程序启动时调用; 这里是兜底, 已启用过会直接忽略错误)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Win8.1+ per-monitor
    except Exception:
        try:
            _user32.SetProcessDPIAware()  # Win7+ process-wide
        except Exception:
            pass
except Exception:  # pragma: no cover
    _user32 = None

_MOUSEEVENTF_WHEEL = 0x0800
_WHEEL_DELTA = 120


def _move_cursor(x: int, y: int) -> bool:
    """用 Win32 SetCursorPos 移动鼠标 (物理像素, 与 ImageGrab 坐标系一致)."""
    if _user32 is None:
        return False
    try:
        _user32.SetCursorPos(int(x), int(y))
        return True
    except Exception:
        return False


# 翻页循环里的鼠标 "安全泊车点": 屏幕左上角, 远离道具网格
# 目的: 避免鼠标停在道具上触发悬浮详情弹框, 遮挡截图里的道具名/价格
_PARK_POS = (10, 10)

# OCR 常见字母→数字误读表 (仅用于 "型号数字" 对比, 不影响最终展示名).
# 挑选原则: 字母形似数字 (O/D/Q/U 看起来像 0; I/l 像 1; S 像 5; B 像 8 等).
# 不包含 T→7 (字形差异实际较大, 加进来反而容易假阳性).
_OCR_LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
    "l": "1", "I": "1", "i": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
})


def _model_digits(name: str) -> str:
    """从道具名开头提取 "型号数字" 序列 (把 OCR 常见字母误读归一化后抽数字).

    例:
      'H07战术头盔' → 前缀 'H07' → 归一化 'H07' → 数字 '07'
      'HU7战木头签' → 前缀 'HU7' → 归一化 'H07' → 数字 '07'  (U→0)
      'DAS防弹头盔' → 前缀 'DAS' → 归一化 '0A5' → 数字 '05'  (D→0, S→5)
      'GN重型头盔' → 前缀 'GN'   → 归一化 'GN'   → 数字 ''    (无数字)
      'H01战术头盔' → 前缀 'H01' → 归一化 'H01' → 数字 '01'

    用于快速判别 "字母前缀相近但模型号数字不同" 的不同道具 (如 H01 vs H07).
    """
    m = re.match(r"^[A-Za-z0-9\-]+", name)
    if not m:
        return ""
    prefix = m.group(0)
    normalized = prefix.translate(_OCR_LETTER_TO_DIGIT)
    return "".join(c for c in normalized if c.isdigit())


def _is_same_item(a: str, b: str) -> bool:
    """判断两个 OCR 得到的道具名是否是同一道具 (容忍常见 OCR 字符误读).

    用于翻页时相邻页重合行的去重. 典型要合并的情况:
        'DAS防弹头盔' vs 'UAS防弹头签'  (D↔U, 盔↔签)
        'H07战术头盔' vs 'HU7战木头签'  (0↔U, 术↔木, 盔↔签)
        '防暴头盔'    vs '防暴头盗'      (盔↔盗)

    但要避免把同类不同型号的真实道具误合, 比如:
        'MHS战术头盔' vs 'H07战术头盔'  (都是战术头盔, 型号不同)
        'MC防弹头盔'  vs 'MC201防弹头盔' (后者多一个 '201' 型号后缀)
        'H01战术头盔' vs 'H07战术头盔' vs 'H09防暴头盔'  (同系列不同型号数字)
        '军用信息终端' vs '军用控制终端'  (同前缀+同后缀, 中间词完全不同 → 不同道具)

    规则 (依次检查, 任一不满足即视为不同道具):
      1. 完全相同 → 合并
      2. 长度差 > 1 → 不合并 (OCR 通常不会丢/加多个字)
      3. 太短 (< 2 字) → 不合并 (误判风险高)
      4. 型号数字检查: 两边前缀都含数字且数字序列不同 → 不合并
         (例如 H01→'01' vs H09→'09', 0 vs 9 不同字符, 是不同型号;
          而 H07→'07' vs HU7→'07' 归一化后相同, 可通过)
      5. "模板式不同道具" 检查 (新增): 如果 SequenceMatcher 只找到两段匹配块
         分别在头部和尾部, 中间两段 **非匹配** 区域都 >= 2 字符 (典型模式:
         '军用信息终端' vs '军用控制终端', 匹配 '军用' + '终端', 中间 '信息'
         vs '控制' 都是 2 字完全不同) → 不合并, 这是不同道具的标配模式.
         关键区别于 OCR 误读: OCR 误读通常散落多个单字, 产生 3+ 个短匹配块,
         不会形成 "头尾 + 中间空洞" 的干净两段式结构.
      6. 整体相似度 >= 0.55 且前 3 字相似度 >= 0.5 → 合并
    """
    if a == b:
        return True
    if len(a) < 2 or len(b) < 2:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    dig_a = _model_digits(a)
    dig_b = _model_digits(b)
    if dig_a and dig_b and dig_a != dig_b:
        return False

    sm = SequenceMatcher(None, a, b)
    # 规则 5: 模板式不同道具检查
    # SequenceMatcher.get_matching_blocks() 末尾一定带一个 size=0 哨兵块, 需要过滤掉
    blocks = [bl for bl in sm.get_matching_blocks() if bl.size > 0]
    if len(blocks) == 2:
        b0, b1 = blocks
        # 两块恰好覆盖首尾, 中间是 "洞" → 典型的同模板不同条目
        covers_prefix = (b0.a == 0 and b0.b == 0)
        covers_suffix = (b1.a + b1.size == len(a) and
                         b1.b + b1.size == len(b))
        mid_a_len = b1.a - (b0.a + b0.size)
        mid_b_len = b1.b - (b0.b + b0.size)
        if covers_prefix and covers_suffix and mid_a_len >= 2 and mid_b_len >= 2:
            return False

    ratio = sm.ratio()
    if ratio < 0.55:
        return False
    head_len = min(3, min(len(a), len(b)))
    head_ratio = SequenceMatcher(None, a[:head_len], b[:head_len]).ratio()
    return head_ratio >= 0.5


def _send_wheel_at(x: int, y: int, clicks: int) -> bool:
    """在屏幕物理像素 (x, y) 处发送滚轮事件 (clicks 负向下正向上).

    - 先 SetCursorPos 定位 (避开 pyautogui 的 DPI 换算坑)
    - 拆成 1-click 为单位的多次发送, 对游戏的滚动动画更友好
    - 用 mouse_event 而非 SendInput, 兼容更老的 Windows 版本
    - 滚动完成后自动把鼠标挪回 _PARK_POS, 避免悬浮详情弹框影响后续截图
    """
    if _user32 is None:
        return False
    try:
        _user32.SetCursorPos(int(x), int(y))
        time.sleep(0.05)
        step = -1 if clicks < 0 else 1
        for _ in range(abs(clicks)):
            _user32.mouse_event(_MOUSEEVENTF_WHEEL, 0, 0,
                                step * _WHEEL_DELTA, 0)
            time.sleep(0.08)
        # 滚轮事件发送完立即泊车, 让悬浮弹框在后续的 sleep 期间淡出
        _user32.SetCursorPos(_PARK_POS[0], _PARK_POS[1])
        return True
    except Exception:
        return False
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QScrollArea, QWidget, QMessageBox, QProgressBar,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor

from config import SHOP_DIR, ITEM_DB_FILE, ITEM_CATEGORIES, COLLECT_OCR_ROI
from gui.theme import (
    CYBER_DIALOG_STYLE, RARITY_COLORS_HEX, set_btn_style,
    NEON_CYAN, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, NEON_RED,
    BG_DARKEST, BG_WIDGET,
    BORDER_GLOW, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
)
from core.item_database import ItemDatabase, ItemEntry

logger = logging.getLogger("collect_dialog")

RARITIES = ["白色", "绿色", "蓝色", "紫色", "金色", "红色"]

RARITY_COLORS = RARITY_COLORS_HEX


def _get_rapid_ocr():
    from core.detector import _get_rapid_ocr as _get
    return _get()


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """针对游戏深色 UI 的 OCR 预处理.

    组合:
    1. CLAHE 局部对比度增强 (LAB 空间只对 L 通道, 保留颜色)
    2. Unsharp Mask 轻度锐化 (增强字符边缘)
    3. 自适应阈值提取的文字掩膜与原图弱混合 (强化小字轮廓, 深色背景上的白字更清晰)

    相较于纯二值化, 这种 "增强 + 弱叠加" 不会抹掉纹理, 对 OCR 更友好.
    """
    # 1) CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2) Unsharp
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)

    # 3) 自适应阈值掩膜 (针对深色背景白字), 与原图弱混合以强化文字
    try:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=-5,
        )
        # 轻度闭操作让字符笔画连贯
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        # 30% 阈值图 + 70% 增强图: 保留纹理同时强化文字轮廓
        out = cv2.addWeighted(out, 0.7, th_bgr, 0.3, 0)
    except Exception:
        logger.debug("自适应阈值混合失败, 使用 CLAHE+Unsharp 结果", exc_info=True)

    return out


def _ocr_screenshot(img_bgr: np.ndarray,
                    roi: Optional[tuple] = None,
                    upscale_to: int = 2400) -> list[dict]:
    """
    对截图执行 OCR, 返回检测到的文本列表 (坐标为原图坐标系).

    优化:
    1. 先按 roi 裁剪, 减小 OCR 输入, 避免侧栏/顶栏干扰
    2. 裁剪结果若宽度小于 upscale_to, 等比放大后再送入 OCR (深色细字显著更清晰)
    3. 通过 CLAHE + Unsharp 预处理增强对比度
    4. OCR 得到的坐标换算回原图坐标, 方便 ROI 过滤和 debug 叠加

    返回: [{"text": str, "box": [[x,y]×4], "score": float}]
    """
    ocr = _get_rapid_ocr()
    if ocr is None:
        logger.warning("RapidOCR 不可用, 无法执行 OCR")
        return []

    h, w = img_bgr.shape[:2]
    # 1) ROI 裁剪
    if roi is None:
        roi = COLLECT_OCR_ROI
    x1_r, y1_r, x2_r, y2_r = roi
    rx1, ry1 = int(w * x1_r), int(h * y1_r)
    rx2, ry2 = int(w * x2_r), int(h * y2_r)
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(w, rx2), min(h, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        logger.warning("ROI 裁剪结果无效, 使用整图")
        crop = img_bgr
        rx1 = ry1 = 0
    else:
        crop = img_bgr[ry1:ry2, rx1:rx2]

    # 2) 放大小图
    ch, cw = crop.shape[:2]
    scale = 1.0
    if cw < upscale_to:
        scale = upscale_to / max(cw, 1)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3) 预处理
    try:
        crop_pp = _preprocess_for_ocr(crop)
    except Exception:
        logger.exception("OCR 预处理失败, 使用原图")
        crop_pp = crop

    # 4) OCR
    try:
        result = ocr(crop_pp)
    except Exception as e:
        logger.exception(f"RapidOCR 调用异常: {e}")
        return []

    if result is None:
        logger.warning("RapidOCR 返回 None")
        return []

    raw_items: list[tuple] = []
    txts = getattr(result, "txts", None)
    if txts is not None:
        boxes = getattr(result, "boxes", None)
        scores = getattr(result, "scores", None)
        for i, txt in enumerate(txts):
            box = boxes[i] if boxes is not None and i < len(boxes) else None
            score = (scores[i] if scores is not None and i < len(scores) else 0.0)
            raw_items.append((txt, box, score))
    else:
        raw = result
        if isinstance(raw, tuple) and len(raw) >= 1:
            raw = raw[0]
        if isinstance(raw, list):
            for entry in raw:
                if not entry or len(entry) < 2:
                    continue
                raw_items.append((entry[1], entry[0],
                                   entry[2] if len(entry) > 2 else 0.0))

    # 5) 坐标换算回原图
    items: list[dict] = []
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    for txt, box, score in raw_items:
        if box is not None:
            mapped = []
            for p in box:
                px = float(p[0]) * inv_scale + rx1
                py = float(p[1]) * inv_scale + ry1
                mapped.append([px, py])
            box_out = mapped
        else:
            box_out = None
        items.append({"text": txt, "box": box_out, "score": float(score)})
    return items


_TOP_BAR_KEYWORDS = {
    "购买", "出售", "交易记录", "收藏",
    "开始游戏", "仓库", "特战干员", "部门", "交易行", "特勤处", "改枪台",
    "请输入", "进入特勤处",
}

# 道具品相后缀 (不是导航关键词! 是名称的一部分, 需要从名称中剥离后保留核心名)
_CONDITION_SUFFIXES = [
    "（全新）", "(全新)", "（几乎全新）", "(几乎全新)",
    "（破损）", "(破损)", "（破旧）", "(破旧)",
    "全新", "几乎全新", "破损", "破旧",
]

# 独立出现时绝不是道具名的碎片 (OCR 拆行后残留的品相/助词等)
_NEVER_NAME_FRAGMENTS = {
    "几乎", "全新", "几乎全新", "破损", "破旧", "新",
    "(全新)", "（全新）", "(几乎全新)", "（几乎全新）",
    "(破损)", "（破损）", "(破旧)", "（破旧)",
    "的", "之", "与", "—", "·",
}

# 单独出现时是品类总称而非具体道具名 (OCR 前缀丢失后残留的尾部品类词)
# 注意: 只匹配完全相同的名称 "头盔" / "步枪", 不会误伤 "战术头盔" / "H09防暴头盔" 等
_GENERIC_CATEGORY_NAMES = {
    "头盔", "护甲", "背包", "胸挂",
    "步枪", "手枪", "冲锋枪", "射手步枪", "组合步枪",
    "弹药", "子弹", "手榴弹",
    "配件", "耗材", "消耗品",
    "收集品", "电子物品", "能源燃料", "资料情报", "钥匙",
}


def _strip_condition_suffix(name: str) -> str:
    """从道具名里剥离 '(全新)' 等品相后缀, 返回规范化的道具名.

    考虑 OCR 有时只识别出半个品相标签 (比如 "GN久战重型夜视头盔（全新" 结尾的
    右括号丢失, 或 "...头盔（" 连文字都丢只剩左括号), 这里多做一轮孤立
    括号/空白的尾部剥离, 保证结果不含 UI 装饰字符.
    """
    s = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in _CONDITION_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)].rstrip(" 　\t")
                changed = True
                break
    # 去掉结尾孤立的未闭合括号 (OCR 有时把 "(全新)" 裁剩一个括号)
    trailing_trash = ("（", "(", "）", ")")
    changed = True
    while changed and s:
        changed = False
        for tr in trailing_trash:
            if s.endswith(tr):
                s = s[: -len(tr)].rstrip(" 　\t")
                changed = True
                break
    return s.strip()


def _merge_wrapped_names(names: list[dict],
                         img_w: int, img_h: int) -> list[dict]:
    """合并因 OCR 分词被拆开的道具名.

    游戏交易行里的道具名经常被 OCR 拆成多段:
    - 水平拆 (最常见): "H70精英" + "头盔" 或 "Mask-1" + "铁壁" + "头盔".
      英文/数字前缀与中文后缀在字符过渡处被切开, 两段同 y 不同 x.
    - 垂直拆 (少见, 长名称换行): "GN久战重型" + "夜视头盔", 同 x 不同 y.

    本函数按 "同行 → 同列" 两轮贪心合并, 把紧邻的名称残片拼回完整名称.

    阈值 (均基于 2560x1440 头盔页实测):
    - 同行 dy < 1.8% 图高 (约 26px), dx 0 < dx < 11% 图宽 (约 280px).
      卡片内拆分文本距离通常 50-150px, 不同卡片间距 > 500px, 阈值给足一点余量.
    - 同列 dx < 8% 图宽, dy 0 < dy < 4% 图高.
    """
    if not names:
        return names

    row_dy_tol = img_h * 0.018
    row_dx_tol = img_w * 0.11
    col_dx_tol = img_w * 0.08
    col_dy_tol = img_h * 0.04

    # 先按 cy, cx 排序 (从上到下, 从左到右), 保证拼接顺序与视觉一致
    sorted_names = sorted(names, key=lambda n: (n["cy"], n["cx"]))
    used = [False] * len(sorted_names)
    merged: list[dict] = []

    for i, nm in enumerate(sorted_names):
        if used[i]:
            continue
        used[i] = True
        cur_name = nm["name"]
        cur_cx, cur_cy = nm["cx"], nm["cy"]
        # 合并过程中追踪包围所有片段的 x_min/x_max (品质色采样用)
        cur_xmin = nm.get("x_min", nm["cx"])
        cur_xmax = nm.get("x_max", nm["cx"])

        # ─── 水平合并: 贪心往右拼接同行片段 ───
        # 循环: 每轮找到距离当前右端最近的同行未用候选, 直到找不到
        while True:
            best_j = -1
            best_dx = float("inf")
            for j in range(len(sorted_names)):
                if used[j] or j == i:
                    continue
                other = sorted_names[j]
                dy = abs(other["cy"] - cur_cy)
                dx = other["cx"] - cur_cx
                if dy < row_dy_tol and 0 < dx < row_dx_tol and dx < best_dx:
                    best_dx = dx
                    best_j = j
            if best_j < 0:
                break
            other = sorted_names[best_j]
            cur_name = cur_name + other["name"]
            cur_cx = other["cx"]  # 右端向右推进, 便于再找更靠右的片段
            cur_xmin = min(cur_xmin, other.get("x_min", other["cx"]))
            cur_xmax = max(cur_xmax, other.get("x_max", other["cx"]))
            used[best_j] = True

        # ─── 垂直合并: 往下找同列紧邻的一条 (长名称换行) ───
        best_j = -1
        best_dy = float("inf")
        for j in range(len(sorted_names)):
            if used[j] or j == i:
                continue
            other = sorted_names[j]
            dx = abs(other["cx"] - nm["cx"])
            dy = other["cy"] - cur_cy
            if dx < col_dx_tol and 0 < dy < col_dy_tol and dy < best_dy:
                best_dy = dy
                best_j = j
        if best_j >= 0:
            other = sorted_names[best_j]
            cur_name = cur_name + other["name"]
            cur_xmin = min(cur_xmin, other.get("x_min", other["cx"]))
            cur_xmax = max(cur_xmax, other.get("x_max", other["cx"]))
            used[best_j] = True

        merged.append({
            "name": cur_name, "cx": nm["cx"], "cy": nm["cy"],
            "x_min": cur_xmin, "x_max": cur_xmax,
        })

    return merged


def _parse_items_from_ocr(ocr_results: list[dict],
                          img_w: int, img_h: int,
                          stats: Optional[dict] = None,
                          roi: Optional[tuple] = None) -> list[dict]:
    """
    从 OCR 结果中解析道具卡片 (名称 + 价格).

    策略:
    - 仅识别 ROI (道具网格红框区域) 内的文本, 忽略侧栏/顶部导航/底部状态
    - 价格: 纯数字 (去逗号后 3 位以上)
    - 名称: 包含中文或英文字母, 至少 2 个字符, 非纯数字
    - 按列和垂直位置配对名称与价格

    roi: 归一化比例 (x1, y1, x2, y2), 默认使用 config.COLLECT_OCR_ROI.
    stats: 若提供则填入诊断信息 (raw/filtered_region/filtered_keyword/names/prices).
    """
    if roi is None:
        roi = COLLECT_OCR_ROI

    names = []
    prices = []

    x1_r, y1_r, x2_r, y2_r = roi
    roi_x1 = img_w * x1_r
    roi_y1 = img_h * y1_r
    roi_x2 = img_w * x2_r
    roi_y2 = img_h * y2_r

    stat_raw = 0
    stat_filt_region = 0
    stat_filt_keyword = 0
    stat_filt_short = 0
    stat_filt_lowconf = 0

    # 低于此置信度的 OCR 文本直接丢弃 (大多是深色背景噪点/乱码).
    # 阈值太高会丢掉英文+中文混排名称的前缀 (如 "H70精英" / "Mask-1" 在暗背景下置信度常在 0.3-0.5),
    # 靠后续的名称规则 + 合并逻辑把残片拼回完整名称.
    _MIN_SCORE = 0.30

    for item in ocr_results:
        text = item["text"].strip()
        if not text:
            continue
        stat_raw += 1

        score = float(item.get("score", 0.0))
        if score > 0 and score < _MIN_SCORE:
            stat_filt_lowconf += 1
            continue

        box = item.get("box")
        if box is not None:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            x_min = float(min(xs))
            x_max = float(max(xs))
            # 仅识别红框 ROI 内的文字
            if cx < roi_x1 or cx > roi_x2 or cy < roi_y1 or cy > roi_y2:
                stat_filt_region += 1
                continue
        else:
            # 无 box 信息时保留 (兼容老版 OCR API), 不过也不做区域过滤
            cx, cy = 0.0, 0.0
            x_min = x_max = 0.0

        if any(kw in text for kw in _TOP_BAR_KEYWORDS):
            stat_filt_keyword += 1
            continue

        # ─── 价格识别 (宽松) ───
        # 先做字符纠正 (常见 OCR 混淆: l→1, O→0, I→1, B→8 等, 仅当原文主要是数字时)
        cleaned = (
            text.replace(",", "")
                .replace("，", "")
                .replace(" ", "")
                .replace(".", "")
        )
        digit_count = sum(c.isdigit() for c in cleaned)
        total_count = len(cleaned)
        # 数字占比 >= 70% 且至少 3 位: 视为价格 (纠正常见误读字符)
        if total_count >= 3 and digit_count / total_count >= 0.7:
            fixed = (
                cleaned.replace("l", "1").replace("I", "1").replace("i", "1")
                       .replace("O", "0").replace("o", "0")
                       .replace("B", "8").replace("S", "5")
                       .replace("Z", "2").replace("z", "2")
                       .replace("n", "").replace("a", "")
            )
            if re.fullmatch(r"\d{3,}", fixed):
                prices.append({
                    "price": int(fixed),
                    "cx": cx, "cy": cy, "text": text,
                })
                continue

        # ─── 名称识别 (严格) ───
        # 先剥离 "(全新)" 等品相后缀, 保留道具的规范名称
        stripped = _strip_condition_suffix(text)
        if not stripped or stripped in _NEVER_NAME_FRAGMENTS:
            # 丢弃空串 / 独立的品相碎片 ("几乎" / "全新" 等跨行拆分残留)
            stat_filt_short += 1
            continue

        has_letter = bool(re.search(r"[A-Za-z]", stripped))
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", stripped))
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        letter_count = len(re.findall(r"[A-Za-z]", stripped))
        non_digit_count = chinese_count + letter_count

        # 名称必须: 含中文或字母 + 长度≥2 + 非数字字符占比>50%
        # 且非数字字符至少 2 个 (避免把 "a1234" 这种误分类的价格当作名称)
        if (
            (has_chinese or has_letter)
            and len(stripped) >= 2
            and non_digit_count >= 2
            and non_digit_count / len(stripped) > 0.5
        ):
            names.append({
                "name": stripped,
                "cx": cx, "cy": cy,
                "x_min": x_min, "x_max": x_max,
            })
        else:
            stat_filt_short += 1

    # 合并前再过滤一次独立品相碎片 (名称里可能含空格被 strip 后仍是 "几乎" 等)
    names = [n for n in names if n["name"] not in _NEVER_NAME_FRAGMENTS]

    # 合并因换行被拆成多行的名称 (如 "GN重型夜视" + "头盔")
    names = _merge_wrapped_names(names, img_w, img_h)

    # 合并后再剥离一次品相后缀 (跨行合并后才会出现的 "...头盔全新" 等)
    cleaned_names = []
    for nm in names:
        canon = _strip_condition_suffix(nm["name"])
        if not canon or canon in _NEVER_NAME_FRAGMENTS:
            continue
        cleaned_names.append({
            "name": canon, "cx": nm["cx"], "cy": nm["cy"],
            "x_min": nm.get("x_min", nm["cx"]),
            "x_max": nm.get("x_max", nm["cx"]),
        })
    names = cleaned_names

    # ─── 名称-价格配对 ───
    # 卡片内布局: 名称左上, 价格右下. 价格 cx 永远 > 名称 cx.
    # 若不加方向约束, 当前卡片的名称可能与左侧相邻卡片的价格距离更近 (误配对),
    # 所以要求 price_cx - name_cx 在 [min_dx_signed, max_dx_signed] 范围内 (价格必须在名称右侧).
    min_dx_signed = img_w * 0.03   # 价格至少比名称靠右 ~77px @2560
    max_dx_signed = img_w * 0.22   # 最远 ~563px @2560 (约一个卡片宽度内)
    max_dy = img_h * 0.20          # 价格离名称最多下移 ~288px @1440 (一个卡片高度内)

    # 底部截断: ROI 底部 15% 区域内的道具视为 "卡片底部被截断", 跳过
    # 这样游戏 HUD 遮挡的最下一行道具 (价格不全/图标不全) 不会被错误采集
    bottom_cutoff_y = roi_y1 + (roi_y2 - roi_y1) * 0.85

    parsed = []
    used_prices = set()
    stat_filt_cutoff = 0
    stat_filt_generic = 0
    stat_filt_noprice = 0

    for nm in names:
        # 1) 跳过底部截断区的道具卡 (价格很可能不全)
        if nm["cy"] > bottom_cutoff_y:
            stat_filt_cutoff += 1
            continue

        # 2) 跳过纯品类总称 (OCR 前缀丢失后残留的 "头盔" / "步枪" 等)
        if nm["name"] in _GENERIC_CATEGORY_NAMES:
            stat_filt_generic += 1
            continue

        # 3) 寻找右下方的价格 (同一张卡片内)
        best_price = None
        best_dist = float("inf")
        for pi, pr in enumerate(prices):
            if pi in used_prices:
                continue
            dx_signed = pr["cx"] - nm["cx"]
            if dx_signed < min_dx_signed or dx_signed > max_dx_signed:
                continue
            dy = pr["cy"] - nm["cy"]
            if 0 < dy < max_dy:
                # 同卡片内 dy (约 210px) 比跨卡片 dy (约 0-30px, 同行邻卡) 大很多;
                # 权重 dy * 0.4 让垂直距离合理而不是一味偏小 dy (否则会选到左邻卡的价格)
                dist = dx_signed + dy * 0.4
                if dist < best_dist:
                    best_dist = dist
                    best_price = pi

        if best_price is None:
            # 4) 没有有效价格配对 → 直接放弃 (用户要求: 价格不全就不采集)
            stat_filt_noprice += 1
            continue

        price_val = prices[best_price]["price"]
        # 记下价格框的 cx/cy, 下游恢复孤儿价格时要用它们来估算相对偏移
        used_prices.add(best_price)
        parsed.append({
            "name": nm["name"],
            "price": price_val,
            "cx": nm["cx"],
            "cy": nm["cy"],
            "price_cx": prices[best_price]["cx"],
            "price_cy": prices[best_price]["cy"],
            "x_min": nm.get("x_min", nm["cx"]),
            "x_max": nm.get("x_max", nm["cx"]),
        })

    # 未被配对的价格 = "孤儿价格" (OCR 漏检了对应的短名称,
    # 常见于 3 字母品牌/简写如 "CPU", 上游可做二次 OCR 尝试恢复)
    orphan_prices = [
        {"price": pr["price"], "cx": pr["cx"], "cy": pr["cy"]}
        for pi, pr in enumerate(prices)
        if pi not in used_prices
    ]

    # 按网格顺序 (从上到下, 从左到右) 排序输出.
    # 同一视觉行的道具 cy 差异较小 (OCR 框中心偏差 < 30px),
    # 将 cy 粗量化到行 (以 img_h * 0.05 = 72px 为行距) 再按 cx 升序
    row_quant = img_h * 0.05
    parsed.sort(key=lambda p: (int(p["cy"] / row_quant), p["cx"]))
    # 保留 cx/cy 返回, 供下游品质色采样 (_detect_rarity_near_name) 使用

    if stats is not None:
        stats.update({
            "raw": stat_raw,
            "filtered_region": stat_filt_region,
            "filtered_keyword": stat_filt_keyword,
            "filtered_short": stat_filt_short,
            "filtered_lowconf": stat_filt_lowconf,
            "filtered_cutoff": stat_filt_cutoff,
            "filtered_generic": stat_filt_generic,
            "filtered_noprice": stat_filt_noprice,
            "names": len(names),
            "prices": len(prices),
            "parsed": len(parsed),
            "orphan_prices": orphan_prices,
        })

    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# 方案 C: 按网格切分, 逐卡片独立 OCR + 品质颜色识别
# ═══════════════════════════════════════════════════════════════════════════
#
# 原理:
#   1. 交易行 [购买] 页的道具网格布局固定: "装备" 大类每页 3 列×4 行, 其它大类 3 列×5 行
#   2. 按固定比例把 ROI 切成单元格, 每张卡片单独处理
#   3. 只对 "左上角标题条" 做 OCR → 避开道具图标/价格/(全新) 标签等干扰
#   4. 只对 "右下角价格条" 做 OCR → 避开名称串扰
#   5. 采样标题条背景 HSV → 分类到 6 种品质颜色 (红/金/紫/蓝/绿/白)
#
# 相较整图 OCR + 后处理, 按卡片切片的好处:
#   - 每次 OCR 只看很小的干净区域, 对小字识别率显著提升
#   - 名称/价格天然不会串卡, 省掉复杂的配对逻辑
#   - 品质颜色可以从像素直接读出, 不再依赖用户手动标注

# 不同大类的网格布局 (rows, cols) 和裁剪 ROI (x1, y1, x2, y2, 归一化比例).
# ROI 需要覆盖该大类完整的 N 行网格, 不包含底部被 HUD 截断的额外行.
#
# 归一化比例基于参考分辨率 2560×1440 实测像素坐标换算 (其它分辨率按比例自动缩放):
#   装备: 像素 (576, 278)→(2435, 1111), 4 行 × 3 列
#   其他: 像素 (576, 205)→(2437, 1250), 5 行 × 3 列
CATEGORY_LAYOUT: dict[str, dict] = {
    "装备": {"rows": 4, "cols": 3, "roi": (0.225, 0.193, 0.951, 0.772)},
}
_DEFAULT_LAYOUT = {"rows": 5, "cols": 3, "roi": (0.225, 0.142, 0.952, 0.868)}

# 翻页模式下的 "画面是否发生变化" 检测用 ROI (只用于前后两帧视觉差分, 不用于 OCR).
# 用户指定的像素范围 (580,280)-(2439,1309) @ 2560x1440, 作为稳定的差分采样区.
_SCROLL_DIFF_ROI = (580.0 / 2560, 280.0 / 1440, 2439.0 / 2560, 1309.0 / 1440)

# 翻页模式 "扫描 OCR" 用的外包围 ROI (比任何单一分类的网格 ROI 都更宽高).
# 为什么翻页不用固定网格: 滚动后游戏不是整行对齐显示, 顶部和底部常常半截卡,
# 用固定 3xN 网格切分会把道具切错位, 把 A 的标题和 B 的价格配到一起.
# 改成: 在这个大 ROI 内跑整图 OCR, 再按空间距离把标题和其右下方价格配对,
# 只有 "标题 + 价格" 都完整的卡片才被采纳, 半截卡自然被丢弃.
#
# 比例范围覆盖: x 22%~95.5% (剔除左侧分类栏/右侧留白),
#              y 13%~94%   (剔除顶部 Tab 栏/底部状态栏, 留足空间给顶行半截价格
#                          和底行半截标题, 反正配对失败时它们会被自动丢弃).
_SCAN_ROI = (0.22, 0.13, 0.955, 0.94)

# 品质颜色 HSV 范围 (基于真实 2560×1440 游戏截图采样实测).
# 每个范围形如 ((H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi)); 红色跨越 H=0/180 故有两段.
#
# 实测名称标签背景典型值 (S>40, V>55):
#   红色: H=170-180, S=70-95, V=70-85 (偶尔 H=0-8 的另一段)
#   金色: H=13-25,   S=60-100, V=70-120
#   紫色: H=125-155, S=40-80,  V=55-90
#   蓝色: H=95-120,  S=40-80,  V=55-90  (注意和卡片正文 H=100-105 撞色, 靠 V 过滤)
#   绿色: H=40-85,   S=40-80,  V=55-90
#
# 关键点: 卡片正文 (item 图片背景) 是 H=100-105 S=85 V=35-45 的深青,
# 阈值里 V_lo 一律用 55 切, 可把这部分底噪完全排除, 不会误判成蓝色.
_RARITY_HSV: dict[str, list[tuple]] = {
    "红色": [((0,   60, 55), (12,  255, 255)),
            ((160, 60, 55), (180, 255, 255))],
    "金色": [((13,  60, 60), (32,  255, 255))],
    "绿色": [((35,  40, 55), (85,  255, 255))],
    "蓝色": [((86,  40, 55), (120, 255, 255))],
    "紫色": [((121, 35, 55), (159, 255, 255))],
}


def _extract_ocr_texts(result) -> list[dict]:
    """把 RapidOCR 多版本返回结果统一成 [{text, cx, cy, x_min, x_max, score}] 列表.

    x_min/x_max 是该段文字框的水平像素范围, 供 "重叠检测" 判断两段是否检测到
    同一块文字 (RapidOCR 偶尔会返回一个短框 + 一个长框都盖在同一数字上).
    """
    items: list[dict] = []
    if result is None:
        return items

    def _pack(txt, box, score):
        cx = cy = 0.0
        x_min = x_max = 0.0
        if box is not None and len(box) > 0:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            x_min = float(min(xs))
            x_max = float(max(xs))
        items.append({
            "text": str(txt).strip(),
            "cx": cx, "cy": cy,
            "x_min": x_min, "x_max": x_max,
            "score": float(score),
        })

    # v3+ 对象风格
    txts = getattr(result, "txts", None)
    if txts is not None:
        boxes = getattr(result, "boxes", None)
        scores = getattr(result, "scores", None)
        for i, txt in enumerate(txts):
            box = boxes[i] if boxes is not None and i < len(boxes) else None
            score = scores[i] if scores is not None and i < len(scores) else 0.0
            _pack(txt, box, score)
        return items
    # 旧版 tuple/list 风格: [(box, text, score), ...] 或 [[(box, text, score), ...]]
    raw = result
    if isinstance(raw, tuple) and len(raw) >= 1:
        raw = raw[0]
    if isinstance(raw, list):
        for entry in raw:
            if not entry or len(entry) < 2:
                continue
            box = entry[0]
            second = entry[1]
            if isinstance(second, (list, tuple)) and len(second) >= 1:
                txt = second[0]
                score = second[1] if len(second) > 1 else 0.0
            else:
                txt = second
                score = entry[2] if len(entry) > 2 else 0.0
            _pack(txt, box, score)
    return items


def _detect_rarity(strip_bgr: np.ndarray) -> str:
    """根据名称标签背景色判定品质 (红/金/紫/蓝/绿/白).

    关键阈值 (基于 2560×1440 真实截图采样):
    - V > 55: 排除卡片正文深青底 (V=35-45), 那部分虽然 S 高 (=85) 但 V 低,
             用 V_lo=55 一刀切干净. 不加这条的话卡片正文的 H=100-105 会被
             误判为 "蓝色".
    - V < 230: 排除标题文字 (白字 V 接近 255)
    - S > 30:  排除灰底白标签

    流程:
    1. 转 HSV, 以上述条件过滤出 "纯品质色像素"
    2. 对每种品质的 HSV 范围统计命中像素数
    3. 命中最多的即为品质; 若彩色像素占比过低则回落 "白色" (灰品质/无标签)
    """
    if strip_bgr is None or strip_bgr.size == 0:
        return "白色"
    try:
        hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return "白色"
    h_all = hsv[..., 0]
    s_all = hsv[..., 1]
    v_all = hsv[..., 2]
    # 卡片正文在 y=240 以下是深青 V≈37, V>55 过滤掉; 但靠近道具图标的地方
    # 正文会被提亮到 V=60-85 (仍保持 H=97-108 深青调+高饱和 S>70).
    # 把这段 "亮化的卡片正文" 也扣掉, 避免它占多数票把标签的品质色盖掉.
    mask_bg = (v_all > 55) & (v_all < 230) & (s_all > 30)
    body_glow = ((h_all >= 96) & (h_all <= 112) &
                 (v_all < 90) & (s_all > 70))
    mask_bg = mask_bg & ~body_glow
    n = int(mask_bg.sum())
    if n < 12:
        return "白色"
    h_vals = h_all[mask_bg]
    s_vals = s_all[mask_bg]
    v_vals = v_all[mask_bg]
    counts: dict[str, int] = {}
    for name, ranges in _RARITY_HSV.items():
        total = 0
        for (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi) in ranges:
            m = ((h_vals >= h_lo) & (h_vals <= h_hi) &
                 (s_vals >= s_lo) & (s_vals <= s_hi) &
                 (v_vals >= v_lo) & (v_vals <= v_hi))
            total += int(np.count_nonzero(m))
        counts[name] = total
    best_name = max(counts.keys(), key=lambda k: counts[k])
    # 命中像素 < 有效像素 20% 时, 视为没有任何品质色主导 → 白色
    if counts[best_name] < max(12, int(n * 0.20)):
        return "白色"
    return best_name


def _ocr_cell_title(title_bgr: np.ndarray, ocr,
                    debug_info: Optional[dict] = None) -> tuple[str, float]:
    """对单卡片的标题条做 OCR, 返回 (清理后的道具名, 置信度均值).

    清理步骤:
    - 剥离 "(全新)"/"(几乎全新)"/"(破损)" 等品相后缀
    - 按 cx 升序拼接同行多段 (OCR 会把 "H70精英" + "头盔" 拆成两段)
    - 过滤纯品类残片 ("头盔" / "步枪" 等独立出现的总称)

    debug_info 若提供, 会写入: raw_texts(原始 OCR 段), merged(合并后), drop_reason.
    """
    if title_bgr is None or title_bgr.size == 0:
        if debug_info is not None:
            debug_info["drop_reason"] = "empty crop"
        return "", 0.0
    h, w = title_bgr.shape[:2]
    # 放大 3x, 让小字在 OCR 模型里占更多像素
    big = cv2.resize(title_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    try:
        pp = _preprocess_for_ocr(big)
    except Exception:
        pp = big
    try:
        result = ocr(pp)
    except Exception:
        logger.debug("卡片标题 OCR 失败", exc_info=True)
        if debug_info is not None:
            debug_info["drop_reason"] = "ocr exception"
        return "", 0.0
    texts = _extract_ocr_texts(result)
    if debug_info is not None:
        debug_info["raw_texts"] = [
            {"text": t["text"], "score": round(t["score"], 2)} for t in texts
        ]
    if not texts:
        if debug_info is not None:
            debug_info["drop_reason"] = "no OCR output"
        return "", 0.0
    # 丢弃极低置信度 (< 0.10) 的片段; 阈值放低让 "GN" / "H09" 这种英文短前缀也能保留
    texts = [t for t in texts if t["text"] and t["score"] >= 0.10]
    if not texts:
        if debug_info is not None:
            debug_info["drop_reason"] = "all texts < 0.10 confidence"
        return "", 0.0
    # 按 cy 分行 (行容差 = big height 的 25%), 行内再按 cx 升序拼接;
    # 若标题跨两行 (例如 GN久战重型 / 夜视头盔), 行顺序按 cy 升序
    texts.sort(key=lambda t: (round(t["cy"] / max(1, big.shape[0] * 0.25)), t["cx"]))
    merged = "".join(t["text"] for t in texts)
    cleaned = _strip_condition_suffix(merged)
    if debug_info is not None:
        debug_info["merged"] = merged
        debug_info["cleaned"] = cleaned
    if not cleaned:
        if debug_info is not None:
            debug_info["drop_reason"] = "empty after cleaning"
        return "", 0.0
    if cleaned in _NEVER_NAME_FRAGMENTS:
        if debug_info is not None:
            debug_info["drop_reason"] = f"in NEVER_NAME_FRAGMENTS"
        return "", 0.0
    if cleaned in _GENERIC_CATEGORY_NAMES:
        if debug_info is not None:
            debug_info["drop_reason"] = f"in GENERIC_CATEGORY_NAMES"
        return "", 0.0
    # 至少 2 字符, 中文+英文字母累计 >= 2
    if len(cleaned) < 2:
        if debug_info is not None:
            debug_info["drop_reason"] = f"too short ({len(cleaned)}<2)"
        return "", 0.0
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    letter_count = len(re.findall(r"[A-Za-z]", cleaned))
    if chinese_count + letter_count < 2:
        if debug_info is not None:
            debug_info["drop_reason"] = (
                f"insufficient letters (CN+EN={chinese_count + letter_count}<2)"
            )
        return "", 0.0
    mean_score = float(np.mean([t["score"] for t in texts]))
    return cleaned, mean_score


def _ocr_cell_price(price_bgr: np.ndarray, ocr,
                    debug_info: Optional[dict] = None) -> int:
    """对单卡片的价格条做 OCR, 返回解析到的价格 (0 表示未取得有效价格).

    三级 pass fallback (哪级先拿到文本就用哪级):
      1. 原图 (不放大不预处理): 价格本身就是大号白字高对比, 原分辨率下 RapidOCR
         的检测器效果最好; 经验证 58x360 的小 crop 放大后反而失败.
      2. 3x upscale (raw): 某些细小数字在放大后才够尺寸让 OCR 检测出来.
      3. 3x upscale + 预处理 (CLAHE+自适应阈值): 深色背景对比度不足时的兜底.
    做宽松的字符纠错: l/I/i→1, O/o→0, B→8, S→5, Z/z→2, 去空格/逗号/点.
    """
    if price_bgr is None or price_bgr.size == 0:
        if debug_info is not None:
            debug_info["price_drop"] = "empty crop"
        return 0
    h, w = price_bgr.shape[:2]

    def _run(image):
        try:
            return _extract_ocr_texts(ocr(image))
        except Exception:
            logger.debug("卡片价格 OCR 失败", exc_info=True)
            return []

    # Pass 1: 原图 (优先, 多数情况够用)
    texts = _run(price_bgr)
    pass_tag = "native"

    # Pass 2: 3x upscale raw (fallback, 字小时可能需要)
    if not texts:
        big = cv2.resize(price_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        texts = _run(big)
        pass_tag = "3x_raw"

        # Pass 3: 3x upscale + 预处理 (深色背景兜底)
        if not texts:
            try:
                pp = _preprocess_for_ocr(big)
            except Exception:
                pp = big
            texts = _run(pp)
            pass_tag = "3x_preprocessed"

    if debug_info is not None:
        debug_info["price_raw"] = [
            {"text": t["text"], "score": round(t["score"], 2),
             "xL": round(t.get("x_min", 0.0), 1),
             "xR": round(t.get("x_max", 0.0), 1)}
            for t in texts
        ]
        debug_info["price_pass"] = pass_tag

    # ─── 价格解析三层策略 ────────────────────────────────────────
    # 每个 OCR 段先做字符纠错 (l/I/i→1, O/o→0, B→8, S→5, Z/z→2) → 抽出纯数字.
    # 得到 [{digits, x_min, x_max, score, raw}] 候选列表, 过滤掉数字占比 < 50% 的段.
    cand: list[dict] = []
    for t in texts:
        raw = t.get("text", "")
        if not raw:
            continue
        cleaned = (raw.replace(",", "").replace("，", "")
                      .replace(" ", "").replace(".", "").replace("·", ""))
        fixed = (cleaned.replace("l", "1").replace("I", "1").replace("i", "1")
                        .replace("O", "0").replace("o", "0")
                        .replace("B", "8").replace("S", "5")
                        .replace("Z", "2").replace("z", "2"))
        digits = re.sub(r"\D", "", fixed)
        if not digits:
            continue
        if len(digits) / max(len(fixed), 1) < 0.5:
            continue
        cand.append({
            "digits": digits,
            "x_min": t.get("x_min", t.get("cx", 0.0)),
            "x_max": t.get("x_max", t.get("cx", 0.0)),
            "score": t.get("score", 0.0),
            "raw": raw,
        })

    # Step 1: 按 x_min 排序
    cand.sort(key=lambda s: s["x_min"])

    # Step 2: 遍历相邻段对做 "重叠感知合并".
    # OCR 有时把价格拆成两段, 而拆点不在逗号上, 导致边界框在像素上重叠 25-30%,
    # 并且两段数字在 "接缝" 处共享 1-2 位 (如 "1,981,6" + "681", 在 "6" 处重叠).
    # 合并规则:
    #   - 若边界框重叠 > 70%: 视为同一段文字被检测两次, 保留数字更多/置信度更高的
    #   - 若边界框有任何重叠 (>0) 或 A 后缀 == B 前缀 >= 2 位:
    #       按最长 suffix-prefix 重叠长度拼接 (A + B[overlap:]), 避免数字重复
    #   - 否则: 保持两段独立 (如金币图标 ghost 与价格完全分离, 不合并)
    def _longest_sp_overlap(a: str, b: str) -> int:
        """返回 a 的后缀与 b 的前缀最长相同子串长度 (0 表示无重叠)."""
        max_k = min(len(a), len(b))
        for k in range(max_k, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    merged_segs = cand
    i = 0
    while i < len(merged_segs) - 1:
        a = merged_segs[i]
        b = merged_segs[i + 1]
        overlap_px = max(0.0, min(a["x_max"], b["x_max"])
                         - max(a["x_min"], b["x_min"]))
        a_w = max(1.0, a["x_max"] - a["x_min"])
        b_w = max(1.0, b["x_max"] - b["x_min"])
        overlap_ratio = overlap_px / min(a_w, b_w)
        sp = _longest_sp_overlap(a["digits"], b["digits"])

        if overlap_ratio > 0.7:
            # 强重叠: 同一段被检测两次, 保留信息更丰富的
            if (len(b["digits"]) > len(a["digits"]) or
                    (len(b["digits"]) == len(a["digits"])
                     and b["score"] > a["score"])):
                a.update(b)
            merged_segs.pop(i + 1)
            # 停留, 用新的 a 与下一段继续比较
        elif overlap_px > 0 or sp >= 2:
            # 部分重叠或数字接缝对齐: 按 suffix-prefix 拼接去重
            a["digits"] = a["digits"] + b["digits"][sp:]
            a["x_max"] = max(a["x_max"], b["x_max"])
            a["score"] = max(a["score"], b["score"])
            merged_segs.pop(i + 1)
        else:
            i += 1

    # Step 3: 从合并后的段里选最终答案.
    #   - 存在 >= 4 位数字的段 → 取最长 (平局取置信度) 作为最终 (防金币图标 ghost 污染)
    #   - 否则 → 按 x_min 拼接全部段 (应对价格被拆成多个 <=3 位短段的场景)
    long_candidates = [d for d in merged_segs if len(d["digits"]) >= 4]
    if long_candidates:
        best = max(long_candidates,
                   key=lambda d: (len(d["digits"]), d["score"]))
        combined = best["digits"]
        strategy = "longest_single"
    else:
        merged_segs.sort(key=lambda s: s["x_min"])
        combined = "".join(d["digits"] for d in merged_segs)
        strategy = "concat_all"

    # 去掉前导 0 (若金币图标被误读成 "0" 仍会出现一个 "0" 开头)
    combined = combined.lstrip("0") or combined  # 全是 0 就保留, 虽然不可能
    if debug_info is not None:
        debug_info["price_combined"] = combined
        debug_info["price_strategy"] = strategy

    # 价格最少 3 位数字 (>=100)
    if len(combined) < 3:
        if debug_info is not None:
            debug_info["price_drop"] = (
                f"digits too few ({len(combined)}<3, combined={combined!r})"
            )
        return 0
    try:
        return int(combined)
    except Exception:
        if debug_info is not None:
            debug_info["price_drop"] = "int convert failed"
        return 0


def _extract_items_grid(img_bgr: np.ndarray,
                        category: str,
                        subcategory: str,
                        stats: Optional[dict] = None,
                        layout_override: Optional[dict] = None) -> list[dict]:
    """方案 C 入口: 按网格逐卡片提取道具名、价格、品质颜色.

    返回 [{name, price, rarity, row, col}] 列表, 已按 (row, col) 排序.
    skipped / ocr_failed / no_price 等统计放进 stats (若提供).

    layout_override: 翻页模式可传 ocr_layout 覆盖分类默认布局 (通常直接走默认).
    """
    h, w = img_bgr.shape[:2]
    layout = layout_override or CATEGORY_LAYOUT.get(category, _DEFAULT_LAYOUT)
    rows = layout["rows"]
    cols = layout["cols"]
    roi = layout["roi"]
    x1 = int(w * roi[0])
    y1 = int(h * roi[1])
    x2 = int(w * roi[2])
    y2 = int(h * roi[3])
    cell_w = (x2 - x1) / cols
    cell_h = (y2 - y1) / rows

    cnt_total = rows * cols
    cnt_no_name = 0
    cnt_no_price = 0
    cnt_ok = 0

    ocr = _get_rapid_ocr()
    if ocr is None:
        logger.warning("RapidOCR 不可用, 无法执行网格 OCR")
        if stats is not None:
            stats.update({"total_cells": cnt_total, "ok": 0,
                          "no_name": 0, "no_price": 0})
        return []

    # 逐格诊断目录: 每次 OCR 用一个新时间戳子目录, 保存每格的标题/价格裁剪图,
    # 以及一个 cells.txt 摘要表 (行号, 列号, OCR 段, 合并后, 丢弃原因)
    cells_dir = None
    cells_log: list[str] = []
    try:
        import time as _t
        from config import _BASE_DIR  # type: ignore
        cells_dir = os.path.join(_BASE_DIR, "debug_ocr", "cells",
                                 _t.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(cells_dir, exist_ok=True)
    except Exception:
        cells_dir = None

    items: list[dict] = []
    for r in range(rows):
        for c in range(cols):
            cx1 = x1 + int(c * cell_w)
            cy1 = y1 + int(r * cell_h)
            cx2 = x1 + int((c + 1) * cell_w)
            cy2 = y1 + int((r + 1) * cell_h)
            cell = img_bgr[cy1:cy2, cx1:cx2]
            if cell is None or cell.size == 0:
                cnt_no_name += 1
                cells_log.append(f"[{r},{c}] EMPTY CELL")
                continue
            ch, cw_ = cell.shape[:2]
            # 左上标题条: top 22% × left 82% (放大一点垂直空间, 避免标题被裁)
            t_h = max(10, int(ch * 0.22))
            t_w = max(20, int(cw_ * 0.82))
            title = cell[0:t_h, 0:t_w]
            # 右下价格条: bottom 28% × right 58%
            p_y1 = int(ch * 0.72)
            p_x1 = int(cw_ * 0.42)
            price_crop = cell[p_y1:ch, p_x1:cw_]

            # 品质色采样: 标题条上部 (y 5%-55%, x 3%-12%), 该区域几乎只有背景色
            rs_y1 = max(1, int(t_h * 0.05))
            rs_y2 = max(rs_y1 + 2, int(t_h * 0.55))
            rs_x1 = max(1, int(t_w * 0.03))
            rs_x2 = max(rs_x1 + 2, int(t_w * 0.12))
            rarity_strip = title[rs_y1:rs_y2, rs_x1:rs_x2]
            rarity = _detect_rarity(rarity_strip)

            # 诊断容器, 由 _ocr_cell_title/_ocr_cell_price 填充
            dbg: dict = {}
            name, _name_score = _ocr_cell_title(title, ocr, debug_info=dbg)
            price_val = 0
            if name:
                price_val = _ocr_cell_price(price_crop, ocr, debug_info=dbg)

            # 写入诊断 (无论成功失败都记录, 便于排查)
            raw_segs = dbg.get("raw_texts", [])
            raw_str = " | ".join(
                f"{s['text']!r}@{s['score']:.2f}" for s in raw_segs
            ) or "(无)"
            price_segs = dbg.get("price_raw", [])
            price_str = " | ".join(
                f"{s['text']!r}@{s['score']:.2f}@xL{s.get('xL', 0)}~{s.get('xR', 0)}"
                for s in price_segs
            ) or "(未尝试)"
            outcome = (f"OK name={name!r} price={price_val} rarity={rarity}"
                       if name and price_val > 0
                       else f"DROP name_reason={dbg.get('drop_reason', '-')} "
                            f"price_reason={dbg.get('price_drop', '-')}")
            cells_log.append(
                f"[{r},{c}] {outcome}\n"
                f"      title_raw: {raw_str}\n"
                f"      title_merged={dbg.get('merged', '-')!r} "
                f"cleaned={dbg.get('cleaned', '-')!r}\n"
                f"      price_pass={dbg.get('price_pass', '-')} "
                f"strategy={dbg.get('price_strategy', '-')} "
                f"combined={dbg.get('price_combined', '-')!r}\n"
                f"      price_raw: {price_str}"
            )
            # 把每格标题/价格裁剪图保存出来 (失败的尤其有用)
            if cells_dir:
                tag = "ok" if (name and price_val > 0) else "fail"
                try:
                    ok1, buf1 = cv2.imencode(".png", title)
                    if ok1:
                        buf1.tofile(os.path.join(
                            cells_dir, f"r{r}c{c}_{tag}_title.png"))
                    ok2, buf2 = cv2.imencode(".png", price_crop)
                    if ok2:
                        buf2.tofile(os.path.join(
                            cells_dir, f"r{r}c{c}_{tag}_price.png"))
                except Exception:
                    pass

            if not name:
                cnt_no_name += 1
                continue
            if price_val <= 0:
                cnt_no_price += 1
                continue

            items.append({
                "name": name,
                "price": price_val,
                "rarity": rarity,
                "row": r,
                "col": c,
            })
            cnt_ok += 1

    # 写诊断摘要文件
    if cells_dir:
        try:
            with open(os.path.join(cells_dir, "cells.txt"), "w",
                      encoding="utf-8") as f:
                f.write(f"Category={category}  Sub={subcategory}  "
                        f"Grid={rows}x{cols}  ROI={roi}\n\n")
                f.write("\n".join(cells_log))
            logger.info(f"逐格 OCR 诊断已保存: {cells_dir}")
        except Exception:
            logger.exception("写入逐格诊断失败")

    if stats is not None:
        stats.update({
            "total_cells": cnt_total,
            "ok": cnt_ok,
            "no_name": cnt_no_name,
            "no_price": cnt_no_price,
        })
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 方案 A (翻页专用): 整图 OCR + 空间配对, 不依赖固定网格
# ═══════════════════════════════════════════════════════════════════════════
#
# 为什么翻页模式要绕开方案 C:
#   方案 C 假定每页道具卡都对齐到固定的 N×3 网格里, scroll=0 时的确如此.
#   但 Delta Force 商店滚动是像素级的 (每次 wheel 大约 1.7 行),
#   滚动后画面上通常是:
#     [上一页末行的半截价格] → [完整 4~5 行] → [下一页首行的半截标题?]
#   固定网格会把半截卡的像素塞进错误的格子, 导致: 格子内同时含 A 的价格和 B
#   的标题, OCR 把它们配对成一个不存在的道具; 或整格被误判空白.
#
# 方案 A 的做法:
#   1. 在一个足够宽的 ROI 里对整张截图跑 OCR
#   2. 把 OCR 文字分为 "名称" 和 "价格" 两类 (已在 _parse_items_from_ocr)
#   3. 对每个名称, 找它右下方最近的价格 (同一张卡片的空间关系)
#   4. 只有名称+价格都在 ROI 内的才算一张完整卡 → 自动过滤半截卡
#   5. 品质色: 取名称中心左侧的标题条背景像素, 走 _detect_rarity

def _detect_rarity_near_name(img_bgr: np.ndarray,
                             cx: float, cy: float,
                             x_min: Optional[float] = None,
                             x_max: Optional[float] = None) -> str:
    """根据名称标签背景色判定品质.

    真机 (2560×1440) 采样分析得出的标签几何 (见诊断脚本 _test_real_tags3.py):
      - 名称标签高度 ≈ 26px, 其 y 中心与名称文本 cy 基本重合
      - 标签宽度随名称长短在 120~350px 之间, 紧贴卡片左边框起
      - 标签背景 = 品质色, 典型 HSV: H 因品质而定, S 60~95, V 60~90
      - 卡片正文紧邻标签下方, HSV = H≈100-105, S=85, V≈37 (深青, V 很低)

    以前的 bug: 把 cy±14 ×  cx+60..110 作为采样区. 这块 y 范围里 14 px
    还 OK (覆盖标签中心), 但 x 偏移到名称右侧 60~110 时, 往往已经
    **超出了标签右边**, 落到卡片背景的深青区上. 于是所有品质都被判成
    "蓝色 (H=100)".

    修正策略:
      1. 以名称 cx/cy 为中心, 采样 **足够宽的横纵带** (y±22, x-150..x+180)
         够覆盖标签整段宽度, 即使名称很长或偏心也能全盖住.
      2. 真正的过滤交给 _detect_rarity: V>55 & S>30 → 只留标签像素,
         卡片深青 V=37 自然被排除, 不管它 H 多少.
      3. 如果 OCR 拿到了名称文本的 bbox (x_min/x_max), 直接用它精确定
         左右边界, 效果更好; 取不到就用经验偏移.
    """
    h, w = img_bgr.shape[:2]
    # y 带: 比 26px 标签略宽, 容错 OCR 的 cy 偏差
    sy1 = max(0, int(cy) - 22)
    sy2 = min(h, int(cy) + 22)
    # x 带
    if x_min is not None and x_max is not None:
        # 用真实名称 bbox: 向左留 8px (容名称离标签左边距), 向右扩 40px
        # (容单字短名如 "CPU" 这类标签右侧还有一截纯品质色)
        sx1 = max(0, int(x_min) - 8)
        sx2 = min(w, int(x_max) + 40)
    else:
        # 没有 bbox 时的经验采样: 名称左侧 150px 到右侧 180px (覆盖常见标签跨度)
        sx1 = max(0, int(cx) - 150)
        sx2 = min(w, int(cx) + 180)
    if sx2 - sx1 < 8 or sy2 - sy1 < 6:
        return "白色"
    patch = img_bgr[sy1:sy2, sx1:sx2]
    return _detect_rarity(patch)


def _recover_orphan_name(img_bgr: np.ndarray,
                         price_cx: float, price_cy: float,
                         avg_dx: float, avg_dy: float,
                         roi_y1_px: float = 0.0,
                         roi_y2_px: Optional[float] = None) -> Optional[dict]:
    """为 "有价格但缺名称" 的孤儿价格做针对性二次 OCR, 尝试捞回短名称.

    典型场景: OCR 第一次扫整图时把 "CPU" 这种 3 字母短文本漏检 (品质标签背景 +
    字符少 + 缩放后更小, RapidOCR 的文本检测阶段直接过滤掉). 价格却能稳定识别,
    所以出现 "孤儿价格" - 位置在右下, 上游没有给它配上任何名称.

    修复: 根据已成功配对的卡片估算 "名称中心相对价格中心的平均偏移 (avg_dx, avg_dy)",
    在孤儿价格左上方裁一个宽标题条, 按 3x 放大 + 无预处理送 OCR, 小字识别率显著提升.

    roi_y1_px / roi_y2_px: 若给出, 估计名称 cy 落在 ROI 外 (顶部/底部半截卡) 直接
    放弃恢复, 避免把游戏顶栏 "交易记录"/"部门" 当成道具名捡回来.

    Returns: dict {name, cx, cy, x_min, x_max} 或 None (恢复失败).
    """
    ocr = _get_rapid_ocr()
    if ocr is None:
        return None
    h, w = img_bgr.shape[:2]

    # 名称中心估计位置 (注意符号: 名称在价格的左上, 所以减掉相对偏移)
    est_nx = price_cx - avg_dx
    est_ny = price_cy - avg_dy

    # 估计名称落在 ROI 之外 → 这是被截断的半截卡, 不应恢复
    if est_ny < roi_y1_px - 10:
        return None
    if roi_y2_px is not None and est_ny > roi_y2_px + 10:
        return None

    # 标题条 ROI: 宽一点 (±200) 以容名称长度差异, 高一点 (±35) 以容 cy 误差.
    # 不能太大, 太大容易把相邻卡片的名称也切进来造成歧义.
    cx1 = max(0, int(est_nx - 200))
    cy1 = max(0, int(est_ny - 35))
    cx2 = min(w, int(est_nx + 200))
    cy2 = min(h, int(est_ny + 35))
    if cx2 - cx1 < 50 or cy2 - cy1 < 20:
        return None
    crop = img_bgr[cy1:cy2, cx1:cx2]

    # 3x 放大 (不做 CLAHE/Unsharp, 短字符用 CUBIC 直接放大即可 —
    # 预处理的自适应阈值会把 3 字母短文本的笔画融掉)
    scaled = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                        interpolation=cv2.INTER_CUBIC)
    try:
        result = ocr(scaled)
    except Exception:
        return None
    if result is None:
        return None

    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if txts is None:
        return None
    if boxes is None:
        boxes = []
    if scores is None:
        scores = []

    # 挑一个 "看起来像道具名" 的候选: 含中文或字母, 非纯数字, 长度 >= 2
    best = None
    best_score = 0.0
    for i, txt in enumerate(txts):
        t = (txt or "").strip()
        if len(t) < 2:
            continue
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", t):
            continue
        # 跳过看起来是价格的 (纯数字/带逗号的数字)
        digits = sum(c.isdigit() for c in t)
        if digits >= len(t) * 0.7:
            continue
        # 跳过独立品相碎片
        stripped = _strip_condition_suffix(t)
        if not stripped or stripped in _NEVER_NAME_FRAGMENTS:
            continue
        # 跳过顶部导航/通用品类名 (孤儿价格若靠近画面顶部,
        # 恢复 crop 会跨进顶栏, 把 "交易记录" / "部门" / "特勤处" 当道具名捡回来)
        if any(kw in stripped for kw in _TOP_BAR_KEYWORDS):
            continue
        if stripped in _GENERIC_CATEGORY_NAMES:
            continue
        sc = float(scores[i]) if i < len(scores) else 0.0
        if sc > best_score:
            best_score = sc
            best = (stripped, boxes[i] if i < len(boxes) else None)

    if best is None or best_score < 0.4:
        return None

    name, box = best
    # 把 box 坐标从 "放大后 crop" 映射回原图
    if box is not None:
        xs = [float(p[0]) / 3 + cx1 for p in box]
        ys = [float(p[1]) / 3 + cy1 for p in box]
        new_cx = (min(xs) + max(xs)) / 2
        new_cy = (min(ys) + max(ys)) / 2
        new_xmin = min(xs)
        new_xmax = max(xs)
    else:
        new_cx, new_cy = est_nx, est_ny
        new_xmin = new_cx - 40
        new_xmax = new_cx + 40
    return {"name": name, "cx": new_cx, "cy": new_cy,
            "x_min": new_xmin, "x_max": new_xmax}


def _extract_items_scan(img_bgr: np.ndarray,
                        stats: Optional[dict] = None,
                        roi: Optional[tuple] = None) -> list[dict]:
    """翻页模式入口 (方案 A): 整图 OCR + 空间配对, 不依赖固定网格.

    道具在画面中任意偏移都能识别; 只采纳同时具备 [名称 + 右下方价格] 的完整卡片,
    半截卡 (顶部只剩价格 / 底部只剩标题) 通过 _parse_items_from_ocr 的配对规则
    被自然过滤掉.

    孤儿价格恢复: 第一遍 OCR 如果把 "CPU" 这种 3 字母短名称漏检了, 会留下价格但
    没有名称. 此处按已成功配对卡片的 "名称→价格" 平均偏移, 针对每个孤儿价格做
    一次放大的定点二次 OCR, 把漏掉的短名称捞回来.

    返回 [{name, price, rarity, cx, cy}] 列表 (已按从上到下、从左到右排序).
    stats 字段会被填: raw/names/prices/parsed 等 + 兼容翻页循环的 ok 字段.
    """
    if roi is None:
        roi = _SCAN_ROI
    h, w = img_bgr.shape[:2]

    ocr_items = _ocr_screenshot(img_bgr, roi=roi)
    inner_stats: dict = {}
    parsed = _parse_items_from_ocr(ocr_items, w, h,
                                   stats=inner_stats, roi=roi)

    # ─── 孤儿价格恢复 ───
    # 从已配对成功的卡片里统计 "名称中心→价格中心" 的平均偏移 (dx, dy),
    # 再用这个偏移给每个孤儿价格反推名称位置做二次 OCR.
    orphan_prices = inner_stats.get("orphan_prices", [])
    recovered: list[dict] = []
    if orphan_prices and parsed:
        dxs = [p["price_cx"] - p["cx"] for p in parsed if "price_cx" in p]
        dys = [p["price_cy"] - p["cy"] for p in parsed if "price_cy" in p]
        if dxs and dys:
            avg_dx = float(np.median(dxs))
            avg_dy = float(np.median(dys))
            used_positions = [(p["cx"], p["cy"]) for p in parsed]
            roi_y1_px = roi[1] * h
            roi_y2_px = roi[3] * h
            for op in orphan_prices:
                rec = _recover_orphan_name(img_bgr, op["cx"], op["cy"],
                                           avg_dx, avg_dy,
                                           roi_y1_px=roi_y1_px,
                                           roi_y2_px=roi_y2_px)
                if rec is None:
                    continue
                # 避免把"同一张卡的名称恰好在第一遍也被捡到过"的场景再次算进去:
                # 如果恢复的名称中心跟某个已采纳卡片非常接近 (<60px), 跳过.
                too_close = any(
                    abs(rec["cx"] - ux) < 60 and abs(rec["cy"] - uy) < 60
                    for ux, uy in used_positions
                )
                if too_close:
                    continue
                recovered.append({
                    "name": rec["name"],
                    "price": op["price"],
                    "cx": rec["cx"],
                    "cy": rec["cy"],
                    "x_min": rec["x_min"],
                    "x_max": rec["x_max"],
                })
                used_positions.append((rec["cx"], rec["cy"]))

    # 合并 + 按网格顺序重排
    all_items = parsed + recovered
    row_quant = h * 0.05
    all_items.sort(key=lambda p: (int(p["cy"] / row_quant), p["cx"]))

    result: list[dict] = []
    for item in all_items:
        rarity = _detect_rarity_near_name(
            img_bgr, item["cx"], item["cy"],
            x_min=item.get("x_min"), x_max=item.get("x_max"),
        )
        result.append({
            "name": item["name"],
            "price": item["price"],
            "rarity": rarity,
            "cx": item["cx"],
            "cy": item["cy"],
        })

    if stats is not None:
        stats.update(inner_stats)
        # 翻页循环通过 stats["ok"] == 0 判断是否空白页重试,
        # 这里把配对成功数当作 ok, 未配对上价格的名称当作 no_price
        stats["ok"] = len(result)
        stats["total_cells"] = inner_stats.get("names", 0)
        stats["no_name"] = 0
        stats["no_price"] = max(0, len(orphan_prices) - len(recovered))
        stats["recovered"] = len(recovered)
    return result


class CollectDialog(QDialog):
    """道具数据采集标注对话框."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("道具数据采集标注")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 900)
        self.setStyleSheet(CYBER_DIALOG_STYLE)
        # 启用标题栏的 最小化 / 最大化 / 关闭 按钮 (默认 QDialog 只有关闭)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self._db = ItemDatabase(ITEM_DB_FILE)
        self._current_category = ""
        self._current_subcategory = ""
        self._current_image: Optional[np.ndarray] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._screenshot_list: list[str] = []
        self._screenshot_idx: int = 0
        self._capture_count: int = 0
        self._table_dirty: bool = False  # 表格是否有未保存的修改

        self._init_ui()
        self._load_shop_tree()
        self._update_db_stats()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        top_info = QLabel(
            "选择分类 → 截取游戏画面 (或从左侧目录加载截图) → "
            "点击 [OCR 提取当前截图] → 手动标注品质颜色 → 保存到数据库"
        )
        top_info.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 4px; font-size: 10pt;")
        top_info.setWordWrap(True)
        main_layout.addWidget(top_info)

        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 10pt; padding: 2px;"
        )
        main_layout.addWidget(self._stats_label)

        capture_bar = QHBoxLayout()
        capture_bar.setSpacing(8)

        capture_bar.addWidget(QLabel("大类:"))
        self._combo_cat = QComboBox()
        self._combo_cat.setMinimumWidth(100)
        self._combo_cat.addItems(list(ITEM_CATEGORIES.keys()))
        self._combo_cat.currentTextChanged.connect(self._on_cat_changed)
        capture_bar.addWidget(self._combo_cat)

        capture_bar.addWidget(QLabel("子类:"))
        self._combo_sub = QComboBox()
        self._combo_sub.setMinimumWidth(120)
        capture_bar.addWidget(self._combo_sub)
        self._on_cat_changed(self._combo_cat.currentText())

        self._btn_capture = QPushButton("截取游戏画面")
        set_btn_style(self._btn_capture, NEON_ORANGE, "11pt", "8px 20px")
        self._btn_capture.clicked.connect(self._on_capture)
        capture_bar.addWidget(self._btn_capture)

        self._btn_capture_scroll = QPushButton("截取并翻页")
        set_btn_style(self._btn_capture_scroll, NEON_PURPLE, "11pt", "8px 20px")
        self._btn_capture_scroll.setToolTip(
            "自动循环: 截屏 → OCR → 滚动 → 再截屏, 按道具名跨页去重\n"
            "直到页面无变化 (到底) 或连续 2 轮无新道具. 最多 30 轮"
        )
        self._btn_capture_scroll.clicked.connect(self._on_capture_and_scroll)
        capture_bar.addWidget(self._btn_capture_scroll)

        self._capture_info = QLabel("")
        self._capture_info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        capture_bar.addWidget(self._capture_info)
        capture_bar.addStretch()

        main_layout.addLayout(capture_bar)

        splitter = QSplitter(Qt.Horizontal)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Shop 目录")
        self._tree.setMinimumWidth(180)
        self._tree.setMaximumWidth(280)
        self._tree.itemClicked.connect(self._on_tree_clicked)
        splitter.addWidget(self._tree)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(2, 2, 2, 2)

        btn_top_row = QHBoxLayout()

        self._btn_ocr = QPushButton("OCR 提取当前截图")
        set_btn_style(self._btn_ocr, NEON_CYAN)
        self._btn_ocr.clicked.connect(self._on_ocr_current)
        btn_top_row.addWidget(self._btn_ocr)

        self._btn_prev = QPushButton("◀ 上一张")
        self._btn_prev.clicked.connect(self._on_prev_image)
        self._btn_prev.setEnabled(False)
        btn_top_row.addWidget(self._btn_prev)

        self._img_info_label = QLabel("")
        self._img_info_label.setAlignment(Qt.AlignCenter)
        self._img_info_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        btn_top_row.addWidget(self._img_info_label)

        self._btn_next = QPushButton("下一张 ▶")
        self._btn_next.clicked.connect(self._on_next_image)
        self._btn_next.setEnabled(False)
        btn_top_row.addWidget(self._btn_next)

        self._btn_zoom = QPushButton("放大查看")
        self._btn_zoom.clicked.connect(self._on_zoom)
        btn_top_row.addWidget(self._btn_zoom)

        center_layout.addLayout(btn_top_row)

        self._img_scroll = QScrollArea()
        self._img_scroll.setWidgetResizable(True)
        self._img_label = QLabel("← 点击左侧目录加载截图")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12pt;")
        self._img_scroll.setWidget(self._img_label)
        center_layout.addWidget(self._img_scroll, 1)

        splitter.addWidget(center_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self._btn_save = QPushButton("保存数据库")
        set_btn_style(self._btn_save, NEON_GREEN)
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)

        self._btn_batch = QPushButton("批量提取全部")
        set_btn_style(self._btn_batch, NEON_PURPLE)
        self._btn_batch.clicked.connect(self._on_batch_extract)
        btn_row.addWidget(self._btn_batch)

        btn_del = QPushButton("删除选中")
        set_btn_style(btn_del, NEON_RED)
        btn_del.clicked.connect(self._on_delete_rows)
        btn_row.addWidget(btn_del)

        btn_minimize = QPushButton("缩小窗口")
        btn_minimize.setToolTip("把道具采集窗口最小化到任务栏 (不关闭, 主界面保持可见)")
        btn_minimize.clicked.connect(self._on_minimize_self)
        btn_row.addWidget(btn_minimize)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        right_layout.addLayout(btn_row)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["道具名称", "大类", "子类", "品质", "价格"]
        )
        self._table.verticalHeader().setStyleSheet(
            f"QHeaderView::section {{ background-color: {BG_WIDGET}; color: {TEXT_SECONDARY}; "
            f"border: 1px solid {BORDER_GLOW}; padding: 2px 4px; }}"
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setMinimumWidth(400)
        right_layout.addWidget(self._table, 1)

        rarity_row = QHBoxLayout()
        rarity_row.addWidget(QLabel("批量品质:"))
        for rarity in RARITIES:
            btn = QPushButton(rarity)
            color = RARITY_COLORS.get(rarity, "#ccc")
            btn.setStyleSheet(
                f"QPushButton {{ color: {color}; font-weight: bold; padding: 3px 6px; }}"
                f"QPushButton:hover {{ background: #333; }}"
            )
            btn.clicked.connect(lambda checked, r=rarity: self._set_selected_rarity(r))
            rarity_row.addWidget(btn)
        rarity_row.addStretch()
        right_layout.addLayout(rarity_row)

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 500, 500])

        main_layout.addWidget(splitter, 1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(18)
        main_layout.addWidget(self._progress)

    def _load_shop_tree(self):
        """扫描 Shop/ 目录, 构建树."""
        self._tree.clear()

        if not os.path.isdir(SHOP_DIR):
            item = QTreeWidgetItem(["(Shop 文件夹不存在)"])
            self._tree.addTopLevelItem(item)
            return

        for cat_name in sorted(os.listdir(SHOP_DIR)):
            cat_path = os.path.join(SHOP_DIR, cat_name)
            if not os.path.isdir(cat_path):
                continue

            cat_item = QTreeWidgetItem([cat_name])
            cat_item.setData(0, Qt.UserRole, cat_path)
            self._tree.addTopLevelItem(cat_item)

            for sub_name in sorted(os.listdir(cat_path)):
                sub_path = os.path.join(cat_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue

                png_count = len([
                    f for f in os.listdir(sub_path)
                    if f.lower().endswith(".png")
                ])
                sub_item = QTreeWidgetItem([f"{sub_name} ({png_count})"])
                sub_item.setData(0, Qt.UserRole, sub_path)
                sub_item.setData(0, Qt.UserRole + 1, cat_name)
                sub_item.setData(0, Qt.UserRole + 2, sub_name)
                cat_item.addChild(sub_item)

            cat_item.setExpanded(True)

    def _on_tree_clicked(self, item: QTreeWidgetItem, column: int):
        """点击树节点, 加载对应截图."""
        path = item.data(0, Qt.UserRole)
        cat = item.data(0, Qt.UserRole + 1)
        sub = item.data(0, Qt.UserRole + 2)

        if not cat or not sub:
            return

        self._current_category = cat
        self._current_subcategory = sub
        self._combo_cat.blockSignals(True)
        self._combo_cat.setCurrentText(cat)
        self._combo_cat.blockSignals(False)
        self._on_cat_changed(cat)
        self._combo_sub.setCurrentText(sub)

        pngs = sorted([
            f for f in os.listdir(path)
            if f.lower().endswith(".png")
        ])
        if not pngs:
            self._img_label.setText("该目录下没有 PNG 截图")
            return

        first_png = os.path.join(path, pngs[0])
        self._screenshot_list = [os.path.join(path, f) for f in pngs]
        self._screenshot_idx = 0

        self._load_image(first_png)

    @staticmethod
    def _bgr_to_pixmap(img: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _fit_image(self):
        """按滚动区域宽度等比缩放显示当前截图."""
        if not hasattr(self, "_current_pixmap") or self._current_pixmap is None:
            return
        vp_w = self._img_scroll.viewport().width() - 4
        if vp_w < 100:
            vp_w = 600
        pm = self._current_pixmap
        if pm.width() > vp_w:
            pm = pm.scaled(vp_w, int(vp_w * pm.height() / pm.width()),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_label.setPixmap(pm)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    @staticmethod
    def _cv_read(img_path: str) -> np.ndarray | None:
        """cv2.imread 的中文路径兼容版本."""
        try:
            data = np.fromfile(img_path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _load_image(self, img_path: str):
        """加载并显示一张截图."""
        img = self._cv_read(img_path)
        if img is None:
            self._img_label.setText(f"无法加载: {img_path}")
            return

        self._current_image = img
        self._current_pixmap = self._bgr_to_pixmap(img)
        self._fit_image()
        self._img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        h, w = img.shape[:2]
        fname = os.path.basename(img_path)
        self.statusBar_msg(
            f"已加载: {self._current_category}/{self._current_subcategory}/{fname} "
            f"({w}x{h})"
        )
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        total = len(self._screenshot_list)
        idx = self._screenshot_idx
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < total - 1)
        if total > 0:
            self._img_info_label.setText(f"第 {idx + 1}/{total} 张")
        else:
            self._img_info_label.setText("")

    def _on_prev_image(self):
        if self._screenshot_idx > 0:
            self._screenshot_idx -= 1
            self._load_image(self._screenshot_list[self._screenshot_idx])

    def _on_next_image(self):
        if self._screenshot_idx < len(self._screenshot_list) - 1:
            self._screenshot_idx += 1
            self._load_image(self._screenshot_list[self._screenshot_idx])

    def _on_zoom(self):
        """全屏显示当前截图."""
        if not hasattr(self, "_current_pixmap") or self._current_pixmap is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("截图预览 (按 ESC 或点击关闭)")
        dlg.setStyleSheet(f"QDialog {{ background: {BG_DARKEST}; }}")
        dlg.showMaximized()

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {BG_DARKEST}; }}")

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"background: {BG_DARKEST};")

        screen_size = QApplication.primaryScreen().availableSize()
        pm = self._current_pixmap.scaled(
            screen_size.width(), screen_size.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        label.setPixmap(pm)
        label.mousePressEvent = lambda e: dlg.close()

        scroll.setWidget(label)
        layout.addWidget(scroll)
        dlg.exec_()

    # ── 实时采集相关 ──────────────────────────────────────────

    def _on_cat_changed(self, cat_text: str):
        """大类下拉框变化时, 刷新子类下拉框."""
        self._combo_sub.clear()
        subs = ITEM_CATEGORIES.get(cat_text, [])
        self._combo_sub.addItems(subs)

    def _capture_game_screen(self) -> np.ndarray:
        """最小化对话框+父窗口 → 截取全屏 → 恢复, 返回 BGR numpy 数组.

        注意: 统一使用 showMinimized() 而不是 hide(), 避免模态对话框下
        hide/show 混用导致的 Qt 窗口状态损坏 (曾引起 0xC000041D 崩溃).
        """
        parent = self.parent()
        if parent is not None:
            parent.showMinimized()
        self.showMinimized()
        QApplication.processEvents()
        time.sleep(0.6)

        try:
            img_pil = ImageGrab.grab()
        finally:
            if parent is not None:
                parent.showNormal()
                parent.activateWindow()
                parent.raise_()
            self.showNormal()
            self.activateWindow()
            self.raise_()
            QApplication.processEvents()

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _show_captured_image(self, img_bgr: np.ndarray):
        """将截取的图像显示到中间区域."""
        self._current_image = img_bgr
        self._current_pixmap = self._bgr_to_pixmap(img_bgr)
        self._fit_image()
        self._img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        h, w = img_bgr.shape[:2]
        self._capture_count += 1
        self._capture_info.setText(f"已截取 {self._capture_count} 张")
        self._img_info_label.setText(f"实时截图 ({w}x{h})")
        self._screenshot_list.clear()
        self._screenshot_idx = 0
        self._update_nav_buttons()

    def _on_capture(self):
        """截取游戏画面 → 显示 (不自动 OCR, 需手动点击 [OCR 提取当前截图])."""
        cat = self._combo_cat.currentText()
        sub = self._combo_sub.currentText()
        if not cat or not sub:
            QMessageBox.warning(self, "未选择分类", "请先选择大类和子类")
            return

        self._current_category = cat
        self._current_subcategory = sub

        self._btn_capture.setEnabled(False)
        self._btn_capture_scroll.setEnabled(False)
        self._btn_capture.setText("正在截取...")
        QApplication.processEvents()

        try:
            img = self._capture_game_screen()
            self._show_captured_image(img)
            self.statusBar_msg("截图完成, 请点击 [OCR 提取当前截图] 手动提取道具")
        except Exception as e:
            logger.exception("截取游戏画面失败")
            QMessageBox.critical(self, "截取失败", f"错误: {e}")
        finally:
            self._btn_capture.setEnabled(True)
            self._btn_capture_scroll.setEnabled(True)
            self._btn_capture.setText("截取游戏画面")

    def _on_capture_and_scroll(self):
        """循环抓取: 截屏 → 扫描 OCR → 按名字去重累加 → 滚动 → 重复, 直到到底.

        OCR 模式: 方案 A (整图 OCR + 空间配对), 不依赖固定网格.
        原因: 滚动后游戏里道具位置是任意像素偏移, 固定 N×3 网格会错位,
        把半截卡/邻卡的内容拼成假道具. 扫描模式下每个名称只配右下方
        最近的价格, 半截卡 (顶行只剩价格/底行只剩标题) 自然被丢弃.

        停止条件 (任一满足即停):
        1. 当前 ROI 与上一次滚动后的 ROI 视觉上几乎一致 (mean abs diff < 6,
           已对商店页面的轻微动画/高亮留出余量)
        2. 连续 2 轮无新道具新增
        3. 达到最大轮数 (30, 兜底防死循环)
        """
        cat = self._combo_cat.currentText()
        sub = self._combo_sub.currentText()
        if not cat or not sub:
            QMessageBox.warning(self, "未选择分类", "请先选择大类和子类")
            return

        self._current_category = cat
        self._current_subcategory = sub

        self._btn_capture.setEnabled(False)
        self._btn_capture_scroll.setEnabled(False)
        self._btn_capture_scroll.setText("正在翻页采集...")
        QApplication.processEvents()

        parent = self.parent()
        # 整段流程只最小化一次, 结束再恢复, 避免每轮 min/restore 打断游戏焦点
        if parent is not None:
            parent.showMinimized()
        self.showMinimized()
        QApplication.processEvents()
        # 进入循环前先把鼠标挪到安全泊车点, 让首张截图也不被悬浮弹框遮挡
        _move_cursor(*_PARK_POS)
        time.sleep(0.6)

        seen_names: set[str] = set()
        total_added = 0
        prev_roi_small: Optional[np.ndarray] = None
        last_img: Optional[np.ndarray] = None
        no_new_streak = 0
        page = 0
        max_pages = 30
        stop_reason = "达到最大轮数"
        # 翻页模式走 "方案 A 扫描": 整图 OCR + 空间配对, 不依赖固定网格切分.
        # 滚动后道具在画面里是任意像素偏移, 固定网格切会错位; 方案 A 让每张
        # 完整卡 (有名称 + 右下方配得上的价格) 被采纳, 半截卡自动丢弃.

        # 翻页诊断目录: 每次 "截取并翻页" 用一个子目录, 按页保存原始全屏截图,
        # 便于事后比对 "空白页" 到底截到的是什么 (游戏中/loading/动画/焦点丢失).
        import time as _t
        scroll_debug_dir = None
        try:
            from config import _BASE_DIR  # type: ignore
            scroll_debug_dir = os.path.join(
                _BASE_DIR, "debug_ocr", "scroll",
                _t.strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(scroll_debug_dir, exist_ok=True)
        except Exception:
            scroll_debug_dir = None

        def _grab_screen() -> np.ndarray:
            """抓一张屏幕全屏截图 (防御性地再次最小化 dialog)."""
            self.showMinimized()
            if parent is not None:
                parent.showMinimized()
            QApplication.processEvents()
            pil = ImageGrab.grab()
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        def _save_scroll_debug(arr: np.ndarray, name: str) -> None:
            if scroll_debug_dir is None:
                return
            try:
                ok_, buf_ = cv2.imencode(".png", arr)
                if ok_:
                    buf_.tofile(os.path.join(scroll_debug_dir, name))
            except Exception:
                pass

        try:
            while page < max_pages:
                page += 1

                img = _grab_screen()
                last_img = img

                # 视觉差分用的 ROI (和 OCR ROI 分开, 用用户指定的固定区域保证翻页检测稳定)
                h, w = img.shape[:2]
                dx1, dy1, dx2, dy2 = _SCROLL_DIFF_ROI
                dpx1 = int(w * dx1)
                dpy1 = int(h * dy1)
                dpx2 = int(w * dx2)
                dpy2 = int(h * dy2)
                diff_roi = img[dpy1:dpy2, dpx1:dpx2]
                roi_small = cv2.resize(diff_roi, (128, 64),
                                       interpolation=cv2.INTER_AREA)

                if prev_roi_small is not None:
                    diff = float(np.mean(np.abs(
                        roi_small.astype(np.int16)
                        - prev_roi_small.astype(np.int16)
                    )))
                    # 阈值 6: 高于真滚动后的页面差异 (通常 >20), 低于同画面+动画的抖动 (<5)
                    if diff < 6.0:
                        stop_reason = f"第 {page} 页与上一页像素基本一致 (diff={diff:.2f})"
                        logger.info(f"[翻页] {stop_reason}, 判定已到底")
                        _save_scroll_debug(
                            img, f"p{page:02d}_stop_diff{diff:.2f}.png"
                        )
                        break
                    logger.debug(f"[翻页] 第 {page} 页与上一页 diff={diff:.2f}")
                prev_roi_small = roi_small

                _save_scroll_debug(img, f"p{page:02d}.png")

                # 正式跑 OCR + 追加到表格. 如果 stats['ok'] == 0 (整页全 DROP),
                # 极可能是截图时动画还没结束 (商店刚滚完、item 还在淡入), 原地重拍最多 2 次.
                stats: dict = {}
                added = self._extract_and_append(
                    img, cat, sub,
                    _out_stats=stats,
                    seen_names=seen_names,
                    use_scan=True,
                )
                if stats.get("ok", 0) == 0 and page > 0:
                    for retry_i in range(2):
                        logger.warning(
                            f"[翻页] 第 {page} 页整页 OCR 全失败, "
                            f"原地重试 #{retry_i + 1}..."
                        )
                        time.sleep(0.8)
                        img = _grab_screen()
                        last_img = img
                        _save_scroll_debug(
                            img, f"p{page:02d}_retry{retry_i + 1}.png"
                        )
                        stats = {}
                        added = self._extract_and_append(
                            img, cat, sub,
                            _out_stats=stats,
                            seen_names=seen_names,
                            use_scan=True,
                        )
                        if stats.get("ok", 0) > 0:
                            # 重试成功, 更新差分基线, 避免下一轮把同一帧误判为 "到底"
                            dx1, dy1, dx2, dy2 = _SCROLL_DIFF_ROI
                            diff_roi = img[dpy1:dpy2, dpx1:dpx2]
                            prev_roi_small = cv2.resize(
                                diff_roi, (128, 64),
                                interpolation=cv2.INTER_AREA
                            )
                            break

                total_added += added
                logger.info(
                    f"[翻页] 第 {page} 页: OCR 成功 {stats.get('ok', 0)} 格, "
                    f"去重后新增 {added} 个 (累计 {total_added})"
                )
                self.statusBar_msg(
                    f"翻页采集中 第 {page} 页: 本页新增 {added}, 累计 {total_added}"
                )
                QApplication.processEvents()

                if added == 0 and page > 1:
                    no_new_streak += 1
                    if no_new_streak >= 2:
                        stop_reason = (
                            f"连续 {no_new_streak} 轮无新道具, 判定到底"
                        )
                        logger.info(f"[翻页] {stop_reason}")
                        break
                else:
                    no_new_streak = 0

                # 鼠标移到差分 ROI 中心再滚动, 避免误滚到左侧分类栏.
                # wheel -3: 实测一次约 1.7 行位移, 相邻两页至少 3 行 overlap,
                # 跨页去重 (seen_names + _is_same_item) 处理重复, 绝不错过任何一行.
                mid_x = (dpx1 + dpx2) // 2
                mid_y = (dpy1 + dpy2) // 2
                ok = _send_wheel_at(mid_x, mid_y, -3)
                if not ok:
                    try:
                        pyautogui.moveTo(mid_x, mid_y, duration=0.1)
                        pyautogui.scroll(-3)
                    except Exception:
                        logger.exception("滚轮事件发送失败")
                        stop_reason = "滚动指令失败"
                        break
                # 滚动后给游戏更多时间完成动画 + 列表重排, 1.5s 实测能让 Delta Force 商店稳定
                time.sleep(1.5)

        except Exception as e:
            logger.exception("截取并翻页失败")
            QMessageBox.critical(self, "翻页采集失败", f"错误: {e}")
        finally:
            if parent is not None:
                parent.showNormal()
                parent.activateWindow()
                parent.raise_()
            self.showNormal()
            self.activateWindow()
            self.raise_()
            QApplication.processEvents()

            if last_img is not None:
                self._show_captured_image(last_img)

            self._btn_capture.setEnabled(True)
            self._btn_capture_scroll.setEnabled(True)
            self._btn_capture_scroll.setText("截取并翻页")

        self.statusBar_msg(
            f"翻页采集完成 - 共 {page} 页, 新增 {total_added} 个道具 ({stop_reason})"
        )
        if total_added > 0:
            QMessageBox.information(
                self, "翻页采集完成",
                f"共翻页 {page} 次, 累计提取 {total_added} 个道具\n"
                f"停止原因: {stop_reason}\n\n"
                f"请在右侧表格里核对品质/价格, 然后点击 [保存数据库]."
            )
        else:
            QMessageBox.warning(
                self, "翻页采集未提取到道具",
                f"共翻页 {page} 次, 未提取到任何道具\n"
                f"停止原因: {stop_reason}\n\n"
                f"请检查: 游戏窗口是否在前台, 画面是否处于交易行购买页面, "
                f"并查看 debug_ocr/scroll/ 下最近的按页截图看当时截到了什么."
            )

    # ── OCR / 图像加载 ─────────────────────────────────────

    def statusBar_msg(self, msg: str):
        parent = self.parent()
        if parent and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg)

    def _on_ocr_current(self):
        """对当前截图执行 OCR 提取."""
        if self._current_image is None:
            QMessageBox.warning(
                self, "无截图",
                "请先点击 [截取游戏画面] 或从左侧目录选择一个截图",
            )
            return

        # 每次 OCR 时从下拉框重新读取分类, 避免用户更改后未刷新
        cat = self._combo_cat.currentText().strip()
        sub = self._combo_sub.currentText().strip()
        if not cat or not sub:
            QMessageBox.warning(
                self, "未选择分类",
                "请先从上方下拉框选择 [大类] 和 [子类], 再执行 OCR 提取",
            )
            return
        self._current_category = cat
        self._current_subcategory = sub

        self._btn_ocr.setEnabled(False)
        self._btn_ocr.setText("正在识别...")
        QApplication.processEvents()

        last_stats: dict = {}
        last_raw: list[dict] = []
        try:
            added = self._extract_and_append(
                self._current_image,
                self._current_category,
                self._current_subcategory,
                _out_stats=last_stats,
                _out_raw=last_raw,
            )
            # 无论成功/失败都保存网格对齐调试图, 便于用户校准 ROI 比例
            debug_path = self._save_ocr_debug(self._current_image, last_raw)
            if debug_path:
                logger.info(f"OCR 网格调试图已保存: {debug_path}")
            if added == 0:
                self._show_ocr_failure_dialog(last_stats, last_raw)
        except Exception as e:
            logger.exception("OCR 提取失败")
            QMessageBox.critical(self, "OCR 提取失败", f"错误: {e}")
        finally:
            self._btn_ocr.setEnabled(True)
            self._btn_ocr.setText("OCR 提取当前截图")

    def _show_ocr_failure_dialog(self, stats: dict, raw: list[dict]):
        """0 个道具被提取时, 弹出详细诊断信息 + 保存 debug 截图."""
        debug_path = self._save_ocr_debug(self._current_image, raw)

        total = stats.get("total_cells", 0)
        ok = stats.get("ok", 0)
        no_name = stats.get("no_name", 0)
        no_price = stats.get("no_price", 0)
        stat_lines = [
            f"网格切分: {total} 个格子 (按 {self._current_category or '?'} 大类布局)",
            f"  → 标题 OCR 失败: {no_name}",
            f"  → 价格 OCR 失败: {no_price}",
            f"  → 成功识别: {ok}",
        ]

        msg = (
            "未能从截图中提取到道具信息.\n\n"
            + "\n".join(stat_lines)
            + "\n\n"
            + (f"已保存 debug 截图到:\n{debug_path}\n\n" if debug_path else "")
            + "可能原因:\n"
            "  1) 截图没有捕获到交易行道具列表页 (请确认游戏画面处于购买/出售页)\n"
            "  2) 游戏分辨率/ROI 比例不匹配, 网格切分错位 (请看 debug 图的格子线是否对准每张卡片)\n"
            "  3) 游戏 HDR/滤镜导致标题条对比度过低\n"
            "  4) 当前大类对应的网格布局 (rows×cols) 与游戏实际显示不一致, 需调整 CATEGORY_LAYOUT"
        )

        QMessageBox.information(self, "OCR 提取结果", msg)

    def _save_ocr_debug(self, img: np.ndarray,
                        raw: list[dict]) -> Optional[str]:
        """把网格切分+标题/价格子区域叠加到截图上保存, 方便排查对齐.

        方案 C 下:
        - 红框: 当前大类布局对应的 ROI
        - 黄框: 每个单元格 (rows × cols)
        - 青框: 每格的左上标题条 (标题 OCR 输入)
        - 绿框: 每格的右下价格条 (价格 OCR 输入)
        """
        try:
            import time as _t
            from config import _BASE_DIR  # type: ignore
        except Exception:
            _BASE_DIR = os.getcwd()
            import time as _t

        debug_dir = os.path.join(_BASE_DIR, "debug_ocr")
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception:
            return None

        stamp = _t.strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(debug_dir, f"ocr_{stamp}.png")

        try:
            annotated = img.copy()
            h, w = annotated.shape[:2]
            category = getattr(self, "_current_category", None) or ""
            layout = CATEGORY_LAYOUT.get(category, _DEFAULT_LAYOUT)
            rows = layout["rows"]
            cols = layout["cols"]
            roi = layout["roi"]
            x1 = int(w * roi[0])
            y1 = int(h * roi[1])
            x2 = int(w * roi[2])
            y2 = int(h * roi[3])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"ROI {category} {rows}x{cols}",
                        (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cell_w = (x2 - x1) / cols
            cell_h = (y2 - y1) / rows
            for r in range(rows):
                for c in range(cols):
                    cx1 = x1 + int(c * cell_w)
                    cy1 = y1 + int(r * cell_h)
                    cx2 = x1 + int((c + 1) * cell_w)
                    cy2 = y1 + int((r + 1) * cell_h)
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2),
                                  (0, 200, 200), 2)
                    cw_ = cx2 - cx1
                    ch_ = cy2 - cy1
                    # 标题条 (青色)
                    tx2 = cx1 + int(cw_ * 0.82)
                    ty2 = cy1 + int(ch_ * 0.20)
                    cv2.rectangle(annotated, (cx1, cy1), (tx2, ty2),
                                  (255, 200, 0), 2)
                    # 价格条 (绿色)
                    px1 = cx1 + int(cw_ * 0.42)
                    py1 = cy1 + int(ch_ * 0.72)
                    cv2.rectangle(annotated, (px1, py1), (cx2, cy2),
                                  (0, 255, 0), 2)
                    cv2.putText(annotated, f"{r},{c}",
                                (cx1 + 6, cy1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".png", annotated)
            if ok:
                buf.tofile(png_path)
            return png_path
        except Exception:
            logger.exception("保存 OCR debug 失败")
            return None

    def _extract_and_append(self, img: np.ndarray, category: str,
                            subcategory: str,
                            _out_stats: Optional[dict] = None,
                            _out_raw: Optional[list] = None,
                            layout_override: Optional[dict] = None,
                            seen_names: Optional[set] = None,
                            use_scan: bool = False) -> int:
        """OCR 提取道具并追加到表格, 返回新增行数.

        两种模式:
        - 网格模式 (默认, 方案 C): 按分类默认布局把 ROI 切成 N×3 单元格,
          每个卡片独立做标题/价格 OCR. 适合 scroll=0 的静态页.
        - 扫描模式 (use_scan=True, 方案 A): 整图 OCR + 空间配对, 不依赖固定网格.
          翻页模式必须用它, 因为滚动后道具位置是任意偏移的.

        _out_stats / _out_raw: 可选输出参数, 用于调用方获取诊断信息.
        layout_override: 仅网格模式生效, 覆盖分类默认布局.
        seen_names: 跨页去重集合. 已命中集合的道具名会被跳过, 新增的会写入集合.
        """
        stats_bucket: dict = {}
        if use_scan:
            parsed = _extract_items_scan(img, stats=stats_bucket)
        else:
            parsed = _extract_items_grid(img, category, subcategory,
                                         stats=stats_bucket,
                                         layout_override=layout_override)
        if _out_stats is not None:
            _out_stats.update(stats_bucket)
        if _out_raw is not None:
            # 两种模式都不再产出整图 OCR 中间结果, 保留接口兼容 (输出空)
            _out_raw.extend([])

        # ─── 词典纠错 (利用历史数据库把 OCR 残片映射回规范名) ───
        # 场景: OCR 把 "MHS战术头盔" 识别成 "术头盔", "Mask-1铁壁头盔" 识别成 "壁头盔",
        # 这里在当前 (category, subcategory) 范围内做低阈值模糊匹配, 命中就替换成规范名.
        # DB 为空或无同类数据时, 该步骤是无操作, 不影响首次采集.
        corrected = 0
        for item in parsed:
            match = self._db.find_fuzzy(
                item["name"], category=category, subcategory=subcategory,
            )
            if match is not None and match.name != item["name"]:
                logger.info(
                    f"  OCR 纠错: '{item['name']}' -> '{match.name}'"
                )
                item["name"] = match.name
                corrected += 1

        # ─── 跨页去重 (翻页模式) ─────────────────────────────────
        # 相邻两次滚动后, 顶部几行道具通常在两次截图里都出现.
        # 同一道具在不同页可能被 OCR 成不同字符 (DAS↔UAS, 盔↔签, 术↔木),
        # 精确匹配无法归并, 用 _is_same_item 做字符容错去重.
        skipped_dup = 0
        if seen_names is not None:
            filtered: list[dict] = []
            for item in parsed:
                name = item.get("name", "")
                if not name:
                    continue
                matched_existing = None
                for existing in seen_names:
                    if _is_same_item(name, existing):
                        matched_existing = existing
                        break
                if matched_existing is not None:
                    skipped_dup += 1
                    if matched_existing != name:
                        logger.info(
                            f"  跨页去重 (模糊): '{name}' 归并到已有 '{matched_existing}'"
                        )
                    continue
                seen_names.add(name)
                filtered.append(item)
            parsed = filtered

        added = 0
        for item in parsed:
            name = item["name"]
            price = item["price"]
            # 优先用像素级检测到的品质颜色; 若 DB 里已有同名记录, 以 DB 为准 (用户可能已校正)
            detected_rarity = item.get("rarity", "白色")
            existing = self._db.get(name)
            rarity = existing.rarity if existing else detected_rarity
            # 规则: 游戏内所有枪械品质均为白色 (标题条底色跟其它分类不一样, 像素检测会误判),
            # 这里无条件覆盖, DB 里的历史错误值也一并修正.
            if category == "枪械":
                rarity = "白色"

            row = self._table.rowCount()
            self._table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            cat_item = QTableWidgetItem(category)
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 1, cat_item)

            sub_item = QTableWidgetItem(subcategory)
            sub_item.setFlags(sub_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 2, sub_item)

            combo = QComboBox()
            for r in RARITIES:
                combo.addItem(r)
                idx = combo.count() - 1
                color = RARITY_COLORS.get(r, "#ccc")
                combo.setItemData(idx, QColor(color), Qt.ForegroundRole)
            combo.setCurrentText(rarity)
            self._table.setCellWidget(row, 3, combo)

            price_item = QTableWidgetItem(f"{price:,}")
            price_item.setFlags(price_item.flags() | Qt.ItemIsEditable)
            self._table.setItem(row, 4, price_item)

            added += 1

        if added > 0:
            self._table_dirty = True

        mode_label = "方案 A 扫描" if use_scan else "方案 C 网格"
        logger.info(
            f"OCR 提取完成 ({mode_label}): {category}/{subcategory}, "
            f"总格子 {stats_bucket.get('total_cells', 0)} | "
            f"成功 {stats_bucket.get('ok', 0)} | "
            f"无名称 {stats_bucket.get('no_name', 0)} | "
            f"无价格 {stats_bucket.get('no_price', 0)} | "
            f"DB 纠错 {corrected} | "
            f"跨页去重 {skipped_dup} | "
            f"新增 {added} 个道具"
        )
        return added

    def _on_batch_extract(self):
        """遍历 Shop/ 下所有截图, 批量 OCR 提取."""
        if not os.path.isdir(SHOP_DIR):
            QMessageBox.warning(self, "目录不存在", f"Shop 目录不存在: {SHOP_DIR}")
            return

        all_files = []
        for cat_name in sorted(os.listdir(SHOP_DIR)):
            cat_path = os.path.join(SHOP_DIR, cat_name)
            if not os.path.isdir(cat_path):
                continue
            for sub_name in sorted(os.listdir(cat_path)):
                sub_path = os.path.join(cat_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                for fname in sorted(os.listdir(sub_path)):
                    if fname.lower().endswith(".png"):
                        all_files.append((
                            os.path.join(sub_path, fname),
                            cat_name,
                            sub_name,
                        ))

        if not all_files:
            QMessageBox.information(self, "无截图", "Shop 目录下未找到任何 PNG 截图")
            return

        confirm = QMessageBox.question(
            self, "批量提取",
            f"共找到 {len(all_files)} 张截图, 开始批量 OCR 提取?\n"
            f"(已有的表格数据将保留, 新数据追加到末尾)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, len(all_files))
        self._btn_batch.setEnabled(False)
        self._btn_batch.setText("正在批量提取...")

        total_items = 0
        try:
            for i, (fpath, cat, sub) in enumerate(all_files):
                self._progress.setValue(i)
                img = self._cv_read(fpath)
                if img is None:
                    continue
                count = self._extract_and_append(img, cat, sub)
                total_items += count
        finally:
            self._progress.setVisible(False)
            self._btn_batch.setEnabled(True)
            self._btn_batch.setText("批量提取全部截图")

        QMessageBox.information(
            self, "批量提取完成",
            f"处理 {len(all_files)} 张截图, 共提取 {total_items} 个道具\n"
            f"请检查并标注品质后保存数据库",
        )

    def _set_selected_rarity(self, rarity: str):
        """为表格中选中的行设置品质."""
        rows = set()
        for idx in self._table.selectedIndexes():
            rows.add(idx.row())

        if not rows:
            QMessageBox.information(self, "未选中", "请先在表格中选中要修改的行")
            return

        for row in rows:
            combo = self._table.cellWidget(row, 3)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(rarity)

    def _on_delete_rows(self):
        """删除表格中选中的行."""
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()),
                      reverse=True)
        if not rows:
            return
        for row in rows:
            self._table.removeRow(row)
        self._table_dirty = True

    def _on_save(self):
        """将表格数据保存到数据库."""
        items = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            cat_item = self._table.item(row, 1)
            sub_item = self._table.item(row, 2)
            combo = self._table.cellWidget(row, 3)
            price_item = self._table.item(row, 4)

            if not name_item or not name_item.text().strip():
                continue

            name = name_item.text().strip()
            category = cat_item.text() if cat_item else ""
            subcategory = sub_item.text() if sub_item else ""
            rarity = combo.currentText() if isinstance(combo, QComboBox) else "白色"

            price_text = price_item.text().replace(",", "") if price_item else "0"
            try:
                price = int(price_text)
            except ValueError:
                price = 0

            items.append(ItemEntry(
                name=name,
                category=category,
                subcategory=subcategory,
                rarity=rarity,
                price=price,
            ))

        if not items:
            QMessageBox.warning(self, "无数据", "表格中没有可保存的数据")
            return

        new_count, update_count = self._db.add_many(items)
        self._db.save()
        self._update_db_stats()
        self._table_dirty = False

        QMessageBox.information(
            self, "保存成功",
            f"新增: {new_count} 个道具\n"
            f"更新: {update_count} 个道具\n"
            f"数据库总计: {self._db.count} 个道具\n"
            f"路径: {ITEM_DB_FILE}",
        )

    def _update_db_stats(self):
        """更新数据库统计信息."""
        stats = self._db.get_stats()
        parts = [f"数据库: {self._db.count} 个道具"]
        for cat, subs in sorted(stats.items()):
            total = sum(subs.values())
            parts.append(f"{cat}({total})")
        self._stats_label.setText("  |  ".join(parts))

    def _on_minimize_self(self):
        """只最小化采集对话框, 主窗口保持原样 (方便用户切回游戏确认信息)."""
        self.showMinimized()

    def _restore_parent(self):
        """关闭对话框前, 确保父窗口从最小化状态完整还原.

        防止截图流程中途关闭对话框时主窗口被遗留在 hidden/minimized 状态,
        后续用户点击最小化触发 0xC000041D 崩溃.
        """
        parent = self.parent()
        if parent is None:
            return
        try:
            if parent.isMinimized() or not parent.isVisible():
                parent.showNormal()
            parent.activateWindow()
            parent.raise_()
        except Exception:
            logger.exception("恢复父窗口状态失败")

    def closeEvent(self, event):
        if not self._table_dirty or self._table.rowCount() == 0:
            self._restore_parent()
            event.accept()
            return
        resp = QMessageBox.question(
            self, "未保存的修改",
            "表格中有未保存的道具数据, 是否先保存到数据库再关闭?\n\n"
            "是 — 保存并关闭\n否 — 放弃修改并关闭\n取消 — 继续编辑",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if resp == QMessageBox.Yes:
            self._on_save()
            self._restore_parent()
            event.accept()
        elif resp == QMessageBox.No:
            self._restore_parent()
            event.accept()
        else:
            event.ignore()
