"""
主控制面板 GUI
赛博朋克主题, 提供筛选设置、坐标校准(含截图拾取)、日志查看
坐标修改后永久保存到 user_config.json, 下次启动自动加载
"""
import os
import logging

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QCheckBox, QLabel, QPushButton, QTextEdit, QTabWidget,
    QSpinBox, QComboBox, QScrollArea, QRadioButton, QButtonGroup,
    QMessageBox, QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QColor, QPixmap, QPainter, QPen

from config import (
    ITEM_CATEGORIES, STORAGE_BOXES, MAX_STORAGE_BOXES, Rarity,
    ITEM_GRID, BOX_SELECTOR, LIST_DIALOG, ORGANIZE_BTN,
    LISTED_ITEMS, LISTING_SLOTS, TAB_COORDS, DETECTION_ROI, HOTKEYS,
    apply_saved_coordinates,
    load_user_config, save_user_config,
    ML_MODEL_FILE, ORIGINAL_DEFAULT_COORDS,
)
from gui.theme import (
    CYBER_STYLE, CYBER_DIALOG_STYLE, CYBER_HELP_CSS,
    NEON_CYAN, NEON_GREEN, NEON_PURPLE, NEON_RED, NEON_GOLD,
    BG_DARKEST, BG_DARK,
    BORDER_GLOW, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    RARITY_COLORS_HEX, set_btn_style,
)
from core.workflow import WorkflowEngine
from core.ml_detector import MLBoundDetector
from core.resolution import GameWindowInfo, REF_WIDTH, REF_HEIGHT

logger = logging.getLogger("gui")


# ═══════════════════════════════════════════════════════════
# 坐标拾取对话框
# ═══════════════════════════════════════════════════════════

class ClickableImageLabel(QLabel):
    """可点击的图片标签, 支持点拾取 (point) 与矩形框选 (region) 两种模式.

    - point 模式: 鼠标左键点击, 发 clicked(rx, ry) 信号
    - region 模式: 按下拖动到释放, 发 region_picked(x1, y1, x2, y2) 信号
                   (四个值都是基于原图坐标的整数, 已处理左右/上下颠倒)
    """
    clicked = pyqtSignal(int, int)
    region_picked = pyqtSignal(int, int, int, int)
    mouse_pos = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._scale = 1.0
        self._base_pixmap = None       # 原始截图
        self._committed_pixmap = None  # 已确认的拾取叠加层 (所有已提交的点/框)
        # 交互模式: "point" 或 "region"
        self._mode = "point"
        # region 模式下正在拖动的临时框 (屏幕坐标)
        self._drag_start = None  # (sx, sy)
        self._drag_current = None

    # ── 外部接口 ──────────────────────────────────
    def set_mode(self, mode: str):
        """设置交互模式. mode ∈ {"point", "region"}"""
        self._mode = "region" if mode == "region" else "point"
        self._drag_start = None
        self._drag_current = None
        if self._committed_pixmap is not None:
            self.setPixmap(self._committed_pixmap)

    def load_image(self, source, max_w: int = 1400, max_h: int = 800) -> bool:
        """加载图片, source 可以是文件路径(str)或BGR numpy数组"""
        if isinstance(source, np.ndarray):
            from PyQt5.QtGui import QImage
            rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = rgb.shape
            qimg = QImage(rgb.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())
        else:
            from PyQt5.QtGui import QImage
            qimg = QImage(str(source))
            if qimg.isNull():
                return False
            pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            return False
        w, h = pixmap.width(), pixmap.height()
        self._scale = min(max_w / w, max_h / h, 1.0)
        sw, sh = int(w * self._scale), int(h * self._scale)
        self._base_pixmap = pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._committed_pixmap = self._base_pixmap.copy()
        self.setPixmap(self._base_pixmap)
        self.setFixedSize(sw, sh)
        return True

    def _to_real(self, x, y):
        return int(x / self._scale), int(y / self._scale)

    # ── 鼠标事件 ──────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._mode == "region":
            # 开始拖动画矩形
            self._drag_start = (event.x(), event.y())
            self._drag_current = self._drag_start
            return
        # 点模式: 立即提交
        rx, ry = self._to_real(event.x(), event.y())
        self._commit_crosshair(event.x(), event.y(), rx, ry)
        self.clicked.emit(rx, ry)

    def mouseMoveEvent(self, event):
        rx, ry = self._to_real(event.x(), event.y())
        self.mouse_pos.emit(rx, ry)
        if self._mode == "region" and self._drag_start is not None:
            # 拖动中: 半透明矩形预览 (在 committed_pixmap 上叠加)
            self._drag_current = (event.x(), event.y())
            self._draw_drag_rect()
        else:
            self._draw_hover_crosshair(event.x(), event.y(), rx, ry)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._mode != "region" or self._drag_start is None:
            return
        end = (event.x(), event.y())
        sx1, sy1 = self._drag_start
        sx2, sy2 = end
        self._drag_start = None
        self._drag_current = None

        # 规范化到左上 -> 右下
        if sx2 < sx1:
            sx1, sx2 = sx2, sx1
        if sy2 < sy1:
            sy1, sy2 = sy2, sy1
        # 拖动太小 (< 6px) 算误触, 不提交
        if (sx2 - sx1) < 6 or (sy2 - sy1) < 6:
            if self._committed_pixmap is not None:
                self.setPixmap(self._committed_pixmap)
            return

        rx1, ry1 = self._to_real(sx1, sy1)
        rx2, ry2 = self._to_real(sx2, sy2)
        self._commit_rect(sx1, sy1, sx2, sy2, rx1, ry1, rx2, ry2)
        self.region_picked.emit(rx1, ry1, rx2, ry2)

    def leaveEvent(self, event):
        """鼠标离开控件, 清除悬停效果, 只保留已提交的拾取."""
        if self._committed_pixmap is not None:
            self.setPixmap(self._committed_pixmap)
        super().leaveEvent(event)

    # ── 绘制: 已提交的点 ────────────────────────────
    def _commit_crosshair(self, sx, sy, rx, ry):
        """左键点击: 在 _committed_pixmap 上绘制不透明十字线 (持久化)."""
        if self._committed_pixmap is None:
            return
        painter = QPainter(self._committed_pixmap)
        pm_w = self._committed_pixmap.width()
        pm_h = self._committed_pixmap.height()

        pen = QPen(QColor(0, 255, 255), 1)
        painter.setPen(pen)
        painter.drawLine(sx, 0, sx, pm_h)
        painter.drawLine(0, sy, pm_w, sy)

        pen2 = QPen(QColor(255, 50, 50), 2)
        painter.setPen(pen2)
        painter.drawEllipse(sx - 8, sy - 8, 16, 16)

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(sx + 14, sy - 5, f"({rx}, {ry})")
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(sx + 12, sy - 7, f"({rx}, {ry})")
        painter.end()

        self.setPixmap(self._committed_pixmap)

    # ── 绘制: 已提交的矩形 ──────────────────────────
    def _commit_rect(self, sx1, sy1, sx2, sy2, rx1, ry1, rx2, ry2):
        """框选完成: 在 _committed_pixmap 上绘制矩形 + 标签."""
        if self._committed_pixmap is None:
            return
        painter = QPainter(self._committed_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(255, 50, 120), 2)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 50, 120, 50))
        painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

        label = f"({rx1}, {ry1}) → ({rx2}, {ry2})"
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(sx1 + 5, sy1 + 16, label)
        painter.setPen(QColor(255, 200, 50))
        painter.drawText(sx1 + 4, sy1 + 15, label)
        painter.end()

        self.setPixmap(self._committed_pixmap)

    # ── 绘制: 拖动预览矩形 ──────────────────────────
    def _draw_drag_rect(self):
        if self._committed_pixmap is None or self._drag_start is None \
           or self._drag_current is None:
            return
        pm = self._committed_pixmap.copy()
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        sx1, sy1 = self._drag_start
        sx2, sy2 = self._drag_current
        if sx2 < sx1:
            sx1, sx2 = sx2, sx1
        if sy2 < sy1:
            sy1, sy2 = sy2, sy1

        pen = QPen(QColor(255, 200, 50), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 200, 50, 40))
        painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

        rx1, ry1 = self._to_real(sx1, sy1)
        rx2, ry2 = self._to_real(sx2, sy2)
        size_txt = f"{rx2 - rx1} × {ry2 - ry1}"
        coord_txt = f"({rx1},{ry1}) → ({rx2},{ry2})"
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(sx1 + 5, sy1 - 4, coord_txt)
        painter.setPen(QColor(0, 255, 255))
        painter.drawText(sx1 + 4, sy1 - 5, coord_txt)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(sx1 + 5, sy2 + 14, size_txt)
        painter.setPen(QColor(255, 200, 50))
        painter.drawText(sx1 + 4, sy2 + 13, size_txt)

        painter.end()
        self.setPixmap(pm)

    def _draw_hover_crosshair(self, sx, sy, rx, ry):
        """鼠标悬停: 在 committed_pixmap 基础上叠加半透明十字线 + 坐标 (不持久化)."""
        if self._committed_pixmap is None:
            return
        pm = self._committed_pixmap.copy()
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(0.45)  # 悬停时整体透明度
        pm_w = pm.width()
        pm_h = pm.height()

        # 十字线: 青色, 较细
        pen = QPen(QColor(0, 255, 255), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(sx, 0, sx, pm_h)
        painter.drawLine(0, sy, pm_w, sy)

        # 小圆圈标记
        pen2 = QPen(QColor(255, 200, 50), 1)
        painter.setPen(pen2)
        painter.drawEllipse(sx - 6, sy - 6, 12, 12)

        # 坐标文字: 半透明黑描边 + 亮青字, 文字稍微更实一点 (不透明度单独调高)
        painter.setOpacity(0.75)
        # 文字默认在右下, 靠近边缘时翻转避免被裁掉
        tx = sx + 12
        ty = sy - 8
        text = f"({rx}, {ry})"
        if tx > pm_w - 120:
            tx = sx - 120
        if ty < 14:
            ty = sy + 16
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(tx + 1, ty + 1, text)
        painter.setPen(QColor(0, 255, 255))
        painter.drawText(tx, ty, text)

        painter.end()
        self.setPixmap(pm)


# ─── 坐标拾取目标注册表 ─────────────────────────────────────────────
# 条目格式:
#   ("point",  key, label, (spin_x_key, spin_y_key))
#   ("region", key, label, (spin_x1, spin_y1, spin_x2, spin_y2))
# 用户在截图上:
#   - point  类型: 单击一下, 坐标填入 spin_x / spin_y
#   - region 类型: 按住鼠标左键拖一个框, 四个值分别填入四个 SpinBox
# 拾取完成后自动跳到下一个目标, 方便一次性校准所有字段.
PICK_TARGETS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # === 道具网格 ===
    ("point",  "grid_start",     "网格左上角 (X起, Y起)",
        ("grid_x_start", "grid_y_start")),
    ("point",  "grid_end",       "网格右下角 (X终, Y终)",
        ("grid_x_end", "grid_y_end")),
    ("point",  "scroll_area",    "网格滚动停鼠点",
        ("scroll_area_x", "scroll_area_y")),

    # === 上架弹窗 — 按钮 ===
    ("point",  "qty_minus",      "数量 − 按钮",
        ("qty_minus_x", "qty_minus_y")),
    ("point",  "qty_plus",       "数量 ＋ 按钮",
        ("qty_plus_x", "qty_plus_y")),
    ("point",  "slider_left",    "数量滑块 左端",
        ("slider_left_x", "slider_left_y")),
    ("point",  "slider_right",   "数量滑块 右端",
        ("slider_right_x", "slider_right_y")),
    ("point",  "qty_max",        "滑动条最右端 (一键拉满)",
        ("qty_max_x", "qty_max_y")),
    ("point",  "price_input",    "价格输入框",
        ("price_input_x", "price_input_y")),
    ("point",  "price_minus",    "价格 − 按钮",
        ("price_minus_x", "price_minus_y")),
    ("point",  "price_plus",     "价格 ＋ 按钮",
        ("price_plus_x", "price_plus_y")),
    ("point",  "list_btn",       "上架确认按钮 (绿色)",
        ("list_btn_x", "list_btn_y")),
    ("point",  "esc_btn",        "弹窗返回/ESC 按钮",
        ("esc_btn_x", "esc_btn_y")),

    # === 下架相关 ===
    ("point",  "delist_btn",     "下架按钮 (左侧列表内)",
        ("delist_btn_x", "delist_btn_y")),
    ("point",  "confirm_btn",    "下架确认按钮 (弹窗内)",
        ("confirm_btn_x", "confirm_btn_y")),

    # === 顶部 tab ===
    ("point",  "tab_trade",      "顶部「交易行」标签",
        ("tab_trade_x", "tab_trade_y")),
    ("point",  "tab_sell",       "二级「出售」标签",
        ("tab_sell_x", "tab_sell_y")),

    # === 仓库管理 ===
    ("point",  "organize_icon",  "整理仓库图标",
        ("org_x", "org_y")),
    ("point",  "sort_btn",       "整理按钮",
        ("sort_x", "sort_y")),

    # === OCR / 检测区域 (框选拖拽) ===
    ("region", "name_region",    "[OCR] 道具名称区",
        ("name_x1", "name_y1", "name_x2", "name_y2")),
    ("region", "income_region",  "[OCR] 预期收入区",
        ("income_x1", "income_y1", "income_x2", "income_y2")),
    ("region", "dialog_crop",    "[OCR] 上架弹窗右面板",
        ("dialog_x1", "dialog_y1", "dialog_x2", "dialog_y2")),
    ("region", "counter_region", "[OCR] 上架槽位计数 (X/15)",
        ("counter_x1", "counter_y1", "counter_x2", "counter_y2")),
    ("region", "page_change",    "[检测] 页面跳转比对区",
        ("page_change_x1", "page_change_y1",
         "page_change_x2", "page_change_y2")),
    ("region", "sell_tab",       "[检测] 出售 tab 高亮条",
        ("sell_tab_x1", "sell_tab_y1", "sell_tab_x2", "sell_tab_y2")),
]
# 为每个箱子追加一个拾取目标: X 共用 box_x, Y 独立为 box{i}_y
for _bi in range(1, MAX_STORAGE_BOXES + 1):
    _label = "主仓库 (箱子1)" if _bi == 1 else f"箱子{_bi}"
    PICK_TARGETS.append(
        ("point", f"box{_bi}", f"{_label} 图标位置",
         ("box_x", f"box{_bi}_y"))
    )
del _bi, _label


def _target_kind(idx: int) -> str:
    """从 PICK_TARGETS[idx] 中提取类型 (兼容工具).

    返回 "point" 或 "region".
    """
    return PICK_TARGETS[idx][0]


def _target_label(idx: int) -> str:
    return PICK_TARGETS[idx][2]


def _target_keys(idx: int) -> tuple[str, ...]:
    return PICK_TARGETS[idx][3]


class CoordPickerDialog(QDialog):
    """截图坐标拾取对话框 — 截图后在图片上点击 (point) 或拖框 (region) 拾取.

    - coord_picked(keys, values): 通用信号, keys 与 values 一一对应.
        * point  类型: keys=(x_key, y_key), values=(x, y)
        * region 类型: keys=(x1_key, y1_key, x2_key, y2_key), values=(x1, y1, x2, y2)
    """
    coord_picked = pyqtSignal(tuple, tuple)

    def __init__(self, image_source, parent=None):
        """image_source: 文件路径(str) 或 BGR numpy 数组"""
        super().__init__(parent)
        self.setWindowTitle("坐标拾取 — 选择目标后在图片上点击/拖框")
        self.setMinimumSize(1000, 650)
        self.resize(1400, 880)
        self.setStyleSheet(CYBER_DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- 顶部: 目标选择 + 坐标显示 ---
        top = QHBoxLayout()
        top.addWidget(QLabel("拾取目标:"))
        self.target_combo = QComboBox()
        for _kind, _key, label, _spins in PICK_TARGETS:
            prefix = "[区域]" if _kind == "region" else "[点  ]"
            self.target_combo.addItem(f"{prefix} {label}")
        self.target_combo.setMinimumWidth(280)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        top.addWidget(self.target_combo)

        self.coord_label = QLabel("  鼠标移到图片上查看坐标, 左键点击拾取")
        self.coord_label.setObjectName("pickerCoord")
        top.addWidget(self.coord_label, 1)

        btn_close = QPushButton("完成")
        btn_close.clicked.connect(self.accept)
        top.addWidget(btn_close)
        layout.addLayout(top)

        # --- 提示栏 (根据模式切换文案) ---
        self.hint_label = QLabel()
        self.hint_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; padding: 2px 4px; font-size: 11px;"
        )
        layout.addWidget(self.hint_label)

        # --- 图片 ---
        self.image_label = ClickableImageLabel()
        ok = self.image_label.load_image(image_source, 1380, 780)
        if not ok:
            self.coord_label.setText("  无法加载图片!")

        self.image_label.mouse_pos.connect(self._on_mouse_move)
        self.image_label.clicked.connect(self._on_click)
        self.image_label.region_picked.connect(self._on_region)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)

        # --- 底部: 已拾取记录 ---
        self.history_label = QLabel("已拾取: (无)")
        self.history_label.setWordWrap(True)
        self.history_label.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 4px;")
        layout.addWidget(self.history_label)

        self._history: list[str] = []
        # 初始化 hint 与交互模式
        self._on_target_changed(0)

    # ── 模式切换 ────────────────────────────────────
    def _on_target_changed(self, idx: int):
        if idx < 0 or idx >= len(PICK_TARGETS):
            return
        kind = _target_kind(idx)
        label = _target_label(idx)
        self.image_label.set_mode(kind)
        if kind == "region":
            self.hint_label.setText(
                f"▣ 区域模式:  在游戏截图上 按住左键拖一个框 框住「{label}」, "
                f"松开即提交"
            )
        else:
            self.hint_label.setText(
                f"◎ 点模式:  在游戏截图上 左键点击「{label}」的中心, "
                f"立即提交"
            )

    def _on_mouse_move(self, rx, ry):
        idx = self.target_combo.currentIndex()
        target_name = _target_label(idx)
        self.coord_label.setText(f"  [{target_name}]  当前: ({rx}, {ry})")

    def _on_click(self, rx, ry):
        """point 模式回调"""
        idx = self.target_combo.currentIndex()
        if _target_kind(idx) != "point":
            return
        target_name = _target_label(idx)
        keys = _target_keys(idx)  # (x_key, y_key)
        self.coord_label.setText(f"  [{target_name}]  已拾取: ({rx}, {ry})")
        self.coord_picked.emit(keys, (rx, ry))

        self._history.append(f"{target_name}: ({rx}, {ry})")
        self.history_label.setText("已拾取: " + "  |  ".join(self._history))
        self._advance_to_next_target(idx)

    def _on_region(self, x1, y1, x2, y2):
        """region 模式回调"""
        idx = self.target_combo.currentIndex()
        if _target_kind(idx) != "region":
            return
        target_name = _target_label(idx)
        keys = _target_keys(idx)  # (x1, y1, x2, y2)
        self.coord_label.setText(
            f"  [{target_name}]  已框选: ({x1},{y1}) → ({x2},{y2})  "
            f"尺寸 {x2 - x1} × {y2 - y1}"
        )
        self.coord_picked.emit(keys, (x1, y1, x2, y2))

        self._history.append(
            f"{target_name}: {x2 - x1}×{y2 - y1} @ ({x1},{y1})"
        )
        self.history_label.setText("已拾取: " + "  |  ".join(self._history))
        self._advance_to_next_target(idx)

    def _advance_to_next_target(self, idx: int):
        if idx < len(PICK_TARGETS) - 1:
            self.target_combo.setCurrentIndex(idx + 1)


# ═══════════════════════════════════════════════════════════
# 上架模式选择对话框
# ═══════════════════════════════════════════════════════════

class ListModeDialog(QDialog):
    """点击 "开始上架" 后的模式选择对话框.

    - 单次模式: 执行一轮所有选中箱子后结束 (原有行为).
    - 挂机模式: 每轮结束后检查上架槽位, 有空位就继续, 槽位满则等待 OCR
                重检间隔后再试, 循环直到用户手动停止或达到最大执行时间.

    结果通过 ``get_result()`` 获取:
        {
            "mode": "single" | "idle",
            "idle_ocr_interval_sec": int,       # 槽位满时的 OCR 重检间隔
            "idle_max_duration_min": int,       # 最大执行时长 (分钟), 0=不限
        }
    """

    # 默认值 (下次打开对话框时会沿用用户上次选择)
    _last_mode = "single"
    _last_ocr_interval_sec = 60
    _last_max_duration_min = 0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择上架模式")
        self.setMinimumWidth(460)
        try:
            self.setStyleSheet(CYBER_DIALOG_STYLE)
        except Exception:
            pass

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- 模式选择 ---
        mode_group = QGroupBox("执行模式")
        mode_layout = QVBoxLayout(mode_group)

        self.btn_group = QButtonGroup(self)
        self.rb_single = QRadioButton("单次上架 — 遍历所有选中箱子一轮后结束")
        self.rb_idle = QRadioButton("挂机模式 — 循环执行, 槽位空出后继续上架")
        self.btn_group.addButton(self.rb_single)
        self.btn_group.addButton(self.rb_idle)
        mode_layout.addWidget(self.rb_single)
        mode_layout.addWidget(self.rb_idle)
        root.addWidget(mode_group)

        # --- 挂机参数 ---
        self.idle_box = QGroupBox("挂机模式设置")
        idle_layout = QGridLayout(self.idle_box)
        idle_layout.setColumnStretch(1, 1)

        idle_layout.addWidget(QLabel("槽位已满时 OCR 重检间隔 (秒):"), 0, 0)
        self.sp_ocr_interval = QSpinBox()
        self.sp_ocr_interval.setRange(10, 3600)
        self.sp_ocr_interval.setSingleStep(10)
        self.sp_ocr_interval.setSuffix(" 秒")
        idle_layout.addWidget(self.sp_ocr_interval, 0, 1)

        idle_layout.addWidget(QLabel("最大执行时长 (0=不限):"), 1, 0)
        self.sp_max_duration = QSpinBox()
        self.sp_max_duration.setRange(0, 1440)
        self.sp_max_duration.setSingleStep(15)
        self.sp_max_duration.setSuffix(" 分钟")
        idle_layout.addWidget(self.sp_max_duration, 1, 1)

        hint = QLabel(
            "说明: 挂机模式会在每轮结束后检测已上架数量, 若有空位立即\n"
            "上架下一件; 若槽位已满则按设定间隔重检, 直到手动停止 (Ctrl+2)\n"
            "或达到最大执行时长."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; padding: 4px;")
        hint.setWordWrap(True)
        idle_layout.addWidget(hint, 2, 0, 1, 2)

        root.addWidget(self.idle_box)

        # --- 确定/取消 ---
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        btns.button(QDialogButtonBox.Ok).setText("开始")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # 初始化: 沿用上次选择
        if ListModeDialog._last_mode == "idle":
            self.rb_idle.setChecked(True)
        else:
            self.rb_single.setChecked(True)
        self.sp_ocr_interval.setValue(ListModeDialog._last_ocr_interval_sec)
        self.sp_max_duration.setValue(ListModeDialog._last_max_duration_min)

        # 挂机参数随单选框联动启用/禁用
        self.rb_single.toggled.connect(self._refresh_idle_enabled)
        self.rb_idle.toggled.connect(self._refresh_idle_enabled)
        self._refresh_idle_enabled()

    def _refresh_idle_enabled(self):
        enabled = self.rb_idle.isChecked()
        self.idle_box.setEnabled(enabled)

    def accept(self):
        ListModeDialog._last_mode = "idle" if self.rb_idle.isChecked() else "single"
        ListModeDialog._last_ocr_interval_sec = self.sp_ocr_interval.value()
        ListModeDialog._last_max_duration_min = self.sp_max_duration.value()
        super().accept()

    def get_result(self) -> dict:
        return {
            "mode": "idle" if self.rb_idle.isChecked() else "single",
            "idle_ocr_interval_sec": self.sp_ocr_interval.value(),
            "idle_max_duration_min": self.sp_max_duration.value(),
        }


# ═══════════════════════════════════════════════════════════
# 工作线程
# ═══════════════════════════════════════════════════════════

class WorkerThread(QThread):
    """后台工作线程, 执行上架/下架流程"""
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()

    def __init__(self, engine: WorkflowEngine, task: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.task = task
        self.kwargs = kwargs
        # 将 engine 的回调替换为通过信号发送, 确保 GUI 操作在主线程执行
        self.engine._on_status = self._safe_status
        self.engine._on_progress = self._safe_progress

    def _safe_status(self, msg: str):
        """线程安全的状态回调: 通过信号发送到主线程."""
        logger.info(msg)
        self.status_signal.emit(msg)

    def _safe_progress(self, current: int, total: int):
        """线程安全的进度回调: 通过信号发送到主线程."""
        self.progress_signal.emit(current, total)

    def run(self):
        try:
            if self.task == "list":
                self.engine.run_list_workflow(**self.kwargs)
            elif self.task == "delist":
                self.engine.run_delist_workflow(**self.kwargs)
            elif self.task == "debug":
                info = self.engine.capture_debug_info()
                self.status_signal.emit(f"调试信息: {info}")
        except StopIteration:
            self.status_signal.emit("操作已被用户中断")
        except Exception as e:
            import traceback
            logger.error(f"工作线程异常: {e}\n{traceback.format_exc()}")
            self.status_signal.emit(f"错误: {e}")
        finally:
            self.finished_signal.emit()


# ═══════════════════════════════════════════════════════════
# 日志处理器
# ═══════════════════════════════════════════════════════════

class _LogSignalBridge(QWidget):
    """桥接: 将后台线程的日志通过信号安全传递到主线程的 QTextEdit."""
    log_signal = pyqtSignal(str)

    def __init__(self, text_edit: QTextEdit, parent=None):
        super().__init__(parent)
        self.log_signal.connect(text_edit.append, Qt.QueuedConnection)


class QTextEditLogHandler(logging.Handler):
    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._bridge = _LogSignalBridge(text_edit)

    def emit(self, record):
        msg = self.format(record)
        self._bridge.log_signal.emit(msg)


# ═══════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.user_config = load_user_config()
        self._game_window = GameWindowInfo()
        self._detect_and_scale_resolution()

        self.engine = WorkflowEngine(
            on_status=self._on_status,
            on_progress=self._on_progress,
        )
        self.engine.auto.set_offset(self._game_window.left, self._game_window.top)
        self.engine.screen.set_game_region(self._game_window.game_region)

        self.worker: WorkerThread | None = None
        self._busy: bool = False  # 防止热键/按钮快速连点产生多个 WorkerThread
        self._auto_minimized: bool = False  # 标记当前任务是否由上架/下架触发自动最小化
        self.ml_detector = MLBoundDetector(model_path=ML_MODEL_FILE)
        self._rarity_checks: dict[str, QCheckBox] = {}
        self._box_checks: dict[str, QCheckBox] = {}
        self._category_checks: dict[str, dict[str, QCheckBox]] = {}
        self._coord_spins: dict[str, QSpinBox] = {}

        self._init_ui()
        self._load_config_to_ui()
        self._warm_up_ocr_background()

    def _detect_and_scale_resolution(self):
        """检测游戏窗口并缩放坐标 (当前已禁用, 固定 2560x1440)"""
        logger.info("分辨率检测已禁用, 使用默认 2560x1440 坐标")

    def _warm_up_ocr_background(self):
        """后台线程预热 RapidOCR，消除首次识别的冷启动延迟"""
        import threading
        from core.detector import warm_up_ocr
        t = threading.Thread(target=warm_up_ocr, daemon=True)
        t.start()

    def _init_ui(self):
        self.setWindowTitle("三角洲行动 - 交易行助手")
        self.setMinimumSize(780, 620)
        self.resize(820, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(CYBER_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        tabs = QTabWidget()
        tabs.addTab(self._create_help_tab(), "使用说明")
        tabs.addTab(self._create_filter_tab(), "筛选设置")
        tabs.addTab(self._create_coord_tab(), "坐标校准")
        tabs.addTab(self._create_log_tab(), "运行日志")
        # 作者信息放 tab 栏右上角 — 顶栏按钮太多放不下, 这里有完整横向空间
        tabs.setCornerWidget(self._create_author_corner(), Qt.TopRightCorner)
        main_layout.addWidget(tabs)

        log_handler = QTextEditLogHandler(self.log_text)
        log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logging.getLogger("workflow").addHandler(log_handler)
        logging.getLogger("workflow").setLevel(logging.INFO)

        ml_log_handler = QTextEditLogHandler(self.log_text)
        ml_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logging.getLogger("ml_detector").addHandler(ml_log_handler)
        logging.getLogger("ml_detector").setLevel(logging.INFO)

        model_status = self.ml_detector.model_info
        self.statusBar().showMessage(
            f"就绪 | 2560x1440 (固定) | {model_status} | "
            "Ctrl+1 上架  Ctrl+2 停止  Ctrl+3 下架  Ctrl+4 调试截图  Ctrl+Z 拾取坐标"
        )

    # ─── 顶部操作栏 ───

    def _create_top_bar(self) -> QWidget:
        widget = QWidget()
        # 用 objectName 限定选择器, 否则 QWidget 通配会传播到子按钮,
        # 导致 border-bottom 覆盖按钮自身边框, 视觉上"按钮底部少了一条线".
        widget.setObjectName("topBar")
        widget.setStyleSheet(
            f"QWidget#topBar {{ background-color: {BG_DARKEST}; "
            f"border-bottom: 1px solid {NEON_CYAN}30; }}"
        )
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)

        title = QLabel("DELTA FORCE TRADER")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        layout.addStretch()

        self.btn_list = QPushButton(f"开始上架 ({HOTKEYS['start_list']})")
        self.btn_list.setObjectName("btnList")
        self.btn_list.clicked.connect(self._on_start_list)
        layout.addWidget(self.btn_list)

        self.btn_stop = QPushButton(f"停止 ({HOTKEYS['stop']})")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(self.btn_stop)

        self.btn_delist = QPushButton(f"一键下架 ({HOTKEYS['start_delist']})")
        self.btn_delist.setObjectName("btnDelist")
        self.btn_delist.clicked.connect(self._on_start_delist)
        layout.addWidget(self.btn_delist)

        self.btn_debug = QPushButton("调试截图")
        self.btn_debug.clicked.connect(self._on_debug)
        layout.addWidget(self.btn_debug)

        self.btn_review = QPushButton("审核标注")
        set_btn_style(self.btn_review, NEON_PURPLE)
        self.btn_review.setToolTip("截取当前出售界面, 审核并修正道具绑定标签, 训练 ML 模型")
        self.btn_review.clicked.connect(self._on_review)
        layout.addWidget(self.btn_review)

        self.btn_collect = QPushButton("道具采集")
        set_btn_style(self.btn_collect, NEON_CYAN)
        self.btn_collect.setToolTip("实时截取游戏画面采集道具数据, 标注品质分类, 保存到数据库")
        self.btn_collect.clicked.connect(self._on_collect)
        layout.addWidget(self.btn_collect)

        return widget

    # ─── Tab 栏右上角: 作者信息 ───

    def _create_author_corner(self) -> QWidget:
        """Tab 栏右上角的作者/联系方式 corner widget.

        放在 tab 栏而不是顶栏是因为: 顶栏有 6 个功能按钮 + 标题, 再塞作者信息
        会被挤成半截. Tab 栏右侧空闲区域天然适合放辅助信息, 还能让主顶栏
        视觉更聚焦.
        """
        widget = QWidget()
        hl = QHBoxLayout(widget)
        hl.setContentsMargins(0, 0, 10, 0)
        hl.setSpacing(0)

        # 作者
        author = QLabel(
            f'<span style="color:{TEXT_SECONDARY};">作者</span> '
            f'<span style="color:{TEXT_PRIMARY}; font-weight:bold;">熊猫不只黑白</span>'
        )
        author.setTextFormat(Qt.RichText)
        hl.addWidget(author)

        _sep1 = QLabel("|")
        _sep1.setStyleSheet(
            f"color: {BORDER_GLOW}; padding: 0 8px; font-size: 11pt;"
        )
        hl.addWidget(_sep1)

        # B站 — 可点击跳转, 链接文字缩短为 "点此访问" 节省横向空间,
        # 悬停 tooltip 显示完整 URL 方便手动复制
        bili = QLabel(
            f'<span style="color:{TEXT_SECONDARY};">B站</span> '
            f'<a href="https://space.bilibili.com/13591468" '
            f'style="color:{NEON_CYAN}; text-decoration:none; font-weight:bold;">'
            f'点此访问</a>'
        )
        bili.setTextFormat(Qt.RichText)
        bili.setOpenExternalLinks(True)
        bili.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.LinksAccessibleByMouse
        )
        bili.setToolTip("https://space.bilibili.com/13591468")
        bili.setCursor(Qt.PointingHandCursor)
        hl.addWidget(bili)

        _sep2 = QLabel("|")
        _sep2.setStyleSheet(
            f"color: {BORDER_GLOW}; padding: 0 8px; font-size: 11pt;"
        )
        hl.addWidget(_sep2)

        # QQ 群 — 点击复制到剪贴板
        self._qq_group_num = "290101314"
        qq = QLabel(
            f'<span style="color:{TEXT_SECONDARY};">QQ群</span> '
            f'<span style="color:{NEON_GOLD}; font-weight:bold;">'
            f'{self._qq_group_num}</span>'
        )
        qq.setTextFormat(Qt.RichText)
        qq.setToolTip("点击复制群号")
        qq.setCursor(Qt.PointingHandCursor)
        qq.mousePressEvent = self._copy_qq_group  # type: ignore[assignment]
        hl.addWidget(qq)

        return widget

    def _copy_qq_group(self, _event):
        """点击 QQ 群号标签时复制到系统剪贴板"""
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.clipboard().setText(self._qq_group_num)
            self.statusBar().showMessage(
                f"已复制 QQ 群号 {self._qq_group_num} 到剪贴板", 3000
            )
        except Exception:
            pass

    # ─── 使用说明选项卡 ───

    def _create_help_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet(
            f"QTextEdit {{"
            f"  background-color: {BG_DARK};"
            f"  color: {TEXT_PRIMARY};"
            f"  border: none;"
            f"  font-size: 10pt;"
            f"  line-height: 1.6;"
            f"  selection-background-color: {NEON_CYAN}40;"
            f"}}"
        )
        help_text.setHtml(self._get_help_html())
        layout.addWidget(help_text)

        scroll.setWidget(content)
        return self._wrap_with_ad(scroll)

    @staticmethod
    def _get_help_html() -> str:
        return f"""
        {CYBER_HELP_CSS}

        <h2>三角洲行动 — 交易行自动上架 / 下架助手</h2>
        <p>识别仓库中的<b>非绑定道具</b>并批量上架交易行,
           支持品质 / 分类筛选, 一键下架, 多箱子轮询挂机.
           内置 ML 检测器 + 多层去重, 不会反复点同一道具.</p>
        <p style="color:{TEXT_DIM};">
            作者: 熊猫不只黑白 ·
            <a href="https://space.bilibili.com/13591468">B 站空间</a> ·
            QQ 交流群 <span class="ok">290101314</span>
        </p>

        <hr>

        <h2>一、环境要求</h2>
        <div class="section">
        <table>
            <tr><th>项目</th><th>要求</th></tr>
            <tr><td>操作系统</td><td>Windows 10 / 11 (64 位)</td></tr>
            <tr><td>Python</td><td>3.10 及以上</td></tr>
            <tr><td>显示器分辨率</td><td><span class="warn">2560 × 1440</span> (当前固定适配此分辨率)</td></tr>
            <tr><td>系统缩放</td><td>150% (推荐, 与开发环境一致)</td></tr>
            <tr><td>游戏显示模式</td><td><span class="warn">无边框窗口</span></td></tr>
            <tr><td>运行权限</td>
                <td><span class="warn">以管理员身份运行</span>
                    (必须, 否则无法监听热键和控制鼠标)</td></tr>
            <tr><td>Tesseract OCR</td>
                <td>安装 <b>tesseract-ocr-w64-setup-5.x</b> 并加入系统 PATH</td></tr>
        </table>
        </div>

        <h3>安装依赖</h3>
        <div class="section">
        <p>在项目根目录打开命令行 (管理员), 执行:</p>
        <pre>pip install -r requirements.txt</pre>
        <p>依赖清单: pyautogui, opencv-python, Pillow, keyboard, PyQt5, numpy,
           pytesseract, rapidocr, scikit-learn, joblib.</p>
        </div>

        <hr>

        <h2>二、快速开始</h2>
        <div class="section">
        <p><span class="step">第 1 步:</span> <b>右键</b> 命令提示符 / PowerShell
           → <b>以管理员身份运行</b>.</p>
        <p><span class="step">第 2 步:</span> 进入项目目录, 执行
           <code>python main.py</code>.</p>
        <p><span class="step">第 3 步:</span> 切到 <b>"筛选设置"</b> 选项卡,
           勾选要上架的<b>仓库箱子</b>和<b>道具品质 / 分类</b>
           (留空表示不限定).</p>
        <p><span class="step">第 4 步:</span> 进入游戏, 打开
           <b>交易行 → 出售</b> 界面.</p>
        <p><span class="step">第 5 步:</span> 按下热键即可开始自动操作.
           启动后软件会<b>自动最小化</b>, 完成后自动恢复显示.</p>
        </div>

        <hr>

        <h2>三、热键说明</h2>
        <div class="section">
        <table>
            <tr><th>热键</th><th>功能</th><th>说明</th></tr>
            <tr>
                <td><span class="key">Ctrl + 1</span></td>
                <td>开始上架</td>
                <td>识别非绑定道具并逐个上架, 支持单次和挂机两种模式</td>
            </tr>
            <tr>
                <td><span class="key">Ctrl + 2</span></td>
                <td>紧急停止</td>
                <td>立即中断当前上架 / 下架操作, 任何阶段均可按</td>
            </tr>
            <tr>
                <td><span class="key">Ctrl + 3</span></td>
                <td>一键下架</td>
                <td>自动下架所有已上架道具并整理仓库</td>
            </tr>
            <tr>
                <td><span class="key">Ctrl + 4</span></td>
                <td>调试截图</td>
                <td>截取当前屏幕, 运行道具检测分析并保存调试图片</td>
            </tr>
            <tr>
                <td><span class="key">Ctrl + Z</span></td>
                <td>截图拾取坐标</td>
                <td>软件最小化后截取游戏画面, 可点取或拖框拾取坐标</td>
            </tr>
        </table>
        </div>

        <hr>

        <h2>四、执行模式</h2>
        <div class="section">
        <p>按下 <span class="key">Ctrl + 1</span> 后会弹出模式选择对话框:</p>
        <table>
            <tr><th>模式</th><th>行为</th></tr>
            <tr>
                <td><b>单次上架</b></td>
                <td>遍历勾选的所有箱子各一次, 上完即停</td>
            </tr>
            <tr>
                <td><b>挂机模式</b></td>
                <td>持续轮询: 上架满 15 槽 → 等待若干分钟 → 自动下架已成交记录
                    → 整理仓库 → 重新开始. 适合放着挂机收菜</td>
            </tr>
        </table>
        <p>挂机模式的等待时长可在对话框右侧 <b>"挂机模式设置"</b> 内修改,
           上限循环次数也可指定.</p>
        </div>

        <hr>

        <h2>五、上架流程 (Ctrl+1)</h2>
        <div class="section">
        <ol>
            <li><b>读取上架状态</b> — OCR 识别 "已上架 / 总槽位"
                (如 "5/15"), 计算剩余可用槽位.</li>
            <li><b>整理仓库</b> — 自动点击 "整理仓库" → "整理",
                让道具有序排列.</li>
            <li><b>遍历箱子</b> — 按 "筛选设置" 中勾选的顺序依次切换箱子,
                每个箱子的坐标可在 "坐标校准" 内独立拾取.</li>
            <li><b>扫描道具</b> — 对当前页面 9 列道具网格做截图分析,
                ML 模型区分非绑定 / 绑定.</li>
            <li><b>连通域合并</b> — 把相邻的非绑定格子合并为<b>逻辑道具</b>
                (一把裸枪可能占 4×2 共 8 格 — 合并后只点 1 次, 不再反复点击
                同一把枪的不同部位).</li>
            <li><b>点击上架</b> — 进入弹窗后:
                <ul>
                    <li>自动拉满出售数量 (点击滑动条最右端)</li>
                    <li>OCR 识别道具名称、预期收入、占用槽位</li>
                    <li>命中筛选 → 点击 "上架" 确认; 不符筛选 → ESC 跳过,
                        并把道具<b>名字 / 图像指纹 / 位置</b>三层加入黑名单,
                        本箱内不再点开</li>
                </ul>
            </li>
            <li><b>翻页继续</b> — 当前页扫完后自动向下滚动. 翻页底部用
                <b>双重确认</b> 检测 (避免界面动画导致误判, 漏扫早期页面).</li>
            <li><b>收入统计</b> — 完成后在 "运行日志" 中输出上架明细 + 总预期
                收入 (已扣手续费和保证金).</li>
        </ol>
        <p><span class="warn">注意:</span> 交易行槽位上限 <b>15 个</b>,
           达到上限后自动停止.</p>
        </div>

        <hr>

        <h2>六、下架流程 (Ctrl+3)</h2>
        <div class="section">
        <ol>
            <li>OCR 读取当前已上架数量, 决定下架次数.</li>
            <li>反复点击列表第一项的<b>下架按钮</b> → 确认下架,
                列表会自动上移, 不需要切换位置.</li>
            <li>下架完成后自动整理仓库.</li>
        </ol>
        <p>仓库空间不足时会自动检测并停止下架, 避免道具丢失.</p>
        </div>

        <hr>

        <h2>七、筛选设置</h2>
        <div class="section">
        <p>在 <b>"筛选设置"</b> 选项卡内可指定:</p>
        <ul>
            <li><b>仓库箱子</b> — 勾选哪些箱子参与本轮上架</li>
            <li><b>品质颜色</b> — 红 / 金 / 紫 / 蓝 / 绿 / 白</li>
            <li><b>道具分类</b> — 枪械、配件、弹药、防具、消耗品 ... 多级分类</li>
        </ul>
        <p>需要先通过 <b>"道具采集"</b> 建立道具数据库, 软件才能根据 OCR
           读到的道具名匹配品质 / 分类 (见第九节).</p>
        <p>未在数据库中的道具<b>默认允许上架</b>, 不会阻断流程.</p>
        </div>

        <hr>

        <h2>八、坐标校准</h2>
        <div class="section">
        <p>如果自动操作出现点击偏移、OCR 区域错位, 需要重新校准坐标.</p>
        <ol>
            <li>切到 <b>"坐标校准"</b> 选项卡, 顶部点
                <b>"截图拾取坐标"</b> (或按 <span class="key">Ctrl+Z</span>).</li>
            <li>软件自动最小化截取当前游戏画面.</li>
            <li>在下拉框选择目标. 目标分两种:
                <ul>
                    <li><b>[点]</b> — 单击截图上的对应位置 (如按钮中心)</li>
                    <li><b>[区域]</b> — 按住左键<b>拖一个矩形框</b>
                        (如 OCR 识别区)</li>
                </ul>
            </li>
            <li>拾取完成后点 <b>"完成"</b>, 数值会回填到对应输入框.</li>
            <li>点 <b>"应用所有坐标修改 (永久保存)"</b> 持久化到
                <code>user_config.json</code>.</li>
        </ol>
        <p>可校准的目标 (覆盖所有点击点和 OCR 识别区, 共 30+ 项):</p>
        <table>
            <tr><th>分组</th><th>包含项</th></tr>
            <tr><td>道具网格</td>
                <td>左上角、右下角、滚动停鼠点</td></tr>
            <tr><td>上架弹窗按钮</td>
                <td>数量 −/+、滑动条最右端、上架确认</td></tr>
            <tr><td>下架相关</td>
                <td>下架按钮、确认按钮</td></tr>
            <tr><td>顶部标签页</td>
                <td>交易行 tab、出售子 tab</td></tr>
            <tr><td>仓库整理</td>
                <td>整理仓库图标、整理按钮</td></tr>
            <tr><td>仓库箱子</td>
                <td>箱子 1 ~ 10 的 Y 坐标 (X 共用)</td></tr>
            <tr><td>OCR / 检测区域</td>
                <td>道具名称、预期收入、上架数量、上架弹窗、
                    页面跳转检测 ROI、出售页 tab ROI</td></tr>
        </table>
        <p>点击 <b>"撤销修改"</b> 可恢复到上次永久保存的坐标值.</p>
        </div>

        <hr>

        <h2>九、ML 模型与审核标注</h2>
        <div class="section">
        <p>ML 模型 (<code>ml_data/bound_model.joblib</code>)
           区分<b>绑定 / 非绑定</b>道具.
           初次使用或识别不准确时可微调:</p>
        <ol>
            <li>在游戏出售界面, 点击主界面上的 <b>"审核标注"</b> 按钮.</li>
            <li>软件截取当前画面, 自动标注每个格子的绑定状态.</li>
            <li>逐个检查, 把错误的标注手动修正.</li>
            <li>点击 <b>"训练模型"</b>, 用累积的标注数据重新训练.</li>
        </ol>
        <p>样本越多准确率越高, 建议首次使用时多做几轮审核
           (覆盖不同种类的枪、配件、弹药、绑定箱).</p>
        </div>

        <hr>

        <h2>十、道具数据采集</h2>
        <div class="section">
        <p><b>"道具采集"</b> 功能用于建立道具数据库, 给筛选设置提供依据.</p>
        <h3>方式一: 实时截取 (推荐)</h3>
        <ol>
            <li>游戏内打开交易行<b>购买</b>页面, 导航到要采集的分类.</li>
            <li>点击主界面 <b>"道具采集"</b> 按钮打开采集对话框.</li>
            <li>顶部下拉框选择 <b>大类</b> 和 <b>子类</b>.</li>
            <li>点击 <b>"截取游戏画面"</b>: 对话框自动隐藏 → 截取全屏 →
                恢复 → OCR 提取道具名.</li>
            <li>点击 <b>"截取并翻页"</b>: 截取后自动滚动商店页面, 方便连续
                采集多页.</li>
            <li>右侧表格中检查结果, 标注品质颜色, 点 <b>"保存数据库"</b>.</li>
        </ol>
        <h3>方式二: 本地截图</h3>
        <ol>
            <li>截图保存到 <b>Shop/大类/子类/</b> 文件夹
                (如 <code>Shop/枪械/步枪/截图.png</code>).</li>
            <li>左侧目录树选择子类加载, 或点 <b>"批量提取全部"</b>.</li>
        </ol>
        </div>

        <hr>

        <h2>十一、常见问题</h2>
        <div class="section">
        <table>
            <tr><th>问题</th><th>解决方案</th></tr>
            <tr>
                <td>按热键无反应</td>
                <td>确认是否以<b>管理员权限</b>运行</td>
            </tr>
            <tr>
                <td>点击位置偏移</td>
                <td>用 <b>"坐标校准"</b> 重新拾取相关坐标</td>
            </tr>
            <tr>
                <td>反复点同一把枪 / 道具</td>
                <td>多格大型道具 (枪) 占多个格子.
                    本版本已用<b>连通域合并 + 名字 / 图像指纹去重</b>解决,
                    若仍出现请确保 ML 模型已最新训练 (审核标注)</td>
            </tr>
            <tr>
                <td>翻页没翻到底就跳箱子</td>
                <td>本版本已用<b>双重底部确认 + 较大滚动幅度</b>解决.
                    如仍漏扫, 在 "坐标校准" 检查 "page_change_*" 区域</td>
            </tr>
            <tr>
                <td>上架数量 OCR 识别错误</td>
                <td>检查 <code>debug_listing_count.png</code>;
                    在 "坐标校准" 微调 <b>counter_x1~y2</b> 区域</td>
            </tr>
            <tr>
                <td>非绑定道具识别不准</td>
                <td>用 <b>"审核标注"</b> 修正错误样本并重新训练模型</td>
            </tr>
            <tr>
                <td>软件崩溃 (0xC0000005)</td>
                <td>通常由截图冲突引起, 关闭其他截图 / 录屏软件后重试</td>
            </tr>
            <tr>
                <td>分辨率不是 2560×1440</td>
                <td>当前版本仅适配 2560×1440 + 150% 缩放, 其他分辨率暂不支持</td>
            </tr>
            <tr>
                <td>提示 "No module named ..."</td>
                <td>执行 <code>pip install -r requirements.txt</code></td>
            </tr>
            <tr>
                <td>提示 "tesseract is not installed"</td>
                <td>安装 Tesseract-OCR 并加入系统 PATH</td>
            </tr>
        </table>
        </div>

        <hr>

        <h2>十二、文件说明</h2>
        <div class="section">
        <table>
            <tr><th>文件 / 目录</th><th>说明</th></tr>
            <tr><td>main.py</td><td>程序入口</td></tr>
            <tr><td>config.py</td><td>坐标 / 分类 / 时间参数等配置</td></tr>
            <tr><td>user_config.json</td>
                <td>用户自定义配置 (坐标、筛选), 自动生成</td></tr>
            <tr><td>core/</td><td>核心逻辑 (截图、检测、自动化、工作流)</td></tr>
            <tr><td>gui/</td><td>GUI 界面</td></tr>
            <tr><td>logs/</td><td>运行日志 (自动生成)</td></tr>
            <tr><td>listings/</td><td>上架弹窗截图记录 (自动生成)</td></tr>
            <tr><td>ml_data/</td><td>ML 训练数据和模型文件</td></tr>
            <tr><td>requirements.txt</td><td>Python 依赖清单</td></tr>
        </table>
        </div>

        <hr>
        <p style="color:{TEXT_DIM}; text-align:center; margin-top:20px;">
            DELTA FORCE TRADER —
            遇到问题请先查看 "运行日志" 选项卡 ·
            建议反馈进 QQ 群 <span class="ok">290101314</span>
        </p>
        """

    # ─── 筛选设置选项卡 ───

    def _create_filter_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setSpacing(12)

        left_col = QVBoxLayout()

        box_group = QGroupBox("仓库箱子")
        box_layout = QVBoxLayout(box_group)
        btn_all_boxes = QPushButton("全选/取消")
        btn_all_boxes.clicked.connect(lambda: self._toggle_all(self._box_checks))
        box_layout.addWidget(btn_all_boxes)
        for box_name in STORAGE_BOXES:
            cb = QCheckBox(box_name)
            self._box_checks[box_name] = cb
            box_layout.addWidget(cb)
        left_col.addWidget(box_group)

        rarity_group = QGroupBox("品质颜色")
        rarity_layout = QVBoxLayout(rarity_group)
        for rarity in Rarity:
            cb = QCheckBox(rarity.value)
            color = RARITY_COLORS_HEX.get(rarity.value, "#cccccc")
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
            self._rarity_checks[rarity.value] = cb
            rarity_layout.addWidget(cb)
        left_col.addWidget(rarity_group)
        left_col.addStretch()
        layout.addLayout(left_col)

        right_col = QVBoxLayout()
        for cat_name, subcats in ITEM_CATEGORIES.items():
            group = QGroupBox(cat_name)
            g_layout = QVBoxLayout(group)
            cat_checks = {}
            btn_row = QHBoxLayout()
            btn_all = QPushButton("全选")
            btn_none = QPushButton("取消")
            btn_all.setFixedHeight(24)
            btn_none.setFixedHeight(24)
            btn_row.addWidget(btn_all)
            btn_row.addWidget(btn_none)
            btn_row.addStretch()
            g_layout.addLayout(btn_row)

            grid = QGridLayout()
            grid.setSpacing(4)
            for i, sub in enumerate(subcats):
                cb = QCheckBox(sub)
                cat_checks[sub] = cb
                grid.addWidget(cb, i // 3, i % 3)
            g_layout.addLayout(grid)
            self._category_checks[cat_name] = cat_checks
            right_col.addWidget(group)
            btn_all.clicked.connect(lambda _, cc=cat_checks: self._set_all(cc, True))
            btn_none.clicked.connect(lambda _, cc=cat_checks: self._set_all(cc, False))

        right_col.addStretch()
        layout.addLayout(right_col)
        scroll.setWidget(content)
        return self._wrap_with_ad(scroll)

    # ─── 坐标校准选项卡 ───

    # ───────────── 辅助: 创建 SpinBox ─────────────
    def _make_spin(self, key: str, default_val: int, max_v: int = 2560) -> QSpinBox:
        """创建一个 SpinBox 并登记到 self._coord_spins[key]."""
        sp = QSpinBox()
        sp.setRange(0, max_v)
        sp.setValue(int(default_val))
        self._coord_spins[key] = sp
        return sp

    def _add_point_row(self, grid: QGridLayout, row: int, label: str,
                       x_key: str, y_key: str,
                       x_val: int, y_val: int):
        """在 2 列网格里加一行 "X:spin  Y:spin"."""
        grid.addWidget(QLabel(f"{label}  X"), row, 0)
        grid.addWidget(self._make_spin(x_key, x_val, 2560), row, 1)
        grid.addWidget(QLabel("Y"), row, 2)
        grid.addWidget(self._make_spin(y_key, y_val, 1440), row, 3)

    def _add_region_row(self, grid: QGridLayout, row: int, label: str,
                        keys: tuple[str, str, str, str],
                        vals: tuple[int, int, int, int]):
        """在 4 列网格里加一行 "X1 Y1 X2 Y2" (紧凑布局)."""
        x1k, y1k, x2k, y2k = keys
        x1, y1, x2, y2 = vals
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(QLabel("X1"), row, 1)
        grid.addWidget(self._make_spin(x1k, x1, 2560), row, 2)
        grid.addWidget(QLabel("Y1"), row, 3)
        grid.addWidget(self._make_spin(y1k, y1, 1440), row, 4)
        grid.addWidget(QLabel("X2"), row, 5)
        grid.addWidget(self._make_spin(x2k, x2, 2560), row, 6)
        grid.addWidget(QLabel("Y2"), row, 7)
        grid.addWidget(self._make_spin(y2k, y2, 1440), row, 8)

    def _create_coord_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- 分辨率提示 ---
        info = QLabel(
            f'<span style="color:{TEXT_SECONDARY};">'
            f'当前版本固定适配 2560x1440 分辨率. 点击"截图拾取坐标"后, '
            f'从下拉框选择目标: <b>[点]</b> 类型=单击中心, '
            f'<b>[区域]</b> 类型=按住左键拖一个框. 修改后按"应用"永久保存.'
            f'</span>'
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- 截图拾取按钮 ---
        pick_row = QHBoxLayout()
        btn_pick = QPushButton("截图拾取坐标")
        set_btn_style(btn_pick, NEON_CYAN)
        btn_pick.setToolTip(
            "截取当前屏幕, 在图上点击(点)或拖框(区域)拾取坐标, 自动填入下方输入框"
        )
        btn_pick.clicked.connect(self._open_coord_picker)
        pick_row.addWidget(btn_pick)
        pick_row.addWidget(QLabel(
            "  流程: 先在游戏中打开相应界面 → 点此按钮截图 → 在图上拾取"
        ))
        pick_row.addStretch()
        layout.addLayout(pick_row)

        # ═════════════════════════════════════════════════════
        # 一、点击坐标
        # ═════════════════════════════════════════════════════

        # --- 道具网格 ---
        grid_group = QGroupBox("道具网格 (出售界面右侧)")
        g = QGridLayout(grid_group)
        self._add_point_row(g, 0, "网格左上",
            "grid_x_start", "grid_y_start",
            ITEM_GRID["x_start"], ITEM_GRID["y_start"])
        self._add_point_row(g, 1, "网格右下",
            "grid_x_end", "grid_y_end",
            ITEM_GRID["x_end"], ITEM_GRID["y_end"])
        self._add_point_row(g, 2, "滚动停鼠点",
            "scroll_area_x", "scroll_area_y",
            ITEM_GRID.get("scroll_area_x", 2060),
            ITEM_GRID.get("scroll_area_y", 700))
        layout.addWidget(grid_group)

        # --- 上架弹窗按钮 ---
        dialog_group = QGroupBox("上架弹窗按钮 (点击道具后的界面)")
        d = QGridLayout(dialog_group)
        self._add_point_row(d, 0, "数量 −",
            "qty_minus_x", "qty_minus_y",
            LIST_DIALOG["qty_minus_btn"][0], LIST_DIALOG["qty_minus_btn"][1])
        self._add_point_row(d, 1, "数量 ＋",
            "qty_plus_x", "qty_plus_y",
            LIST_DIALOG["qty_plus_btn"][0], LIST_DIALOG["qty_plus_btn"][1])
        self._add_point_row(d, 2, "滑块左端",
            "slider_left_x", "slider_left_y",
            LIST_DIALOG["qty_slider_left"][0], LIST_DIALOG["qty_slider_left"][1])
        self._add_point_row(d, 3, "滑块右端",
            "slider_right_x", "slider_right_y",
            LIST_DIALOG["qty_slider_right"][0], LIST_DIALOG["qty_slider_right"][1])
        self._add_point_row(d, 4, "滑块最右 (拉满)",
            "qty_max_x", "qty_max_y",
            LIST_DIALOG["qty_slider_max"][0], LIST_DIALOG["qty_slider_max"][1])
        self._add_point_row(d, 5, "价格输入",
            "price_input_x", "price_input_y",
            LIST_DIALOG["price_input"][0], LIST_DIALOG["price_input"][1])
        self._add_point_row(d, 6, "价格 −",
            "price_minus_x", "price_minus_y",
            LIST_DIALOG["price_minus"][0], LIST_DIALOG["price_minus"][1])
        self._add_point_row(d, 7, "价格 ＋",
            "price_plus_x", "price_plus_y",
            LIST_DIALOG["price_plus"][0], LIST_DIALOG["price_plus"][1])
        self._add_point_row(d, 8, "上架确认",
            "list_btn_x", "list_btn_y",
            LIST_DIALOG["list_btn"][0], LIST_DIALOG["list_btn"][1])
        self._add_point_row(d, 9, "返回/ESC",
            "esc_btn_x", "esc_btn_y",
            LIST_DIALOG["esc_btn"][0], LIST_DIALOG["esc_btn"][1])
        layout.addWidget(dialog_group)

        # --- 下架相关 ---
        delist_group = QGroupBox("下架相关")
        dl = QGridLayout(delist_group)
        self._add_point_row(dl, 0, "下架按钮",
            "delist_btn_x", "delist_btn_y",
            LISTED_ITEMS["delist_btn_x"], LISTED_ITEMS["delist_btn_y"])
        self._add_point_row(dl, 1, "下架确认按钮",
            "confirm_btn_x", "confirm_btn_y",
            LISTED_ITEMS["confirm_btn_x"], LISTED_ITEMS["confirm_btn_y"])
        layout.addWidget(delist_group)

        # --- 顶部 tab ---
        tab_group = QGroupBox("顶部标签页")
        tg = QGridLayout(tab_group)
        trade_xy = TAB_COORDS.get("交易行", (950, 42))
        sell_xy = TAB_COORDS.get("出售", (430, 82))
        self._add_point_row(tg, 0, "交易行 tab",
            "tab_trade_x", "tab_trade_y",
            trade_xy[0], trade_xy[1])
        self._add_point_row(tg, 1, "出售 tab",
            "tab_sell_x", "tab_sell_y",
            sell_xy[0], sell_xy[1])
        layout.addWidget(tab_group)

        # --- 整理仓库 ---
        org_group = QGroupBox("仓库整理")
        og = QGridLayout(org_group)
        self._add_point_row(og, 0, "整理仓库图标",
            "org_x", "org_y",
            ORGANIZE_BTN["icon_x"], ORGANIZE_BTN["icon_y"])
        self._add_point_row(og, 1, "整理按钮 (右下角)",
            "sort_x", "sort_y",
            ORGANIZE_BTN["sort_btn_x"], ORGANIZE_BTN["sort_btn_y"])
        layout.addWidget(org_group)

        # --- 箱子选择器 ---
        box_group = QGroupBox(
            f"箱子选择器 (共 {MAX_STORAGE_BOXES} 个, "
            f"第 1 个为主仓库, 每个箱子独立校准 Y 坐标)"
        )
        b = QGridLayout(box_group)
        b.addWidget(QLabel("箱子图标 X (共用)"), 0, 0)
        b.addWidget(self._make_spin("box_x", BOX_SELECTOR["x"], 2560), 0, 1, 1, 3)
        positions = BOX_SELECTOR.get("positions", [])
        for idx in range(MAX_STORAGE_BOXES):
            row = 1 + idx // 2
            col = (idx % 2) * 2
            key = f"box{idx + 1}_y"
            label = "主仓库 Y" if idx == 0 else f"箱子{idx + 1} Y"
            b.addWidget(QLabel(label), row, col)
            default_y = (positions[idx] if idx < len(positions)
                         else ORIGINAL_DEFAULT_COORDS.get(key, 118 + idx * 45))
            b.addWidget(self._make_spin(key, default_y, 1440), row, col + 1)
        layout.addWidget(box_group)

        # ═════════════════════════════════════════════════════
        # 二、OCR / 检测区域 (矩形, 8 列紧凑布局)
        # ═════════════════════════════════════════════════════
        region_group = QGroupBox("OCR / 检测区域 (x1, y1, x2, y2) — 建议用拖框拾取")
        rg = QGridLayout(region_group)
        # 表头
        rg.addWidget(QLabel("<b>区域</b>"), 0, 0)
        for col, hdr in enumerate(["X1", "", "Y1", "", "X2", "", "Y2"], start=1):
            if hdr:
                rg.addWidget(QLabel(f"<b>{hdr}</b>"), 0, col + (col - 1) // 2)

        self._add_region_row(rg, 1, "道具名称",
            ("name_x1", "name_y1", "name_x2", "name_y2"),
            LIST_DIALOG["item_name_region"])
        self._add_region_row(rg, 2, "预期收入",
            ("income_x1", "income_y1", "income_x2", "income_y2"),
            LIST_DIALOG["income_region"])
        self._add_region_row(rg, 3, "弹窗面板",
            ("dialog_x1", "dialog_y1", "dialog_x2", "dialog_y2"),
            LIST_DIALOG["dialog_crop"])
        self._add_region_row(rg, 4, "槽位计数",
            ("counter_x1", "counter_y1", "counter_x2", "counter_y2"),
            LISTING_SLOTS["counter_region"])
        self._add_region_row(rg, 5, "页面跳转",
            ("page_change_x1", "page_change_y1",
             "page_change_x2", "page_change_y2"),
            DETECTION_ROI["page_change"])
        self._add_region_row(rg, 6, "出售 tab 高亮",
            ("sell_tab_x1", "sell_tab_y1", "sell_tab_x2", "sell_tab_y2"),
            DETECTION_ROI["sell_tab"])
        layout.addWidget(region_group)

        # --- 应用 + 重置按钮 ---
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("应用所有坐标修改 (永久保存)")
        set_btn_style(btn_apply, NEON_GREEN)
        btn_apply.clicked.connect(self._apply_coord_changes)
        btn_row.addWidget(btn_apply)

        btn_reset = QPushButton("撤销修改 (恢复已保存坐标)")
        btn_reset.clicked.connect(self._reset_coords_to_default)
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(widget)
        return self._wrap_with_ad(scroll)

    # ─── 运行日志选项卡 ───

    def _create_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("=== 三角洲行动 交易行助手 ===")
        self.log_text.append("准备就绪, 请配置筛选条件后按 Ctrl+1 开始上架")
        self.log_text.append("")
        layout.addWidget(self.log_text, stretch=1)

        # 日志下方的"广告招租"栏目, 固定高度, 不挤占日志显示区
        layout.addWidget(self._create_ad_panel())

        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(self.log_text.clear)
        layout.addWidget(btn_clear)
        return widget

    def _wrap_with_ad(self, content: QWidget) -> QWidget:
        """把一个内容控件 (通常是 QScrollArea) 与广告位上下堆叠.

        广告位固定在底部, 不会随 scroll 内容滚走. 用于所有 tab 共享同一个
        广告位视觉位置. 每次调用都新建独立的广告位实例 (Qt 的 widget 不能
        跨 parent 复用).
        """
        wrapper = QWidget()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(content, stretch=1)
        lay.addWidget(self._create_ad_panel())
        return wrapper

    def _create_ad_panel(self) -> QWidget:
        """日志栏下方的广告招租面板. 金色霓虹风, 固定高度 ~80px."""
        box = QGroupBox("✦ 广告招租 ✦")
        box.setObjectName("adPanel")
        box.setStyleSheet(
            f"QGroupBox#adPanel {{"
            f"  border: 1px solid {NEON_GOLD}; "
            f"  border-radius: 6px; "
            f"  margin-top: 10px; "
            f"  background-color: {BG_DARKEST}; "
            f"}}"
            f"QGroupBox#adPanel::title {{"
            f"  subcontrol-origin: margin; "
            f"  left: 12px; padding: 0 6px; "
            f"  color: {NEON_GOLD}; font-weight: bold; "
            f"}}"
        )
        box.setFixedHeight(80)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(14, 6, 14, 8)
        inner.setSpacing(2)

        slogan = QLabel("此处广告位招租 · Your Ad Here")
        slogan.setStyleSheet(
            f"color: {NEON_GOLD}; font-size: 12pt; font-weight: bold;"
        )
        slogan.setAlignment(Qt.AlignCenter)
        inner.addWidget(slogan)

        contact = QLabel(
            f'<span style="color:{TEXT_SECONDARY};">推广合作请联系作者 · </span>'
            f'<a href="https://space.bilibili.com/13591468" '
            f'  style="color:{NEON_CYAN}; text-decoration:none;">B 站 @熊猫不只黑白</a>'
            f'<span style="color:{TEXT_SECONDARY};">  ·  QQ 交流群 </span>'
            f'<span style="color:{NEON_GREEN}; font-weight:bold;">290101314</span>'
        )
        contact.setOpenExternalLinks(True)
        contact.setAlignment(Qt.AlignCenter)
        contact.setTextInteractionFlags(Qt.TextBrowserInteraction)
        inner.addWidget(contact)

        return box

    # ═══════════════════════════════════════════════════════
    # 坐标拾取器
    # ═══════════════════════════════════════════════════════

    @pyqtSlot()
    def _open_coord_picker(self):
        """最小化软件窗口 → 截取游戏画面 → 打开坐标拾取对话框"""
        if self._is_worker_active():
            QMessageBox.information(self, "任务运行中",
                                    "请先停止当前上架/下架任务后再拾取坐标")
            return
        self.statusBar().showMessage("正在截图 (1秒后截取游戏画面)...")
        self.showMinimized()
        QTimer.singleShot(800, self._do_coord_picker_grab)

    def _do_coord_picker_grab(self):
        """实际执行截图和弹出拾取器"""
        try:
            full = self.engine.screen.grab_full()
        except Exception as e:
            self.showNormal()
            self.activateWindow()
            QMessageBox.warning(self, "截图失败", f"无法截取屏幕: {e}")
            return

        self.showNormal()
        self.activateWindow()

        dialog = CoordPickerDialog(full, self)
        dialog.coord_picked.connect(self._on_coord_picked)
        self.statusBar().showMessage("坐标拾取中... 选择目标后在图上点击")
        dialog.exec_()
        self.statusBar().showMessage("坐标拾取完成")

    def _on_coord_picked(self, keys: tuple, values: tuple):
        """拾取器回调: 将拾取到的坐标/区域写入对应的 SpinBox.

        keys 和 values 是一一对应的等长元组:
          - point:  keys=(x_key, y_key), values=(x, y)
          - region: keys=(x1, y1, x2, y2), values=(x1, y1, x2, y2)
        """
        for key, val in zip(keys, values):
            if key in self._coord_spins:
                self._coord_spins[key].setValue(int(val))

    # ═══════════════════════════════════════════════════════
    # 坐标持久化
    # ═══════════════════════════════════════════════════════

    def _collect_coord_values(self) -> dict:
        """从所有 SpinBox 收集坐标值"""
        return {key: sp.value() for key, sp in self._coord_spins.items()}

    def _apply_coords_to_globals(self):
        """将当前 SpinBox 值同步到全局配置字典 (运行时生效)"""
        apply_saved_coordinates(self._collect_coord_values())

    def _apply_coord_changes(self):
        """应用坐标修改: 同步到全局字典 + 永久保存到 user_config.json"""
        self._apply_coords_to_globals()

        self.user_config["coordinates"] = self._collect_coord_values()
        self.user_config["coordinates_resolution"] = [
            self._game_window.width, self._game_window.height,
        ]
        save_user_config(self.user_config)

        self.log_text.append(
            f"坐标已保存 ({self._game_window.width}x{self._game_window.height}) | "
            f"网格: ({ITEM_GRID['x_start']},{ITEM_GRID['y_start']})->"
            f"({ITEM_GRID['x_end']},{ITEM_GRID['y_end']}) | "
            f"上架按钮: {LIST_DIALOG['list_btn']} | 数量+: {LIST_DIALOG['qty_plus_btn']}"
        )
        self.statusBar().showMessage("坐标已永久保存到 user_config.json")

    def _reset_coords_to_default(self):
        """撤销未保存的修改, 恢复到上次永久保存的坐标"""
        saved_coords = self.user_config.get("coordinates", {})

        if saved_coords:
            for key, val in saved_coords.items():
                if key in self._coord_spins:
                    self._coord_spins[key].setValue(val)
            apply_saved_coordinates(saved_coords)
            self.log_text.append("坐标已恢复为上次保存的值")
            self.statusBar().showMessage("坐标已恢复为上次保存的值")
        else:
            for key, val in ORIGINAL_DEFAULT_COORDS.items():
                if key in self._coord_spins:
                    self._coord_spins[key].setValue(val)
            apply_saved_coordinates(dict(ORIGINAL_DEFAULT_COORDS))
            self.log_text.append("无已保存坐标, 已恢复为初始默认值")
            self.statusBar().showMessage("无已保存坐标, 已恢复为初始默认值")

    # ═══════════════════════════════════════════════════════
    # 事件处理
    # ═══════════════════════════════════════════════════════

    def _is_worker_active(self) -> bool:
        """worker 是否正在运行 (或尚未被回收)."""
        if self._busy:
            return True
        return bool(self.worker and self.worker.isRunning())

    def _start_worker(self, task: str, **kwargs):
        """
        统一启动 WorkerThread, 处理 busy 标记、signal 连接和资源回收.
        调用者需确保已通过 _is_worker_active 检查过.
        """
        self._busy = True
        self._set_running(True)

        worker = WorkerThread(self.engine, task, **kwargs)
        worker.status_signal.connect(self._on_status)
        worker.progress_signal.connect(self._on_progress)
        worker.finished_signal.connect(self._on_finished)
        # 线程退出后自动回收 Qt 对象, 避免信号/槽链条在内存中堆积
        worker.finished.connect(worker.deleteLater)
        self.worker = worker
        worker.start()

    @pyqtSlot()
    def _on_start_list(self):
        if self._is_worker_active():
            self.statusBar().showMessage("已有任务在运行, 请先停止")
            return

        # 先让用户选择模式 (单次/挂机) 和挂机参数
        mode_dlg = ListModeDialog(self)
        if mode_dlg.exec_() != QDialog.Accepted:
            self.statusBar().showMessage("已取消")
            return
        mode_cfg = mode_dlg.get_result()

        self._save_config_from_ui()

        selected_boxes = [
            name for name, cb in self._box_checks.items() if cb.isChecked()
        ]
        if not selected_boxes:
            selected_boxes = list(STORAGE_BOXES)

        selected_rarities = [
            r for r in Rarity
            if r.value in self._rarity_checks and self._rarity_checks[r.value].isChecked()
        ]
        if not selected_rarities:
            selected_rarities = list(Rarity)

        allowed_rarities: set[str] = set()
        allowed_categories: dict[str, set[str]] = {}
        for cat, checks in self._category_checks.items():
            selected_subs = {sub for sub, cb in checks.items() if cb.isChecked()}
            if selected_subs:
                allowed_categories[cat] = selected_subs

        for name, cb in self._rarity_checks.items():
            if cb.isChecked():
                allowed_rarities.add(name)

        rarity_desc = ", ".join(allowed_rarities) if allowed_rarities else "全部"
        cat_desc = ", ".join(allowed_categories.keys()) if allowed_categories else "全部"
        if mode_cfg["mode"] == "idle":
            mode_desc = (
                f"挂机 (槽位满则 {mode_cfg['idle_ocr_interval_sec']}s 重检, "
                f"上限 {mode_cfg['idle_max_duration_min'] or '不限'}min)"
            )
        else:
            mode_desc = "单次"
        self.log_text.append(
            f"\n--- 开始上架 [{mode_desc}] | 箱子: {len(selected_boxes)}个 | "
            f"品质: {rarity_desc} | 分类: {cat_desc} ---"
        )

        self._start_worker(
            "list",
            selected_boxes=selected_boxes,
            selected_rarities=selected_rarities,
            max_slots=15,
            allowed_rarities=allowed_rarities if allowed_rarities else None,
            allowed_categories=allowed_categories if allowed_categories else None,
            mode=mode_cfg["mode"],
            idle_ocr_interval_sec=mode_cfg["idle_ocr_interval_sec"],
            idle_max_duration_min=mode_cfg["idle_max_duration_min"],
        )
        self._minimize_for_task()

    @pyqtSlot()
    def _on_start_delist(self):
        if self._is_worker_active():
            self.statusBar().showMessage("已有任务在运行, 请先停止")
            return

        self.log_text.append("\n--- 开始下架 ---")
        self._start_worker("delist", delist_all=True)
        self._minimize_for_task()

    def _minimize_for_task(self):
        """任务启动后把主窗口最小化, 让游戏窗口占据前景.

        使用 QTimer 延后到下一轮事件循环, 避免打断当前点击信号的派发.
        同时打上 _auto_minimized 标记, 任务结束时会自动还原窗口.
        """
        self._auto_minimized = True
        QTimer.singleShot(120, self.showMinimized)

    def _restore_after_task(self):
        """任务结束时恢复主窗口 (若之前由本软件自动最小化).

        用户手动最小化/关闭的情况不做处理, 以免抢焦点.
        """
        if not self._auto_minimized:
            return
        self._auto_minimized = False
        # showNormal 会取消最小化但保持原先是否最大化的状态
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @pyqtSlot()
    def _on_stop(self):
        self.engine.stop()
        self.log_text.append(">>> 用户请求停止 <<<")

    @pyqtSlot()
    def _on_debug(self):
        if self._is_worker_active():
            self.statusBar().showMessage("已有任务在运行, 请先停止")
            return

        self.log_text.append("\n--- 调试截图 ---")
        self._start_worker("debug")

    @pyqtSlot()
    def _on_review(self):
        """打开审核标注对话框."""
        if self._is_worker_active():
            QMessageBox.information(self, "任务运行中",
                                    "请先停止当前上架/下架任务后再打开审核标注")
            return
        self.statusBar().showMessage("正在截图...")
        try:
            screenshot = self.engine.screen.grab_full()
        except Exception as e:
            QMessageBox.warning(self, "截图失败", f"无法截取屏幕: {e}")
            return

        from gui.review_dialog import ReviewDialog
        dialog = ReviewDialog(screenshot, self.ml_detector, self)
        dialog.model_trained.connect(self._on_model_trained)
        self.statusBar().showMessage("审核标注中...")
        dialog.exec_()
        self.statusBar().showMessage("审核标注完成")

    @pyqtSlot()
    def _on_collect(self):
        """打开道具数据采集标注对话框."""
        if self._is_worker_active():
            QMessageBox.information(self, "任务运行中",
                                    "请先停止当前上架/下架任务后再打开道具采集")
            return
        from gui.collect_dialog import CollectDialog
        dialog = CollectDialog(self)
        self.statusBar().showMessage("道具采集中...")
        try:
            dialog.exec_()
        finally:
            # 兜底: 采集对话框会在截图流程中最小化主窗口,
            # 若中途异常或用户直接关闭, 这里确保主窗口还原到正常状态,
            # 避免遗留 minimized/hidden 导致用户再点最小化时触发 0xC000041D.
            if self.isMinimized() or not self.isVisible():
                self.showNormal()
            self.activateWindow()
            self.raise_()
        self.statusBar().showMessage("道具采集完成")
        self.log_text.append("道具数据采集标注完成")

    def _on_model_trained(self):
        """模型训练完成回调."""
        self.log_text.append(f"ML 模型已更新: {self.ml_detector.model_info}")
        self.statusBar().showMessage(f"模型已更新 | {self.ml_detector.model_info}")

    def _on_status(self, msg: str):
        self.log_text.append(msg)
        self.statusBar().showMessage(msg)

    def _on_progress(self, current: int, total: int):
        self.statusBar().showMessage(f"进度: {current}/{total}")

    def _on_finished(self):
        self._busy = False
        self._set_running(False)
        # worker 已连接 deleteLater, 这里只解除引用避免悬空访问
        self.worker = None
        # 如果任务期间自动最小化过, 任务完成后把窗口还原出来
        self._restore_after_task()

    def _set_running(self, running: bool):
        self.btn_list.setEnabled(not running)
        self.btn_delist.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_debug.setEnabled(not running)
        self.btn_review.setEnabled(not running)
        self.btn_collect.setEnabled(not running)

    # ═══════════════════════════════════════════════════════
    # 配置读写
    # ═══════════════════════════════════════════════════════

    def _save_config_from_ui(self):
        config = self.user_config.copy()
        config["selected_categories"] = {}
        config["selected_rarities"] = []
        config["selected_boxes"] = []

        for cat, checks in self._category_checks.items():
            selected = [name for name, cb in checks.items() if cb.isChecked()]
            if selected:
                config["selected_categories"][cat] = selected

        config["selected_rarities"] = [
            name for name, cb in self._rarity_checks.items() if cb.isChecked()
        ]
        config["selected_boxes"] = [
            name for name, cb in self._box_checks.items() if cb.isChecked()
        ]

        self.user_config = config
        save_user_config(config)

    def _load_config_to_ui(self):
        cfg = self.user_config

        for cat, subs in cfg.get("selected_categories", {}).items():
            if cat in self._category_checks:
                for sub in subs:
                    if sub in self._category_checks[cat]:
                        self._category_checks[cat][sub].setChecked(True)

        for rarity_name in cfg.get("selected_rarities", []):
            if rarity_name in self._rarity_checks:
                self._rarity_checks[rarity_name].setChecked(True)

        for box_name in cfg.get("selected_boxes", []):
            if box_name in self._box_checks:
                self._box_checks[box_name].setChecked(True)

        # 加载保存的坐标 → 检查分辨率是否匹配 → 换算 → 填入 SpinBox
        saved_coords = cfg.get("coordinates", {})
        if saved_coords:
            # 兼容迁移: 旧版仅保存 box_y_start/box_y_step, 展开为 10 个 box{i}_y
            if "box_y_start" in saved_coords and not any(
                f"box{i}_y" in saved_coords for i in range(1, MAX_STORAGE_BOXES + 1)
            ):
                y_start = int(saved_coords.get("box_y_start", 118))
                y_step = int(saved_coords.get("box_y_step", 45))
                for i in range(MAX_STORAGE_BOXES):
                    saved_coords[f"box{i + 1}_y"] = y_start + i * y_step

            saved_res = cfg.get("coordinates_resolution", [REF_WIDTH, REF_HEIGHT])
            cur_w, cur_h = self._game_window.width, self._game_window.height
            need_rescale = (saved_res[0] != cur_w or saved_res[1] != cur_h)

            if need_rescale and saved_res[0] > 0 and saved_res[1] > 0:
                sx = cur_w / saved_res[0]
                sy = cur_h / saved_res[1]
                _x_suffixes = ("_x", "x_start", "x_end")
                _y_suffixes = ("_y", "y_start", "y_step")
                rescaled = {}
                for key, val in saved_coords.items():
                    if any(key.endswith(s) for s in _x_suffixes):
                        rescaled[key] = round(val * sx)
                    elif any(key.endswith(s) for s in _y_suffixes):
                        rescaled[key] = round(val * sy)
                    else:
                        rescaled[key] = val
                saved_coords = rescaled
                self.log_text.append(
                    f"坐标从 {saved_res[0]}x{saved_res[1]} "
                    f"换算到 {cur_w}x{cur_h}"
                )

            for key, value in saved_coords.items():
                if key in self._coord_spins:
                    self._coord_spins[key].setValue(value)
            self._apply_coords_to_globals()

    @staticmethod
    def _toggle_all(checks: dict[str, QCheckBox]):
        any_checked = any(cb.isChecked() for cb in checks.values())
        for cb in checks.values():
            cb.setChecked(not any_checked)

    @staticmethod
    def _set_all(checks: dict[str, QCheckBox], state: bool):
        for cb in checks.values():
            cb.setChecked(state)

    def closeEvent(self, event):
        # 保存 UI 状态
        try:
            self._save_config_from_ui()
        except Exception:
            logger.exception("保存配置失败")

        # 如果 worker 还在运行, 询问用户
        if self.worker and self.worker.isRunning():
            resp = QMessageBox.question(
                self, "任务运行中",
                "当前有任务正在运行, 确定要退出吗?\n(退出会先尝试安全停止, 最多等待 5 秒)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return

        # 停止工作线程并关闭后台资源
        try:
            self.engine.stop()
        except Exception:
            logger.exception("停止 engine 失败")

        if self.worker and self.worker.isRunning():
            if not self.worker.wait(5000):
                logger.warning("WorkerThread 5 秒内未结束, 强制终止")
                self.worker.terminate()
                self.worker.wait(1000)

        # 关闭 OCR 线程池, 防止进程残留
        try:
            self.engine.shutdown()
        except Exception:
            logger.exception("关闭 engine 资源失败")

        event.accept()
