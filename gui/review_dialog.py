"""
审核标注对话框 — 网格显示所有格子, 用户可修正预测标签, 保存并训练模型.

流程:
  1. 截取当前出售界面 → 按网格切出所有格子
  2. 模型 (或规则法) 预测每格: 绑定/非绑定
  3. 以 9×N 网格显示, 红边=绑定, 绿边=非绑定
  4. 用户左键点击翻转标签
  5. 点"保存标签" → 格子图保存到 samples/, 标签写入 labels.json
  6. 点"训练模型" → 用所有已标注样本训练 RandomForest
"""
import json
import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QWidget,
    QMessageBox, QProgressBar, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QFont

from config import CELL_GRID, ML_SAMPLES_DIR, ML_LABELS_FILE, ML_MODEL_FILE
from gui.theme import (
    CYBER_DIALOG_STYLE, set_btn_style,
    NEON_CYAN, NEON_GREEN, NEON_RED, NEON_PINK, NEON_PURPLE,
    BG_DARKEST, BG_DARK, BG_MID, BG_WIDGET,
    BORDER_GLOW, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
)

logger = logging.getLogger("review_dialog")


def numpy_to_qpixmap(img_bgr: np.ndarray, target_size: int = 72) -> QPixmap:
    """BGR numpy 图像 -> QPixmap, 缩放到 target_size x target_size."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
    pm = QPixmap.fromImage(qimg)
    return pm.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CellWidget(QFrame):
    """单个格子的显示控件: 缩略图 + 标签边框 + 置信度."""
    clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, row: int, col: int, cell_bgr: np.ndarray,
                 is_bound: bool, confidence: float, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        # 只在构造时把原始图像转换为 QPixmap 并缓存, 避免每次 paintEvent 重建
        self._pixmap = numpy_to_qpixmap(cell_bgr, 72)
        self.is_bound = is_bound
        self.initial_is_bound = is_bound
        self.confidence = confidence

        self.setFixedSize(80, 100)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

    def _update_style(self):
        color = NEON_RED if self.is_bound else NEON_GREEN
        self.setStyleSheet(
            f"CellWidget {{ border: 3px solid {color}; "
            f"border-radius: 4px; background: {BG_MID}; }}"
        )

    def toggle_label(self):
        self.is_bound = not self.is_bound
        self._update_style()
        self.update()

    def set_label(self, is_bound: bool):
        if self.is_bound != is_bound:
            self.is_bound = is_bound
            self._update_style()
            self.update()

    @property
    def is_modified(self) -> bool:
        return self.is_bound != self.initial_is_bound

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_label()
            self.clicked.emit(self.row, self.col)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pm = self._pixmap
        x_off = (self.width() - pm.width()) // 2
        painter.drawPixmap(x_off, 2, pm)

        label_text = "绑定" if self.is_bound else "可售"
        color = QColor(220, 50, 50) if self.is_bound else QColor(50, 180, 50)
        painter.setPen(color)
        painter.setFont(QFont("Microsoft YaHei", 7, QFont.Bold))
        painter.drawText(2, self.height() - 4, label_text)

        conf_text = f"{self.confidence:.0%}"
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(self.width() - 30, self.height() - 4, conf_text)

        painter.end()


class ReviewDialog(QDialog):
    """审核标注对话框."""
    model_trained = pyqtSignal()

    def __init__(self, screenshot: np.ndarray, detector, parent=None):
        super().__init__(parent)
        self.setWindowTitle("道具绑定标注审核")
        self.setMinimumSize(960, 700)
        self.resize(1100, 850)
        self.setStyleSheet(CYBER_DIALOG_STYLE)

        self._screenshot = screenshot
        self._detector = detector
        self._cells: List[dict] = []
        self._cell_widgets: dict = {}
        self._labels_saved: bool = True  # 标注未修改时不需要保存提示

        self._init_ui()
        self._scan_and_display()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- 顶部说明 ---
        info = QLabel(
            "左键点击格子可翻转标签 (绑定 ↔ 可售)  |  "
            "红边 = 绑定  |  绿边 = 可售(非绑定)  |  "
            "修正完成后点击下方按钮保存"
        )
        info.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 6px; font-size: 10pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- 统计栏 ---
        self._stats_label = QLabel("正在扫描...")
        self._stats_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11pt; padding: 4px;")
        layout.addWidget(self._stats_label)

        # --- 网格区域 ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(4)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll, 1)

        # --- 底部按钮 ---
        btn_row = QHBoxLayout()

        self._btn_save = QPushButton("保存标签到 samples/")
        set_btn_style(self._btn_save, NEON_CYAN, "11pt", "8px 16px")
        self._btn_save.clicked.connect(self._save_labels)
        btn_row.addWidget(self._btn_save)

        self._btn_train = QPushButton("训练模型")
        set_btn_style(self._btn_train, NEON_GREEN, "11pt", "8px 16px")
        self._btn_train.clicked.connect(self._train_model)
        btn_row.addWidget(self._btn_train)

        self._btn_toggle_all_bound = QPushButton("全部标为绑定")
        set_btn_style(self._btn_toggle_all_bound, NEON_RED)
        self._btn_toggle_all_bound.clicked.connect(lambda: self._set_all(True))
        btn_row.addWidget(self._btn_toggle_all_bound)

        self._btn_toggle_all_unbound = QPushButton("全部标为可售")
        set_btn_style(self._btn_toggle_all_unbound, NEON_GREEN)
        self._btn_toggle_all_unbound.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(self._btn_toggle_all_unbound)

        btn_row.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

        # --- 进度条 ---
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(20)
        layout.addWidget(self._progress)

    def _scan_and_display(self):
        """扫描截图并显示所有格子."""
        predictions = self._detector.predict_cells(self._screenshot)

        grid = CELL_GRID
        gx, gy = grid["origin_x"], grid["origin_y"]
        cw, ch = grid["cell_w"], grid["cell_h"]

        self._cells.clear()
        self._cell_widgets.clear()

        for pred in predictions:
            row, col = pred["row"], pred["col"]
            x1 = gx + col * cw
            y1 = gy + row * ch
            cell_bgr = self._screenshot[y1:y1 + ch, x1:x1 + cw].copy()

            cell_info = {
                "row": row, "col": col,
                "cell_bgr": cell_bgr,
                "is_bound": pred["is_bound"],
                "confidence": pred["confidence"],
            }
            self._cells.append(cell_info)

            widget = CellWidget(
                row, col, cell_bgr,
                pred["is_bound"], pred["confidence"]
            )
            widget.clicked.connect(self._on_cell_clicked)
            self._grid_layout.addWidget(widget, row, col)
            self._cell_widgets[(row, col)] = widget

        self._update_stats()

    def _on_cell_clicked(self, row: int, col: int):
        self._labels_saved = False
        self._update_stats()

    def _update_stats(self):
        n_total = len(self._cell_widgets)
        n_bound = sum(1 for w in self._cell_widgets.values() if w.is_bound)
        n_unbound = n_total - n_bound
        mode = self._detector.model_info
        self._stats_label.setText(
            f"总计: {n_total} 格  |  "
            f"绑定: {n_bound}  |  可售: {n_unbound}  |  "
            f"检测模式: {mode}"
        )

    def _set_all(self, is_bound: bool):
        changed = False
        for w in self._cell_widgets.values():
            if w.is_bound != is_bound:
                w.set_label(is_bound)
                changed = True
        if changed:
            self._labels_saved = False
        self._update_stats()

    def _save_labels(self):
        """保存格子图像和标签."""
        os.makedirs(ML_SAMPLES_DIR, exist_ok=True)

        existing_labels = {}
        if os.path.exists(ML_LABELS_FILE):
            try:
                with open(ML_LABELS_FILE, "r", encoding="utf-8") as f:
                    existing_labels = json.load(f)
            except Exception:
                pass

        ts = int(time.time())
        saved_count = 0

        for cell in self._cells:
            row, col = cell["row"], cell["col"]
            widget = self._cell_widgets.get((row, col))
            if widget is None:
                continue

            fname = f"R{row:02d}C{col:02d}_{ts}.png"
            fpath = os.path.join(ML_SAMPLES_DIR, fname)
            cv2.imwrite(fpath, cell["cell_bgr"])

            existing_labels[fname] = 1 if widget.is_bound else 0
            saved_count += 1

        with open(ML_LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_labels, f, ensure_ascii=False, indent=2)

        total_labels = len(existing_labels)
        n_bound = sum(1 for v in existing_labels.values() if v == 1)
        n_unbound = total_labels - n_bound

        self._labels_saved = True

        QMessageBox.information(
            self, "保存完成",
            f"本次保存: {saved_count} 个格子\n"
            f"累计样本: {total_labels} 个 (绑定={n_bound}, 可售={n_unbound})\n"
            f"路径: {ML_SAMPLES_DIR}/\n\n"
            f"样本足够后可点击「训练模型」提升识别准确率"
        )

    def _train_model(self):
        """训练模型."""
        if not os.path.exists(ML_LABELS_FILE):
            QMessageBox.warning(self, "无标注数据", "请先保存标签后再训练")
            return

        with open(ML_LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)

        if len(labels) < 4:
            QMessageBox.warning(
                self, "样本不足",
                f"当前只有 {len(labels)} 个样本, 至少需要 4 个才能训练"
            )
            return

        n_bound = sum(1 for v in labels.values() if v == 1)
        n_unbound = len(labels) - n_bound
        if n_bound == 0 or n_unbound == 0:
            QMessageBox.warning(
                self, "标签不均衡",
                f"绑定={n_bound}, 可售={n_unbound}\n"
                "至少需要每类各1个样本"
            )
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._btn_train.setEnabled(False)
        self._btn_train.setText("训练中...")

        try:
            from core.ml_detector import MLBoundDetector
            stats = MLBoundDetector.train_model(
                samples_dir=ML_SAMPLES_DIR,
                labels_file=ML_LABELS_FILE,
                model_out=ML_MODEL_FILE,
            )

            self._detector.reload_model()
            self.model_trained.emit()

            QMessageBox.information(
                self, "训练完成",
                f"样本总数: {stats['total_samples']}\n"
                f"绑定样本: {stats['bound_samples']}\n"
                f"可售样本: {stats['unbound_samples']}\n"
                f"模型已保存: {stats['model_path']}\n\n"
                f"下次检测将自动使用 ML 模式"
            )

        except Exception as e:
            logger.exception("模型训练失败")
            QMessageBox.critical(self, "训练失败", str(e))

        finally:
            self._progress.setVisible(False)
            self._btn_train.setEnabled(True)
            self._btn_train.setText("训练模型")
            self._update_stats()

    def closeEvent(self, event):
        if self._labels_saved:
            event.accept()
            return
        resp = QMessageBox.question(
            self, "未保存的修改",
            "有标注尚未保存, 是否先保存再关闭?\n\n"
            "是 — 保存并关闭\n否 — 放弃修改并关闭\n取消 — 继续编辑",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if resp == QMessageBox.Yes:
            self._save_labels()
            event.accept()
        elif resp == QMessageBox.No:
            event.accept()
        else:
            event.ignore()
