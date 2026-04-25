"""
ML 绑定道具检测器 — RandomForest 多特征分类

支持两种模式:
  - 规则模式 (fallback): 无训练数据时, 沿用 HSV V 阈值法
  - ML 模式: 加载训练好的 RandomForest 模型, 用 19 维特征预测

特征体系 (19 维, 来自 ChatGPT 方案验证有效):
  亮度 (4): v_mean, v_std, border_mean, border_std
  中心区域 (2): center_v_mean, center_v_std
  饱和度 (3): s_mean, s_std, center_s_mean
  灰度 (2): gray_mean, gray_std
  纹理 (4): grad_mean, lap_abs_mean, gabor_45, gabor_135
  统计 (4): dark_ratio, diag_diff_mean, bright_ratio, low_sat_ratio
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import CELL_GRID, BIND_DETECTION

logger = logging.getLogger("ml_detector")

FEATURE_NAMES = [
    "v_mean", "v_std",
    "border_mean", "border_std",
    "center_v_mean", "center_v_std",
    "s_mean", "s_std", "center_s_mean",
    "gray_mean", "gray_std",
    "grad_mean", "lap_abs_mean",
    "gabor_45", "gabor_135",
    "dark_ratio", "diag_diff_mean",
    "bright_ratio", "low_sat_ratio",
]

# ─── Gabor 内核预计算 (参数固定, 每格复用) ───
_GABOR_KERNEL_45 = cv2.getGaborKernel(
    (15, 15), 3.0, np.pi / 4, 8.0, 0.7, 0, ktype=cv2.CV_32F
)
_GABOR_KERNEL_135 = cv2.getGaborKernel(
    (15, 15), 3.0, 3 * np.pi / 4, 8.0, 0.7, 0, ktype=cv2.CV_32F
)


def extract_features(cell_bgr: np.ndarray) -> np.ndarray:
    """从单个格子图像提取 19 维特征向量."""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, s, v = cv2.split(hsv)
    cell_h, cell_w = v.shape

    border_w = max(6, cell_w // 10)
    border_h = max(6, cell_h // 10)

    border_mask = np.zeros_like(v, dtype=np.uint8)
    border_mask[:border_h, :] = 1
    border_mask[-border_h:, :] = 1
    border_mask[:, :border_w] = 1
    border_mask[:, -border_w:] = 1

    border = v[border_mask.astype(bool)]
    center_v = v[border_h:cell_h - border_h, border_w:cell_w - border_w]
    center_s = s[border_h:cell_h - border_h, border_w:cell_w - border_w]
    center_g = gray[border_h:cell_h - border_h, border_w:cell_w - border_w]

    gx = cv2.Sobel(center_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(center_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(center_g, cv2.CV_32F)

    gabor_45 = float(np.mean(np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNEL_45))))
    gabor_135 = float(np.mean(np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNEL_135))))

    dark_thr = np.percentile(v, 30)
    bright_thr = np.percentile(v, 75)
    sat_thr = np.percentile(s, 30)
    dark_ratio = float((center_v < dark_thr).mean()) if center_v.size else 0.0
    bright_ratio = float((center_v > bright_thr).mean()) if center_v.size else 0.0
    low_sat_ratio = float((center_s < sat_thr).mean()) if center_s.size else 0.0

    diag_main = np.diagonal(center_g.astype(np.float32))
    diag_other = np.diagonal(np.fliplr(center_g).astype(np.float32))
    diag_diff_mean = float(abs(diag_main.mean() - diag_other.mean())) if diag_main.size else 0.0

    return np.array([
        float(v.mean()), float(v.std()),
        float(border.mean()), float(border.std()),
        float(center_v.mean()), float(center_v.std()),
        float(s.mean()), float(s.std()), float(center_s.mean()),
        float(gray.mean()), float(gray.std()),
        float(grad_mag.mean()), float(np.mean(np.abs(lap))),
        gabor_45, gabor_135,
        dark_ratio, diag_diff_mean,
        bright_ratio, low_sat_ratio,
    ], dtype=np.float32)


class MLBoundDetector:
    """
    可训练的绑定道具检测器.

    有模型时用 ML 预测, 无模型时降级为规则法 (HSV V 阈值).
    """

    def __init__(self, model_path: str = "models/bound_model.joblib"):
        self._grid = CELL_GRID
        self._det = BIND_DETECTION
        self._model_path = model_path
        self._model = None
        self._is_ml_mode = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self._model_path):
            logger.info(f"模型文件不存在: {self._model_path}, 使用规则模式")
            return
        try:
            import joblib
            self._model = joblib.load(self._model_path)
            self._is_ml_mode = True
            logger.info(f"已加载 ML 模型: {self._model_path}")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}, 降级为规则模式")

    def reload_model(self):
        """重新加载模型 (训练完成后调用)."""
        self._model = None
        self._is_ml_mode = False
        self._load_model()

    @property
    def is_ml_mode(self) -> bool:
        return self._is_ml_mode

    @property
    def model_info(self) -> str:
        if self._is_ml_mode:
            return f"ML模式 ({self._model_path})"
        return "规则模式 (HSV V阈值)"

    def iter_cells(self, screenshot: np.ndarray):
        """遍历网格, 返回每个非空格子的 (row, col, x, y, cell_bgr)."""
        gx = self._grid["origin_x"]
        gy = self._grid["origin_y"]
        cw = self._grid["cell_w"]
        ch = self._grid["cell_h"]
        cols = self._grid["cols"]
        rows = self._grid["visible_rows"]
        h, w = screenshot.shape[:2]
        empty_thr = self._det["empty_threshold"]

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        for row in range(rows):
            for col in range(cols):
                x1 = gx + col * cw
                y1 = gy + row * ch
                if x1 + cw > w or y1 + ch > h:
                    continue

                cell_gray = gray[y1:y1 + ch, x1:x1 + cw]
                center = cell_gray[20:65, 10:75]
                if center.size > 0 and center.mean() < empty_thr:
                    continue

                cell_bgr = screenshot[y1:y1 + ch, x1:x1 + cw]
                yield row, col, x1, y1, cell_bgr

    def predict_cells(self, screenshot: np.ndarray) -> List[dict]:
        """
        预测截图中所有格子的绑定/非绑定状态.

        Returns:
            列表, 每项: {row, col, cx, cy, is_bound, confidence, v_mean}
        """
        if self._is_ml_mode:
            return self._predict_ml(screenshot)
        return self._predict_rule(screenshot)

    def _predict_ml(self, screenshot: np.ndarray) -> List[dict]:
        """ML 模式: 用 RandomForest 预测."""
        cw = self._grid["cell_w"]
        ch = self._grid["cell_h"]
        results = []

        for row, col, x1, y1, cell_bgr in self.iter_cells(screenshot):
            feat = extract_features(cell_bgr).reshape(1, -1)
            bound_proba = float(self._model.predict_proba(feat)[0, 1])
            is_bound = bound_proba >= 0.5
            confidence = bound_proba if is_bound else 1.0 - bound_proba

            results.append({
                "row": row, "col": col,
                "cx": x1 + cw // 2, "cy": y1 + ch // 2,
                "is_bound": is_bound,
                "confidence": round(confidence, 3),
                "bound_score": round(bound_proba, 3),
                "v_mean": round(float(
                    cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
                ), 1),
            })

        return results

    def _predict_rule(self, screenshot: np.ndarray) -> List[dict]:
        """规则模式: 沿用 HSV V 阈值法."""
        cw = self._grid["cell_w"]
        ch = self._grid["cell_h"]
        v_thr = self._det["v_threshold"]
        sy1, sy2 = self._det["sample_y1"], self._det["sample_y2"]
        sx1, sx2 = self._det["sample_x1"], self._det["sample_x2"]
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        results = []

        for row, col, x1, y1, cell_bgr in self.iter_cells(screenshot):
            bg_v = float(hsv[y1 + sy1:y1 + sy2, x1 + sx1:x1 + sx2, 2].mean())
            is_bound = bg_v <= v_thr

            results.append({
                "row": row, "col": col,
                "cx": x1 + cw // 2, "cy": y1 + ch // 2,
                "is_bound": is_bound,
                "confidence": 0.7,
                "bound_score": 0.8 if is_bound else 0.2,
                "v_mean": round(bg_v, 1),
            })

        return results

    def get_unbound_items(self, screenshot: np.ndarray) -> List[dict]:
        """
        返回非绑定道具列表 (兼容旧接口).
        只返回 is_bound=False 的格子.
        """
        all_cells = self.predict_cells(screenshot)
        return [c for c in all_cells if not c["is_bound"]]

    # ═══════════════════════════════════════════════════════
    # 训练相关
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def train_model(
        samples_dir: str,
        labels_file: str,
        model_out: str,
    ) -> dict:
        """
        从标注样本训练 RandomForest 模型.

        Args:
            samples_dir: 格子图像目录
            labels_file: 标签 JSON 文件 {"filename": 1_or_0, ...}
            model_out: 模型输出路径

        Returns:
            训练统计信息 dict
        """
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)

        features = []
        targets = []
        skipped = 0

        for filename, label in labels.items():
            img_path = os.path.join(samples_dir, filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            feat = extract_features(img)
            features.append(feat)
            targets.append(int(label))

        if len(features) < 4:
            raise ValueError(f"样本数量不足 (需要至少4个, 当前 {len(features)})")

        X = np.vstack(features).astype(np.float32)
        y = np.array(targets, dtype=np.int32)

        n_bound = int(y.sum())
        n_unbound = int((1 - y).sum())
        if n_bound == 0 or n_unbound == 0:
            raise ValueError(
                f"标签只有一类 (绑定={n_bound}, 非绑定={n_unbound}), "
                "至少需要每类各1个样本"
            )

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            max_depth=None,
        )
        clf.fit(X, y)

        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        joblib.dump(clf, model_out)

        stats = {
            "total_samples": len(y),
            "bound_samples": n_bound,
            "unbound_samples": n_unbound,
            "skipped_files": skipped,
            "model_path": model_out,
            "feature_count": len(FEATURE_NAMES),
        }

        stats_path = model_out.replace(".joblib", ".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"模型训练完成: {len(y)} 样本 "
            f"(绑定={n_bound}, 非绑定={n_unbound}), "
            f"保存到 {model_out}"
        )
        return stats

    @staticmethod
    def save_cell_images(
        screenshot: np.ndarray,
        output_dir: str,
        prefix: str = "",
    ) -> List[dict]:
        """
        将截图按网格切分, 保存每个非空格子到文件.

        Returns:
            列表 [{filename, row, col, cx, cy}, ...]
        """
        grid = CELL_GRID
        det = BIND_DETECTION
        gx, gy = grid["origin_x"], grid["origin_y"]
        cw, ch = grid["cell_w"], grid["cell_h"]
        cols, rows = grid["cols"], grid["visible_rows"]
        h, w = screenshot.shape[:2]
        empty_thr = det["empty_threshold"]

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for row in range(rows):
            for col in range(cols):
                x1 = gx + col * cw
                y1 = gy + row * ch
                if x1 + cw > w or y1 + ch > h:
                    continue

                cell_gray = gray[y1:y1 + ch, x1:x1 + cw]
                center = cell_gray[20:65, 10:75]
                if center.size > 0 and center.mean() < empty_thr:
                    continue

                cell_bgr = screenshot[y1:y1 + ch, x1:x1 + cw]
                fname = f"{prefix}R{row:02d}C{col:02d}.png"
                cv2.imwrite(os.path.join(output_dir, fname), cell_bgr)

                saved.append({
                    "filename": fname,
                    "row": row, "col": col,
                    "cx": x1 + cw // 2, "cy": y1 + ch // 2,
                })

        return saved
