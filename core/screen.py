"""
屏幕截图与图像处理工具

使用 PIL.ImageGrab 替代 mss, 彻底解决 mss 在 Windows 上
高频截图导致的 GDI 句柄泄漏 / 堆损坏 (0xC0000374) / 访问违规 (0xC0000005).
"""
import ctypes
import logging
import threading

import cv2
import numpy as np
from PIL import ImageGrab

logger = logging.getLogger("screen")

_grab_lock = threading.Lock()


def set_dpi_awareness():
    """设置DPI感知, 确保坐标映射到物理像素"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class ScreenCapture:
    """
    屏幕截图管理器 — 基于 PIL.ImageGrab.

    ImageGrab 每次调用独立获取/释放 GDI 资源, 不持有长期句柄,
    在多线程高频调用下远比 mss 稳定.

    game_region: 游戏窗口在屏幕上的区域 (left, top, right, bottom).
                 None 表示全屏 (无边框/全屏模式).
    """

    def __init__(self, game_region: tuple[int, int, int, int] | None = None):
        self._game_region = game_region

    def set_game_region(self, region: tuple[int, int, int, int] | None):
        self._game_region = region

    def grab_full(self) -> np.ndarray:
        """截取游戏窗口区域, 返回BGR格式numpy数组"""
        with _grab_lock:
            img = ImageGrab.grab(bbox=self._game_region, all_screens=False)
        frame = np.array(img)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def grab_region(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """截取游戏窗口内的指定区域 (窗口内坐标), 返回BGR格式numpy数组"""
        ox = self._game_region[0] if self._game_region else 0
        oy = self._game_region[1] if self._game_region else 0
        with _grab_lock:
            img = ImageGrab.grab(
                bbox=(ox + x1, oy + y1, ox + x2, oy + y2),
                all_screens=False,
            )
        frame = np.array(img)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def grab_item_grid(self, grid_cfg: dict) -> np.ndarray:
        """截取道具网格区域"""
        return self.grab_region(
            grid_cfg["x_start"], grid_cfg["y_start"],
            grid_cfg["x_end"], grid_cfg["y_end"],
        )


def find_template(screenshot: np.ndarray, template: np.ndarray,
                  threshold: float = 0.8) -> list[tuple[int, int, float]]:
    """
    在截图中查找模板图片, 返回所有匹配位置
    Returns: [(x_center, y_center, confidence), ...]
    """
    if screenshot is None or template is None:
        return []

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    h, w = template.shape[:2]
    matches = []
    for pt in zip(*locations[::-1]):
        cx = pt[0] + w // 2
        cy = pt[1] + h // 2
        conf = result[pt[1], pt[0]]
        matches.append((cx, cy, float(conf)))

    # 非极大值抑制: 去除重叠匹配
    return _nms(matches, w, h)


def _nms(matches: list, w: int, h: int, overlap_thresh: float = 0.5) -> list:
    """非极大值抑制, 去除重叠的匹配结果"""
    if not matches:
        return []

    matches.sort(key=lambda m: m[2], reverse=True)
    keep = []
    for m in matches:
        too_close = False
        for k in keep:
            if abs(m[0] - k[0]) < w * overlap_thresh and abs(m[1] - k[1]) < h * overlap_thresh:
                too_close = True
                break
        if not too_close:
            keep.append(m)
    return keep


def load_template(path: str) -> np.ndarray | None:
    """加载模板图片"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def save_debug_screenshot(frame: np.ndarray, path: str):
    """保存调试截图"""
    cv2.imwrite(path, frame)
