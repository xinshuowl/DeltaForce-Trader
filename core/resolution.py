"""
游戏窗口检测与多分辨率坐标缩放

通过 Win32 API 检测 "Delta Force" 窗口的实际大小和位置,
将基于 2560x1440 参考分辨率的坐标按比例缩放到实际分辨率.
"""
import copy
import ctypes
import ctypes.wintypes
import logging
from dataclasses import dataclass

logger = logging.getLogger("resolution")

REF_WIDTH = 2560
REF_HEIGHT = 1440

GAME_WINDOW_TITLES = ["三角洲行动", "DeltaForce", "Delta Force"]
GAME_PROCESS_NAME = "DeltaForceClient-Win64-Shipping.exe"

PRESETS = {
    "1920x1080 (1080p)": (1920, 1080),
    "2560x1080 (21:9)":  (2560, 1080),
    "2560x1440 (1440p)": (2560, 1440),
    "3440x1440 (21:9)":  (3440, 1440),
    "3840x2160 (4K)":    (3840, 2160),
}


@dataclass
class GameWindowInfo:
    width: int = REF_WIDTH
    height: int = REF_HEIGHT
    left: int = 0
    top: int = 0
    detected: bool = False

    @property
    def scale_x(self) -> float:
        return self.width / REF_WIDTH

    @property
    def scale_y(self) -> float:
        return self.height / REF_HEIGHT

    @property
    def needs_offset(self) -> bool:
        return self.left != 0 or self.top != 0

    @property
    def game_region(self):
        """ImageGrab bbox, None if fullscreen at (0,0)"""
        if not self.needs_offset:
            return None
        return (self.left, self.top,
                self.left + self.width, self.top + self.height)

    @property
    def summary(self) -> str:
        sx = self.scale_x
        mode = "自动检测" if self.detected else "手动设置"
        offset = f" 偏移({self.left},{self.top})" if self.needs_offset else ""
        return f"{self.width}x{self.height} (缩放 {sx:.2f}x) [{mode}]{offset}"


def _find_hwnd_by_process() -> int:
    """通过进程名枚举所有窗口, 找到属于游戏进程的主窗口."""
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    psapi = ctypes.windll.psapi

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    result = [0]

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return True

        hproc = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not hproc:
            return True

        try:
            buf = ctypes.create_unicode_buffer(512)
            size = ctypes.wintypes.DWORD(512)
            if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
                exe_path = buf.value
                if exe_path.lower().endswith(GAME_PROCESS_NAME.lower()):
                    rect = ctypes.wintypes.RECT()
                    user32.GetClientRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 100 and h > 100:
                        result[0] = hwnd
                        return False  # stop enumeration
        finally:
            kernel32.CloseHandle(hproc)
        return True

    user32.EnumWindows(enum_cb, 0)
    return result[0]


def _get_screen_resolution(user32) -> tuple[int, int]:
    """获取主显示器当前分辨率 (物理像素)"""
    try:
        SM_CXSCREEN, SM_CYSCREEN = 0, 1
        w = user32.GetSystemMetrics(SM_CXSCREEN)
        h = user32.GetSystemMetrics(SM_CYSCREEN)
        return w, h
    except Exception:
        return 0, 0


def _is_admin() -> bool:
    """检查当前进程是否以管理员权限运行"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def detect_game_window() -> GameWindowInfo:
    """
    检测 Delta Force 游戏窗口.
    分两步:
    1. 临时切 DPI-unaware 找到 HWND (解决 Per-Monitor v2 看不到游戏窗口)
    2. 切回 DPI-aware 获取物理像素尺寸
    """
    try:
        user32 = ctypes.windll.user32
        DPI_AWARENESS_CONTEXT_UNAWARE = ctypes.c_void_p(-1)

        # ── 第1步: DPI-unaware 上下文中查找 HWND ──
        hwnd = 0
        old_ctx = None
        try:
            old_ctx = user32.SetThreadDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_UNAWARE
            )
        except Exception:
            pass

        try:
            for title in GAME_WINDOW_TITLES:
                hwnd = user32.FindWindowW(None, title)
                if hwnd:
                    logger.info(f"通过标题匹配到窗口: '{title}'")
                    break

            if not hwnd:
                hwnd = _find_hwnd_by_process()
                if hwnd:
                    title_buf = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(hwnd, title_buf, 256)
                    logger.info(f"通过进程名匹配到窗口: '{title_buf.value}'")
        finally:
            if old_ctx is not None:
                try:
                    user32.SetThreadDpiAwarenessContext(
                        ctypes.c_void_p(old_ctx)
                    )
                except Exception:
                    pass

        if hwnd:
            # ── 第2步: DPI-aware 上下文中获取物理像素尺寸 ──
            rect = ctypes.wintypes.RECT()
            if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top

                if width > 0 and height > 0:
                    point = ctypes.wintypes.POINT(0, 0)
                    user32.ClientToScreen(hwnd, ctypes.byref(point))
                    left, top = point.x, point.y
                    logger.info(
                        f"检测到游戏窗口: {width}x{height} @ ({left},{top})"
                    )
                    return GameWindowInfo(
                        width=width, height=height,
                        left=left, top=top, detected=True,
                    )

        # ── 兜底: 全屏模式下窗口不可见, 读当前屏幕分辨率 ──
        width, height = _get_screen_resolution(user32)
        if width > 0 and height > 0:
            logger.info(
                f"未找到游戏窗口, 使用当前屏幕分辨率: {width}x{height} "
                f"(全屏模式下游戏分辨率 = 屏幕分辨率)"
            )
            return GameWindowInfo(
                width=width, height=height,
                left=0, top=0, detected=True,
            )

        admin = _is_admin()
        logger.warning(
            f"未找到游戏窗口 (标题: {GAME_WINDOW_TITLES}, "
            f"进程: {GAME_PROCESS_NAME}, "
            f"管理员权限: {'是' if admin else '否 — 请用管理员权限启动本软件'})"
        )
        return GameWindowInfo()

    except Exception as e:
        logger.error(f"窗口检测异常: {e}")
        return GameWindowInfo()


# ═══════════════════════════════════════════════════════════
# 坐标缩放
# ═══════════════════════════════════════════════════════════

# 每个配置字典中哪些键属于 x 轴、哪些属于 y 轴
_SCALE_KEYS = {
    "TAB_COORDS": None,  # dict of tuples, handled specially
    "ORGANIZE_BTN": {
        "x": ["icon_x", "sort_btn_x"],
        "y": ["icon_y", "sort_btn_y"],
    },
    "BOX_SELECTOR": {
        "x": ["x"],
        "y": ["y_start", "y_step"],
        "y_list": ["positions"],  # list of Y coords, scaled element-wise
    },
    "ITEM_GRID": {
        "x": ["x_start", "x_end", "scroll_area_x"],
        "y": ["y_start", "y_end", "scroll_area_y"],
    },
    "CELL_GRID": {
        "x": ["origin_x", "cell_w"],
        "y": ["origin_y", "cell_h"],
    },
    "BIND_DETECTION": {
        "x": ["sample_x1", "sample_x2"],
        "y": ["sample_y1", "sample_y2"],
    },
    "LISTED_ITEMS": {
        "x": ["x_start", "delist_btn_x", "scroll_area_x", "confirm_btn_x"],
        "y": ["y_start", "item_height", "delist_btn_y",
               "delist_btn_offset_y", "view_btn_offset_y",
               "scroll_area_y", "confirm_btn_y"],
    },
    "LISTING_SLOTS": {
        "region": ["counter_region"],  # (x1,y1,x2,y2) tuple
    },
    "LIST_DIALOG": {
        "tuple_xy": [
            "qty_minus_btn", "qty_plus_btn",
            "qty_slider_left", "qty_slider_right", "qty_slider_max",
            "price_input", "price_minus", "price_plus",
            "list_btn", "esc_btn",
        ],
        "region": [
            "item_name_region", "income_region", "dialog_crop",
        ],
    },
    "DETECTION_ROI": {
        "region": ["page_change", "sell_tab"],
    },
}


def _scale_val(val: int, factor: float) -> int:
    return round(val * factor)


def _scale_tuple_xy(t: tuple, sx: float, sy: float) -> tuple:
    """Scale (x, y) tuple."""
    return (_scale_val(t[0], sx), _scale_val(t[1], sy))


def _scale_region(t: tuple, sx: float, sy: float) -> tuple:
    """Scale (x1, y1, x2, y2) region."""
    return (
        _scale_val(t[0], sx), _scale_val(t[1], sy),
        _scale_val(t[2], sx), _scale_val(t[3], sy),
    )


def scale_all_configs(width: int, height: int):
    """
    从参考值重建并按比例缩放所有坐标配置字典.
    先 reset 到参考值再缩放, 避免累积误差.
    """
    from config import (
        reset_to_reference,
        TAB_COORDS, ORGANIZE_BTN, BOX_SELECTOR, ITEM_GRID,
        CELL_GRID, BIND_DETECTION, LISTED_ITEMS, LISTING_SLOTS,
        LIST_DIALOG, DETECTION_ROI, ORIGINAL_DEFAULT_COORDS, COORD_KEY_MAP,
        SCREEN,
    )

    reset_to_reference()

    sx = width / REF_WIDTH
    sy = height / REF_HEIGHT

    if sx == 1.0 and sy == 1.0:
        logger.info("分辨率与参考值一致, 无需缩放")
        return

    logger.info(f"缩放坐标: {width}x{height}, sx={sx:.3f}, sy={sy:.3f}")

    SCREEN["width"] = width
    SCREEN["height"] = height

    # TAB_COORDS: dict of str -> (x, y) tuples
    for key in list(TAB_COORDS.keys()):
        TAB_COORDS[key] = _scale_tuple_xy(TAB_COORDS[key], sx, sy)

    # Simple x/y key dicts
    _config_map = {
        "ORGANIZE_BTN": ORGANIZE_BTN,
        "BOX_SELECTOR": BOX_SELECTOR,
        "ITEM_GRID": ITEM_GRID,
        "CELL_GRID": CELL_GRID,
        "BIND_DETECTION": BIND_DETECTION,
        "LISTED_ITEMS": LISTED_ITEMS,
    }
    for dict_name, cfg_dict in _config_map.items():
        rules = _SCALE_KEYS.get(dict_name)
        if not rules:
            continue
        for key in rules.get("x", []):
            if key in cfg_dict and isinstance(cfg_dict[key], (int, float)):
                cfg_dict[key] = _scale_val(cfg_dict[key], sx)
        for key in rules.get("y", []):
            if key in cfg_dict and isinstance(cfg_dict[key], (int, float)):
                cfg_dict[key] = _scale_val(cfg_dict[key], sy)
        for key in rules.get("y_list", []):
            if key in cfg_dict and isinstance(cfg_dict[key], list):
                cfg_dict[key] = [_scale_val(v, sy) for v in cfg_dict[key]]

    # LISTING_SLOTS: counter_region is (x1, y1, x2, y2)
    for key in _SCALE_KEYS["LISTING_SLOTS"].get("region", []):
        if key in LISTING_SLOTS and isinstance(LISTING_SLOTS[key], tuple):
            LISTING_SLOTS[key] = _scale_region(LISTING_SLOTS[key], sx, sy)

    # LIST_DIALOG: mix of (x,y) tuples, (x1,y1,x2,y2) regions, and scalars
    for key in _SCALE_KEYS["LIST_DIALOG"].get("tuple_xy", []):
        if key in LIST_DIALOG and isinstance(LIST_DIALOG[key], tuple):
            LIST_DIALOG[key] = _scale_tuple_xy(LIST_DIALOG[key], sx, sy)
    for key in _SCALE_KEYS["LIST_DIALOG"].get("region", []):
        if key in LIST_DIALOG and isinstance(LIST_DIALOG[key], tuple):
            LIST_DIALOG[key] = _scale_region(LIST_DIALOG[key], sx, sy)

    # DETECTION_ROI: 所有条目都是 (x1, y1, x2, y2) 区域
    for key in _SCALE_KEYS["DETECTION_ROI"].get("region", []):
        if key in DETECTION_ROI and isinstance(DETECTION_ROI[key], tuple):
            DETECTION_ROI[key] = _scale_region(DETECTION_ROI[key], sx, sy)

    # ORIGINAL_DEFAULT_COORDS: flat dict of key -> int
    # 按键名后缀判断轴向 (注意顺序: _x1/_x2 必须在 _x 之前判断以避免误匹配)
    _x_suffixes = ("_x1", "_x2", "_x", "x_start", "x_end")
    _y_suffixes = ("_y1", "_y2", "_y", "y_start", "y_step")
    for key in ORIGINAL_DEFAULT_COORDS:
        if any(key.endswith(s) for s in _x_suffixes):
            ORIGINAL_DEFAULT_COORDS[key] = _scale_val(
                ORIGINAL_DEFAULT_COORDS[key], sx
            )
        elif any(key.endswith(s) for s in _y_suffixes):
            ORIGINAL_DEFAULT_COORDS[key] = _scale_val(
                ORIGINAL_DEFAULT_COORDS[key], sy
            )

    logger.info("坐标缩放完成")
