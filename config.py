"""
三角洲行动 - 交易行自动上架/下架工具 配置文件
参考分辨率: 2560x1440 (所有坐标基于此分辨率, 运行时按实际分辨率缩放)
"""
import copy
import json
import os
from enum import Enum

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "user_config.json")

# 开启后将写出 debug_*.png (OCR 调试图). 生产环境建议关闭以减少磁盘 I/O.
DEBUG = os.environ.get("AUTOSHOP_DEBUG", "0") == "1"

# ═══════════════════════════════════════════════════════════
# 道具品质颜色定义 (BGR格式, 用于OpenCV匹配)
# ═══════════════════════════════════════════════════════════

class Rarity(Enum):
    RED = "红色"
    GOLD = "金色"
    PURPLE = "紫色"
    BLUE = "蓝色"
    GREEN = "绿色"
    WHITE = "白色"

# 道具名称文字颜色范围 (HSV色彩空间, [H_min, S_min, V_min], [H_max, S_max, V_max])
RARITY_HSV_RANGES = {
    Rarity.RED:    ([0, 100, 150],   [10, 255, 255]),
    Rarity.GOLD:   ([15, 100, 150],  [35, 255, 255]),
    Rarity.PURPLE: ([125, 50, 100],  [155, 255, 255]),
    Rarity.BLUE:   ([95, 80, 100],   [125, 255, 255]),
    Rarity.GREEN:  ([40, 60, 100],   [85, 255, 255]),
    Rarity.WHITE:  ([0, 0, 180],     [180, 30, 255]),
}

# ═══════════════════════════════════════════════════════════
# 道具分类定义
# ═══════════════════════════════════════════════════════════

ITEM_CATEGORIES = {
    "枪械": ["步枪", "冲锋枪", "狙击步枪", "霰弹枪", "射手步枪", "轻机枪", "手枪", "特殊武器"],
    "装备": ["头盔", "护甲", "胸挂", "背包"],
    "配件": ["弹匣", "瞄具", "护木", "枪管", "枪口", "枪托", "前握把", "后握把", "功能性配件"],
    "弹药": [
        ".357 Magnum", ".45 ACP", ".50AE", "12 Gauge", "12.7x55mm",
        "5.45x39mm", "5.56x45mm", "5.7x28mm", "5.8x42mm", "6.8x51mm",
        "7.62x39mm", "7.62x51mm", "7.62x54mm", "9x19mm", "9x39mm",
        "4.6x30mm", ".300 BLK", "箭矢", "45-70 Govt",
    ],
    "收集品": ["电子物品", "医疗道具", "工具材料", "家居物品", "工艺藏品", "资料情报", "能源燃料"],
    "消耗品": ["药品", "维修套件", "针剂"],
    "钥匙": ["长弓溪谷", "零号大坝", "航天基地", "巴克什", "潮汐监狱"],
}

# ═══════════════════════════════════════════════════════════
# 仓库箱子定义 (从上到下, 默认顺序)
# ═══════════════════════════════════════════════════════════

# 最多 10 个箱子 (含主仓库). 第一个固定为"主仓库", 其余只用序号命名, 由用户校准坐标.
MAX_STORAGE_BOXES = 10
STORAGE_BOXES = ["主仓库"] + [f"箱子{i}" for i in range(2, MAX_STORAGE_BOXES + 1)]

# ═══════════════════════════════════════════════════════════
# 屏幕坐标配置 (2560x1440 物理像素)
# 所有坐标均为游戏窗口内的绝对坐标
# *** 以下为初始估算值, 首次使用请通过"调试截图"校准 ***
# ═══════════════════════════════════════════════════════════

SCREEN = {
    "width": 2560,
    "height": 1440,
}

# 交易行 → 出售 标签页坐标
TAB_COORDS = {
    "交易行": (950, 42),
    "出售":   (430, 82),
}

# ─── "整理仓库" 功能按钮 (xj.png 红框) ───
# 点击后展开所有箱子列表, 再点"整理"按钮回到主视图
ORGANIZE_BTN = {
    "icon_x": 1558,      # "整理仓库"小图标 x (红框内的眼睛图标)
    "icon_y": 1218,       # "整理仓库"小图标 y
    "sort_btn_x": 2370,   # 右下角"整理"按钮 x
    "sort_btn_y": 1224,   # 右下角"整理"按钮 y
}

# ─── 右侧面板: 仓库箱子选择器 (竖排小图标, 见 3.png) ───
# positions 为每个箱子的 Y 坐标列表 (长度 = MAX_STORAGE_BOXES).
# 不同箱子间距可能不一致, 用户可在坐标校准里单独为每个箱子拾取 Y 坐标.
# 默认值按 45px 等距生成, 仅作为占位; 首次使用请用"截图拾取坐标"校准每个箱子.
BOX_SELECTOR = {
    "x": 1575,
    "positions": [118 + i * 45 for i in range(MAX_STORAGE_BOXES)],
    "count": MAX_STORAGE_BOXES,
}

# ─── 右侧面板: 道具网格区域 ───
ITEM_GRID = {
    "x_start": 1642,     # 网格左边界 (箱子图标右侧)
    "y_start": 160,      # 网格上边界 (标题文字下方)
    "x_end": 2488,       # 网格右边界 (滚动条左侧)
    "y_end": 1360,       # 网格下边界 (可见区域底部)
    "scroll_area_x": 2060,  # 滚动时鼠标放置的x位置
    "scroll_area_y": 700,   # 滚动时鼠标放置的y位置
}

# ─── 9列标准化网格 (出售界面右侧面板) ───
# 出售界面固定 9列 × 12行可见, 总行数取决于仓库容量 (如459格/9=51行)
CELL_GRID = {
    "origin_x": 1651,    # 第一个格子(R0C0)左上角 x
    "origin_y": 244,     # 第一个格子(R0C0)左上角 y
    "cell_w": 85,        # 每格宽度 (像素)
    "cell_h": 85,        # 每格高度 (像素)
    "cols": 9,           # 固定9列
    "visible_rows": 12,  # 一屏最多可见12行
}

# ─── 非绑定道具检测参数 (出售界面) ───
# 在出售界面中, 非绑定道具背景更亮 (HSV V > 阈值)
BIND_DETECTION = {
    "v_threshold": 41,          # HSV V值阈值: V > 此值 = 非绑定
    "sample_y1": 3,             # 背景采样区域 y起始 (格子内偏移)
    "sample_y2": 25,            # 背景采样区域 y结束
    "sample_x1": 3,             # 背景采样区域 x起始
    "sample_x2": 40,            # 背景采样区域 x结束
    "empty_threshold": 20,      # 格子中心区域均值低于此值 = 空格
}

# ─── 左侧面板: 已上架道具列表 ───
LISTED_ITEMS = {
    "x_start": 50,
    "y_start": 135,       # 第一个已上架道具的y起始
    "item_height": 190,   # 每个已上架道具的间距
    "delist_btn_x": 1000, # "下架"按钮x中心
    "delist_btn_y": 380,  # 第一个"下架"按钮y (始终点第一个, 下架后列表上移)
    "delist_btn_offset_y": 150,  # "下架"按钮相对于道具起始y的偏移
    "view_btn_offset_y": 120,    # "查看"按钮相对于道具起始y的偏移
    "max_visible": 5,     # 一页最多可见的已上架道具数
    "scroll_area_x": 300, # 左侧列表滚动区域x
    "scroll_area_y": 400, # 左侧列表滚动区域y
    "confirm_btn_x": 1496,  # 下架确认弹窗中"下架"按钮 x
    "confirm_btn_y": 907,   # 下架确认弹窗中"下架"按钮 y
}

# ─── 上架槽位信息 ───
LISTING_SLOTS = {
    "max": 15,
    "counter_region": (222, 204, 296, 230),
}

# ─── 跨模块 / 工作流内部使用的检测 ROI ───
# 这些矩形区域用于 "页面状态判定" 而不是 OCR, 但同样放进 DETECTION_ROI 里,
# 以便在 GUI 坐标校准里统一让用户框选:
#   page_change : 出售界面 ↔ 上架弹窗的视觉差异区
#                 (出售页 = 深色道具网格; 上架弹窗 = 白/金大字 + 滑块 + 绿色确认钮)
#                 两者缩略图 mean_diff 通常 > 25, 远大于 has_page_changed 的 10 阈值
#                 用局部抓取代替全屏, 单次从 ~50-80ms 降到 ~10-20ms, 每件上架省 ~150ms
#   sell_tab    : "出售" 二级 tab 按钮所在小块 (亮绿高亮时 mean_brightness > 35;
#                 被弹窗/增加槽位页覆盖时亮度骤降, 用于判定是否回到出售页)
DETECTION_ROI = {
    "page_change": (1440, 390, 2080, 960),
    "sell_tab":    (340, 132, 520, 180),
}

# 保留旧符号名做向后兼容 (导入处仍可用 from config import PAGE_CHANGE_ROI).
# 注意: 用户通过坐标校准修改后, 实时生效的是 DETECTION_ROI["page_change"],
# 代码里需要最新值时一律从 DETECTION_ROI 读取, 不要再读这个模块级常量.
PAGE_CHANGE_ROI = DETECTION_ROI["page_change"]

# ─── 上架弹窗 (sj1.png / sj2.png 的 "上架物品" 对话框) ───
# 点击道具后弹出的上架物品界面
LIST_DIALOG = {
    "qty_minus_btn": (1540, 700),   # 出售数量 "-" 按钮
    "qty_plus_btn":  (2020, 700),   # 出售数量 "+" 按钮
    "qty_slider_left":  (1580, 700),  # 滑块左端
    "qty_slider_right": (1980, 700),  # 滑块右端
    "qty_slider_max":   (1936, 747),  # 滑动条最右端 (点击一次即拉满数量)
    "price_input":   (1790, 775),   # 价格输入框 (中心)
    "price_minus":   (1540, 775),   # 价格 "-" 按钮
    "price_plus":    (2020, 775),   # 价格 "+" 按钮
    "list_btn":      (1792, 900),   # 绿色 "上架" 按钮
    "esc_btn":       (1280, 1280),  # "ESC 返回" 按钮
    "max_qty_clicks": 50,           # 最多点击"+"次数 (超过实际上限无影响)

    # OCR 识别区域 (x1, y1, x2, y2) — 用于读取道具名称和预期收入
    "item_name_region": (1440, 400, 2080, 470),   # 道具名称文字 (如 "JAD下挂激光")
    "income_region":    (1580, 840, 1860, 900),    # 预期收入数字 (如 "180,338")
    "dialog_crop":      (1440, 390, 2080, 960),    # 弹窗右侧面板截图区域
}

# ═══════════════════════════════════════════════════════════
# 自动化参数
# ═══════════════════════════════════════════════════════════

TIMING = {
    "click_delay": 0.06,        # 普通点击后等待 (秒)
    "action_delay": 0.3,        # 重要操作后等待 (上架确认等)
    "scroll_delay": 0.10,       # 滚动后等待
    "box_switch_delay": 0.5,    # 切换箱子后等待
    "page_load_delay": 0.5,     # 页面加载等待
    "dialog_open_delay": 0.4,   # 弹窗打开等待
    "dialog_close_delay": 0.25, # 弹窗关闭后等待
    "qty_click_delay": 0.04,    # 数量"+"按钮连点间隔
    "random_min": 0.01,         # 随机延时最小值
    "random_max": 0.06,         # 随机延时最大值
    # 滑动条拉满出售数量点击次数: 1=默认(省 0.4~0.7s/件),
    # 2=保守 (游戏动画罕见吞点击时设为 2)
    "maximize_clicks": 1,
    # 下架时每 N 次才做一次 OCR 检查上架剩余数量 (N=1 即每次都查).
    # 调高到 3 可省 ~60% OCR 时间; 仓库满/异常时靠后续重复读数兜底.
    "delist_ocr_every": 3,
}

HOTKEYS = {
    "start_list": "ctrl+1",
    "stop": "ctrl+2",
    "start_delist": "ctrl+3",
    "debug": "ctrl+4",
    "coord_pick": "ctrl+z",
}

# ═══════════════════════════════════════════════════════════
# ML 模型路径配置
# ═══════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(__file__)
ML_SAMPLES_DIR = os.path.join(_BASE_DIR, "samples")
ML_LABELS_FILE = os.path.join(_BASE_DIR, "samples", "labels.json")
ML_MODEL_FILE = os.path.join(_BASE_DIR, "models", "bound_model.joblib")
ITEM_DB_FILE = os.path.join(_BASE_DIR, "item_database.json")
SHOP_DIR = os.path.join(_BASE_DIR, "Shop")

# ─── 道具采集 OCR 识别区域 (归一化比例, 适配任意分辨率) ───
# 只识别红框内的道具网格区域, 忽略左侧分类栏/顶部导航栏/底部状态栏.
# 实际大类 ROI 在 gui/collect_dialog.py 的 CATEGORY_LAYOUT 中按大类分别定义,
# 本常量是旧 API 的默认值 (非装备大类), 像素参考: (576, 205)→(2437, 1250) @2560×1440.
COLLECT_OCR_ROI = (0.225, 0.142, 0.952, 0.868)

# ═══════════════════════════════════════════════════════════
# 原始默认坐标 (不可变, 供"恢复默认"按钮使用)
# ═══════════════════════════════════════════════════════════

ORIGINAL_DEFAULT_COORDS = {
    # 道具网格 (出售界面右侧)
    "grid_x_start": 1642, "grid_y_start": 160,
    "grid_x_end":   2488, "grid_y_end":   1360,
    "scroll_area_x": 2060, "scroll_area_y": 700,

    # 上架弹窗 — 点击按钮
    "qty_plus_x":  2020, "qty_plus_y":  700,
    "qty_minus_x": 1540, "qty_minus_y": 700,
    "slider_left_x":  1580, "slider_left_y":  700,
    "slider_right_x": 1980, "slider_right_y": 700,
    "qty_max_x":   1936, "qty_max_y":   747,
    "price_input_x": 1790, "price_input_y": 775,
    "price_minus_x": 1540, "price_minus_y": 775,
    "price_plus_x":  2020, "price_plus_y":  775,
    "list_btn_x":   1792, "list_btn_y":   900,
    "esc_btn_x":    1280, "esc_btn_y":    1280,

    # 仓库管理
    "org_x":  1558, "org_y":  1218,
    "sort_x": 2370, "sort_y": 1224,

    # 箱子选择器共用 X
    "box_x":  1575,

    # 下架相关 (左侧已上架列表 + 下架确认弹窗)
    "delist_btn_x":  1000, "delist_btn_y":  380,
    "confirm_btn_x": 1496, "confirm_btn_y": 907,

    # 顶部标签页 (主 tab 栏 / 出售 tab)
    "tab_trade_x": 950, "tab_trade_y": 42,
    "tab_sell_x":  430, "tab_sell_y":  82,

    # OCR 识别区域 (矩形 x1,y1,x2,y2) — 由用户在游戏截图里手动框选
    "name_x1": 1440, "name_y1": 400, "name_x2": 2080, "name_y2": 470,
    "income_x1": 1580, "income_y1": 840, "income_x2": 1860, "income_y2": 900,
    "dialog_x1": 1440, "dialog_y1": 390, "dialog_x2": 2080, "dialog_y2": 960,
    "counter_x1": 222, "counter_y1": 204, "counter_x2": 296, "counter_y2": 230,

    # 检测 ROI (不跑 OCR 但同样需要区域校准: 页面跳转 / 出售 tab 亮度)
    "page_change_x1": 1440, "page_change_y1": 390,
    "page_change_x2": 2080, "page_change_y2": 960,
    "sell_tab_x1": 340, "sell_tab_y1": 132,
    "sell_tab_x2": 520, "sell_tab_y2": 180,
}
# 默认每个箱子的 Y 坐标: box1_y ... box10_y
for _i in range(MAX_STORAGE_BOXES):
    ORIGINAL_DEFAULT_COORDS[f"box{_i + 1}_y"] = 118 + _i * 45
del _i

# ═══════════════════════════════════════════════════════════
# 用户配置 持久化
# ═══════════════════════════════════════════════════════════

DEFAULT_USER_CONFIG = {
    "selected_categories": {},   # {"枪械": ["步枪", "冲锋枪"], ...}
    "selected_rarities": ["紫色", "蓝色", "绿色", "白色"],
    "selected_boxes": ["主仓库"],
    "auto_price": True,         # 使用系统推荐价格
    "confirm_before_list": True,
}


def load_user_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_USER_CONFIG.copy()


def save_user_config(config: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ─── 坐标 ↔ 全局字典 映射表 ───
# key: SpinBox键名  →  映射说明
#   (cfg_dict, sub_key)           : cfg_dict[sub_key] = value  (简单值)
#   (cfg_dict, sub_key, idx)      : cfg_dict[sub_key][idx] = value  (元组/列表某位)
COORD_KEY_MAP = {
    # --- 道具网格 ---
    "grid_x_start": (ITEM_GRID, "x_start"),
    "grid_y_start": (ITEM_GRID, "y_start"),
    "grid_x_end":   (ITEM_GRID, "x_end"),
    "grid_y_end":   (ITEM_GRID, "y_end"),
    "scroll_area_x": (ITEM_GRID, "scroll_area_x"),
    "scroll_area_y": (ITEM_GRID, "scroll_area_y"),

    # --- 上架弹窗按钮 ---
    "qty_plus_x":     (LIST_DIALOG, "qty_plus_btn", 0),
    "qty_plus_y":     (LIST_DIALOG, "qty_plus_btn", 1),
    "qty_minus_x":    (LIST_DIALOG, "qty_minus_btn", 0),
    "qty_minus_y":    (LIST_DIALOG, "qty_minus_btn", 1),
    "slider_left_x":  (LIST_DIALOG, "qty_slider_left", 0),
    "slider_left_y":  (LIST_DIALOG, "qty_slider_left", 1),
    "slider_right_x": (LIST_DIALOG, "qty_slider_right", 0),
    "slider_right_y": (LIST_DIALOG, "qty_slider_right", 1),
    "qty_max_x":      (LIST_DIALOG, "qty_slider_max", 0),
    "qty_max_y":      (LIST_DIALOG, "qty_slider_max", 1),
    "price_input_x":  (LIST_DIALOG, "price_input", 0),
    "price_input_y":  (LIST_DIALOG, "price_input", 1),
    "price_minus_x":  (LIST_DIALOG, "price_minus", 0),
    "price_minus_y":  (LIST_DIALOG, "price_minus", 1),
    "price_plus_x":   (LIST_DIALOG, "price_plus", 0),
    "price_plus_y":   (LIST_DIALOG, "price_plus", 1),
    "list_btn_x":     (LIST_DIALOG, "list_btn", 0),
    "list_btn_y":     (LIST_DIALOG, "list_btn", 1),
    "esc_btn_x":      (LIST_DIALOG, "esc_btn", 0),
    "esc_btn_y":      (LIST_DIALOG, "esc_btn", 1),

    # --- 仓库管理 ---
    "box_x":  (BOX_SELECTOR, "x"),
    "org_x":  (ORGANIZE_BTN, "icon_x"),
    "org_y":  (ORGANIZE_BTN, "icon_y"),
    "sort_x": (ORGANIZE_BTN, "sort_btn_x"),
    "sort_y": (ORGANIZE_BTN, "sort_btn_y"),

    # --- 下架相关 ---
    "delist_btn_x":  (LISTED_ITEMS, "delist_btn_x"),
    "delist_btn_y":  (LISTED_ITEMS, "delist_btn_y"),
    "confirm_btn_x": (LISTED_ITEMS, "confirm_btn_x"),
    "confirm_btn_y": (LISTED_ITEMS, "confirm_btn_y"),

    # --- 顶部 tab ---
    "tab_trade_x": (TAB_COORDS, "交易行", 0),
    "tab_trade_y": (TAB_COORDS, "交易行", 1),
    "tab_sell_x":  (TAB_COORDS, "出售",   0),
    "tab_sell_y":  (TAB_COORDS, "出售",   1),

    # --- OCR 区域: 道具名称 ---
    "name_x1": (LIST_DIALOG, "item_name_region", 0),
    "name_y1": (LIST_DIALOG, "item_name_region", 1),
    "name_x2": (LIST_DIALOG, "item_name_region", 2),
    "name_y2": (LIST_DIALOG, "item_name_region", 3),

    # --- OCR 区域: 预期收入 ---
    "income_x1": (LIST_DIALOG, "income_region", 0),
    "income_y1": (LIST_DIALOG, "income_region", 1),
    "income_x2": (LIST_DIALOG, "income_region", 2),
    "income_y2": (LIST_DIALOG, "income_region", 3),

    # --- OCR 区域: 弹窗面板 (也用作 dialog_crop 的 RapidOCR 范围) ---
    "dialog_x1": (LIST_DIALOG, "dialog_crop", 0),
    "dialog_y1": (LIST_DIALOG, "dialog_crop", 1),
    "dialog_x2": (LIST_DIALOG, "dialog_crop", 2),
    "dialog_y2": (LIST_DIALOG, "dialog_crop", 3),

    # --- OCR 区域: 上架槽位计数 ---
    "counter_x1": (LISTING_SLOTS, "counter_region", 0),
    "counter_y1": (LISTING_SLOTS, "counter_region", 1),
    "counter_x2": (LISTING_SLOTS, "counter_region", 2),
    "counter_y2": (LISTING_SLOTS, "counter_region", 3),

    # --- 检测 ROI: 页面跳转 ---
    "page_change_x1": (DETECTION_ROI, "page_change", 0),
    "page_change_y1": (DETECTION_ROI, "page_change", 1),
    "page_change_x2": (DETECTION_ROI, "page_change", 2),
    "page_change_y2": (DETECTION_ROI, "page_change", 3),

    # --- 检测 ROI: 出售 tab 高亮 ---
    "sell_tab_x1": (DETECTION_ROI, "sell_tab", 0),
    "sell_tab_y1": (DETECTION_ROI, "sell_tab", 1),
    "sell_tab_x2": (DETECTION_ROI, "sell_tab", 2),
    "sell_tab_y2": (DETECTION_ROI, "sell_tab", 3),
}
# 每个箱子的 Y 坐标独立校准: box1_y .. box10_y -> BOX_SELECTOR["positions"][i]
for _i in range(MAX_STORAGE_BOXES):
    COORD_KEY_MAP[f"box{_i + 1}_y"] = (BOX_SELECTOR, "positions", _i)
del _i


def apply_saved_coordinates(saved: dict):
    """将保存的坐标值写回全局配置字典.

    兼容旧版 user_config.json: 若仅保存了 box_y_start/box_y_step (无 box{i}_y),
    则按等距展开为 10 个 positions.
    """
    # 1) 兼容迁移: 旧配置 -> 新 box{i}_y
    if "box_y_start" in saved and not any(
        f"box{i}_y" in saved for i in range(1, MAX_STORAGE_BOXES + 1)
    ):
        y_start = int(saved.get("box_y_start", 118))
        y_step = int(saved.get("box_y_step", 45))
        for i in range(MAX_STORAGE_BOXES):
            saved[f"box{i + 1}_y"] = y_start + i * y_step

    # 2) 常规写回
    for key, value in saved.items():
        mapping = COORD_KEY_MAP.get(key)
        if not mapping:
            continue
        if len(mapping) == 3:
            cfg_dict, cfg_key, idx = mapping
            old = cfg_dict[cfg_key]
            if isinstance(old, list):
                if 0 <= idx < len(old):
                    old[idx] = value
            else:
                lst = list(old)
                if 0 <= idx < len(lst):
                    lst[idx] = value
                    cfg_dict[cfg_key] = tuple(lst)
        else:
            cfg_dict, cfg_key = mapping
            cfg_dict[cfg_key] = value


# ═══════════════════════════════════════════════════════════
# 参考分辨率坐标快照 (2560x1440 原始值)
# 用于多分辨率缩放时从原始值出发, 避免累积误差
# ═══════════════════════════════════════════════════════════

_ALL_COORD_DICTS = {
    "SCREEN": SCREEN,
    "TAB_COORDS": TAB_COORDS,
    "ORGANIZE_BTN": ORGANIZE_BTN,
    "BOX_SELECTOR": BOX_SELECTOR,
    "ITEM_GRID": ITEM_GRID,
    "CELL_GRID": CELL_GRID,
    "BIND_DETECTION": BIND_DETECTION,
    "LISTED_ITEMS": LISTED_ITEMS,
    "LISTING_SLOTS": LISTING_SLOTS,
    "LIST_DIALOG": LIST_DIALOG,
    "DETECTION_ROI": DETECTION_ROI,
    "ORIGINAL_DEFAULT_COORDS": ORIGINAL_DEFAULT_COORDS,
}

_REFERENCE_SNAPSHOT = copy.deepcopy(_ALL_COORD_DICTS)


def reset_to_reference():
    """将所有坐标字典恢复到 2560x1440 参考值"""
    for name, ref_dict in _REFERENCE_SNAPSHOT.items():
        target = _ALL_COORD_DICTS[name]
        target.clear()
        target.update(copy.deepcopy(ref_dict))
