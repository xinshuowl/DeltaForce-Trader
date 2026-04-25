"""
鼠标/键盘自动化操作封装
坐标为游戏窗口内坐标, 通过 offset 映射到屏幕绝对坐标
"""
import time
import random
import threading
import pyautogui
from config import TIMING, LIST_DIALOG, ORGANIZE_BTN

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


class GameAutomation:
    """游戏自动化操作器"""

    def __init__(self, offset_x: int = 0, offset_y: int = 0):
        self._stop_event = threading.Event()
        self._timing = TIMING
        self._ox = offset_x
        self._oy = offset_y

    def set_offset(self, offset_x: int, offset_y: int):
        self._ox = offset_x
        self._oy = offset_y

    def _abs(self, x: int, y: int) -> tuple[int, int]:
        """游戏窗口坐标 -> 屏幕绝对坐标"""
        return x + self._ox, y + self._oy

    def stop(self):
        self._stop_event.set()

    def resume(self):
        self._stop_event.clear()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def check_stop(self):
        if self._stop_event.is_set():
            raise StopIteration("操作已被用户中断")

    def _random_delay(self):
        time.sleep(random.uniform(
            self._timing["random_min"],
            self._timing["random_max"],
        ))

    # ═══════════════════════════════════════════════════════
    # 基础操作
    # ═══════════════════════════════════════════════════════

    def move_to(self, x: int, y: int, duration: float = 0.05):
        self.check_stop()
        ax, ay = self._abs(x, y)
        pyautogui.moveTo(ax, ay, duration=duration)

    # 鼠标 "停车位" — 出售页面左下角空白处 (y=1350 超过网格 y_end=1360 外, x=30 远离网格)
    # 必须满足:
    #   1. 不在 ITEM_GRID (1642-2488, 160-1360) 内, 否则触发道具悬浮框
    #   2. 不在 PAGE_CHANGE_ROI (1440, 390, 2080, 960) 内, 否则鼠标光标
    #      会出现在页面跳变比对的截图中影响 mean_diff
    #   3. 不在 "等待上架" 槽位 / "整理仓库" 按钮 / tab 栏等可点击元素上,
    #      避免不小心 hover 弹 tooltip
    _PARK_POS = (30, 1350)

    def park_mouse(self, duration: float = 0.04):
        """把鼠标挪到安全区, 避免悬浮在道具网格上触发道具详情弹窗.

        应在每次 "点击上架确认" 后、翻页滚动后调用, 防止下一次截图时
        因道具悬浮框挡住网格 → ML 识别失败 / 页面变化检测误判.
        """
        self.check_stop()
        # PARK_POS 是 "游戏窗口坐标", 经 _abs 换算到屏幕坐标:
        # - 全屏模式 offset=(0,0): 落在屏幕左下 (30, 1350), 游戏内空白区
        # - 窗口模式有 offset: 仍落在游戏窗口内的左下空白区, 不会漂出游戏
        ax, ay = self._abs(*self._PARK_POS)
        pyautogui.moveTo(ax, ay, duration=duration)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        self.check_stop()
        ax, ay = self._abs(x, y)
        jx = random.randint(-1, 1)
        jy = random.randint(-1, 1)
        pyautogui.moveTo(ax + jx, ay + jy, duration=random.uniform(0.05, 0.12))
        self._random_delay()
        for _ in range(clicks):
            pyautogui.mouseDown(button=button)
            time.sleep(random.uniform(0.03, 0.06))
            pyautogui.mouseUp(button=button)
            if clicks > 1:
                time.sleep(random.uniform(0.02, 0.05))
        time.sleep(self._timing["click_delay"])

    def fast_click(self, x: int, y: int):
        """快速点击, 用于连续点击"+"按钮等场景"""
        self.check_stop()
        ax, ay = self._abs(x, y)
        pyautogui.click(ax, ay)
        time.sleep(self._timing["qty_click_delay"])

    def scroll(self, x: int, y: int, clicks: int = -3):
        self.check_stop()
        ax, ay = self._abs(x, y)
        pyautogui.moveTo(ax, ay, duration=0.05)
        time.sleep(0.05)
        pyautogui.scroll(clicks, ax, ay)
        time.sleep(self._timing["scroll_delay"])

    def press_key(self, key: str):
        self.check_stop()
        pyautogui.press(key)
        self._random_delay()

    def wait(self, seconds: float):
        end_time = time.time() + seconds
        while time.time() < end_time:
            self.check_stop()
            time.sleep(min(0.1, end_time - time.time()))

    # ═══════════════════════════════════════════════════════
    # 导航操作
    # ═══════════════════════════════════════════════════════

    def navigate_to_sell_tab(self, tab_coords: dict):
        """导航到 交易行 → 出售 页面"""
        self.click(*tab_coords["交易行"])
        self.wait(self._timing["page_load_delay"])
        self.click(*tab_coords["出售"])
        self.wait(self._timing["page_load_delay"])

    def click_organize_storage(self):
        """点击"整理仓库"图标按钮 (展开箱子列表)"""
        self.click(ORGANIZE_BTN["icon_x"], ORGANIZE_BTN["icon_y"])
        self.wait(self._timing["action_delay"])

    def click_sort_button(self):
        """点击"整理"按钮 (整理箱子并返回主视图)"""
        self.click(ORGANIZE_BTN["sort_btn_x"], ORGANIZE_BTN["sort_btn_y"])
        self.wait(self._timing["page_load_delay"])

    def click_box_selector(self, box_index: int, box_cfg: dict):
        """点击箱子选择器图标 (0-based index).

        箱子 Y 坐标从 box_cfg["positions"] 读取 (每个箱子独立校准).
        若缺失 positions 字段, 退化到旧的 y_start + y_step 计算.
        """
        x = box_cfg["x"]
        positions = box_cfg.get("positions")
        if positions and 0 <= box_index < len(positions):
            y = positions[box_index]
        else:
            y = box_cfg.get("y_start", 118) + box_index * box_cfg.get("y_step", 45)
        self.click(x, y)
        self.wait(self._timing["box_switch_delay"])

    # ═══════════════════════════════════════════════════════
    # 道具网格操作
    # ═══════════════════════════════════════════════════════

    def click_grid_cell(self, grid_cfg: dict, cell_x: int, cell_y: int,
                        cell_w: int, cell_h: int):
        """点击道具网格中的一个格子 (打开上架弹窗)"""
        abs_x = grid_cfg["x_start"] + cell_x + cell_w // 2
        abs_y = grid_cfg["y_start"] + cell_y + cell_h // 2
        self.click(abs_x, abs_y)
        self.wait(self._timing["dialog_open_delay"])

    _SCROLL_BATCH = 50  # 单次滚轮最大格数 (超过部分会被 OS 吞掉)

    def scroll_item_grid(self, grid_cfg: dict, direction: str = "down",
                         amount: int = 3, fast: bool = False):
        """滚动道具网格.

        amount 为鼠标滚轮总格数, 会分批执行以避免单次滚轮量过大被吞掉.
        每批最多滚 _SCROLL_BATCH 格.

        fast=True: 快速模式, 批次间只留 0.02s (够 OS 处理事件队列即可),
                   跳过每批的 moveTo + 0.05s setup 等开销. 用于 "翻页判空"
                   这种不需要对滚动位置特别精确的场景 (只在乎总体滚完了).
                   对比默认模式每批 ~0.2s, 3 批省 ~0.5s.
        """
        x = grid_cfg["scroll_area_x"]
        y = grid_cfg["scroll_area_y"]
        sign = -1 if direction == "down" else 1

        if fast:
            # 只 moveTo 一次, 后续连续发滚轮事件, 批次间仅短暂停顿避免 OS 合并
            ax, ay = self._abs(x, y)
            self.check_stop()
            pyautogui.moveTo(ax, ay, duration=0.03)
            time.sleep(0.03)
            remaining = amount
            while remaining > 0:
                self.check_stop()
                batch = min(remaining, self._SCROLL_BATCH)
                pyautogui.scroll(sign * batch, ax, ay)
                remaining -= batch
                if remaining > 0:
                    time.sleep(0.02)
            return

        remaining = amount
        while remaining > 0:
            self.check_stop()
            batch = min(remaining, self._SCROLL_BATCH)
            self.scroll(x, y, clicks=sign * batch)
            remaining -= batch

    # ═══════════════════════════════════════════════════════
    # 上架弹窗操作 (xj2.png / xj3.png)
    # ═══════════════════════════════════════════════════════

    def maximize_quantity(self, clicks: int = 1):
        """点击滑动条最右端拉满出售数量.

        clicks=1 (默认): 单次可靠点击, 总耗时 ~0.45s, 比原 2 次 (1.0s+) 显著更快.
        clicks=2: 保守模式, 两次可靠点击. 可通过 config.TIMING["maximize_clicks"] 切换.

        为什么 "单次点击偶尔不成功":
          - 进入上架弹窗后 UI 仍有渐入动画 (~100ms), 此时点击会被动画层吞掉
          - 游戏引擎需要至少 ~40-60ms (3-4 frame @ 60fps) 的按下时长才把点击当作
            "确定的左键", 过短的 mouseDown→mouseUp (~40ms) 偶尔在输入队列合并时
            被识别成悬停而非点击
          - pyautogui.moveTo 结束后立即 mouseDown 会被 Windows 视作 "鼠标轨迹尚
            未稳定", 游戏引擎对非稳态坐标的点击容错低

        针对性修复 (相比旧的 0.04~0.08s 按下 + 0.06~0.12s 移到点击间隔):
          1. 进入前先 sleep 0.08~0.14s 等 UI 动画渲染完成
          2. 移动后停顿 0.14~0.22s 让鼠标位置 "稳定" (原 0.06~0.12s 太短)
          3. 按下时长 0.10~0.18s, 确保游戏把它当 "实打实的点击" (原 0.04~0.08s)
          4. 连续点击间隔 0.14~0.20s, 让上一次点击的滑块动画有时间推进

        仍保留鼠标抖动 jx/jy (±2px) 和随机化延时, 防止反作弊系统把几何完美的
        点击位置 + 恒定时序识别成脚本.
        """
        gx, gy = LIST_DIALOG["qty_slider_max"]
        ax, ay = self._abs(gx, gy)

        # UI 渲染稳定窗口: 弹窗打开后的渐入动画约 100ms, 在此之前的点击会被吞
        time.sleep(random.uniform(0.08, 0.14))

        # 第一次: 带 moveTo 模拟人工走位
        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        pyautogui.moveTo(ax + jx, ay + jy, duration=random.uniform(0.18, 0.28))
        # 移动后足够的稳定时间 (重要: 原 0.06~0.12s 太短, 游戏偶尔认为坐标还在
        # 运动中, 不触发点击落位逻辑)
        time.sleep(random.uniform(0.14, 0.22))
        pyautogui.mouseDown(button="left")
        # 按下时长 >= 100ms, 保证游戏引擎把它当成 "有意的左键" 而不是悬停扫过
        time.sleep(random.uniform(0.10, 0.18))
        pyautogui.mouseUp(button="left")

        # 保守模式下的额外点击: 不再做慢 moveTo, 但保留足够的 "滑块动画间隔"
        for _ in range(max(0, clicks - 1)):
            time.sleep(random.uniform(0.14, 0.20))
            pyautogui.mouseDown(button="left")
            time.sleep(random.uniform(0.10, 0.18))
            pyautogui.mouseUp(button="left")

        # 点击后的稳定窗口, 给后续 grab_full 留一点让 UI 刷新的时间
        time.sleep(random.uniform(0.10, 0.16))

    def click_list_confirm(self):
        """点击上架弹窗中的绿色"上架"按钮"""
        x, y = LIST_DIALOG["list_btn"]
        self.click(x, y)

    def close_list_dialog(self):
        """按ESC关闭上架弹窗 (不上架)"""
        self.press_key("escape")
        self.wait(self._timing["dialog_close_delay"])

    def perform_listing(self):
        """
        完整的单次上架操作:
        1. (已在弹窗中) 点击"+"按钮将数量拉满
        2. 点击绿色"上架"按钮确认
        """
        self.maximize_quantity()
        time.sleep(0.2)
        self.click_list_confirm()

    # ═══════════════════════════════════════════════════════
    # 下架操作
    # ═══════════════════════════════════════════════════════

    def _human_click(self, x: int, y: int):
        """模拟真人点击: 自然移动→停顿→按下→松开"""
        self.check_stop()
        ax, ay = self._abs(x, y)
        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        pyautogui.moveTo(ax + jx, ay + jy, duration=random.uniform(0.15, 0.25))
        time.sleep(random.uniform(0.05, 0.10))
        pyautogui.mouseDown(button="left")
        time.sleep(random.uniform(0.04, 0.08))
        pyautogui.mouseUp(button="left")
        time.sleep(random.uniform(0.08, 0.15))

    def click_delist_button(self, listed_cfg: dict):
        """
        完成单次下架: 点击列表第一个"下架"按钮 → 等待确认弹窗 → 点击确认.
        始终点第一个, 因为下架后列表自动上移.
        """
        x = listed_cfg["delist_btn_x"]
        y = listed_cfg["delist_btn_y"]
        self._human_click(x, y)
        self.wait(random.uniform(0.4, 0.6))

        cx = listed_cfg["confirm_btn_x"]
        cy = listed_cfg["confirm_btn_y"]
        self._human_click(cx, cy)
        self.wait(random.uniform(0.4, 0.6))

    def scroll_listed_items(self, listed_cfg: dict, direction: str = "down",
                            amount: int = 2):
        """滚动左侧已上架列表"""
        clicks = -amount if direction == "down" else amount
        self.scroll(
            listed_cfg["scroll_area_x"],
            listed_cfg["scroll_area_y"],
            clicks=clicks,
        )
