"""
上架/下架业务流程引擎 v4
- 9列网格 + HSV背景亮度检测非绑定道具
- 从左到右、从上到下逐格扫描
- 点击后检测弹窗判断是否可上架
- 最大化数量后 OCR 读取道具名称和预期收入
- 上架完成后汇总收入统计
"""
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config import (
    Rarity, ITEM_GRID, BOX_SELECTOR, LISTED_ITEMS,
    STORAGE_BOXES, TIMING, CELL_GRID, ML_MODEL_FILE, ITEM_DB_FILE,
    DETECTION_ROI,
)
from core.screen import ScreenCapture, save_debug_screenshot
from core.detector import UnboundItemDetector
from core.ml_detector import MLBoundDetector
from core.automation import GameAutomation
from core.item_database import ItemDatabase

logger = logging.getLogger("workflow")


# ─── 道具图像指纹 ────────────────────────────────────────────────
# 用于在扫描时识别 "同一道具的不同 stack 格子".
#
# 背景: 一个道具如果数量超过单格堆叠上限 (大多数 50 / 100), 会被拆成多个
#       独立的格子并排显示, ML 检测器只能告诉我们"这格是非绑定", 不知道
#       两格是不是同一种道具. 用 (row, col) 去重无效 — 每个 stack 占的
#       row/col 不同, 但都是同一个道具, 不符筛选时会被反复点开.
#
# 解决: 计算每个格子图标的 dHash 指纹 (64-bit, 横向梯度差).
#       - 同一道具图标在不同位置/不同页, 指纹一致 (像素稳定渲染时精确匹配)
#       - 不同道具图标指纹不同
#       - 占 8 字节, 用作 set 的 key 极快
def _cell_fingerprint(cell_bgr: np.ndarray) -> bytes | None:
    """计算格子图像的 dHash 指纹 (64-bit, 8 字节).

    实现: 转灰度 -> 裁去 1/8 边距 (聚焦中心图标, 避免边框装饰干扰) ->
          缩到 9×8 -> 横向相邻像素亮度差的二值化 -> 打包成 bytes.

    返回 None 表示输入太小或无效, 调用方应跳过指纹判断.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return None
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 16 or w < 16:
        return None
    my = h // 8
    mx = w // 8
    cropped = gray[my:h - my, mx:w - mx]
    try:
        small = cv2.resize(cropped, (9, 8), interpolation=cv2.INTER_AREA)
    except cv2.error:
        return None
    diff = small[:, 1:] > small[:, :-1]
    # packbits: 64 bool -> 8 字节真正的 64-bit 指纹
    return np.packbits(diff.flatten()).tobytes()


def _crop_cell(screenshot: np.ndarray, row: int, col: int) -> np.ndarray | None:
    """从全屏截图裁出 (row, col) 格子的 BGR 小图. 越界返回 None."""
    cg = CELL_GRID
    x = cg["origin_x"] + col * cg["cell_w"]
    y = cg["origin_y"] + row * cg["cell_h"]
    cw, ch = cg["cell_w"], cg["cell_h"]
    h, w = screenshot.shape[:2]
    if x < 0 or y < 0 or x + cw > w or y + ch > h:
        return None
    return screenshot[y:y + ch, x:x + cw]


def _merge_into_groups(items: list[dict]) -> list[dict]:
    """把 ML 给出的非绑定格子按 4-邻居连通域合并成 "逻辑道具".

    背景: 多格大型道具 (如裸枪 4×2 占 8 格) 在 ML 输出里是多个独立的
    (row, col) cell — ML 只能告诉你 "这格非绑定", 不能告诉你哪几格属
    于同一道具. 但游戏里它们其实是同一个: 点击任意一格都跳到同一个
    上架弹窗, 读到的也是同一个名字. 不合并 → 一把枪被点 N 次.

    合并策略:
    - 4-邻居 (上下左右), 不含对角. 对角合并会把相邻但独立的道具串成一组.
    - 每个连通块选最左上 (row,col 字典序最小) 的 cell 作为"代表", 后续
      点击/计算指纹都用代表 cell.
    - 代表 cell 上挂 _group 字段, 列出整块所有 (row, col), 后续 skip 时
      整组一并加入位置黑名单.

    返回的列表长度 ≤ 输入. 顺序按代表 cell 的 (row, col) 升序.
    """
    if not items:
        return items

    cells = {(it["row"], it["col"]): it for it in items}
    visited: set[tuple[int, int]] = set()
    representatives: list[dict] = []

    # 按 (row, col) 升序遍历, 保证每块第一个被访问的就是左上角
    for start in sorted(cells.keys()):
        if start in visited:
            continue
        stack = [start]
        group: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or (r, c) not in cells:
                continue
            visited.add((r, c))
            group.append((r, c))
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        group.sort()
        rep_key = group[0]
        rep = dict(cells[rep_key])  # copy 避免影响原 item
        rep["_group"] = group
        representatives.append(rep)

    return representatives


@dataclass
class ListingRecord:
    """单次上架记录"""
    index: int
    name: str
    income: int
    box_name: str
    row: int
    col: int
    screenshot_path: str = ""


class WorkflowEngine:
    """上架/下架工作流引擎"""

    def __init__(self, on_status: Callable[[str], None] | None = None,
                 on_progress: Callable[[int, int], None] | None = None):
        self.screen = ScreenCapture()
        self.detector = UnboundItemDetector()
        self.ml_detector = MLBoundDetector(model_path=ML_MODEL_FILE)
        self.auto = GameAutomation()
        self._ocr_executor = ThreadPoolExecutor(max_workers=1)

        self._on_status = on_status or (lambda s: logger.info(s))
        self._on_progress = on_progress or (lambda c, t: None)
        self._listed_count = 0
        self._listing_records: list[ListingRecord] = []

    def _status(self, msg: str):
        self._on_status(msg)

    def _progress(self, current: int, total: int):
        self._on_progress(current, total)

    def stop(self):
        self.auto.stop()
        self._status("已停止")

    def resume(self):
        self.auto.resume()

    def shutdown(self):
        """应用退出时调用, 关闭线程池等资源."""
        try:
            self._ocr_executor.shutdown(wait=False)
        except Exception:
            pass

    @property
    def listing_records(self) -> list[ListingRecord]:
        return self._listing_records

    # ═══════════════════════════════════════════════════════
    # 上架流程
    # ═══════════════════════════════════════════════════════

    def _detect_listing_status(self) -> tuple[int, int, int]:
        """
        检测当前上架状态 (OCR 辅助, 不保证准确).
        Returns:
            (已上架数, 总槽位, 剩余可用槽位)
            识别失败返回 (-1, -1, -1)
        """
        screenshot = self.screen.grab_full()
        listed, total = self.detector.read_listing_count(screenshot)
        if listed >= 0 and total > 0:
            remaining = total - listed
            self._status(
                f"OCR 检测上架状态: {listed}/{total} "
                f"(剩余 {remaining} 个槽位)"
            )
            return listed, total, remaining
        self._status("OCR 无法识别上架数量")
        return -1, -1, -1

    def _detect_listing_status_reliable(self, retries: int = 3,
                                        delay: float = 0.3) -> tuple[int, int, int]:
        """
        可靠版本的槽位检测: 最多 retries 次读取, 返回最后一次成功结果.
        任意一次识别成功即返回 (避免因偶发 OCR 误判而误停流程).
        """
        last = (-1, -1, -1)
        for i in range(retries):
            listed, total, remaining = self._detect_listing_status()
            if listed >= 0 and total > 0:
                return listed, total, remaining
            last = (listed, total, remaining)
            if i < retries - 1:
                time.sleep(delay)
        return last

    def run_list_workflow(self,
                         selected_boxes: list[str],
                         selected_rarities: list[Rarity],
                         max_slots: int = 15,
                         organize_first: bool = True,
                         allowed_rarities: set[str] | None = None,
                         allowed_categories: dict[str, set[str]] | None = None,
                         mode: str = "single",
                         idle_ocr_interval_sec: int = 60,
                         idle_max_duration_min: int = 0):
        """上架工作流入口.

        Args:
            mode: "single" (默认, 执行一轮) 或 "idle" (挂机, 循环直到停止).
            idle_ocr_interval_sec: 挂机模式下槽位已满时的 OCR 重检间隔 (秒).
            idle_max_duration_min: 挂机模式最大执行时长 (分钟, 0=不限).
        """
        self.auto.resume()
        self._listed_count = 0
        self._listing_records.clear()
        total_boxes = len(selected_boxes)
        rarity_names = [r.value for r in selected_rarities]

        self._item_db = ItemDatabase(ITEM_DB_FILE)
        self._filter_rarities = allowed_rarities or set()
        self._filter_categories = allowed_categories or {}

        mode_tag = "挂机" if mode == "idle" else "单次"
        if self._item_db.count > 0 and self._filter_rarities:
            self._status(
                f"开始上架 [{mode_tag}] | 箱子: {total_boxes}个 | "
                f"品质: {rarity_names} | "
                f"数据库: {self._item_db.count}个道具 | "
                f"品质筛选: {self._filter_rarities}"
            )
        else:
            self._status(
                f"开始上架 [{mode_tag}] | 箱子: {total_boxes}个 | "
                f"品质: {rarity_names}"
            )

        os.makedirs("listings", exist_ok=True)

        try:
            if mode == "idle":
                self._run_idle_loop(
                    selected_boxes, max_slots, organize_first,
                    idle_ocr_interval_sec, idle_max_duration_min,
                )
            else:
                self._run_single_round(
                    selected_boxes, max_slots, organize_first
                )

            self._show_summary()

        except StopIteration:
            self._status(f"\n上架被中断, 已上架 {self._listed_count} 件")
            self._show_summary()
        except Exception as e:
            self._status(f"\n上架出错: {e}")
            logger.exception("上架工作流异常")
            self._show_summary()

    def _run_single_round(self,
                          selected_boxes: list[str],
                          max_slots: int,
                          organize_first: bool) -> int:
        """执行一轮上架 (所有选中箱子扫一遍). 返回本轮实际上架件数."""
        total_boxes = len(selected_boxes)
        round_start_count = self._listed_count

        # 尝试 OCR 检测当前上架数量 (可能不准, 最多重试 3 次避免偶发误读)
        already_listed, total_slots, remaining = self._detect_listing_status_reliable()
        if remaining == 0:
            # 首次读到 0 再额外确认一次, 避免 "0/15" 被误读成 "15/15"
            self._status("OCR 显示槽位已满, 二次确认中...")
            time.sleep(0.4)
            _l, _t, remaining2 = self._detect_listing_status()
            if remaining2 == 0:
                self._status("槽位已满, 无法继续上架! 请先下架部分道具.")
                return 0
            remaining = remaining2
        # 本轮最多上架 = 当前剩余槽位 + 本轮已经累计的上架数
        # (因为 _listed_count 会在每件上架后 +1, 需跟 max_slots 对齐)
        round_max_slots = self._listed_count + (remaining if remaining > 0 else max_slots)
        self._status(
            f"本轮最多上架 {round_max_slots - self._listed_count} 件 "
            f"(OCR 识别失败时默认 {max_slots})"
        )

        if organize_first:
            self._status("步骤1: 整理仓库...")
            self.auto.click_organize_storage()
            time.sleep(0.3)
            self.auto.click_sort_button()
            self._status("  仓库整理完成")

        for box_idx, box_name in enumerate(selected_boxes):
            self.auto.check_stop()
            self._status(f"\n[箱子 {box_idx + 1}/{total_boxes}] {box_name}")

            storage_index = self._get_box_index(box_name)
            if storage_index < 0:
                self._status(f"  未找到 '{box_name}', 跳过")
                continue

            self.auto.click_box_selector(storage_index, BOX_SELECTOR)
            listed_in_box = self._scan_and_list_box(box_name, round_max_slots)
            self._status(f"  {box_name}: 上架了 {listed_in_box} 件")

            if self._listed_count >= round_max_slots:
                self._status(f"已达上架上限 ({round_max_slots})")
                break

        return self._listed_count - round_start_count

    def _run_idle_loop(self,
                       selected_boxes: list[str],
                       max_slots: int,
                       organize_first: bool,
                       ocr_interval_sec: int,
                       max_duration_min: int):
        """挂机模式: 循环执行上架直到用户停止或达到最大时长.

        每轮完整地遍历选中箱子, 然后:
          - 若当轮实际上架 > 0 且仍有槽位: 立即继续下一轮
          - 若槽位已满: 按 ocr_interval_sec 节奏等待 + 重检
          - 若当轮实际上架 = 0 且槽位未满但仓库已无可上架道具:
                也按 ocr_interval_sec 等待, 给用户机会补货
        """
        start_time = time.time()
        round_idx = 0
        total_target_slots: int | None = None  # 挂机整体目标, 不限就是 None

        deadline = None
        if max_duration_min > 0:
            deadline = start_time + max_duration_min * 60
            self._status(
                f"挂机模式: 最大时长 {max_duration_min} 分钟 "
                f"(至 {time.strftime('%H:%M:%S', time.localtime(deadline))})"
            )
        else:
            self._status("挂机模式: 无时长限制, Ctrl+2 或停止按钮可中断")

        self._status(f"挂机模式: 槽位满时每 {ocr_interval_sec}s 重检一次")

        while True:
            self.auto.check_stop()
            if deadline and time.time() >= deadline:
                self._status(f"\n已达最大执行时长 ({max_duration_min} 分钟), 停止挂机")
                return

            round_idx += 1
            self._status(f"\n══ 第 {round_idx} 轮上架 ══")

            # 本轮开始前先 OCR 一次槽位状态: 若已满, 进入等待循环
            listed, total, remaining = self._detect_listing_status_reliable()
            if remaining == 0:
                self._status(f"  槽位已满 ({listed}/{total}), 等待重检")
                if not self._idle_wait_until_slot_free(ocr_interval_sec, deadline):
                    return  # 超时或被中断
                # 等到有空位了, 继续本轮

            # 执行一轮上架
            before_round = self._listed_count
            self._run_single_round(
                selected_boxes,
                max_slots,
                organize_first,
            )
            listed_this_round = self._listed_count - before_round

            self._status(
                f"  第 {round_idx} 轮小计: 上架 {listed_this_round} 件 | "
                f"累计 {self._listed_count} 件"
            )

            # 本轮为 0 件: 意味着要么槽位满, 要么仓库无可上架道具.
            # 两种情况都按 ocr_interval_sec 等待给外部条件变化的机会.
            if listed_this_round == 0:
                self._status(
                    f"  本轮未上架任何道具, {ocr_interval_sec}s 后重试"
                )
                if not self._idle_wait_until_slot_free(ocr_interval_sec, deadline,
                                                        reason="empty_round"):
                    return
            # 本轮有产出: 轮间留一个短暂呼吸 (避免 CPU 满速 + 让游戏 UI 刷新)
            else:
                self.auto.wait(1.0)

    def _idle_wait_until_slot_free(self,
                                   ocr_interval_sec: int,
                                   deadline: float | None,
                                   reason: str = "slot_full") -> bool:
        """挂机模式: 等待直到至少有 1 个空槽位可用.

        每 ocr_interval_sec 秒做一次 OCR 检查. 期间定期 check_stop 以便用户中断.
        Returns:
            True: 等到空位, 可以继续下一轮
            False: 超过最大时长或被用户停止 (调用方应直接 return)
        """
        while True:
            # 分段 sleep, 每段不超过 1s 以便及时响应 check_stop/停止事件
            waited = 0.0
            while waited < ocr_interval_sec:
                self.auto.check_stop()
                if deadline and time.time() >= deadline:
                    self._status("已达最大执行时长, 退出挂机等待")
                    return False
                step = min(1.0, ocr_interval_sec - waited)
                time.sleep(step)
                waited += step

            listed, total, remaining = self._detect_listing_status_reliable()
            if remaining > 0:
                if reason == "slot_full":
                    self._status(
                        f"  检测到空位 ({listed}/{total}, 剩 {remaining} 个), 继续上架"
                    )
                else:
                    self._status(
                        f"  再次扫描 ({listed}/{total}, 剩 {remaining} 个)"
                    )
                return True
            # 否则槽位依然满, 继续等
            if reason == "slot_full":
                self._status(
                    f"  槽位依然满 ({listed}/{total}), 再等 {ocr_interval_sec}s"
                )

    def _scan_and_list_box(self, box_name: str, max_slots: int) -> int:
        """
        扫描一个箱子, 按从左到右、从上到下的顺序逐个尝试上架.
        每次成功上架后回到出售界面会重新扫描 (因为上架后物品消失, 网格坐标变化).

        关键: 上架一种道具会一次清除该道具的所有实例 (如密令斜角 x5 → 5格变空),
        必须等页面完全刷新后再重新扫描, 并跳过刚清空的位置.
        """
        listed = 0
        skipped = 0
        scroll_count = 0
        max_scroll = 30
        empty_scans = 0
        # 连续 N 次空扫描后退出. 提升到 6 是为了覆盖 "顶部几页全绑定 / 全不在
        # 筛选范围" 的箱子 — 之前 2 太短会直接误判成空箱提前跳下一个.
        # 真正到底的箱子靠 _scroll_and_at_bottom 的双重确认提前 break, 不会
        # 因为 empty_scans_limit 大而拖慢空箱处理.
        empty_scans_limit = 6
        # 三层 skip 机制. 依次失效时由下一层兜底, 互相补充:
        #
        # 1) recently_listed_positions: 刚上架的格子残影 (本轮扫描内有效, 翻页清空).
        #    上架成功后页面尚未刷新, ML 可能仍把空格判为非绑定, 用位置临时屏蔽.
        recently_listed_positions: set[tuple[int, int]] = set()
        #
        # 2) skip_positions: 已知不可上架的连通块位置 (本箱页内有效, 翻页清空).
        #    经过 _merge_into_groups 后, 多格大型道具 (枪等) 会被合并成一组,
        #    skip 时整组 (row, col) 一并加入. 这是修复"反复点同一把枪不同部位"
        #    bug 的关键.
        skip_positions: set[tuple[int, int]] = set()
        #
        # 3) skip_fingerprints: 道具图像 dHash 指纹 (整箱有效, 跨页保留).
        #    解决堆叠拆 stack 道具翻页前后的去重 (同道具图标指纹一致).
        skip_fingerprints: set[bytes] = set()
        #
        # 4) skip_names: 已读到名字、确认不符筛选的道具名 (整箱有效, 跨页保留).
        #    终极兜底 — 即使前 3 层都漏过 (ML 漏判致连通块断裂、翻页后 cell
        #    位置变化、枪不同部位指纹不同), 只要点开后 OCR 拿到名字, 命中
        #    skip_names 就立刻 ESC, 避免走完整的"最大化数量+读收入"流程.
        skip_names: set[str] = set()

        # "到底候选" 状态: 上一次滚动后 diff 很小 → 记一次候选;
        # 连续两次候选才真正判定到底并 break. 避免滚动动画未结束导致的
        # 单次假阳性让扫描提前终止.
        bottom_candidate = False

        def _check_bottom(pre_shot):
            """滚动并做到底判定 (双重确认).

            返回 True 仅当 "上一次 + 这一次" 都判到底 — 对应确实没内容可滚;
            任何一次判定 "还能滚" 都会重置候选状态, 继续扫描.
            """
            nonlocal bottom_candidate
            reached = self._scroll_and_at_bottom(pre_shot)
            if not reached:
                bottom_candidate = False
                return False
            if not bottom_candidate:
                # 第一次检测到 "到底" — 先当候选, 再滚一轮验证
                bottom_candidate = True
                self._status("  到底检测: 候选 (再验证 1 次)")
                return False
            # 连续两次 "到底" — 真正到底
            bottom_candidate = False
            return True

        while scroll_count < max_scroll and empty_scans < empty_scans_limit:
            self.auto.check_stop()

            screenshot = self.screen.grab_full()
            raw_items = self.ml_detector.get_unbound_items(screenshot)
            # 关键: 先做连通域合并, 把多 cell 大型道具 (枪/装备) 折叠成一组.
            # 后续过滤、点击、skip 都以"组"为单位, 而不是单格.
            items = _merge_into_groups(raw_items)

            mode_tag = "ML" if self.ml_detector.is_ml_mode else "规则"

            # 过滤候选道具组:
            # 任一 cell 命中 recently_listed_positions / skip_positions → 整组跳过.
            # 同时为代表 cell 预计算指纹挂到 item["_fp"], 后续 skip 时直接复用.
            if items:
                before_count = len(items)
                kept = []
                for it in items:
                    group = it["_group"]
                    if any(pos in recently_listed_positions for pos in group):
                        continue
                    if any(pos in skip_positions for pos in group):
                        continue
                    cell_img = _crop_cell(screenshot, it["row"], it["col"])
                    fp = _cell_fingerprint(cell_img) if cell_img is not None else None
                    if fp is not None and fp in skip_fingerprints:
                        continue
                    it["_fp"] = fp
                    kept.append(it)
                items = kept
                dropped = before_count - len(items)
                if dropped > 0:
                    self._status(
                        f"  跳过 {dropped} 组已处理道具 "
                        f"(待上架 {len(items)} 组 / "
                        f"位置 {len(skip_positions)} / "
                        f"指纹 {len(skip_fingerprints)} / "
                        f"名字 {len(skip_names)})"
                    )

            if not items:
                # 位置类集合翻页清空 (新页面 (row, col) 含义变了).
                # 指纹/名字集合保留 — 它们与位置无关, 跨页继续生效.
                recently_listed_positions.clear()
                skip_positions.clear()
                self._status("  当前页无候选, 翻页...")
                # 滚动前留下当前网格快照, 滚动后做双重到底确认
                if _check_bottom(screenshot):
                    self._status("  已到达箱子底部 (双重确认), 结束扫描")
                    break
                scroll_count += 1
                empty_scans += 1
                continue

            empty_scans = 0
            listed_this_scan = False

            for item in items:
                self.auto.check_stop()
                if self._listed_count >= max_slots:
                    return listed

                conf_str = f" conf={item['confidence']:.0%}" if self.ml_detector.is_ml_mode else ""
                self._status(
                    f"  [{mode_tag}] R{item['row']}C{item['col']} "
                    f"V={item['v_mean']}{conf_str}"
                )

                # 复用 ML 扫描截图作为 before，省一次 grab_full
                before = screenshot
                # 从全屏截图切出 "页面特征 ROI", 做跳转判定的基准.
                # 后续 after 只抓这个小区域, 每次比 grab_full 省 ~30-60ms.
                px1, py1, px2, py2 = DETECTION_ROI["page_change"]
                before_feat = before[py1:py2, px1:px2]

                self.auto.click(item["cx"], item["cy"])

                # — 检测是否跳转到上架界面 (最多2轮) —
                page_changed = False
                after_feat = None
                for wait_ms in (0.45, 0.35):
                    self.auto.check_stop()
                    self.auto.wait(wait_ms)
                    # 只抓页面特征 ROI (~640×570 vs 2560×1440)
                    after_feat = self.screen.grab_region(px1, py1, px2, py2)
                    if self.detector.has_page_changed(
                        before_feat, after_feat, save_debug=False
                    ):
                        page_changed = True
                        break

                if not page_changed:
                    skipped += 1
                    # 整个连通块位置全部加入 skip — 同一道具的所有 cell 一并
                    # 屏蔽, 避免 ML 扫到组内其它格子时又点开.
                    skip_positions.update(item["_group"])
                    if item.get("_fp") is not None:
                        skip_fingerprints.add(item["_fp"])
                    self._status(
                        f"  -> 跳过 R{item['row']}C{item['col']} "
                        f"(无反应, 累计 {skipped}, 整组 {len(item['_group'])} 格)"
                    )
                    continue

                # — 已进入上架界面 —
                # listing_page 仅用于 click_list_confirm 后的反向跳转比对,
                # 所以只保留特征 ROI 即可.
                listing_page_feat = after_feat

                self.auto.maximize_quantity(
                    clicks=TIMING.get("maximize_clicks", 1)
                )
                self.auto.wait(0.12)

                info_screenshot = self.screen.grab_full()

                ocr_future = self._ocr_executor.submit(
                    self.detector.read_dialog_info, info_screenshot
                )

                record_idx = len(self._listing_records) + 1
                shot_path = f"listings/listing_{record_idx:03d}.png"
                panel = self.detector.crop_dialog_panel(info_screenshot)
                save_debug_screenshot(panel, shot_path)

                # 同步等 OCR 仅在启用筛选时才必要 (需要用名称判断是否跳过).
                # 无筛选时让 OCR 并行跑 — 直接点确认, 等上架完成后再取结果填记录,
                # 单件省 0.5~2.0s.
                filter_active = bool(self._filter_rarities) and self._item_db.count > 0
                info: dict | None = None

                if filter_active:
                    try:
                        info = ocr_future.result(timeout=3.0)
                    except Exception:
                        info = {"name": "", "income": 0}

                    name = info.get("name") or ""
                    # 名字命中 skip_names 直接走 ESC 流程 (跳过 matches_filter),
                    # 同时把当前位置/指纹补充进各 skip 集合, 提高下次过滤命中率.
                    name_hit = name and name in skip_names
                    if name and (name_hit or not self._item_db.matches_filter(
                        name,
                        self._filter_rarities,
                        self._filter_categories,
                    )):
                        tag = "(已知不符)" if name_hit else "不符合筛选条件"
                        self._status(
                            f"  ✗ R{item['row']}C{item['col']} "
                            f"[{name}] {tag}, 跳过"
                        )
                        self.auto.press_key("escape")
                        self.auto.wait(0.6)
                        skipped += 1
                        # 三重标记: 名字 (跨页跨指纹兜底) + 位置 (整组连通块) +
                        # 指纹 (代表 cell, 处理跨页 stack 拆格场景).
                        skip_names.add(name)
                        skip_positions.update(item["_group"])
                        if item.get("_fp") is not None:
                            skip_fingerprints.add(item["_fp"])
                        continue

                self.auto.click_list_confirm()

                # 立刻把鼠标挪出道具网格, 避免点完确认后鼠标停在 (1792, 900)
                # (list_btn 在 ITEM_GRID 内) 触发道具悬浮详情框, 导致:
                #   - PAGE_CHANGE_ROI 截图被弹框遮挡 → mean_diff 误判
                #   - 下轮 grab_full 时悬浮框覆盖网格 → ML 检测不到非绑定道具
                self.auto.park_mouse()

                # — 等待回到出售界面 —
                #
                # 判定逻辑 v7: 两级判据
                #   Fast path: before_feat ↔ current_feat 差异小 → 同一家族出售页
                #              (对 1~6 件清空的常规情况 mean_diff < 8, 通过率高)
                #   Slow path: 快速检查失败时, 查 "顶部 tab 栏是否可见"
                #              (出售页顶部有 开始游戏/仓库/.../交易行 高亮 tab 栏;
                #               "增加槽位" 全屏暗色覆盖, tab 栏被盖住)
                #
                # 为什么需要 Slow path:
                #   上架一次可能一次清空多个同类道具 (比如 18 个营养粥罐头),
                #   before↔current 的 mean_diff 会飙到 ~22, 超过 is_back_on_sell
                #   的阈值 (8), 被误判成 "没回到出售页" → 误按 ESC → 踢出交易行
                #   → 工作流异常结束 (之前日志里的主界面 bug).
                #
                # 覆盖的失败场景:
                #   (1) 卡在上架弹窗 (tab 栏仍被弹窗盖住 → Slow path 也否决)
                #   (2) 槽位满跳 "增加槽位" 页 (全屏暗色 → Slow path 否决)
                #   (3) 大 stack 清空的正常上架 (tab 栏可见 → Slow path 通过)
                page_after_confirm = False
                current_feat = None
                for wait_ms in (0.4, 0.35, 0.3):
                    self.auto.check_stop()
                    self.auto.wait(wait_ms)
                    current_feat = self.screen.grab_region(px1, py1, px2, py2)
                    if self._is_back_on_sell(before_feat, current_feat):
                        page_after_confirm = True
                        break

                if not page_after_confirm:
                    # Slow path: "出售" 二级 tab 是否还处于高亮选中状态
                    if self._on_sell_page():
                        self._status(
                            "  上架成功 (多件同类一次清空, 通过 出售 tab 高亮确认)"
                        )
                        page_after_confirm = True

                if not page_after_confirm:
                    # 快路 + 慢路都失败: 要么卡弹窗, 要么跳到 "增加槽位" 页
                    # (槽位已满时的游戏硬引导). 两种情况都用 ESC 恢复.
                    self._status("  ⚠ 未回到出售界面, 按 ESC 恢复 (可能槽位已满)")
                    self.auto.press_key("escape")
                    self.auto.wait(0.6)

                    # 检查 ESC 之后 "出售" tab 是否重新高亮 (= 回到出售页)
                    if not self._on_sell_page():
                        # 极少见叠了两层页面 (比如弹窗里点了个二级确认框), 再按一次
                        self.auto.press_key("escape")
                        self.auto.wait(0.6)

                    if self._on_sell_page():
                        self._status("  ✓ ESC 已返回出售界面, 本轮提前结束")
                    else:
                        self._status("  ⚠ ESC 后仍未回到出售界面, 放弃本轮")
                    # 本件其实没真正上架 → 不累加到 listed / _listed_count,
                    # 已经保存的 listing_{idx}.png 是 "增加槽位" 或弹窗截图, 无妨.
                    return listed

                # 等待网格刷新
                self.auto.wait(0.3)

                self._record_cleared_positions(
                    recently_listed_positions, item, before
                )

                # 异步路径: 现在 (上架已完成, 页面已刷新) 再取 OCR 结果,
                # 这段时间 OCR 线程基本已经跑完了, timeout 给 1.5s 兜底.
                if info is None:
                    try:
                        info = ocr_future.result(timeout=1.5)
                    except Exception:
                        info = {"name": "", "income": 0}

                record = ListingRecord(
                    index=record_idx,
                    name=info["name"] or f"R{item['row']}C{item['col']}",
                    income=info["income"],
                    box_name=box_name,
                    row=item["row"],
                    col=item["col"],
                    screenshot_path=shot_path,
                )
                self._listing_records.append(record)

                listed += 1
                self._listed_count += 1
                self._progress(self._listed_count, max_slots)

                income_str = f" | 预期收入: {record.income:,}" if record.income else ""
                self._status(
                    f"  ✓ [{record.name}] "
                    f"({self._listed_count}/{max_slots}){income_str}"
                )

                listed_this_scan = True
                break

            if not listed_this_scan:
                recently_listed_positions.clear()
                skip_positions.clear()
                # skip_fingerprints / skip_names 跨页保留, 不清空
                self._status("  当前页无可上架道具, 翻页...")
                if _check_bottom(screenshot):
                    self._status("  已到达箱子底部 (双重确认), 结束扫描")
                    break
                scroll_count += 1
            else:
                # 本轮成功上架过 -> 重置到底候选 (页面已变化, 旧的候选失效)
                bottom_candidate = False
                # 注意: 不符筛选的道具仍然不符筛选, skip_* 系列保留.

        if empty_scans >= empty_scans_limit:
            self._status(f"  连续 {empty_scans} 页无候选道具, 结束扫描")
        elif scroll_count >= max_scroll:
            self._status(f"  已翻页 {scroll_count} 次达到上限, 结束扫描")

        return listed

    # "到底检测" 阈值: 滚动前后网格 ROI 的平均像素差 < 此值 认为未发生变化.
    # 实测:
    #   - 静止帧 mean_diff < 1
    #   - 正常滚一页 mean_diff 通常 > 10 (整屏道具全部换位置)
    #   - 但如果当前页大段全绑定/全空格 + 滚动后仍是暗背景, mean_diff 可能 5-8
    #     (只有少量道具图标移动, 大片黑底没变)
    # 所以阈值不能太大, 否则会把 "还有几件道具+大片空格" 的页面误判成到底;
    # 也不能太小, 否则动画未结束时假阴性. 配合外层的 "双重确认" 机制,
    # 这里取 2.5 稳妥, 只要不是明显静止就让外层继续滚.
    _GRID_BOTTOM_DIFF_THRESHOLD = 2.5

    # 单次翻页的滚轮刻度量:
    # - 值越大 = 一次翻得越远, 扫完整箱所需迭代次数越少 (用户感知更快)
    # - 由于游戏常把连续滚轮事件合并, 并非 1 刻度=1 行; 实测 144 刻度约覆盖
    #   2~3 屏内容, 对 450 格的大箱子基本 3~4 轮就能扫完.
    # - 配合外层双重确认, 即使"过冲越过少量道具"的情况也靠 recently_listed
    #   跟踪和下一轮扫描的重叠视野补回来.
    _SCROLL_AMOUNT = 144  # = visible_rows * 12

    # 滚动后等待 UI 稳定的时间 (秒):
    # - 0.35 太保守, 用户体感明显卡顿
    # - 0.20 偶尔会在动画中抓到帧 → 假阳性 "到底", 但外层 _check_bottom
    #   的双重确认会再滚一次验证, 所以假阳性代价 = 再花 1 轮 (~0.45s), 可接受
    _SCROLL_SETTLE_WAIT = 0.20

    def _scroll_and_at_bottom(self, pre_scroll_screenshot: np.ndarray) -> bool:
        """在道具网格上向下滚动一页, 并判断是否已到底部 (单次).

        注意: 本函数只做 "单次判定". 外层 _scan_and_list_box 会用
        双重确认 (连续两次到底才真正 break) 规避动画未结束的假阳性.

        流程:
          1. 滚 _SCROLL_AMOUNT 刻度 (经验值 144, 覆盖 2~3 屏)
          2. park_mouse 避免悬浮框遮挡截图
          3. wait _SCROLL_SETTLE_WAIT 让 UI 滑动动画结束
          4. 取网格 ROI 跟滚动前对比, 像素差极小即视为到底

        返回:
          True  — 本次检测认为已到底部 (调用方需要外部再确认一次)
          False — 仍有内容可显示, 继续扫描
        """
        gx1 = ITEM_GRID["x_start"]
        gy1 = ITEM_GRID["y_start"]
        gx2 = ITEM_GRID["x_end"]
        gy2 = ITEM_GRID["y_end"]
        before_roi = pre_scroll_screenshot[gy1:gy2, gx1:gx2]

        self.auto.scroll_item_grid(
            ITEM_GRID, "down", amount=self._SCROLL_AMOUNT, fast=True
        )
        self.auto.park_mouse()
        self.auto.wait(self._SCROLL_SETTLE_WAIT)

        after_roi = self.screen.grab_region(gx1, gy1, gx2, gy2)
        if after_roi.shape != before_roi.shape:
            # 理论上不会发生, 保守返回 False 让外层按老逻辑走
            return False
        try:
            diff = float(cv2.absdiff(before_roi, after_roi).mean())
        except Exception:
            return False
        at_bottom = diff < self._GRID_BOTTOM_DIFF_THRESHOLD
        logger.debug(
            f"到底检测: 网格 ROI 滚动前后 mean_diff={diff:.2f} "
            f"(阈值 {self._GRID_BOTTOM_DIFF_THRESHOLD}) → "
            f"{'到底(候选)' if at_bottom else '可继续'}"
        )
        return at_bottom

    # 判定 "出售 tab 高亮" 的平均亮度阈值 (V 通道 0-255):
    # - 出售页 (亮绿底 + 白字): 实测 mean > 50 (高亮时 G 通道尤其强)
    # - 被弹窗/蒙版覆盖, 或切到其他 tab: mean < 25
    # 取 35 能稳稳把两者分开
    # 注: ROI 坐标本身放在 config.DETECTION_ROI["sell_tab"], 用户可在坐标校准里框选.
    _SELL_TAB_BRIGHTNESS_THRESHOLD = 35.0

    def _on_sell_page(self) -> bool:
        """检测是否处在 "交易行 → 出售" 界面.

        方法: 抓 "出售" 二级 tab 按钮所在的一小块 ROI, 看平均亮度是否达到
        "高亮选中状态" 的阈值.

        用途: 区分
          - 出售页 (tab 高亮 → True)
          - 增加槽位页 (全屏暗蒙版, tab 被盖 → False)
          - 卡在上架弹窗 (弹窗覆盖顶栏 → False)
          - 误切到 商家 / 交易记录 tab (出售 tab 失去高亮 → False)

        比 PAGE_CHANGE_ROI 的帧差更稳 — 不受大 stack 一次清空影响.
        """
        x1, y1, x2, y2 = DETECTION_ROI["sell_tab"]
        strip = self.screen.grab_region(x1, y1, x2, y2)
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        is_sell = brightness >= self._SELL_TAB_BRIGHTNESS_THRESHOLD
        logger.debug(
            f"出售 tab 高亮检测: 平均亮度={brightness:.1f} "
            f"(阈值 {self._SELL_TAB_BRIGHTNESS_THRESHOLD}) "
            f"→ {'在出售页' if is_sell else '不在出售页'}"
        )
        return is_sell

    @staticmethod
    def _is_back_on_sell(before_feat, current_feat,
                         thumb_size: tuple[int, int] = (128, 72),
                         threshold: float = 8.0) -> bool:
        """判断 current_feat 是否跟 before_feat (原出售页面) 属于同一家族.

        出售页面上架一件道具后:
          - 少了 1 格的物品 (右下或视野内某一格变暗)
          - 其他格保持不变
          → PAGE_CHANGE_ROI 缩略图 mean_diff 通常 < 6

        若 current_feat 还停留在上架弹窗:
          - 两张图分别是 "出售网格" vs "上架弹窗详情", 视觉差异巨大
          → mean_diff 通常 > 20

        threshold=8 介于两者之间, 对 "正常上架完成" 绝对是负值, 对 "卡弹窗"
        绝对是正值, 容错大.
        """
        gray_b = cv2.cvtColor(before_feat, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(current_feat, cv2.COLOR_BGR2GRAY)
        thumb_b = cv2.resize(gray_b, thumb_size, interpolation=cv2.INTER_AREA)
        thumb_c = cv2.resize(gray_c, thumb_size, interpolation=cv2.INTER_AREA)
        mean_diff = float(cv2.absdiff(thumb_b, thumb_c).mean())
        is_same_family = mean_diff < threshold
        logger.debug(
            f"回到出售页判定: before↔current mean_diff={mean_diff:.1f} "
            f"→ {'回到出售页' if is_same_family else '未回到出售页'}"
        )
        return is_same_family

    def _record_cleared_positions(
        self, positions: set, listed_item: dict, pre_screenshot
    ):
        """
        上架后同名道具会全部消失, 记录这些可能被清空的格子位置.
        通过比较上架前截图中与已上架道具相似的格子来确定.

        判定条件: 灰度平均差 < 灰度阈值 且 HSV V 通道差 < V 阈值,
        双重条件降低不同道具因背景相似被误判为同类的概率.
        """
        grid = CELL_GRID
        gx, gy = grid["origin_x"], grid["origin_y"]
        cw, ch = grid["cell_w"], grid["cell_h"]
        cols, rows = grid["cols"], grid["visible_rows"]
        h, w = pre_screenshot.shape[:2]

        listed_row, listed_col = listed_item["row"], listed_item["col"]
        positions.add((listed_row, listed_col))
        # 上架的是连通块代表 cell, 整组格子 (整把枪/整套装备) 一并消失,
        # 把整组都加进清空集合, 防止 ML 残影把组内其它格判为非绑定再点击.
        positions.update(listed_item.get("_group", []))

        lx1 = gx + listed_col * cw
        ly1 = gy + listed_row * ch
        if lx1 + cw > w or ly1 + ch > h:
            return

        gray_full = cv2.cvtColor(pre_screenshot, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv_full = cv2.cvtColor(pre_screenshot, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_gray = gray_full[ly1:ly1 + ch, lx1:lx1 + cw]
        ref_hsv = hsv_full[ly1:ly1 + ch, lx1:lx1 + cw]

        for row in range(rows):
            for col in range(cols):
                if row == listed_row and col == listed_col:
                    continue
                x1 = gx + col * cw
                y1 = gy + row * ch
                if x1 + cw > w or y1 + ch > h:
                    continue

                cell_gray = gray_full[y1:y1 + ch, x1:x1 + cw]
                cell_hsv = hsv_full[y1:y1 + ch, x1:x1 + cw]

                gray_diff = cv2.absdiff(ref_gray, cell_gray).mean()
                # 色相(H) + 饱和度(S) + 亮度(V) 综合差异, 防止不同道具背景相似误判
                hsv_diff = cv2.absdiff(ref_hsv, cell_hsv).mean()
                if gray_diff < 6.0 and hsv_diff < 6.0:
                    positions.add((row, col))

        if len(positions) > 1:
            logger.info(
                f"检测到 {len(positions)} 个同类道具位置将被跳过: "
                f"{sorted(positions)}"
            )

    def _show_summary(self):
        """显示上架收入汇总统计"""
        records = self._listing_records
        if not records:
            self._status("\n上架完成, 无成功上架记录")
            return

        total_income = sum(r.income for r in records)

        self._status("\n" + "=" * 40)
        self._status("  上架收入统计")
        self._status("=" * 40)

        for r in records:
            income_str = f"{r.income:,}" if r.income else "待识别"
            self._status(f"  {r.index}. {r.name:<16s} 预期收入: {income_str}")

        self._status("-" * 40)
        self._status(f"  总计上架: {len(records)} 件道具")
        if total_income > 0:
            self._status(f"  总预期收入: {total_income:,}")
        self._status(f"  弹窗截图保存在: listings/")
        self._status("=" * 40)

    # ═══════════════════════════════════════════════════════
    # 下架流程
    # ═══════════════════════════════════════════════════════

    def run_delist_workflow(self, delist_all: bool = True,
                           max_delist: int = 15):
        """
        一键下架: 反复点击列表第一个"下架"按钮 → 确认弹窗 → 确认.
        每次下架后列表自动上移, 所以始终点第一个位置即可.
        """
        self.auto.resume()
        delisted = 0
        fail_count = 0

        # 尝试 OCR 确定需要下架的数量 (仅参考, 最多重试 3 次)
        already_listed, total_slots, _ = self._detect_listing_status_reliable()
        if already_listed == 0:
            self._status("当前没有已上架道具, 无需下架")
            return
        if already_listed > 0:
            max_delist = already_listed
        self._status(f"开始下架 (共 {max_delist} 件)...")

        try:
            recent_readings = []
            prev_listed = already_listed if already_listed > 0 else -1

            # 每 ocr_every 次下架才做一次 OCR 检查, 其余轮次直接乐观计数.
            # N=3 时总 OCR 次数 / 下架数 从 1.0 降到 ~0.33, 省 ~60% OCR 时间.
            # 最后一轮强制 OCR, 确保知道是否真的下架完成.
            ocr_every = max(1, TIMING.get("delist_ocr_every", 3))

            for i in range(max_delist):
                self.auto.check_stop()
                self._status(f"  正在下架第 {i + 1}/{max_delist} 件...")
                self.auto.click_delist_button(LISTED_ITEMS)
                self.auto.wait(0.5)

                # 决定这一轮要不要做 OCR: 按 ocr_every 节奏 + 最后一轮强制
                should_ocr = (
                    (i + 1) % ocr_every == 0
                    or (i + 1) == max_delist
                )
                if should_ocr:
                    cur_listed, _, _ = self._detect_listing_status()
                else:
                    cur_listed = -1

                # 只有在 OCR 能确认已上架数量减少时才视为真正下架成功, 避免虚增计数.
                # 跳过 OCR 的轮次按 "乐观计数 + 1" 处理, 等下次 OCR 时用实际读数纠正.
                if cur_listed >= 0 and prev_listed >= 0:
                    if cur_listed < prev_listed:
                        delisted += (prev_listed - cur_listed)
                    # cur_listed == prev_listed 时说明前几轮乐观计数偏多,
                    # 交给后面的 "连续读数相同" 分支判定仓库满
                elif cur_listed >= 0 and prev_listed < 0:
                    # 第一次 OCR 刚成功, 不计数只建立基线
                    pass
                else:
                    # 本轮未做 OCR 或 OCR 失败: 乐观估计为下架 1 件
                    delisted += 1

                if cur_listed >= 0:
                    prev_listed = cur_listed

                self._progress(delisted, max_delist)

                if cur_listed == 0:
                    self._status("  上架数量已归零, 下架完毕")
                    break
                if cur_listed > 0:
                    recent_readings.append(cur_listed)
                    if len(recent_readings) > 3:
                        recent_readings.pop(0)
                    if (len(recent_readings) >= 3
                            and recent_readings[0] == recent_readings[1]
                            == recent_readings[2]):
                        self._status(
                            f"  连续3次OCR读数相同 ({cur_listed}), "
                            f"仓库空间可能不足, 停止下架"
                        )
                        break

            self._status(f"\n下架完成! 共成功下架 {delisted} 件")

            self._status("整理仓库...")
            self.auto.click_organize_storage()
            time.sleep(0.3)
            self.auto.click_sort_button()
            self._status("仓库整理完成")

        except StopIteration:
            self._status(f"\n下架被中断, 已下架 {delisted} 件")
        except Exception as e:
            self._status(f"\n下架出错: {e}")
            logger.exception("下架工作流异常")

    # ═══════════════════════════════════════════════════════
    # 调试
    # ═══════════════════════════════════════════════════════

    def capture_debug_info(self) -> dict:
        """截图 + 网格检测, 保存调试图片"""
        self._status("正在截图分析...")

        full = self.screen.grab_full()
        save_debug_screenshot(full, "debug_full.png")

        # 旧规则法调试 (保存热力图和标注图)
        rule_items = self.detector.find_unbound_items(full, debug=True)

        # ML 检测器结果
        ml_items = self.ml_detector.get_unbound_items(full)
        mode_tag = "ML" if self.ml_detector.is_ml_mode else "规则"

        cw = CELL_GRID["cell_w"]
        ch = CELL_GRID["cell_h"]
        summary = {
            "grid": f"{CELL_GRID['cols']}x{CELL_GRID['visible_rows']} cells ({cw}x{ch}px)",
            "mode": mode_tag,
            "unbound_count": len(ml_items),
            "rule_count": len(rule_items),
            "items": [
                f"R{it['row']}C{it['col']} V={it['v_mean']}"
                f" conf={it['confidence']:.0%}"
                f" @ ({it['cx']},{it['cy']})"
                for it in ml_items
            ],
        }

        self._status(f"网格: {summary['grid']} | 检测模式: {mode_tag}")
        self._status(f"检测到 {len(ml_items)} 个非绑定道具 ({mode_tag}):")
        for desc in summary["items"]:
            self._status(f"  {desc}")
        if mode_tag == "ML":
            self._status(f"  (规则法对比: {len(rule_items)} 个)")
        self._status(
            "调试图已保存:\n"
            "  debug_full.png (全屏截图)\n"
            "  debug_4_annotated.png (网格+检测标注)\n"
            "  debug_2_v_heatmap.png (V值热力图)"
        )
        return summary

    @staticmethod
    def _get_box_index(box_name: str) -> int:
        for i, name in enumerate(STORAGE_BOXES):
            if name == box_name:
                return i
        return -1
