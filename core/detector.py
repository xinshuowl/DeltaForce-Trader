"""
道具检测器 v13 — 9列网格 + HSV背景亮度法 + RapidOCR弹窗识别

在交易行出售界面中:
  - 所有道具背景都有对角线条纹 (UI统一风格)
  - 非绑定道具背景明显更亮 (HSV V ≈ 42-84)
  - 绑定道具背景更暗 (HSV V ≈ 27-40)

策略:
  1. 按固定 9列 网格将右侧面板切分为独立格子
  2. 跳过空格子 (中心区域灰度均值 < 20)
  3. 对每个非空格子, 取左上背景区域的 HSV V 通道均值
  4. V > 阈值 → 非绑定; V ≤ 阈值 → 绑定
  5. 上架弹窗: RapidOCR (PaddleOCR) 读取道具名称和预期收入
"""
import re
import logging
import threading
import numpy as np
import cv2

from config import CELL_GRID, BIND_DETECTION, LIST_DIALOG, LISTING_SLOTS, DEBUG

logger = logging.getLogger("detector")

# ─── RapidOCR 懒加载单例 ───
_rapid_ocr_instance = None
_rapid_ocr_lock = threading.Lock()

# ─── Tesseract 串行锁: pytesseract 内部调用外部进程, 并发时临时文件可能冲突 ───
_tesseract_lock = threading.Lock()


def _get_rapid_ocr():
    """懒加载 RapidOCR 实例 (线程安全, 全局单例)"""
    global _rapid_ocr_instance
    if _rapid_ocr_instance is not None:
        return _rapid_ocr_instance
    with _rapid_ocr_lock:
        if _rapid_ocr_instance is not None:
            return _rapid_ocr_instance
        try:
            import logging as _logging
            _logging.getLogger("RapidOCR").setLevel(_logging.WARNING)
            from rapidocr import RapidOCR
            _rapid_ocr_instance = RapidOCR()
            logger.info("RapidOCR 引擎初始化成功")
            return _rapid_ocr_instance
        except Exception as e:
            logger.warning(f"RapidOCR 初始化失败, 将降级为 Tesseract: {e}")
            return None


def warm_up_ocr():
    """预热 RapidOCR，在后台线程调用以消除首次识别的冷启动延迟"""
    import numpy as np
    ocr = _get_rapid_ocr()
    if ocr is not None:
        dummy = np.zeros((64, 200, 3), dtype=np.uint8)
        try:
            ocr(dummy)
        except Exception:
            pass
        logger.info("RapidOCR 预热完成")


class UnboundItemDetector:

    def __init__(self):
        self._grid = CELL_GRID
        self._det = BIND_DETECTION

    def find_unbound_items(self, full_screenshot: np.ndarray,
                           debug: bool = False) -> list[dict]:
        """
        在完整截图上按 9列网格 逐格检测非绑定道具.

        Returns:
            列表, 每项包含 row/col/cx/cy (绝对屏幕坐标) 和 v_mean.
        """
        gx = self._grid["origin_x"]
        gy = self._grid["origin_y"]
        cw = self._grid["cell_w"]
        ch = self._grid["cell_h"]
        cols = self._grid["cols"]
        rows = self._grid["visible_rows"]
        v_thr = self._det["v_threshold"]

        hsv = cv2.cvtColor(full_screenshot, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(full_screenshot, cv2.COLOR_BGR2GRAY)

        sy1, sy2 = self._det["sample_y1"], self._det["sample_y2"]
        sx1, sx2 = self._det["sample_x1"], self._det["sample_x2"]
        empty_thr = self._det["empty_threshold"]

        items = []
        debug_cells = []

        for row in range(rows):
            for col in range(cols):
                x1 = gx + col * cw
                y1 = gy + row * ch

                if x1 + cw > full_screenshot.shape[1] or y1 + ch > full_screenshot.shape[0]:
                    continue

                cell_gray = gray[y1:y1 + ch, x1:x1 + cw]
                center = cell_gray[20:65, 10:75]
                if center.mean() < empty_thr:
                    continue

                bg_v = hsv[y1 + sy1:y1 + sy2, x1 + sx1:x1 + sx2, 2].astype(float).mean()
                is_unbound = bg_v > v_thr

                # 格子中心的绝对屏幕坐标
                cx = x1 + cw // 2
                cy = y1 + ch // 2

                if debug:
                    debug_cells.append((row, col, x1, y1, bg_v, is_unbound))

                if is_unbound:
                    items.append({
                        "row": row,
                        "col": col,
                        "cx": cx,
                        "cy": cy,
                        "v_mean": round(bg_v, 1),
                    })

        if debug:
            self._save_debug(full_screenshot, debug_cells, items)

        return items

    def has_page_changed(self, before: np.ndarray, after: np.ndarray,
                         save_debug: bool = True) -> bool:
        """
        判断点击道具后是否跳转到了上架物品界面.

        出售界面 → 上架物品界面 是完全不同的两个页面:
          - 出售界面: 右侧是 9列物品网格, 左侧是上架槽位列表
          - 上架物品界面: 左侧价格图表 + 右侧道具详情/上架按钮

        方法: 将两张截图都缩小为 128×72 缩略图后比较灰度差异.
        缩略图对比不依赖任何具体像素坐标, 对分辨率/DPI变化天然免疫.
          - 页面跳转: 整屏内容完全不同, mean_diff 通常 > 25
          - 未跳转 (不可上架提示): 只有一小块提示文字变化, mean_diff < 8

        before / after 可以是全屏截图, 也可以是 PAGE_CHANGE_ROI 的局部截图
        (两者尺寸一致即可). 用局部截图每次可省 ~30-60ms.
        """
        thumb_size = (128, 72)

        gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        thumb_b = cv2.resize(gray_b, thumb_size, interpolation=cv2.INTER_AREA)
        thumb_a = cv2.resize(gray_a, thumb_size, interpolation=cv2.INTER_AREA)

        diff = cv2.absdiff(thumb_b, thumb_a)
        mean_diff = float(diff.mean())

        changed = mean_diff > 10

        logger.debug(
            f"页面变化检测: mean_diff={mean_diff:.1f} → {'跳转' if changed else '未跳转'}"
        )

        if save_debug and DEBUG:
            try:
                cv2.imwrite("debug_page_before.png", before)
                cv2.imwrite("debug_page_after.png", after)
                diff_vis = cv2.resize(diff, (512, 288))
                cv2.imwrite("debug_page_diff.png", diff_vis)
            except Exception as e:
                logger.warning(f"保存调试截图失败: {e}")

        return changed

    def read_listing_count(self, screenshot: np.ndarray) -> tuple[int, int]:
        """
        从出售界面截图中读取当前上架数量, 如 "5/15".

        Returns:
            (已上架数, 总槽位数), 识别失败返回 (-1, -1)
        """
        x1, y1, x2, y2 = LISTING_SLOTS["counter_region"]
        h, w = screenshot.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = screenshot[y1:y2, x1:x2]

        if crop.size == 0:
            return (-1, -1)

        try:
            import pytesseract
        except ImportError:
            logger.debug("pytesseract 未安装, 无法读取上架数量")
            return (-1, -1)

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)
            _, binarized = cv2.threshold(scaled, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binarized = cv2.erode(binarized, kernel, iterations=1)
            with _tesseract_lock:
                text = pytesseract.image_to_string(
                    binarized,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789/",
                ).strip()

            if DEBUG:
                cv2.imwrite("debug_listing_count.png", crop)
                cv2.imwrite("debug_listing_count_bin.png", binarized)
            logger.info(f"上架数量 OCR 原文: '{text}'")

            match = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if match:
                listed = int(match.group(1))
                total = int(match.group(2))

                if total < 1 or total > 15:
                    logger.warning(f"OCR 总槽位异常: {total}, 原文: '{text}'")
                    return (-1, -1)

                if listed > total:
                    raw_listed = match.group(1)
                    for trim in range(1, len(raw_listed)):
                        candidate = int(raw_listed[trim:])
                        if 0 <= candidate <= total:
                            logger.info(
                                f"OCR 修正前导噪声: '{raw_listed}' → {candidate}"
                            )
                            listed = candidate
                            break
                    else:
                        logger.warning(
                            f"OCR 已上架数异常: {listed}/{total}, 原文: '{text}'"
                        )
                        return (-1, -1)

                return (listed, total)

            logger.warning(f"上架数量格式不匹配: '{text}'")
            return (-1, -1)

        except Exception as e:
            logger.warning(f"读取上架数量失败: {e}")
            return (-1, -1)

    def read_dialog_info(self, screenshot: np.ndarray) -> dict:
        """
        从上架弹窗截图中读取道具名称和预期收入.
        优先使用 RapidOCR (PaddleOCR, 中文识别精度高),
        降级为 Tesseract.

        Returns:
            {"name": str, "income": int}
        """
        info = {"name": "", "income": 0}
        h, w = screenshot.shape[:2]

        ocr = _get_rapid_ocr()
        if ocr is not None:
            info = self._read_dialog_rapid(screenshot, ocr)
        else:
            info = self._read_dialog_tesseract(screenshot)

        logger.info(f"OCR 结果: 名称='{info['name']}', 收入={info['income']}")
        return info

    def _read_dialog_rapid(self, screenshot: np.ndarray, ocr) -> dict:
        """RapidOCR 读名称 + 弹窗下半屏 OCR 定位 "预期收入" 标签后取数字.

        之前的做法是直接在硬编码 income_region 上跑 Tesseract, 对游戏 UI
        布局的上下漂移非常敏感 (实测 y 轴偏差 20-40px 就会读到价格输入框
        "71384" 或出售总价区). 新做法:

          1. 在弹窗右侧面板的下半部分 (dialog_crop 下半区) 跑 RapidOCR,
             得到多行文字 + 每行的 y 中心坐标
          2. 找包含 "预期收入" 或 "预期收益" 的行
          3. 在同一 y 带 (±30px) 寻找纯数字文本块 → 这就是净收益
          4. 若没找到标签行, 退回到原来的硬编码区 Tesseract 作为兜底

        这样无论 186,097 所在行上下偏移多少, 只要它跟 "预期收入" 标签
        y 对齐就能被正确抓到.
        """
        info = {"name": "", "income": 0}
        h, w = screenshot.shape[:2]

        try:
            # 道具名称: RapidOCR (中文识别精度高)
            nx1, ny1, nx2, ny2 = LIST_DIALOG["item_name_region"]
            nx1, ny1 = max(0, nx1), max(0, ny1)
            nx2, ny2 = min(w, nx2), min(h, ny2)
            name_crop = screenshot[ny1:ny2, nx1:nx2]

            if name_crop.size > 0:
                result = ocr(name_crop)
                if result and result.txts:
                    info["name"] = "".join(result.txts).strip()

            # 预期收入: 用 RapidOCR 读 "弹窗下半部分" 定位标签 + 数字
            info["income"] = self._read_expected_income(screenshot, ocr)

        except Exception as e:
            logger.warning(f"OCR 读取失败: {e}")

        return info

    # "预期收入" 标签可能的写法 (游戏不同版本或 OCR 近似识别可能落在这些变体)
    _INCOME_LABEL_KEYWORDS = ("预期收入", "预期收益", "预期总收益", "预计收入")

    def _read_expected_income(self, screenshot: np.ndarray, ocr) -> int:
        """在上架弹窗里定位 "预期收入 XXX" 行, 返回净收入数字.

        思路: RapidOCR 同时返回 文本 + 每个文本块的 bounding box, 我们扫描
        下半面板:
          - 找 "预期收入" 标签所在的 y 中心 (cy_label)
          - 再找 y 中心接近 cy_label (±30px) 的 "纯数字" 文本块
          - 取其中数字位数最多的 (避免把 "3/3" "1/6" 这种分数文本当成收入)

        返回 0 表示失败, 由调用方决定是否兜底.
        """
        h, w = screenshot.shape[:2]

        # 弹窗右侧面板的 "下半部分": 参考价格/价格输入/预期收入/上架按钮 都在这
        dx1, dy1, dx2, dy2 = LIST_DIALOG["dialog_crop"]
        y_mid = dy1 + (dy2 - dy1) // 2
        ix1 = max(0, dx1)
        iy1 = max(0, y_mid)
        ix2 = min(w, dx2)
        iy2 = min(h, dy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0

        crop = screenshot[iy1:iy2, ix1:ix2]

        # 调试保存: 每次识别覆盖 listings/last_income_crop.png, 肉眼就能核对
        # 当前 crop 是否包含 "预期收入 XXX" 完整一行
        if DEBUG:
            try:
                import os
                os.makedirs("listings", exist_ok=True)
                cv2.imwrite("listings/last_income_crop.png", crop)
            except Exception:
                pass

        try:
            result = ocr(crop)
        except Exception as e:
            logger.debug(f"income OCR 失败: {e}")
            return self._income_tesseract_fallback(screenshot)

        txts = getattr(result, "txts", None) or []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            boxes = []
        if not txts or len(txts) != len(boxes):
            return self._income_tesseract_fallback(screenshot)

        # 1) 找 "预期收入" 标签的 y 中心
        label_cy = None
        for txt, box in zip(txts, boxes):
            if any(kw in txt for kw in self._INCOME_LABEL_KEYWORDS):
                ys = [pt[1] for pt in box]
                label_cy = (min(ys) + max(ys)) / 2
                logger.debug(
                    f"预期收入标签定位: '{txt}' cy={label_cy:.1f} (crop 局部)"
                )
                break

        # 2) 收集候选: 至少 3 位数字, 排除分数/出售份数类 (3/3, 1/6)
        candidates: list[tuple[int, float, str]] = []
        for txt, box in zip(txts, boxes):
            ys = [pt[1] for pt in box]
            cy = (min(ys) + max(ys)) / 2
            if "/" in txt or "×" in txt:
                continue
            digits_only = re.sub(r"[^\d]", "", txt)
            if len(digits_only) < 3:
                continue
            candidates.append((int(digits_only), cy, txt))

        if not candidates:
            logger.debug("预期收入 OCR: 无数字候选, 降级 Tesseract")
            return self._income_tesseract_fallback(screenshot)

        if label_cy is not None:
            same_line = [c for c in candidates if abs(c[1] - label_cy) <= 30]
            if same_line:
                candidates = same_line

        # 位数最多的那个优先 (e.g. 186,097 6 位 > 71,384 5 位);
        # 同位数时取 y 最接近标签的
        best = max(
            candidates,
            key=lambda c: (
                len(str(c[0])),
                -abs(c[1] - label_cy) if label_cy is not None else 0,
            ),
        )
        logger.debug(
            f"预期收入候选 {len(candidates)} 个, 选中 '{best[2]}' → {best[0]:,} "
            f"(cy={best[1]:.1f})"
        )
        return best[0]

    def _income_tesseract_fallback(self, screenshot: np.ndarray) -> int:
        """原硬编码区 + Tesseract, 仅在 RapidOCR 流水线失败时使用"""
        h, w = screenshot.shape[:2]
        ix1, iy1, ix2, iy2 = LIST_DIALOG["income_region"]
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(w, ix2), min(h, iy2)
        inc_crop = screenshot[iy1:iy2, ix1:ix2]
        if inc_crop.size == 0:
            return 0
        return self._ocr_income_tesseract(inc_crop)

    @staticmethod
    def _ocr_income_tesseract(crop: np.ndarray) -> int:
        """用 Tesseract 快速识别纯数字收入"""
        try:
            import pytesseract
        except ImportError:
            return 0
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray, None, fx=3, fy=3,
                                interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(
                scaled, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            binary = cv2.copyMakeBorder(
                binary, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=255
            )
            with _tesseract_lock:
                text = pytesseract.image_to_string(
                    binary,
                    config="--psm 7 --oem 3 "
                           "-c tessedit_char_whitelist=0123456789,.",
                )
            digits = re.sub(r"[^\d]", "", text)
            return int(digits) if digits else 0
        except Exception:
            return 0

    def _read_dialog_tesseract(self, screenshot: np.ndarray) -> dict:
        """降级: 使用 Tesseract 读取道具名称和预期收入"""
        info = {"name": "", "income": 0}

        try:
            import pytesseract
        except ImportError:
            logger.debug("pytesseract 未安装, 跳过OCR")
            return info

        try:
            h, w = screenshot.shape[:2]

            nx1, ny1, nx2, ny2 = LIST_DIALOG["item_name_region"]
            nx1, ny1 = max(0, nx1), max(0, ny1)
            nx2, ny2 = min(w, nx2), min(h, ny2)
            name_crop = screenshot[ny1:ny2, nx1:nx2]
            if name_crop.size > 0:
                name_processed = self._preprocess_for_tesseract(name_crop)
                if DEBUG:
                    cv2.imwrite("debug_ocr_name.png", name_processed)
                with _tesseract_lock:
                    name_text = pytesseract.image_to_string(
                        name_processed, lang="chi_sim+eng",
                        config="--psm 7 --oem 3"
                    )
                info["name"] = re.sub(r'[|\\_\[\]{}()（）]', '', name_text).strip()

            ix1, iy1, ix2, iy2 = LIST_DIALOG["income_region"]
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(w, ix2), min(h, iy2)
            inc_crop = screenshot[iy1:iy2, ix1:ix2]
            if inc_crop.size > 0:
                inc_processed = self._preprocess_for_tesseract(inc_crop)
                if DEBUG:
                    cv2.imwrite("debug_ocr_income.png", inc_processed)
                with _tesseract_lock:
                    inc_text = pytesseract.image_to_string(
                        inc_processed,
                        config="--psm 7 --oem 3 "
                               "-c tessedit_char_whitelist=0123456789,.",
                    )
                digits = re.sub(r"[^\d]", "", inc_text)
                if digits:
                    info["income"] = int(digits)

        except Exception as e:
            logger.warning(f"Tesseract 读取失败: {e}")

        return info

    @staticmethod
    def _preprocess_for_tesseract(crop: np.ndarray) -> np.ndarray:
        """Tesseract 专用预处理: 灰度 → 放大 → OTSU反色 → 白边框"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(
            scaled, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        binary = cv2.copyMakeBorder(
            binary, 15, 15, 15, 15,
            cv2.BORDER_CONSTANT, value=255
        )
        return binary

    def crop_dialog_panel(self, screenshot: np.ndarray) -> np.ndarray:
        """裁剪上架弹窗右侧面板区域 (包含道具名、数量、价格、收入)"""
        x1, y1, x2, y2 = LIST_DIALOG["dialog_crop"]
        h, w = screenshot.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return screenshot[y1:y2, x1:x2]

    def _save_debug(self, img, debug_cells, items):
        cw = self._grid["cell_w"]
        ch = self._grid["cell_h"]
        rows = self._grid["visible_rows"]
        v_thr = self._det["v_threshold"]

        annotated = img.copy()

        for row, col, x1, y1, v_mean, is_unbound in debug_cells:
            x2 = x1 + cw
            y2 = y1 + ch

            if is_unbound:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2),
                              (0, 200, 0), -1)
                annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 180), 1)

            cv2.putText(annotated, f"{v_mean:.0f}",
                        (x1 + 5, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, 255, 0) if is_unbound else (0, 0, 200), 1)

        # 图例
        lx, ly = self._grid["origin_x"], self._grid["origin_y"] + ch * rows + 10
        cv2.putText(annotated, f"GREEN = Unbound (V>{v_thr})",
                    (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"RED = Bound (V<={v_thr})",
                    (lx, ly + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        # 裁剪右侧面板保存
        gx = self._grid["origin_x"]
        gy = self._grid["origin_y"]
        cols = self._grid["cols"]
        panel = annotated[gy - 60:gy + ch * rows + 50,
                          gx - 30:gx + cw * cols + 30]
        cv2.imwrite("debug_4_annotated.png", panel)

        # V值热力图
        v_map = np.zeros((ch * rows, cw * cols), dtype=np.uint8)
        for row, col, x1, y1, v_mean, _ in debug_cells:
            r1 = row * ch
            c1 = col * cw
            v_map[r1:r1 + ch, c1:c1 + cw] = min(int(v_mean * 3), 255)
        cv2.imwrite("debug_2_v_heatmap.png",
                    cv2.applyColorMap(v_map, cv2.COLORMAP_JET))
