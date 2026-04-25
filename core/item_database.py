"""
道具数据库 — 存储从交易行采集的道具信息 (名称/大类/子类/品质/价格)
供筛选设置在上架流程中进行精确匹配.
"""
import json
import logging
import os
from dataclasses import dataclass, fields, asdict
from difflib import SequenceMatcher
from typing import Optional

from config import ITEM_DB_FILE

logger = logging.getLogger("item_db")

# 模糊匹配阈值: OCR 结果与数据库项名称相似度 >= 此值才视为命中
_FUZZY_THRESHOLD = 0.82
_MIN_NAME_LEN = 2


@dataclass
class ItemEntry:
    name: str
    category: str
    subcategory: str
    rarity: str = "白色"
    price: int = 0

    @property
    def key(self) -> str:
        return self.name.strip()


class ItemDatabase:
    """道具数据库, 以道具名称为主键, 支持增删改查和持久化."""

    def __init__(self, db_path: str = ITEM_DB_FILE):
        self._path = db_path
        self._items: dict[str, ItemEntry] = {}
        self.load()

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def all_items(self) -> list[ItemEntry]:
        return list(self._items.values())

    def load(self):
        self._items.clear()
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_fields = {f.name for f in fields(ItemEntry)}
            loaded = 0
            for entry in data.get("items", []):
                if not isinstance(entry, dict) or "name" not in entry:
                    continue
                # 过滤掉未知字段, 避免 dataclass 构造器 TypeError
                clean = {k: v for k, v in entry.items() if k in valid_fields}
                try:
                    item = ItemEntry(**clean)
                except Exception as ex:
                    logger.debug(f"跳过损坏的道具条目 {entry}: {ex}")
                    continue
                self._items[item.key] = item
                loaded += 1
            logger.info(f"道具数据库已加载: {loaded} 个道具")
        except Exception as e:
            logger.warning(f"加载道具数据库失败: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        data = {
            "version": 1,
            "items": [asdict(it) for it in self._items.values()],
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"道具数据库已保存: {self.count} 个道具")

    def add(self, item: ItemEntry) -> bool:
        """添加或更新一个道具, 返回 True 表示新增, False 表示更新."""
        key = item.key
        is_new = key not in self._items
        self._items[key] = item
        return is_new

    def add_many(self, items: list[ItemEntry]) -> tuple[int, int]:
        """批量添加, 返回 (新增数, 更新数)."""
        new_count = 0
        update_count = 0
        for item in items:
            if self.add(item):
                new_count += 1
            else:
                update_count += 1
        return new_count, update_count

    def remove(self, name: str) -> bool:
        key = name.strip()
        if key in self._items:
            del self._items[key]
            return True
        return False

    def get(self, name: str) -> Optional[ItemEntry]:
        return self._items.get(name.strip())

    def find(self, name: str) -> Optional[ItemEntry]:
        """
        模糊匹配: 先精确匹配, 再用 SequenceMatcher 按相似度排序取最高.
        太短的查询 (<_MIN_NAME_LEN) 或最高相似度低于 _FUZZY_THRESHOLD 时返回 None,
        避免把 OCR 噪声误匹到无关道具上.
        """
        key = name.strip()
        if not key or len(key) < _MIN_NAME_LEN:
            return None
        if key in self._items:
            return self._items[key]

        best_item = None
        best_score = 0.0
        for item_key, item in self._items.items():
            score = SequenceMatcher(None, key, item_key).ratio()
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None and best_score >= _FUZZY_THRESHOLD:
            return best_item
        return None

    def find_fuzzy(self,
                   name: str,
                   category: Optional[str] = None,
                   subcategory: Optional[str] = None,
                   cutoff: float = 0.45,
                   tie_margin: float = 0.12) -> Optional[ItemEntry]:
        """OCR 纠错专用的低阈值模糊匹配.

        与 find() 的区别:
        - 可按 (category, subcategory) 缩小搜索空间 (同类道具内纠错, 避免跨类误匹)
        - 阈值默认 0.45 (远低于 find 的 0.82), 专门应对 OCR 前缀丢失 / 字符错读
        - 子串加成: OCR 结果 (>=3 字) 完整出现在 DB 名称里 → +0.75 下限
        - 公共后缀加成: OCR 与 DB 名共享 >=3 字尾部 → 按尾部长度加分 (前缀丢失常见场景)
        - 歧义拒绝: 最优分与次优分差距 < tie_margin 时放弃纠错 (避免瞎猜)

        命中返回 DB 中的规范 ItemEntry, 否则 None.
        """
        key = name.strip()
        if not key or len(key) < _MIN_NAME_LEN:
            return None
        if key in self._items:
            return self._items[key]

        candidates = [
            it for it in self._items.values()
            if (not category or it.category == category)
            and (not subcategory or it.subcategory == subcategory)
        ]
        if not candidates:
            return None

        scored: list[tuple[float, ItemEntry]] = []
        for item in candidates:
            ratio = SequenceMatcher(None, key, item.key).ratio()
            score = ratio
            # 子串加成: 前缀丢失场景, OCR 结果是 DB 名的 substring
            if len(key) >= 3 and key in item.key:
                score = max(score, 0.75)
            # 公共尾部加成 (>=3 字)
            max_n = min(len(key), len(item.key))
            common_suffix = 0
            for n in range(max_n, 2, -1):
                if key[-n:] == item.key[-n:]:
                    common_suffix = n
                    break
            if common_suffix >= 3:
                # 按尾部长度相对 OCR 全长的占比加分; 越接近 "OCR = 尾部整体" 加分越大
                score += (common_suffix / max(len(key), 1)) * 0.30
            scored.append((score, item))

        scored.sort(key=lambda x: -x[0])
        best_score, best_item = scored[0]

        # 短名称 (<4 字) 风险大: "单头盔" / "术头盔" 这种 OCR 残片对同类里所有 "XX头盔"
        # 都有 0.5-0.6 的基础相似度, 必须要求更强的证据 (子串/长公共尾部) 才认定.
        effective_cutoff = cutoff if len(key) >= 4 else max(cutoff, 0.70)
        if best_score < effective_cutoff:
            return None
        # 歧义拒绝: 多个候选的分数都很接近时, 无法可靠纠错 → 保留原 OCR 结果
        if len(scored) > 1 and (best_score - scored[1][0]) < tie_margin:
            return None
        return best_item

    def matches_filter(self, name: str,
                       allowed_rarities: set[str],
                       allowed_categories: dict[str, set[str]]) -> bool:
        """
        检查道具是否符合筛选条件.
        - allowed_rarities: 允许的品质集合, 如 {"红色", "金色"}
        - allowed_categories: 允许的大类→子类, 如 {"枪械": {"步枪","冲锋枪"}}
        未在数据库中的道具默认允许上架 (不阻断).
        """
        item = self.find(name)
        if item is None:
            return True

        if allowed_rarities and item.rarity not in allowed_rarities:
            return False

        if allowed_categories:
            cat = item.category
            if cat not in allowed_categories:
                return False
            subs = allowed_categories[cat]
            if subs and item.subcategory not in subs:
                return False

        return True

    def get_stats(self) -> dict[str, dict[str, int]]:
        """按大类统计道具数量."""
        stats: dict[str, dict[str, int]] = {}
        for item in self._items.values():
            cat = item.category
            sub = item.subcategory
            if cat not in stats:
                stats[cat] = {}
            stats[cat][sub] = stats[cat].get(sub, 0) + 1
        return stats
