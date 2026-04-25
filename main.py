"""
三角洲行动 - 交易行自动上架/下架助手
主入口文件

使用方法:
  1. pip install -r requirements.txt
  2. python main.py
  3. 在 GUI 中配置筛选条件
  4. 在游戏中打开 交易行 → 出售 页面
  5. 按 Ctrl+1 开始上架, Ctrl+3 开始下架, Ctrl+2 紧急停止

注意: 需要以管理员权限运行以确保键盘钩子和鼠标操作正常工作
"""
import sys
import os
import logging
import traceback
from datetime import datetime

# 设置DPI感知 (确保 pyautogui/mss 使用物理像素坐标)
from core.screen import set_dpi_awareness
set_dpi_awareness()

# 启用Qt高DPI自动缩放 (必须在 QApplication 实例化之前)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QMetaObject
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

import keyboard

from gui.main_window import MainWindow
from config import HOTKEYS

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"autoshop_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def _global_exception_hook(exc_type, exc_value, exc_tb):
    """全局未捕获异常处理, 写入日志文件而非直接崩溃"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"未捕获异常:\n{text}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _global_exception_hook


class HotkeyManager:
    """全局热键管理, 桥接 keyboard 库与 Qt 主窗口"""

    def __init__(self, window: MainWindow):
        self.window = window
        self._registered = False

    def register(self):
        if self._registered:
            return
        keyboard.add_hotkey(HOTKEYS["start_list"], self._on_f1, suppress=False)
        keyboard.add_hotkey(HOTKEYS["stop"], self._on_f2, suppress=False)
        keyboard.add_hotkey(HOTKEYS["start_delist"], self._on_f3, suppress=False)
        keyboard.add_hotkey(HOTKEYS["debug"], self._on_f4, suppress=False)
        keyboard.add_hotkey(HOTKEYS["coord_pick"], self._on_coord_pick, suppress=False)
        self._registered = True
        logger.info(
            f"热键已注册: {HOTKEYS['start_list']}=上架, "
            f"{HOTKEYS['stop']}=停止, {HOTKEYS['start_delist']}=下架, "
            f"{HOTKEYS['debug']}=调试截图, {HOTKEYS['coord_pick']}=截图拾取坐标"
        )

    def unregister(self):
        if not self._registered:
            return
        keyboard.unhook_all()
        self._registered = False

    def _invoke(self, slot_name: str):
        """
        从 keyboard 库后台线程安全地调用主窗口槽函数.
        QMetaObject.invokeMethod + QueuedConnection 会把调用排队到 Qt 主线程执行;
        直接用 QTimer.singleShot 在非 Qt 线程调用是未定义行为 (可能导致崩溃).
        """
        try:
            QMetaObject.invokeMethod(self.window, slot_name, Qt.QueuedConnection)
        except Exception as e:
            logger.warning(f"热键 {slot_name} 转发失败: {e}")

    def _on_f1(self):
        self._invoke("_on_start_list")

    def _on_f2(self):
        self._invoke("_on_stop")

    def _on_f3(self):
        self._invoke("_on_start_delist")

    def _on_f4(self):
        self._invoke("_on_debug")

    def _on_coord_pick(self):
        self._invoke("_open_coord_picker")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("三角洲行动交易行助手")

    from gui.theme import CYBER_APP_BASE
    app.setStyleSheet(CYBER_APP_BASE)

    window = MainWindow()
    window.show()

    hotkey_mgr = HotkeyManager(window)
    hotkey_mgr.register()

    def cleanup():
        hotkey_mgr.unregister()

    app.aboutToQuit.connect(cleanup)

    logger.info("程序启动完成")
    logger.info("分辨率: 2560x1440 (固定, 分辨率检测已禁用)")
    logger.info(f"热键: Ctrl+1上架 Ctrl+2停止 Ctrl+3下架 Ctrl+4调试截图 Ctrl+Z拾取坐标")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
