"""
赛博朋克主题模块 — 统一全项目 GUI 样式

所有 QSS、颜色常量、品质颜色、HTML 内联 CSS 和按钮工具函数集中于此。
主窗口、对话框、子控件均从此模块导入样式。
"""
from PyQt5.QtWidgets import QPushButton

# ═══════════════════════════════════════════════════════════
# 霓虹色常量
# ═══════════════════════════════════════════════════════════

NEON_CYAN = "#00f0ff"
NEON_PINK = "#ff2d7b"
NEON_GREEN = "#39ff14"
NEON_PURPLE = "#b026ff"
NEON_GOLD = "#ffb700"
NEON_ORANGE = "#ff6a00"
NEON_RED = "#ff1744"

BG_DARKEST = "#06060e"
BG_DARK = "#0a0a14"
BG_MID = "#12122a"
BG_PANEL = "#16163a"
BG_WIDGET = "#1a1a44"
BG_HOVER = "#24245c"
BORDER_DIM = "#2a2a5c"
BORDER_GLOW = "#3a3a7c"
TEXT_PRIMARY = "#e8e8f0"
TEXT_SECONDARY = "#a0a0c0"
TEXT_DIM = "#606080"

# ═══════════════════════════════════════════════════════════
# 品质颜色 (道具稀有度)
# ═══════════════════════════════════════════════════════════

RARITY_COLORS_HEX = {
    "白色": "#c8c8c8",
    "绿色": "#39ff14",
    "蓝色": "#00b8ff",
    "紫色": "#b026ff",
    "金色": "#ffb700",
    "红色": "#ff1744",
}

# ═══════════════════════════════════════════════════════════
# 按钮工厂
# ═══════════════════════════════════════════════════════════


def styled_btn(text: str, color: str = NEON_CYAN,
               size: str = "10pt", padding: str = "7px 18px") -> QPushButton:
    """创建带霓虹色样式的按钮."""
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{ background: {BG_WIDGET}; color: {color}; "
        f"border: 1px solid {color}; border-radius: 4px; "
        f"font-size: {size}; font-weight: bold; padding: {padding}; }}"
        f"QPushButton:hover {{ background: {color}; color: {BG_DARKEST}; "
        f"border-color: {color}; }}"
        f"QPushButton:pressed {{ background: {BG_DARKEST}; color: {color}; }}"
        f"QPushButton:disabled {{ background: {BG_DARK}; color: {TEXT_DIM}; "
        f"border-color: {BORDER_DIM}; }}"
    )
    return btn


def set_btn_style(btn: QPushButton, color: str = NEON_CYAN,
                  size: str = "10pt", padding: str = "7px 18px"):
    """为已有按钮设置霓虹色样式."""
    btn.setStyleSheet(
        f"QPushButton {{ background: {BG_WIDGET}; color: {color}; "
        f"border: 1px solid {color}; border-radius: 4px; "
        f"font-size: {size}; font-weight: bold; padding: {padding}; }}"
        f"QPushButton:hover {{ background: {color}; color: {BG_DARKEST}; "
        f"border-color: {color}; }}"
        f"QPushButton:pressed {{ background: {BG_DARKEST}; color: {color}; }}"
        f"QPushButton:disabled {{ background: {BG_DARK}; color: {TEXT_DIM}; "
        f"border-color: {BORDER_DIM}; }}"
    )


# ═══════════════════════════════════════════════════════════
# 主窗口全局 QSS  (QMainWindow 及其子控件)
# ═══════════════════════════════════════════════════════════

CYBER_STYLE = f"""
/* ── 基础 ── */
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 10pt;
}}

/* ── 分组框 ── */
QGroupBox {{
    border: 1px solid {NEON_PURPLE};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: bold;
    color: {NEON_CYAN};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {NEON_CYAN};
}}

/* ── 复选框 ── */
QCheckBox {{
    spacing: 6px;
    color: {TEXT_SECONDARY};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {BORDER_GLOW};
    border-radius: 3px;
    background-color: {BG_MID};
}}
QCheckBox::indicator:checked {{
    background-color: {NEON_CYAN};
    border-color: {NEON_CYAN};
}}
QCheckBox::indicator:hover {{
    border-color: {NEON_CYAN};
}}

/* ── 按钮 (默认) ── */
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {NEON_CYAN};
    color: {NEON_CYAN};
}}
QPushButton:pressed {{
    background-color: {NEON_CYAN};
    color: {BG_DARKEST};
}}
QPushButton:disabled {{
    background-color: {BG_DARK};
    color: {TEXT_DIM};
    border-color: {BORDER_DIM};
}}

/* ── 主操作按钮 (objectName 样式) ── */
QPushButton#btnList {{
    background-color: {BG_WIDGET};
    color: {NEON_GREEN};
    border: 2px solid {NEON_GREEN};
}}
QPushButton#btnList:hover {{
    background-color: {NEON_GREEN};
    color: {BG_DARKEST};
}}
QPushButton#btnDelist {{
    background-color: {BG_WIDGET};
    color: {NEON_RED};
    border: 2px solid {NEON_RED};
}}
QPushButton#btnDelist:hover {{
    background-color: {NEON_RED};
    color: {BG_DARKEST};
}}
QPushButton#btnStop {{
    background-color: {BG_WIDGET};
    color: {NEON_ORANGE};
    border: 2px solid {NEON_ORANGE};
}}
QPushButton#btnStop:hover {{
    background-color: {NEON_ORANGE};
    color: {BG_DARKEST};
}}

/* ── 文本编辑区 (日志) ── */
QTextEdit {{
    background-color: {BG_DARKEST};
    color: {NEON_GREEN};
    border: 1px solid {NEON_GREEN}40;
    border-radius: 4px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
    padding: 6px;
    selection-background-color: {NEON_CYAN}40;
}}

/* ── Tab 栏 ── */
QTabWidget::pane {{
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    background-color: {BG_DARK};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {BG_MID};
    color: {TEXT_DIM};
    border: 1px solid {BORDER_DIM};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background-color: {BG_DARK};
    color: {NEON_CYAN};
    border-color: {NEON_CYAN};
    border-bottom: 3px solid {NEON_CYAN};
}}
QTabBar::tab:hover:!selected {{
    color: {NEON_PURPLE};
    border-color: {NEON_PURPLE};
}}

/* ── 状态栏 ── */
QStatusBar {{
    background-color: {BG_DARKEST};
    color: {NEON_CYAN};
    border-top: 1px solid {NEON_CYAN}30;
    font-size: 9pt;
}}

/* ── 滚动区域 ── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background-color: {BG_DARKEST};
    border: none;
    width: 8px;
    height: 8px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {BORDER_GLOW};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background-color: {NEON_CYAN};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px;
    width: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ── SpinBox ── */
QSpinBox {{
    background-color: {BG_MID};
    color: {NEON_CYAN};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    padding: 4px;
    font-family: "Cascadia Code", "Consolas", monospace;
}}
QSpinBox:focus {{
    border-color: {NEON_CYAN};
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox:hover {{
    border-color: {NEON_CYAN};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    selection-background-color: {NEON_CYAN};
    selection-color: {BG_DARKEST};
    border: 1px solid {NEON_CYAN};
}}

/* ── 热键标签 / 坐标标签 ── */
QLabel#hotkey {{
    color: {NEON_GOLD};
    font-weight: bold;
    font-size: 13px;
}}
QLabel#pickerCoord {{
    color: {NEON_GREEN};
    font-size: 14pt;
    font-weight: bold;
    font-family: "Cascadia Code", "Consolas", monospace;
}}
QLabel#appTitle {{
    color: {NEON_CYAN};
    font-size: 14pt;
    font-weight: bold;
    font-family: "Cascadia Code", "Consolas", monospace;
    padding: 0 8px;
}}
"""

# ═══════════════════════════════════════════════════════════
# 对话框通用 QSS  (CollectDialog, ReviewDialog, CoordPickerDialog)
# ═══════════════════════════════════════════════════════════

CYBER_DIALOG_STYLE = f"""
QDialog, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 10pt;
}}
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {NEON_CYAN};
    color: {NEON_CYAN};
}}
QPushButton:pressed {{
    background-color: {NEON_CYAN};
    color: {BG_DARKEST};
}}
QPushButton:disabled {{
    background-color: {BG_DARK};
    color: {TEXT_DIM};
    border-color: {BORDER_DIM};
}}
QTreeWidget {{
    background-color: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
}}
QTreeWidget::item:selected {{
    background-color: {NEON_CYAN};
    color: {BG_DARKEST};
}}
QTreeWidget::item:hover {{
    background-color: {BG_HOVER};
}}
QTableWidget {{
    background-color: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    gridline-color: {BG_PANEL};
}}
QTableCornerButton::section {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_GLOW};
}}
QTableWidget::item {{
    padding: 4px;
}}
QTableWidget::item:selected {{
    background-color: {NEON_CYAN};
    color: {BG_DARKEST};
}}
QHeaderView::section {{
    background-color: {BG_PANEL};
    color: {NEON_CYAN};
    border: 1px solid {BORDER_GLOW};
    padding: 4px 8px;
    font-weight: bold;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background-color: {BG_DARKEST};
    border: none;
    width: 8px;
    height: 8px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {BORDER_GLOW};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background-color: {NEON_CYAN};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px; width: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
QComboBox {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    padding: 3px 8px;
}}
QComboBox:hover {{
    border-color: {NEON_CYAN};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    selection-background-color: {NEON_CYAN};
    selection-color: {BG_DARKEST};
    border: 1px solid {NEON_CYAN};
}}
QProgressBar {{
    background-color: {BG_DARKEST};
    border: 1px solid {BORDER_GLOW};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
}}
QProgressBar::chunk {{
    background-color: {NEON_CYAN};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: {BORDER_GLOW};
    width: 2px;
}}
"""

# ═══════════════════════════════════════════════════════════
# 全局兜底样式 (app.setStyleSheet)
# ═══════════════════════════════════════════════════════════

CYBER_APP_BASE = f"""
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}}
QToolTip {{
    background-color: {BG_PANEL};
    color: {NEON_CYAN};
    border: 1px solid {NEON_CYAN};
    padding: 4px 8px;
    font-size: 9pt;
}}
"""

# ═══════════════════════════════════════════════════════════
# 使用说明 HTML 内联 CSS
# ═══════════════════════════════════════════════════════════

CYBER_HELP_CSS = f"""
<style>
    body {{ color: {TEXT_PRIMARY}; font-family: "Microsoft YaHei", sans-serif; }}
    h2 {{ color: {NEON_CYAN}; margin-top: 20px; margin-bottom: 8px;
          border-bottom: 1px solid {NEON_CYAN}30; padding-bottom: 4px; }}
    h3 {{ color: {NEON_PURPLE}; margin-top: 16px; margin-bottom: 6px; }}
    .section {{ margin-bottom: 14px; }}
    .key {{ background-color: {BG_PANEL}; color: {NEON_CYAN}; padding: 2px 10px;
            border: 1px solid {NEON_CYAN}60; border-radius: 4px;
            font-weight: bold; font-family: "Cascadia Code", "Consolas", monospace; }}
    .warn {{ color: {NEON_ORANGE}; font-weight: bold; }}
    .ok {{ color: {NEON_GREEN}; }}
    table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
    td, th {{ border: 1px solid {BORDER_GLOW}; padding: 8px 14px; text-align: left; }}
    th {{ background-color: {BG_PANEL}; color: {NEON_CYAN}; font-weight: bold; }}
    tr:hover td {{ background-color: {BG_MID}; }}
    ul {{ margin: 4px 0; padding-left: 22px; }}
    li {{ margin: 4px 0; }}
    ol {{ padding-left: 22px; }}
    ol li {{ margin: 5px 0; }}
    .step {{ color: {NEON_PURPLE}; font-weight: bold; }}
    hr {{ border: none; border-top: 1px solid {BORDER_GLOW}; margin: 16px 0; }}
    pre {{ background: {BG_MID}; padding: 10px; border-radius: 4px;
           color: {NEON_GREEN}; border: 1px solid {NEON_GREEN}30;
           font-family: "Cascadia Code", "Consolas", monospace; }}
    code {{ color: {NEON_CYAN}; font-family: "Cascadia Code", "Consolas", monospace; }}
    a {{ color: {NEON_PURPLE}; text-decoration: none; }}
    a:hover {{ color: {NEON_CYAN}; text-decoration: underline; }}
</style>
"""
