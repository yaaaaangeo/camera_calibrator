"""Shared engineering design system (dark or light, sky-blue accent) for the Qt UI.

Qt Style Sheets do not support CSS custom properties, so colors and dimensions
live here as Python tokens.  Global widget styling is generated from these
tokens; custom painters and dynamic status colors import the same values.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


import json
import os
from pathlib import Path

# 화면 테마: "dark"(기본) / "light"(흰 바탕에 검은 글씨).
# 정하는 순서: 환경변수 CALIB_THEME > 설정 파일(메뉴 보기 > 화면 테마) > dark.
# 위젯이 만들어질 때 색을 스타일시트에 넣으므로 테마는 앱을 시작할 때 정해지고,
# 바꾸면 앱을 다시 시작한다 (main_window 의 보기 메뉴).
UI_CONFIG_PATH = Path.home() / ".config" / "calibration-tool" / "ui.json"
MODES = ("dark", "light")

_DARK = dict(
    BG_PRIMARY="#111111", BG_SECONDARY="#1A1A1A", BG_TERTIARY="#242424", BG_ELEVATED="#202020",
    BG_HOVER="#292929", BG_SELECTED="#0F2D3D",
    BORDER="#333333", BORDER_STRONG="#3A3A3A",
    TEXT_PRIMARY="#F0F0F0", TEXT_VALUE="#FFFFFF", TEXT_SECONDARY="#A0A0A0", TEXT_MUTED="#777777",
    TEXT_DISABLED="#666666",
    ACCENT="#38BDF8", ACCENT_HOVER="#7DD3FC", ACCENT_PRESSED="#0EA5E9", ACCENT_TEXT="#111111",
    GOOD="#4ADE80", WARNING="#E0A800", BAD="#D9534F", INFO="#D8D8D8",
    GRAPH_BG="#161616", GRAPH_GRID="#303030", GRAPH_AXIS="#BDBDBD", CANVAS_BG="#1E1E1E",
    TABLE_ODD="#151515", TABLE_EVEN="#1A1A1A", TABLE_HOVER="#242424", TABLE_BEST="#13293A",
    TABLE_WINNER="#173A52", TABLE_TIE="#242424", TABLE_TEXT="#DDDDDD", TABLE_GRID="#2A2A2A",
    HEADER_BG="#202020", HEADER_TEXT="#EAEAEA",
    COVERAGE_LOW="#3A1D1D", COVERAGE_HIGH="#263A14", HEATMAP_HIGH="#722A28",
    TABLE_HEADER_VARIANTS=("#202820", "#292020", "#242424", "#1F2B18", "#292618", "#1D281B"),
    TAB_TEXT="#AAAAAA", BUTTON_TEXT="#EEEEEE", BUTTON_HOVER="#323232", BUTTON_HOVER_BORDER="#4A4A4A",
    BUTTON_PRESSED="#202020", BUTTON_DISABLED_BG="#1C1C1C", BUTTON_DISABLED_BORDER="#292929",
    PRIMARY_DISABLED_BG="#15303F", PRIMARY_DISABLED_BORDER="#1E4255",
    INPUT_TEXT="#F5F5F5", INPUT_BORDER="#3B3B3B", INPUT_DISABLED_BG="#1D1D1D", INPUT_DISABLED_BORDER="#2A2A2A",
    SPIN_HOVER="#2C2C2C", ARROW="#F5F5F5", CHECK_BORDER="#4A4A4A", SCROLL_HANDLE="#454545", MENU_BG="#151515",
)

_LIGHT = dict(
    BG_PRIMARY="#FFFFFF", BG_SECONDARY="#F6F7F8", BG_TERTIARY="#FFFFFF", BG_ELEVATED="#F0F1F3",
    BG_HOVER="#F0F1F3", BG_SELECTED="#DCEEFB",
    BORDER="#D5D8DC", BORDER_STRONG="#BFC4CA",
    TEXT_PRIMARY="#111111", TEXT_VALUE="#000000", TEXT_SECONDARY="#555555", TEXT_MUTED="#6B6B6B",
    TEXT_DISABLED="#A3A3A3",
    ACCENT="#0284C7", ACCENT_HOVER="#0369A1", ACCENT_PRESSED="#075985", ACCENT_TEXT="#FFFFFF",
    GOOD="#15803D", WARNING="#B45309", BAD="#C62828", INFO="#374151",
    GRAPH_BG="#FFFFFF", GRAPH_GRID="#E3E5E8", GRAPH_AXIS="#333333", CANVAS_BG="#EEEEEE",
    TABLE_ODD="#FFFFFF", TABLE_EVEN="#F7F8F9", TABLE_HOVER="#EEF5FB", TABLE_BEST="#E0F2FE",
    TABLE_WINNER="#BAE6FD", TABLE_TIE="#EEEEEE", TABLE_TEXT="#111111", TABLE_GRID="#E3E5E8",
    HEADER_BG="#F0F1F3", HEADER_TEXT="#111111",
    COVERAGE_LOW="#FBE3E3", COVERAGE_HIGH="#DFF2D3", HEATMAP_HIGH="#F2A29E",
    TABLE_HEADER_VARIANTS=("#E7F2E7", "#F6E6E6", "#EEEEEE", "#E3F0DA", "#F5F0DC", "#E1EFDD"),
    TAB_TEXT="#555555", BUTTON_TEXT="#111111", BUTTON_HOVER="#E6E8EB", BUTTON_HOVER_BORDER="#9AA1A9",
    BUTTON_PRESSED="#D9DCE0", BUTTON_DISABLED_BG="#F3F4F5", BUTTON_DISABLED_BORDER="#E2E4E7",
    PRIMARY_DISABLED_BG="#B9DDF0", PRIMARY_DISABLED_BORDER="#B9DDF0",
    INPUT_TEXT="#111111", INPUT_BORDER="#BFC4CA", INPUT_DISABLED_BG="#F3F4F5", INPUT_DISABLED_BORDER="#E2E4E7",
    SPIN_HOVER="#E6E8EB", ARROW="#333333", CHECK_BORDER="#8A9099", SCROLL_HANDLE="#C4C8CD", MENU_BG="#F6F7F8",
)

PALETTES = {"dark": _DARK, "light": _LIGHT}


def saved_mode() -> str:
    try:
        mode = json.loads(UI_CONFIG_PATH.read_text(encoding="utf-8")).get("theme", "")
    except (OSError, ValueError, AttributeError):
        mode = ""
    return mode if mode in MODES else "dark"


def current_mode() -> str:
    env = os.environ.get("CALIB_THEME", "").strip().lower()
    return env if env in MODES else saved_mode()


def save_mode(mode: str) -> None:
    """다음 실행부터 쓸 테마를 저장한다."""
    if mode not in MODES:
        raise ValueError(mode)
    try:
        data = json.loads(UI_CONFIG_PATH.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    data["theme"] = mode
    UI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Theme:
    """현재 테마의 색 토큰. 값은 앱 시작 때 `PALETTES[current_mode()]` 로 채워진다."""

    MODE = "dark"
    RADIUS = 4


def _use(mode: str) -> None:
    Theme.MODE = mode
    for key, value in PALETTES[mode].items():
        setattr(Theme, key, value)


_use(current_mode())


def qcolor(value: str) -> QColor:
    return QColor(value)


def tone_style(tone: str) -> str:
    color = {
        "good": Theme.GOOD,
        "warning": Theme.WARNING,
        "bad": Theme.BAD,
        "info": Theme.INFO,
        "muted": Theme.TEXT_SECONDARY,
    }.get(tone, Theme.TEXT_PRIMARY)
    return f"color: {color};"


def set_tone(widget: QWidget, tone: str) -> None:
    """Set a dynamic semantic tone and immediately refresh QSS matching."""
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {Theme.BG_PRIMARY};
        color: {Theme.TEXT_PRIMARY};
        selection-background-color: {Theme.BG_SELECTED};
        selection-color: {Theme.TEXT_VALUE};
        font-size: 10pt;
    }}
    
    QMainWindow, QDialog {{ background-color: {Theme.BG_PRIMARY}; }}
    
    QLabel {{ background: transparent; }}
    QLabel[tone="muted"] {{ color: {Theme.TEXT_SECONDARY}; }}
    QLabel[tone="good"] {{ color: {Theme.GOOD}; }}
    QLabel[tone="warning"] {{ color: {Theme.WARNING}; }}
    QLabel[tone="bad"] {{ color: {Theme.BAD}; }}
    QLabel[tone="info"] {{ color: {Theme.INFO}; }}
    QLabel[role="sectionTitle"] {{ color: {Theme.TEXT_VALUE}; font-weight: 600; }}
    QLabel[surface="image"] {{
        background-color: {Theme.GRAPH_BG};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS}px;
    }}
    
    QGroupBox {{
        background-color: {Theme.BG_SECONDARY};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS}px;
        margin-top: 18px;
        padding: 10px 8px 8px 8px;
        font-weight: 600;
        color: {Theme.TEXT_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: {Theme.TEXT_PRIMARY};
        background-color: {Theme.BG_SECONDARY};
    }}
    QGroupBox#settingsPanel {{ background-color: {Theme.BG_SECONDARY}; }}
    QGroupBox#settingsPanel::indicator {{ width: 0px; height: 0px; image: none; }}
    QGroupBox#settingsPanel::title {{ color: {Theme.ACCENT}; }}
    QGroupBox#advancedCalibrationPanel::indicator {{ width: 0px; height: 0px; image: none; }}
    QGroupBox#advancedCalibrationPanel::title {{ color: {Theme.ACCENT}; }}
    
    QTabWidget::pane {{
        border: 1px solid {Theme.BORDER};
        background-color: {Theme.BG_PRIMARY};
        top: -1px;
    }}
    QTabBar {{ background-color: {Theme.BG_PRIMARY}; }}
    QTabBar::tab {{
        background-color: {Theme.BG_PRIMARY};
        color: {Theme.TAB_TEXT};
        border: none;
        border-bottom: 2px solid transparent;
        padding: 9px 13px;
        margin-right: 1px;
    }}
    QTabBar::tab:hover {{ background-color: {Theme.BG_ELEVATED}; color: {Theme.TEXT_PRIMARY}; }}
    QTabBar::tab:selected {{
        color: {Theme.ACCENT};
        background-color: {Theme.BG_SECONDARY};
        border-bottom: 2px solid {Theme.ACCENT};
        font-weight: 600;
    }}
    
    QPushButton {{
        background-color: {Theme.BG_HOVER};
        color: {Theme.BUTTON_TEXT};
        border: 1px solid {Theme.BORDER_STRONG};
        border-radius: {Theme.RADIUS}px;
        padding: 6px 11px;
        min-height: 20px;
    }}
    QPushButton:hover {{ background-color: {Theme.BUTTON_HOVER}; border-color: {Theme.BUTTON_HOVER_BORDER}; }}
    QPushButton:pressed {{ background-color: {Theme.BUTTON_PRESSED}; }}
    QPushButton:disabled {{ color: {Theme.TEXT_DISABLED}; background-color: {Theme.BUTTON_DISABLED_BG}; border-color: {Theme.BUTTON_DISABLED_BORDER}; }}
    QPushButton[role="primary"] {{
        background-color: {Theme.ACCENT};
        color: {Theme.ACCENT_TEXT};
        border-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{ background-color: {Theme.ACCENT_HOVER}; border-color: {Theme.ACCENT_HOVER}; }}
    QPushButton[role="primary"]:pressed {{ background-color: {Theme.ACCENT_PRESSED}; }}
    QPushButton[role="primary"]:disabled {{
        color: {Theme.TEXT_DISABLED}; background-color: {Theme.PRIMARY_DISABLED_BG}; border-color: {Theme.PRIMARY_DISABLED_BORDER};
    }}
    QPushButton[role="danger"] {{ background-color: {Theme.BAD}; color: white; border-color: {Theme.BAD}; }}
    
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
        background-color: {Theme.BG_TERTIARY};
        color: {Theme.INPUT_TEXT};
        border: 1px solid {Theme.INPUT_BORDER};
        border-radius: 3px;
        padding: 4px 7px;
        min-height: 20px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {Theme.ACCENT}; }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        color: {Theme.TEXT_DISABLED}; background-color: {Theme.INPUT_DISABLED_BG}; border-color: {Theme.INPUT_DISABLED_BORDER};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background-color: {Theme.BG_TERTIARY};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BORDER_STRONG};
        selection-background-color: {Theme.BG_SELECTED};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {Theme.BG_ELEVATED}; border-left: 1px solid {Theme.BORDER}; width: 18px;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {Theme.SPIN_HOVER};
    }}
    /* 기본 스타일의 화살표는 어두운 배경 위에서 거의 안 보였다 - 테두리만으로
       삼각형을 직접 그려서 확실히 보이게 한다. */
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        width: 0px; height: 0px;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid {Theme.ARROW};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 0px; height: 0px;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {Theme.ARROW};
    }}
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
        border-bottom-color: {Theme.TEXT_DISABLED};
    }}
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
        border-top-color: {Theme.TEXT_DISABLED};
    }}
    
    QCheckBox, QRadioButton {{ color: {Theme.TEXT_PRIMARY}; spacing: 7px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px; height: 15px; background-color: {Theme.BG_TERTIARY};
        border: 1px solid {Theme.CHECK_BORDER};
    }}
    QCheckBox::indicator {{ border-radius: 2px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {Theme.ACCENT}; border: 3px solid {Theme.BG_TERTIARY};
    }}
    
    QTableView, QTableWidget {{
        background-color: {Theme.TABLE_ODD};
        alternate-background-color: {Theme.TABLE_EVEN};
        color: {Theme.TABLE_TEXT};
        gridline-color: {Theme.TABLE_GRID};
        border: 1px solid {Theme.BORDER};
        selection-background-color: {Theme.BG_SELECTED};
        selection-color: {Theme.TEXT_VALUE};
    }}
    QTableView::item, QTableWidget::item {{ padding: 5px; border-bottom: 1px solid {Theme.TABLE_GRID}; }}
    QTableView::item:hover, QTableWidget::item:hover {{ background-color: {Theme.TABLE_HOVER}; }}
    QHeaderView::section {{
        background-color: {Theme.HEADER_BG};
        color: {Theme.HEADER_TEXT};
        border: none;
        border-right: 1px solid {Theme.BORDER};
        border-bottom: 1px solid {Theme.BORDER_STRONG};
        padding: 6px;
        font-weight: 600;
    }}
    QTableCornerButton::section {{ background-color: {Theme.HEADER_BG}; border: 1px solid {Theme.BORDER}; }}
    
    QListView, QListWidget, QTreeView {{
        background-color: {Theme.TABLE_ODD}; color: {Theme.TABLE_TEXT};
        border: 1px solid {Theme.BORDER}; alternate-background-color: {Theme.TABLE_EVEN};
    }}
    QListView::item, QListWidget::item, QTreeView::item {{ padding: 5px; }}
    QListView::item:hover, QListWidget::item:hover, QTreeView::item:hover {{ background-color: {Theme.TABLE_HOVER}; }}
    QListView::item:selected, QListWidget::item:selected, QTreeView::item:selected {{
        background-color: {Theme.BG_SELECTED}; color: {Theme.TEXT_VALUE}; border-left: 2px solid {Theme.ACCENT};
    }}
    
    QProgressBar {{
        background-color: {Theme.BG_TERTIARY}; color: {Theme.TEXT_VALUE};
        border: 1px solid {Theme.BORDER_STRONG}; border-radius: 3px;
        text-align: center; min-height: 16px;
    }}
    QProgressBar::chunk {{ background-color: {Theme.ACCENT}; border-radius: 2px; }}
    
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: {Theme.BG_PRIMARY}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {Theme.SCROLL_HANDLE}; min-height: 28px; border-radius: 4px; }}
    QScrollBar::handle:vertical:hover {{ background: {Theme.ACCENT}; }}
    QScrollBar:horizontal {{ background: {Theme.BG_PRIMARY}; height: 11px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {Theme.SCROLL_HANDLE}; min-width: 28px; border-radius: 4px; }}
    QScrollBar::handle:horizontal:hover {{ background: {Theme.ACCENT}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    
    QMenuBar, QMenu, QStatusBar {{ background-color: {Theme.MENU_BG}; color: {Theme.TEXT_SECONDARY}; }}
    QMenuBar::item:selected, QMenu::item:selected {{ background-color: {Theme.BG_SELECTED}; color: {Theme.ACCENT}; }}
    QMenu {{ border: 1px solid {Theme.BORDER}; }}
    QToolTip {{ background-color: {Theme.BG_TERTIARY}; color: {Theme.TEXT_PRIMARY}; border: 1px solid {Theme.BORDER_STRONG}; }}
    QProgressDialog {{ background-color: {Theme.BG_SECONDARY}; }}
"""


APP_STYLESHEET = build_stylesheet()


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, qcolor(Theme.BG_PRIMARY))
    palette.setColor(QPalette.WindowText, qcolor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, qcolor(Theme.BG_TERTIARY))
    palette.setColor(QPalette.AlternateBase, qcolor(Theme.TABLE_EVEN))
    palette.setColor(QPalette.Text, qcolor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, qcolor(Theme.BG_HOVER))
    palette.setColor(QPalette.ButtonText, qcolor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight, qcolor(Theme.BG_SELECTED))
    palette.setColor(QPalette.HighlightedText, qcolor(Theme.TEXT_VALUE))
    palette.setColor(QPalette.PlaceholderText, qcolor(Theme.TEXT_MUTED))
    palette.setColor(QPalette.Disabled, QPalette.Text, qcolor(Theme.TEXT_DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, qcolor(Theme.TEXT_DISABLED))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)
