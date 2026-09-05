"""Shared Limelight-inspired dark engineering design system for the Qt UI.

Qt Style Sheets do not support CSS custom properties, so colors and dimensions
live here as Python tokens.  Global widget styling is generated from these
tokens; custom painters and dynamic status colors import the same values.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


class Theme:
    BG_PRIMARY = "#111111"
    BG_SECONDARY = "#1A1A1A"
    BG_TERTIARY = "#242424"
    BG_ELEVATED = "#202020"
    BG_HOVER = "#292929"
    BG_SELECTED = "#20330F"

    BORDER = "#333333"
    BORDER_STRONG = "#3A3A3A"

    TEXT_PRIMARY = "#F0F0F0"
    TEXT_VALUE = "#FFFFFF"
    TEXT_SECONDARY = "#A0A0A0"
    TEXT_MUTED = "#777777"
    TEXT_DISABLED = "#666666"

    ACCENT = "#76B900"
    ACCENT_HOVER = "#86C900"
    ACCENT_PRESSED = "#659E00"

    GOOD = "#76B900"
    WARNING = "#E0A800"
    BAD = "#D9534F"
    INFO = "#D8D8D8"

    GRAPH_BG = "#161616"
    GRAPH_GRID = "#303030"
    GRAPH_AXIS = "#BDBDBD"

    TABLE_ODD = "#151515"
    TABLE_EVEN = "#1A1A1A"
    TABLE_HOVER = "#242424"
    TABLE_BEST = "#203315"
    TABLE_WINNER = "#294516"
    TABLE_TIE = "#242424"

    COVERAGE_LOW = "#3A1D1D"
    COVERAGE_HIGH = "#263A14"
    HEATMAP_HIGH = "#722A28"

    TABLE_HEADER_VARIANTS = (
        "#202820",
        "#292020",
        "#242424",
        "#1F2B18",
        "#292618",
        "#1D281B",
    )

    RADIUS = 4


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


APP_STYLESHEET = f"""
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
    color: #AAAAAA;
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
    color: #EEEEEE;
    border: 1px solid {Theme.BORDER_STRONG};
    border-radius: {Theme.RADIUS}px;
    padding: 6px 11px;
    min-height: 20px;
}}
QPushButton:hover {{ background-color: #323232; border-color: #4A4A4A; }}
QPushButton:pressed {{ background-color: #202020; }}
QPushButton:disabled {{ color: {Theme.TEXT_DISABLED}; background-color: #1C1C1C; border-color: #292929; }}
QPushButton[role="primary"] {{
    background-color: {Theme.ACCENT};
    color: {Theme.BG_PRIMARY};
    border-color: {Theme.ACCENT};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background-color: {Theme.ACCENT_HOVER}; border-color: {Theme.ACCENT_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background-color: {Theme.ACCENT_PRESSED}; }}
QPushButton[role="primary"]:disabled {{
    color: {Theme.TEXT_DISABLED}; background-color: #263018; border-color: #354126;
}}
QPushButton[role="danger"] {{ background-color: {Theme.BAD}; color: white; border-color: {Theme.BAD}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background-color: {Theme.BG_TERTIARY};
    color: #F5F5F5;
    border: 1px solid #3B3B3B;
    border-radius: 3px;
    padding: 4px 7px;
    min-height: 20px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {Theme.ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {Theme.TEXT_DISABLED}; background-color: #1D1D1D; border-color: #2A2A2A;
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
    background-color: #2C2C2C;
}}
/* 기본 스타일의 화살표는 어두운 배경 위에서 거의 안 보였다 - 테두리만으로
   흰색 삼각형을 직접 그려서 확실히 보이게 한다. */
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0px; height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #F5F5F5;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0px; height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #F5F5F5;
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
    border: 1px solid #4A4A4A;
}}
QCheckBox::indicator {{ border-radius: 2px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {Theme.ACCENT}; border: 3px solid {Theme.BG_TERTIARY};
}}

QTableView, QTableWidget {{
    background-color: {Theme.TABLE_ODD};
    alternate-background-color: {Theme.TABLE_EVEN};
    color: #DDDDDD;
    gridline-color: #2A2A2A;
    border: 1px solid {Theme.BORDER};
    selection-background-color: {Theme.BG_SELECTED};
    selection-color: {Theme.TEXT_VALUE};
}}
QTableView::item, QTableWidget::item {{ padding: 5px; border-bottom: 1px solid #2A2A2A; }}
QTableView::item:hover, QTableWidget::item:hover {{ background-color: {Theme.TABLE_HOVER}; }}
QHeaderView::section {{
    background-color: #202020;
    color: #EAEAEA;
    border: none;
    border-right: 1px solid {Theme.BORDER};
    border-bottom: 1px solid {Theme.BORDER_STRONG};
    padding: 6px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: #202020; border: 1px solid {Theme.BORDER}; }}

QListView, QListWidget, QTreeView {{
    background-color: {Theme.TABLE_ODD}; color: #DDDDDD;
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
QScrollBar::handle:vertical {{ background: #454545; min-height: 28px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: {Theme.ACCENT}; }}
QScrollBar:horizontal {{ background: {Theme.BG_PRIMARY}; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #454545; min-width: 28px; border-radius: 4px; }}
QScrollBar::handle:horizontal:hover {{ background: {Theme.ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QMenuBar, QMenu, QStatusBar {{ background-color: #151515; color: {Theme.TEXT_SECONDARY}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background-color: {Theme.BG_SELECTED}; color: {Theme.ACCENT}; }}
QMenu {{ border: 1px solid {Theme.BORDER}; }}
QToolTip {{ background-color: {Theme.BG_TERTIARY}; color: {Theme.TEXT_PRIMARY}; border: 1px solid {Theme.BORDER_STRONG}; }}
QProgressDialog {{ background-color: {Theme.BG_SECONDARY}; }}
"""


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
