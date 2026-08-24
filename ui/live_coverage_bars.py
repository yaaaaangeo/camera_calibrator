"""
camera_calibrator.ui.live_coverage_bars
===========================================

실시간 캡처 중 "어디가 부족한지"를 X/Y/Size/Skew 4개 막대로 보여주는 위젯.
ROS camera_calibration 패키지의 cameracalibrator.py GUI가 보여주는 형태를
참고했다 - 계산 로직은 calibration.quality.compute_live_coverage_bars가
이 프로젝트 기존 지표(distance_diversity, rotation_diversity 등)를 그대로
재사용해서 만든다 (ui/live_coverage_bars.py는 표시만 담당, 계산 없음 -
이 프로젝트의 "UI는 계산하지 않는다" 원칙을 따름).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from calibration.quality import LiveCoverageBars
from ui.theme import Theme


class LiveCoverageBarsWidget(QWidget):
    """X / Y / Size / Skew 4개 진행률 바.

    각 바는 0~100(%)로 표시하고, 70% 이상이면 초록, 그 아래는 주황으로
    색을 바꿔서 "아직 부족한 축이 뭔지" 한눈에 보이게 한다.
    """

    _LABELS = [
        ("x", "X"),
        ("y", "Y"),
        ("size", "Size"),
        ("skew", "Skew"),
    ]
    _GOOD_THRESHOLD = 70  # 이 이상이면 초록

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._bars: dict[str, QProgressBar] = {}
        for row, (key, label_text) in enumerate(self._LABELS):
            label = QLabel(label_text)
            label.setMinimumWidth(40)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            self._bars[key] = bar
            layout.addWidget(label, row, 0)
            layout.addWidget(bar, row, 1)

        self._apply_style(0)  # 초기값(전부 0%) 스타일 적용

    def set_bars(self, bars: LiveCoverageBars) -> None:
        """calibration.quality.compute_live_coverage_bars() 결과를 그대로 반영."""
        values = {
            "x": bars.x_coverage,
            "y": bars.y_coverage,
            "size": bars.size_coverage,
            "skew": bars.skew_coverage,
        }
        for key, score in values.items():
            pct = max(0, min(100, round(score * 100)))
            bar = self._bars[key]
            bar.setValue(pct)
            self._style_bar(bar, pct)

    def reset(self) -> None:
        for bar in self._bars.values():
            bar.setValue(0)
            self._style_bar(bar, 0)

    # ------------------------------------------------------------------

    def _apply_style(self, pct: int) -> None:
        for bar in self._bars.values():
            self._style_bar(bar, pct)

    def _style_bar(self, bar: QProgressBar, pct: int) -> None:
        color = Theme.GOOD if pct >= self._GOOD_THRESHOLD else Theme.WARNING
        bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {Theme.BORDER_STRONG}; border-radius: 3px; "
            f"text-align: center; color: {Theme.TEXT_VALUE}; background: {Theme.BG_TERTIARY}; }}"
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
