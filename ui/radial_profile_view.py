"""
camera_calibrator.ui.radial_profile_view
=============================================

설계 문서 4번 - Radial Error Profile을 그래프로 표시.
백엔드(calibration/radial_profile.py) 결과를 그리기만 한다 - coverage_view.py와
동일한 원칙(계산은 backend, UI는 시각화만).

외부 차트 라이브러리(matplotlib, QtCharts) 의존성을 추가하지 않기 위해
QPainter로 직접 막대그래프를 그린다 (coverage_view.py의 CoverageGridWidget과
같은 접근 방식).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from calibration.types import CalibrationResult, CameraModelType, RadialErrorProfile

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]

_BAR_COLOR_LOW = QColor(46, 125, 50)     # 초록 (오차 작음)
_BAR_COLOR_HIGH = QColor(198, 40, 40)    # 빨강 (오차 큼)
_AXIS_COLOR = QColor(80, 80, 80)
_EMPTY_COLOR = QColor(230, 230, 230)


def _lerp_color(t: float) -> QColor:
    """0(낮음, 초록) ~ 1(높음, 빨강) 사이 선형 보간. coverage_view.py의
    _score_to_color와 반대 방향(여기선 오차가 클수록 나쁨=빨강)."""
    t = max(0.0, min(1.0, t))
    r = int(_BAR_COLOR_LOW.red() + (_BAR_COLOR_HIGH.red() - _BAR_COLOR_LOW.red()) * t)
    g = int(_BAR_COLOR_LOW.green() + (_BAR_COLOR_HIGH.green() - _BAR_COLOR_LOW.green()) * t)
    b = int(_BAR_COLOR_LOW.blue() + (_BAR_COLOR_HIGH.blue() - _BAR_COLOR_LOW.blue()) * t)
    return QColor(r, g, b)


class RadialProfileChartWidget(QWidget):
    """반지름 구간별 평균 재투영 오차를 막대그래프로 표시.

    설계 문서 4번: "렌즈 외곽에서 모델이 잘 동작하는지 바로 확인 가능"이
    핵심 목적이므로, 막대 색상도 오차 크기에 따라 초록(양호)~빨강(주의)으로
    바로 눈에 띄게 한다.
    """

    _MARGIN_LEFT = 55
    _MARGIN_BOTTOM = 40
    _MARGIN_TOP = 20
    _MARGIN_RIGHT = 15

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._profile: RadialErrorProfile | None = None
        self._fixed_max_error: float | None = None
        self.setMinimumHeight(260)

    def set_profile(self, profile: RadialErrorProfile | None, fixed_max_error: float | None = None) -> None:
        """fixed_max_error를 주면 y축 최댓값을 이 값으로 고정한다 (모델 간
        비교용 - 안 주면 이 프로필 자기 자신의 최댓값으로 스케일된다,
        기존 동작과 동일)."""
        self._profile = profile
        self._fixed_max_error = fixed_max_error
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        plot_rect = QRectF(
            self._MARGIN_LEFT,
            self._MARGIN_TOP,
            rect.width() - self._MARGIN_LEFT - self._MARGIN_RIGHT,
            rect.height() - self._MARGIN_TOP - self._MARGIN_BOTTOM,
        )

        if not self._profile or not self._profile.bins:
            painter.setPen(QPen(_AXIS_COLOR))
            painter.drawText(rect, Qt.AlignCenter, "표시할 데이터가 없습니다.\n(캘리브레이션 실행 후 표시됩니다)")
            return

        bins = self._profile.bins
        if self._fixed_max_error is not None:
            # 모델 3개 중 하나라도 가장 큰 오차 구간을 기준으로 y축을 고정해서,
            # 콤보박스로 모델을 바꿔도 막대 높이를 그대로 비교할 수 있게 한다.
            # (예전 버그: 모델마다 자기 자신의 최댓값으로 다시 스케일해서
            # 매번 축이 바뀌었고, 그래서 어느 모델이 실제로 더 나은지
            # 막대 높이만 봐서는 알 수 없었다.)
            max_error = self._fixed_max_error
        else:
            valid_errors = [b.mean_error for b in bins if b.mean_error is not None]
            max_error = max(valid_errors) if valid_errors else 1.0
            max_error = max(max_error, 1e-6) * 1.15  # 여유 15%

        # --- 축 ---
        painter.setPen(QPen(_AXIS_COLOR, 1))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())

        # y축 눈금 (0, 25%, 50%, 75%, 100%)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot_rect.bottom() - frac * plot_rect.height()
            painter.setPen(QPen(QColor(220, 220, 220)))
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
            painter.setPen(QPen(_AXIS_COLOR))
            painter.drawText(
                QRectF(0, y - 8, self._MARGIN_LEFT - 6, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{frac*max_error:.2f}",
            )

        # --- 막대 ---
        n = len(bins)
        bar_gap = 4
        bar_w = (plot_rect.width() - bar_gap * (n - 1)) / n if n > 0 else 0

        for i, b in enumerate(bins):
            x = plot_rect.left() + i * (bar_w + bar_gap)

            if b.mean_error is None:
                # 포인트가 없는 구간(대개 화각 밖 극외곽) - 빈 칸으로 표시
                painter.setPen(QPen(_EMPTY_COLOR))
                painter.drawText(
                    QRectF(x, plot_rect.bottom() - 20, bar_w, 20), Qt.AlignCenter, "N/A"
                )
            else:
                bar_h = (b.mean_error / max_error) * plot_rect.height()
                t = b.mean_error / max_error
                painter.setBrush(_lerp_color(t))
                painter.setPen(Qt.NoPen)
                painter.drawRect(QRectF(x, plot_rect.bottom() - bar_h, bar_w, bar_h))

                painter.setPen(QPen(QColor(30, 30, 30)))
                painter.drawText(
                    QRectF(x, plot_rect.bottom() - bar_h - 16, bar_w, 14),
                    Qt.AlignCenter,
                    f"{b.mean_error:.2f}",
                )

            # x축 라벨 (반지름 구간, px)
            painter.setPen(QPen(_AXIS_COLOR))
            label = f"{b.radius_min:.0f}\n~{b.radius_max:.0f}"
            painter.drawText(
                QRectF(x, plot_rect.bottom() + 4, bar_w, self._MARGIN_BOTTOM - 4),
                Qt.AlignCenter,
                label,
            )

        painter.setPen(QPen(_AXIS_COLOR))
        painter.drawText(
            QRectF(0, rect.height() - 14, rect.width(), 14),
            Qt.AlignCenter,
            "반지름(px, 이미지 중심으로부터 거리) →",
        )


class RadialProfileView(QWidget):
    """모델 선택 콤보 + 차트 + 요약 라벨을 묶은 뷰."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._results: dict[CameraModelType, CalibrationResult] = {}
        self._global_max_error: float | None = None

        layout = QVBoxLayout(self)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        self.model_combo.currentIndexChanged.connect(self._refresh_chart)
        combo_row.addWidget(self.model_combo)
        combo_row.addStretch(1)
        layout.addLayout(combo_row)

        group = QGroupBox("Edge Error Map (Radial Error Profile)")
        group_layout = QVBoxLayout(group)
        self.chart = RadialProfileChartWidget()
        group_layout.addWidget(self.chart)
        self.summary_label = QLabel(
            "설계 문서 4번: 이미지 중심으로부터의 거리에 따른 재투영 오차 분포. "
            "외곽으로 갈수록 막대가 빨갛고 높아지면 렌즈 외곽에서 모델이 잘 안 맞는다는 신호입니다."
        )
        self.summary_label.setWordWrap(True)
        group_layout.addWidget(self.summary_label)
        layout.addWidget(group)

    def set_results(self, calibration_results: dict[CameraModelType, CalibrationResult]) -> None:
        self._results = calibration_results
        # 3개 모델 전체를 통틀어 가장 큰 구간 평균 오차를 y축 상한으로 고정한다.
        # _refresh_chart()가 모델 콤보 변경마다 호출되는데, 그때마다 축이
        # 다시 스케일되면 막대 높이로 모델 간 비교가 안 되기 때문
        # (사용자 피드백: "세로 축이 계속 바뀌니깐 뭐가 더 좋은지 잘 모르겠어").
        all_errors: list[float] = []
        for result in calibration_results.values():
            if not result or not result.success or not result.radial_profile:
                continue
            all_errors.extend(
                b.mean_error for b in result.radial_profile.bins if b.mean_error is not None
            )
        self._global_max_error = max(all_errors) * 1.15 if all_errors else None
        self._refresh_chart()

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)  # currentIndexChanged가 _refresh_chart 호출
        else:
            self._refresh_chart()

    def _refresh_chart(self) -> None:
        model = self.model_combo.currentData()
        result = self._results.get(model)

        if not result or not result.success:
            self.chart.set_profile(None)
            reason = result.error_message if result and result.error_message else "아직 계산되지 않았습니다."
            self.summary_label.setText(f"[{_MODEL_LABELS.get(model, '')}] {reason}")
            return

        profile = result.radial_profile
        self.chart.set_profile(profile, fixed_max_error=self._global_max_error)

        if profile and profile.bins:
            valid = [b for b in profile.bins if b.mean_error is not None]
            if len(valid) >= 2:
                center_err = valid[0].mean_error
                edge_err = valid[-1].mean_error
                trend = "외곽으로 갈수록 오차가 커지는 경향" if edge_err > center_err else "외곽과 중심의 오차 차이가 크지 않음"
                self.summary_label.setText(
                    f"[{_MODEL_LABELS[model]}] 중심 구간 {center_err:.3f}px → "
                    f"외곽 구간 {edge_err:.3f}px  ({trend})"
                )
            else:
                self.summary_label.setText(f"[{_MODEL_LABELS[model]}] 구간별로 비교할 데이터가 충분하지 않습니다.")
        else:
            self.summary_label.setText(f"[{_MODEL_LABELS[model]}] Radial Error Profile 데이터가 없습니다.")
