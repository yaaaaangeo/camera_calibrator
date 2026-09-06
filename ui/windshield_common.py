"""
camera_calibrator.ui.windshield_common
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)를 패널별
파일로 분리하면서, 여러 패널이 공유하는 최소 레벨 상수/헬퍼만 이 모듈에
남긴다.

`ui/windshield_workspace.py`와 각 `ui/windshield_*_panel.py`가 서로를
import하는 순환 참조를 피하기 위한 목적이 크다 - workspace.py는 패널
mixin 클래스들을 import해야 하고, 패널들은 이 공용 헬퍼들을 import해야
하므로, 공용 헬퍼는 어느 쪽도 아닌 별도 모듈에 둔다(calibration/
project_codecs/common.py와 동일한 패턴).

이 파일은 로직을 하나도 새로 추가하지 않았다 - `ui/windshield_workspace.py`
에 있던 코드를 그대로 옮겼을 뿐이다.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTableWidget

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldModelType

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}

_WINDSHIELD_MODEL_LABELS = {
    WindshieldModelType.BASELINE: "Baseline",
    WindshieldModelType.SPHERICAL: "Spherical",
    WindshieldModelType.RESIDUAL_RAY: "Residual Ray",
    WindshieldModelType.SPLINE: "Spline [Advanced]",
}

_STATS_ROWS = [
    ("RMS", "rmse"),
    ("Median", "median"),
    ("P95", "p95"),
    ("P99", "p99"),
    ("Max", "max"),
]

_REGIONAL_ROWS = ["center", "left", "right", "top", "bottom", "corner"]

# spinbox의 "설정 안 함" sentinel - 사용자가 값을 만지지 않으면 config에
# 아무것도 쓰지 않는다(Baseline 등 굴절률/sphere 개념이 없는 모델을 실행할
# 때 의미 없는 값이 끼어들지 않게).
_UNSET_SPINBOX_VALUE = 0.0


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


def _fmt_deg(v) -> str:
    return f"{v:.3f}°" if isinstance(v, (int, float)) else "N/A"


class _ScrollTable(QTableWidget):
    """마우스 휠을 항상 페이지 스크롤로 넘기는 QTableWidget.
    ui/result_view.py::_PageScrollTableWidget과 동일한 패턴(표 자체가
    스크롤하지 않고 곧바로 부모 페이지 스크롤로 넘어가게 함)."""

    def wheelEvent(self, event) -> None:
        event.ignore()


def _fit_table_to_rows(table: QTableWidget) -> None:
    """모든 행을 펼쳐 페이지 스크롤만 쓰도록 table 높이를 맞춘다.
    ui/result_view.py::_fit_table_to_rows와 동일한 패턴."""
    table.resizeRowsToContents()
    header_height = table.horizontalHeader().sizeHint().height()
    rows_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
    table.setFixedHeight(header_height + rows_height + table.frameWidth() * 2 + 4)
