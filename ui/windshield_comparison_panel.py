"""
camera_calibrator.ui.windshield_comparison_panel
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)에서 분리한
④ Comparison 탭 전용 UI. `ComparisonPanelMixin`은 `WindshieldWorkspace`에
다른 패널 mixin들과 함께 다중 상속되어 같은 `self`를 공유한다(자세한
설명은 ui/windshield_common.py, ui/windshield_geometry_panel.py 참고).

이 파일은 로직을 하나도 새로 추가/변경하지 않았다 - `ui/windshield_workspace.py`
에 있던 메서드를 그대로 옮겼을 뿐이다.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTableWidgetItem, QVBoxLayout, QWidget

from calibration.models.common import regional_edge_average
from calibration.windshield.base import WindshieldModelType, WindshieldResultKey, windshield_result_key, windshield_result_key_label
from ui.windshield_common import _ScrollTable, _fit_table_to_rows, _fmt, _fmt_deg


class ComparisonPanelMixin:
    """④ Comparison 탭 - Baseline/Spherical/Residual Grid/Residual RBF/
    Spline 결과를 나란히 비교하는 표만 그린다."""

    def _build_comparison_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "WINDSHIELD MODEL COMPARISON - Camera Model(Pinhole/Brown/Rational/Fisheye) "
            "비교와는 완전히 별개의 표입니다."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.comparison_table = _ScrollTable(5, 0)
        self.comparison_table.setVerticalHeaderLabels(
            ["Hold-out RMS", "P95", "Edge RMS", "Ray Angular Error", "Improvement %"]
        )
        layout.addWidget(self.comparison_table)
        layout.addStretch(1)
        return page

    def _refresh_comparison_table(self) -> None:
        order: list[WindshieldResultKey] = [
            WindshieldModelType.BASELINE,
            WindshieldModelType.SPHERICAL,
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "grid"),
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "rbf"),
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "neural"),
            WindshieldModelType.SPLINE,
        ]
        present = [m for m in order if m in self._windshield_results]
        self.comparison_table.setColumnCount(len(present))
        self.comparison_table.setHorizontalHeaderLabels([windshield_result_key_label(m) for m in present])

        baseline = self._windshield_results.get(WindshieldModelType.BASELINE)
        baseline_rms = (
            baseline.test_residual_stats.rmse
            if baseline and baseline.test_residual_stats and baseline.test_residual_stats.rmse
            else None
        )

        for col, model in enumerate(present):
            result = self._windshield_results[model]
            test_stats = result.test_residual_stats
            rms = test_stats.rmse if test_stats else None
            p95 = test_stats.p95 if test_stats else None
            # Hold-out(Test) 기준으로 일관되게 비교한다 - Train 쪽 regional_error를
            # 쓰면 "Hold-out RMS/P95"와 기준이 달라져 비교표 의미가 흐려진다.
            edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
            if model == WindshieldModelType.BASELINE or baseline_rms is None or rms is None:
                improvement = None
            else:
                improvement = (baseline_rms - rms) / baseline_rms * 100.0

            self.comparison_table.setItem(0, col, QTableWidgetItem(_fmt(rms)))
            self.comparison_table.setItem(1, col, QTableWidgetItem(_fmt(p95)))
            self.comparison_table.setItem(2, col, QTableWidgetItem(_fmt(edge)))
            self.comparison_table.setItem(3, col, QTableWidgetItem(_fmt_deg(result.test_ray_angular_error_deg)))
            self.comparison_table.setItem(
                4, col, QTableWidgetItem(f"{improvement:.0f}%" if improvement is not None else "-")
            )
        _fit_table_to_rows(self.comparison_table)

    # ------------------------------------------------------------------
    # STEP 8A - Ghost Evaluation handlers
    # ------------------------------------------------------------------

