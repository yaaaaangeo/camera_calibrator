"""
camera_calibrator.calibration.compare
=========================================

설계 문서 17번 Step4/Step5 - 3개 모델 동시 계산 + Model Comparison.

주의: 여기서는 "비교표를 보여주는 것"까지만 한다.
"어느 모델이 낫다"는 자동 판정(Model Score, 추천)은 recommender.py의 몫이며,
recommender.py는 Hold-out Validation(5단계) 결과가 있어야 의미가 있으므로
아직은 만들지 않는다 (설계 문서 8번: "RMS가 가장 낮은 모델 = 정답"은 금지).
"""

from __future__ import annotations

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
)
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye
from calibration.models.common import fmt_optional, regional_edge_average

# 모델 복잡도 (자유도) - 참고용 별점. Score 공식은 recommender.py에서 사용.
_COMPLEXITY_STARS = {
    CameraModelType.PINHOLE: "*",
    CameraModelType.EXTENDED_PINHOLE: "**",
    CameraModelType.FISHEYE: "***",
}


def run_all_models(
    dataset: Dataset,
    camera_config: CameraConfig,
    use_rational_model: bool = False,
    estimate_fisheye_uncertainty: bool = True,
) -> list[CalibrationResult]:
    """세 모델을 정해진 순서로 계산.

    Pinhole을 가장 먼저 계산하는 이유: Fisheye 초기값으로 넘겨줘야 하기 때문
    (설계 문서 2번 발산 방지). 순서를 바꾸면 안 된다.

    estimate_fisheye_uncertainty: 기본 True. 이 함수는 "사용자에게 보여줄 결과"를
    만드는 1차 실행 경로(app/cli.py, ui/worker.py)에서만 호출되므로, 여기서는
    fisheye의 bootstrap 불확실성 추정(비용이 있음, calibration/models/fisheye.py
    참고)을 기본으로 켠다 - Pinhole/Extended가 공짜로 표준편차를 받는 것과
    체감 상 동일하게 보이도록. validation.py(hold-out 교차검증)와
    outlier.py(반복 재계산)는 fisheye 캘리브레이션을 여러 번 돌리므로 이 함수를
    거치지 않고 calibrate_fisheye()를 직접 호출하며, 거기서는 기본값(False)이 유지된다.
    """
    pinhole_result = calibrate_pinhole(dataset, camera_config)
    extended_result = calibrate_extended_pinhole(
        dataset, camera_config, use_rational_model=use_rational_model
    )
    fisheye_result = calibrate_fisheye(
        dataset, camera_config, initial_guess=pinhole_result,
        estimate_uncertainty=estimate_fisheye_uncertainty,
    )
    return [pinhole_result, extended_result, fisheye_result]


def _model_label(model: CameraModelType) -> str:
    return {
        CameraModelType.PINHOLE: "Pinhole",
        CameraModelType.EXTENDED_PINHOLE: "Extended",
        CameraModelType.FISHEYE: "Fisheye",
    }[model]


def format_comparison_table(results: list[CalibrationResult]) -> str:
    """설계 문서 17번 Step5 형식의 비교표.

                      Pinhole  Extended  Fisheye
        Train RMS        1.12      0.46     0.39
        Mean Error       0.91      0.38     0.34
        Edge Error       2.13      0.61     0.42
        Max Error        3.21      1.82     1.31
        Complexity          *        **      ***
    """
    labels = [_model_label(r.model_name) for r in results]
    col_w = max(10, max(len(l) for l in labels) + 2)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<14}" + "".join(f"{v:>{col_w}}" for v in values)

    header = row("", labels)

    def train_rms(r: CalibrationResult) -> str:
        return f"{r.rms_error:.3f}" if r.success and r.rms_error is not None else "FAIL"

    def mean_error(r: CalibrationResult) -> str:
        if not r.success or not r.per_frame_error:
            return "N/A"
        vals = list(r.per_frame_error.values())
        return f"{sum(vals)/len(vals):.3f}"

    def max_error(r: CalibrationResult) -> str:
        if not r.success or not r.per_frame_error:
            return "N/A"
        return f"{max(r.per_frame_error.values()):.3f}"

    def edge_error(r: CalibrationResult) -> str:
        if not r.success or r.regional_error is None:
            return "N/A"
        return fmt_optional(regional_edge_average(r.regional_error))

    def complexity(r: CalibrationResult) -> str:
        return _COMPLEXITY_STARS.get(r.model_name, "")

    lines = [
        header,
        row("Train RMS", [train_rms(r) for r in results]),
        row("Mean Error", [mean_error(r) for r in results]),
        row("Edge Error", [edge_error(r) for r in results]),
        row("Max Error", [max_error(r) for r in results]),
        row("Complexity", [complexity(r) for r in results]),
    ]

    # 실패한 모델이 있으면 이유를 아래에 별도로 표시 (표 안에 우겨넣지 않음)
    failures = [r for r in results if not r.success]
    if failures:
        lines.append("")
        for r in failures:
            lines.append(f"  [{_model_label(r.model_name)} 실패] {r.error_message}")

    return "\n".join(lines)
