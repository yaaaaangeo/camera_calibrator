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

import logging
from pathlib import Path

from calibration.cache import PersistentResultCache, model_results_cache_key
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
)
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye
from calibration.models.common import fmt_optional, regional_edge_average
from calibration.observability import attach_observability_report
from calibration.undistortion_quality import attach_undistortion_quality_report

logger = logging.getLogger(__name__)

# 모델 복잡도 (자유도) - 참고용 별점. Score 공식은 recommender.py에서 사용.
_COMPLEXITY_STARS = {
    CameraModelType.PINHOLE: "*",
    CameraModelType.BROWN_CONRADY: "**",
    CameraModelType.EXTENDED_PINHOLE: "**",
    CameraModelType.FISHEYE: "***",
}


def run_all_models(
    dataset: Dataset,
    camera_config: CameraConfig,
    use_rational_model: bool = False,
    estimate_fisheye_uncertainty: bool = True,
    bootstrap_jobs: int = 1,
    model_jobs: int = 1,
    persistent_cache_dir: str | Path | None = None,
    models: list[CameraModelType] | tuple[CameraModelType, ...] | None = None,
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
    requested_models = tuple(models or (
        CameraModelType.PINHOLE,
        CameraModelType.BROWN_CONRADY,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    ))
    requested_set = set(requested_models)
    needs_pinhole = CameraModelType.PINHOLE in requested_set or CameraModelType.FISHEYE in requested_set
    needs_brown = CameraModelType.BROWN_CONRADY in requested_set
    needs_extended = CameraModelType.EXTENDED_PINHOLE in requested_set

    cache = PersistentResultCache(persistent_cache_dir, namespace="model-results") if persistent_cache_dir else None
    cache_key = model_results_cache_key(
        dataset, camera_config, use_rational_model, estimate_fisheye_uncertainty, bootstrap_jobs,
        tuple(m.value for m in requested_models),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("모델 결과 persistent cache hit: %s", cache_key[:12])
            return cached

    pinhole_result = None
    brown_result = None
    extended_result = None
    if needs_pinhole:
        pinhole_result = calibrate_pinhole(dataset, camera_config)
    if needs_brown:
        brown_result = calibrate_brown_conrady(dataset, camera_config)
    if needs_extended:
        extended_result = calibrate_extended_pinhole(
            dataset, camera_config, use_rational_model=True
        )
    fisheye_result = None
    if CameraModelType.FISHEYE in requested_set:
        fisheye_result = calibrate_fisheye(
            dataset, camera_config, initial_guess=pinhole_result,
            estimate_uncertainty=estimate_fisheye_uncertainty,
            bootstrap_jobs=bootstrap_jobs,
        )
    by_model = {
        CameraModelType.PINHOLE: pinhole_result,
        CameraModelType.BROWN_CONRADY: brown_result,
        CameraModelType.EXTENDED_PINHOLE: extended_result,
        CameraModelType.FISHEYE: fisheye_result,
    }
    results = [by_model[m] for m in requested_models if by_model.get(m) is not None]
    for result in results:
        attach_observability_report(result, dataset)
        attach_undistortion_quality_report(result, dataset, camera_config)
    if cache is not None:
        cache.set(cache_key, results)
        logger.info("모델 결과 persistent cache 저장: %s", cache_key[:12])
    return results


def _model_label(model: CameraModelType) -> str:
    return {
        CameraModelType.PINHOLE: "Ideal Pinhole",
        CameraModelType.BROWN_CONRADY: "Brown-Conrady",
        CameraModelType.EXTENDED_PINHOLE: "Rational",
        CameraModelType.FISHEYE: "Fisheye",
    }[model]


def format_comparison_table(results: list[CalibrationResult]) -> str:
    """설계 문서 17번 Step5 형식의 비교표.

                      Pinhole  Extended  Fisheye
        Train RMS        1.12      0.46     0.39
        Mean Error       0.91      0.38     0.34
        Edge Error       2.13      0.61     0.42
        Max Error        3.21      1.82     1.31
        P95 (pt)         2.05      0.98     0.81
        P99 (pt)         2.98      1.55     1.20
        Complexity          *        **      ***

    P95/P99(pt)는 설계 문서 11번 - 코너 포인트 단위 재투영 오차의 percentile
    (residual_stats.py). Mean/Max Error는 프레임 단위 RMS의 평균/최댓값이라
    성격이 다르다 - 코너 단위 분포에서 "꼬리"가 얼마나 긴지는 P95/P99만 보여준다.
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

    def p95_pt(r: CalibrationResult) -> str:
        if not r.success or r.residual_stats is None or r.residual_stats.n == 0:
            return "N/A"
        return fmt_optional(r.residual_stats.p95)

    def p99_pt(r: CalibrationResult) -> str:
        if not r.success or r.residual_stats is None or r.residual_stats.n == 0:
            return "N/A"
        return fmt_optional(r.residual_stats.p99)

    def valid_pixels(r: CalibrationResult) -> str:
        if not r.success or r.undistortion_quality is None:
            return "N/A"
        return f"{r.undistortion_quality.valid_pixel_ratio * 100:.1f}%"

    def roi_loss(r: CalibrationResult) -> str:
        if not r.success or r.undistortion_quality is None:
            return "N/A"
        return f"{r.undistortion_quality.roi_loss_ratio * 100:.1f}%"

    def complexity(r: CalibrationResult) -> str:
        return _COMPLEXITY_STARS.get(r.model_name, "")

    lines = [
        header,
        row("Train RMS", [train_rms(r) for r in results]),
        row("Mean Error", [mean_error(r) for r in results]),
        row("Edge Error", [edge_error(r) for r in results]),
        row("Max Error", [max_error(r) for r in results]),
        row("P95 (pt)", [p95_pt(r) for r in results]),
        row("P99 (pt)", [p99_pt(r) for r in results]),
        row("Valid Pixels", [valid_pixels(r) for r in results]),
        row("ROI Loss", [roi_loss(r) for r in results]),
        row("Complexity", [complexity(r) for r in results]),
    ]

    # 실패한 모델이 있으면 이유를 아래에 별도로 표시 (표 안에 우겨넣지 않음)
    failures = [r for r in results if not r.success]
    if failures:
        lines.append("")
        for r in failures:
            lines.append(f"  [{_model_label(r.model_name)} 실패] {r.error_message}")

    return "\n".join(lines)
