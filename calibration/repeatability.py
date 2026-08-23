"""
camera_calibrator.calibration.repeatability
================================================

설계 문서 40번 - Calibration Repeatability.

같은 dataset으로 calibration을 여러 번 수행해도 결과가 같은지 확인한다.
이 모듈은 프레임 순서를 매번 다르게 섞어서 재캘리브레이션을 반복하고,
fx/fy/cx/cy가 얼마나 일관되게 나오는지를 변동계수(CV = std/mean)로 측정한다.

정직한 기대치: cv2.calibrateCamera/cv2.fisheye.calibrate의 최적화는
결정론적이다(초기값도 선형 근사로 고정 계산되고, Levenberg-Marquardt는
같은 비용함수/같은 시작점이면 항상 같은 지역해로 수렴한다) - 그래서
"프레임 순서만" 바꾸는 이 테스트는 대부분의 정상적인 데이터셋에서 거의
100%에 가까운 repeatability를 보이는 게 자연스러운 결과다. 이게 "당연한
결과라 의미 없다"는 뜻은 아니다 - 오히려 "이 계산 파이프라인이 실제로
재현 가능하다"는 것 자체가 검증해야 할 사실이고, repeatability가 낮게
나온다면(드물지만) 데이터가 병적으로 부실하거나(프레임이 너무 적음,
심한 outlier가 안 걸러짐) 모델이 여러 국소해 사이에서 갈팡질팡한다는
뜻이므로 그 자체로 유용한 진단이다.
"""

from __future__ import annotations

import random

import numpy as np

from calibration.types import CameraConfig, CameraModelType, Dataset, RepeatabilityResult

_MIN_SUCCESSFUL_RUNS = 2


def compute_repeatability(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    n_runs: int = 5,
    seed: int = 42,
    use_rational_model: bool = False,
) -> RepeatabilityResult:
    """dataset.frames의 순서를 n_runs번 다르게 섞어 각각 재캘리브레이션하고,
    fx/fy/cx/cy의 변동계수(CV)로 반복 재현성을 측정한다.

    프레임의 "내용"은 절대 바꾸지 않는다 - 오직 리스트 안에서의 순서만
    바뀐다(cv2.calibrateCamera 계열 함수에 들어가는 object_points/image_points
    리스트 순서가 이걸 통해 바뀐다). 각 실행은 완전히 독립적인 캘리브레이션
    이며, 이전 실행 결과를 초기값으로 재사용하지 않는다.
    """
    from calibration.models.pinhole import calibrate_pinhole
    from calibration.models.extended_pinhole import calibrate_extended_pinhole
    from calibration.models.fisheye import calibrate_fisheye

    base_frames = list(dataset.frames)
    rng = random.Random(seed)

    fx_list: list[float] = []
    fy_list: list[float] = []
    cx_list: list[float] = []
    cy_list: list[float] = []
    rms_list: list[float] = []

    for _ in range(n_runs):
        shuffled = base_frames[:]
        rng.shuffle(shuffled)
        shuffled_dataset = Dataset(
            frames=shuffled, coverage_grid=dataset.coverage_grid, diversity=dataset.diversity,
        )

        if model == CameraModelType.PINHOLE:
            result = calibrate_pinhole(shuffled_dataset, camera_config)
        elif model == CameraModelType.EXTENDED_PINHOLE:
            result = calibrate_extended_pinhole(shuffled_dataset, camera_config, use_rational_model=use_rational_model)
        elif model == CameraModelType.FISHEYE:
            result = calibrate_fisheye(shuffled_dataset, camera_config)
        else:
            raise ValueError(f"알 수 없는 모델: {model}")

        if not result.success or result.camera_matrix is None:
            continue

        fx_list.append(float(result.camera_matrix[0, 0]))
        fy_list.append(float(result.camera_matrix[1, 1]))
        cx_list.append(float(result.camera_matrix[0, 2]))
        cy_list.append(float(result.camera_matrix[1, 2]))
        if result.rms_error is not None:
            rms_list.append(result.rms_error)

    if len(fx_list) < _MIN_SUCCESSFUL_RUNS:
        return RepeatabilityResult(n_runs=n_runs, n_successful=len(fx_list))

    def _cv(values: list[float]) -> float:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        return abs(std / mean) if mean != 0 else 0.0

    fx_cv, fy_cv, cx_cv, cy_cv = _cv(fx_list), _cv(fy_list), _cv(cx_list), _cv(cy_list)
    mean_cv = float(np.mean([fx_cv, fy_cv, cx_cv, cy_cv]))
    repeatability_pct = float(max(0.0, min(100.0, 100.0 * (1.0 - mean_cv))))

    return RepeatabilityResult(
        n_runs=n_runs,
        n_successful=len(fx_list),
        fx_cv=fx_cv, fy_cv=fy_cv, cx_cv=cx_cv, cy_cv=cy_cv,
        rms_std=float(np.std(rms_list, ddof=1)) if len(rms_list) > 1 else None,
        repeatability_pct=repeatability_pct,
    )


def format_repeatability(result: RepeatabilityResult) -> str:
    """설계 문서 40번 출력 형식.

        Repeatability = 99.2% (5/5회 성공)
        fx CV=0.08%  fy CV=0.09%  cx CV=0.12%  cy CV=0.10%
        RMS std = 0.004px
    """
    if result.repeatability_pct is None:
        return f"Repeatability: 계산할 수 없습니다 (성공 {result.n_successful}/{result.n_runs}회 - 최소 2회 필요)."

    def pct(v: float | None) -> str:
        return f"{v*100:.2f}%" if v is not None else "N/A"

    lines = [
        f"Repeatability = {result.repeatability_pct:.1f}% ({result.n_successful}/{result.n_runs}회 성공)",
        f"fx CV={pct(result.fx_cv)}  fy CV={pct(result.fy_cv)}  cx CV={pct(result.cx_cv)}  cy CV={pct(result.cy_cv)}",
    ]
    if result.rms_std is not None:
        lines.append(f"RMS std = {result.rms_std:.4f}px")
    return "\n".join(lines)
