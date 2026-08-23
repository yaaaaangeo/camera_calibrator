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

from concurrent.futures import ThreadPoolExecutor
import logging
import random

import cv2
import numpy as np

from calibration.models.common import (
    DEFAULT_TERM_CRITERIA,
    collect_calibration_inputs,
    infer_image_size,
    validate_finite_calibration_output,
)
from calibration.performance import resolve_worker_count
from calibration.types import CameraConfig, CameraModelType, Dataset, RepeatabilityResult

logger = logging.getLogger(__name__)

_MIN_SUCCESSFUL_RUNS = 2
_PINHOLE_FLAGS = (
    cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
)


def _sample_from_result(result) -> tuple[float, float, float, float, float | None] | None:
    if not result.success or result.camera_matrix is None:
        return None
    return (
        float(result.camera_matrix[0, 0]),
        float(result.camera_matrix[1, 1]),
        float(result.camera_matrix[0, 2]),
        float(result.camera_matrix[1, 2]),
        result.rms_error,
    )


def _run_repeatability_sample(args: tuple) -> tuple[float, float, float, float, float | None] | None:
    dataset, camera_config, model, order, use_rational_model = args
    from calibration.models.pinhole import calibrate_pinhole
    from calibration.models.extended_pinhole import calibrate_extended_pinhole
    from calibration.models.fisheye import calibrate_fisheye

    shuffled = [dataset.frames[i] for i in order]
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

    return _sample_from_result(result)


def _calibrate_reference(dataset: Dataset, camera_config: CameraConfig, model: CameraModelType, use_rational_model: bool):
    from calibration.models.pinhole import calibrate_pinhole
    from calibration.models.extended_pinhole import calibrate_extended_pinhole
    from calibration.models.fisheye import calibrate_fisheye

    if model == CameraModelType.PINHOLE:
        return calibrate_pinhole(dataset, camera_config)
    if model == CameraModelType.EXTENDED_PINHOLE:
        return calibrate_extended_pinhole(dataset, camera_config, use_rational_model=use_rational_model)
    if model == CameraModelType.FISHEYE:
        pinhole = calibrate_pinhole(dataset, camera_config)
        return calibrate_fisheye(dataset, camera_config, initial_guess=pinhole)
    raise ValueError(f"알 수 없는 모델: {model}")


def _perturb_initial_guess(
    K_ref: np.ndarray,
    D_ref: np.ndarray,
    image_size: tuple[int, int],
    rng: np.random.Generator,
    perturbation: float,
    perturb_distortion: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    K = K_ref.copy().astype(np.float64)
    D = D_ref.copy().astype(np.float64)
    K[0, 0] *= float(1.0 + rng.uniform(-perturbation, perturbation))
    K[1, 1] *= float(1.0 + rng.uniform(-perturbation, perturbation))
    K[0, 2] += float(rng.uniform(-perturbation, perturbation) * width)
    K[1, 2] += float(rng.uniform(-perturbation, perturbation) * height)
    K[0, 2] = float(np.clip(K[0, 2], 0.0, max(width - 1.0, 0.0)))
    K[1, 2] = float(np.clip(K[1, 2], 0.0, max(height - 1.0, 0.0)))
    if perturb_distortion and D.size:
        scale = np.maximum(np.abs(D), 1e-4) * perturbation
        D = D + rng.normal(0.0, scale, size=D.shape)
    return K, D


def _run_initial_condition_sample(args: tuple) -> tuple[float, float, float, float, float | None] | None:
    (
        object_points,
        image_points,
        image_size,
        model,
        K_init,
        D_init,
        use_rational_model,
    ) = args
    try:
        if model == CameraModelType.FISHEYE:
            flags = getattr(cv2.fisheye, "CALIB_USE_INTRINSIC_GUESS", 0)
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                object_points, image_points, image_size,
                K_init.copy(), D_init.copy(), flags=flags,
            )
        else:
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            if model == CameraModelType.PINHOLE:
                flags |= _PINHOLE_FLAGS
            elif model == CameraModelType.EXTENDED_PINHOLE and use_rational_model:
                flags |= cv2.CALIB_RATIONAL_MODEL
            try:
                rms, K, D, _, _ = cv2.calibrateCamera(
                    object_points, image_points, image_size,
                    K_init.copy(), D_init.copy(), flags=flags, criteria=DEFAULT_TERM_CRITERIA,
                )
            except cv2.error:
                if model != CameraModelType.PINHOLE:
                    raise
                rms, K, D, _, _ = cv2.calibrateCamera(
                    object_points, image_points, image_size,
                    K_init.copy(), None, flags=flags, criteria=DEFAULT_TERM_CRITERIA,
                )
    except cv2.error as e:
        logger.debug("Initial-condition repeatability sample failed for %s: %s", model.value, e)
        return None

    if validate_finite_calibration_output(K, D):
        return None
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), float(rms)


def compute_repeatability(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    n_runs: int = 5,
    seed: int = 42,
    use_rational_model: bool = False,
    n_jobs: int = 1,
    vary_initial_conditions: bool = True,
    initial_condition_perturbation: float = 0.05,
) -> RepeatabilityResult:
    """dataset.frames의 순서를 n_runs번 다르게 섞어 각각 재캘리브레이션하고,
    fx/fy/cx/cy의 변동계수(CV)로 반복 재현성을 측정한다.

    프레임의 "내용"은 절대 바꾸지 않는다 - 오직 리스트 안에서의 순서만
    바뀐다(cv2.calibrateCamera 계열 함수에 들어가는 object_points/image_points
    리스트 순서가 이걸 통해 바뀐다). 각 실행은 완전히 독립적인 캘리브레이션
    이며, 이전 실행 결과를 초기값으로 재사용하지 않는다.
    """
    rng = random.Random(seed)

    fx_list: list[float] = []
    fy_list: list[float] = []
    cx_list: list[float] = []
    cy_list: list[float] = []
    rms_list: list[float] = []

    orders: list[list[int]] = []
    base_order = list(range(len(dataset.frames)))
    for _ in range(n_runs):
        order = base_order[:]
        rng.shuffle(order)
        orders.append(order)

    tasks = [(dataset, camera_config, model, order, use_rational_model) for order in orders]
    workers = resolve_worker_count(n_jobs, len(tasks))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            run_results = list(executor.map(_run_repeatability_sample, tasks))
    else:
        run_results = [_run_repeatability_sample(task) for task in tasks]

    order_successful = sum(1 for sample in run_results if sample is not None)

    initial_results = []
    if vary_initial_conditions:
        ref = _calibrate_reference(dataset, camera_config, model, use_rational_model)
        if ref.success and ref.camera_matrix is not None and ref.distortion is not None:
            _, object_points, image_points = collect_calibration_inputs(dataset)
            image_size = infer_image_size(dataset, camera_config)
            np_rng = np.random.default_rng(seed + 7919)
            init_tasks = []
            for _ in range(n_runs):
                K_init, D_init = _perturb_initial_guess(
                    ref.camera_matrix,
                    ref.distortion,
                    image_size,
                    np_rng,
                    initial_condition_perturbation,
                    perturb_distortion=model != CameraModelType.PINHOLE,
                )
                init_tasks.append(
                    (object_points, image_points, image_size, model, K_init, D_init, use_rational_model)
                )
            init_workers = resolve_worker_count(n_jobs, len(init_tasks))
            if init_workers > 1:
                with ThreadPoolExecutor(max_workers=init_workers) as executor:
                    initial_results = list(executor.map(_run_initial_condition_sample, init_tasks))
            else:
                initial_results = [_run_initial_condition_sample(task) for task in init_tasks]

    initial_successful = sum(1 for sample in initial_results if sample is not None)
    all_results = run_results + initial_results

    for sample in all_results:
        if sample is None:
            continue
        fx, fy, cx, cy, rms = sample
        fx_list.append(fx)
        fy_list.append(fy)
        cx_list.append(cx)
        cy_list.append(cy)
        if rms is not None:
            rms_list.append(rms)

    if len(fx_list) < _MIN_SUCCESSFUL_RUNS:
        return RepeatabilityResult(
            n_runs=len(all_results),
            n_successful=len(fx_list),
            order_runs=len(run_results),
            order_successful=order_successful,
            initial_condition_runs=len(initial_results),
            initial_condition_successful=initial_successful,
            initial_condition_perturbation=initial_condition_perturbation if vary_initial_conditions else 0.0,
        )

    def _cv(values: list[float]) -> float:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        return abs(std / mean) if mean != 0 else 0.0

    fx_cv, fy_cv, cx_cv, cy_cv = _cv(fx_list), _cv(fy_list), _cv(cx_list), _cv(cy_list)
    mean_cv = float(np.mean([fx_cv, fy_cv, cx_cv, cy_cv]))
    repeatability_pct = float(max(0.0, min(100.0, 100.0 * (1.0 - mean_cv))))

    return RepeatabilityResult(
        n_runs=len(all_results),
        n_successful=len(fx_list),
        order_runs=len(run_results),
        order_successful=order_successful,
        initial_condition_runs=len(initial_results),
        initial_condition_successful=initial_successful,
        initial_condition_perturbation=initial_condition_perturbation if vary_initial_conditions else 0.0,
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
    if result.order_runs or result.initial_condition_runs:
        lines.append(
            "Runs: "
            f"order shuffle {result.order_successful}/{result.order_runs}, "
            f"initial condition {result.initial_condition_successful}/{result.initial_condition_runs}"
            + (
                f" (perturbation ±{result.initial_condition_perturbation * 100:.1f}%)"
                if result.initial_condition_runs else ""
            )
        )
    if result.rms_std is not None:
        lines.append(f"RMS std = {result.rms_std:.4f}px")
    return "\n".join(lines)
