"""
camera_calibrator.calibration.bootstrap
===========================================

설계 문서 20/21/22번 - Bootstrap Stability / Parameter Confidence Interval을
Fisheye 전용이 아니라 Standard 4모델 전부에서 쓸 수 있게 일반화한 모듈.

원래 이 로직은 calibration/models/fisheye.py의 _bootstrap_fisheye_uncertainty()
안에만 있었다 - fisheye는 cv2.fisheye.calibrate가 stdDeviations를 안 줘서
(Pinhole/Brown-Conrady/Extended처럼 "공짜로" 못 얻어서) 어쩔 수 없이
bootstrap을 썼던 것. 하지만 "이 데이터셋으로 추정한 파라미터가 얼마나
안정적인가"는 Pinhole/Brown-Conrady/Extended에도 똑같이 유용한 질문이다 -
covariance 기반 표준편차는 선형화된 근사치일 뿐이고, bootstrap은 실제
재표본화로 얻은 경험적 분포라 서로 다른 관점의 교차검증 역할을 한다. 그래서
이 함수를 모델 무관하게 만들어 네 모델 다 (선택적으로) bootstrap 불확실성을
계산할 수 있게 했다.

방법론(기존 fisheye 전용 버전과 동일, 정직하게 한계도 그대로 명시):
    전체 데이터로 얻은 K_ref/D_ref를 초기값 삼아 프레임을 복원추출(bootstrap)로
    재표본화해 n_bootstrap번 재캘리브레이션하고, fx/fy/cx/cy의 표준편차와
    95% CI(2.5/97.5 percentile)를 구한다.

    한계: 각 재표본이 전체 데이터의 K_ref를 초기값(CALIB_USE_INTRINSIC_GUESS)
    으로 재사용하므로, "전체 데이터 추정치 근방에서의 국소적 분산"을 재는
    셈이라 완전히 독립적인 붓스트랩보다 분산을 다소 과소평가할 수 있다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

import cv2
import numpy as np

from calibration.models.common import distortion_coeff_labels, expected_distortion_coeff_count
from calibration.performance import resolve_worker_count
from calibration.types import CameraModelType, DistortionCoeffStat, ParameterUncertainty

logger = logging.getLogger(__name__)

_MIN_SUCCESSFUL_SAMPLES = 5
_STABILITY_EPS = 1e-9
# Paper Evidence 단계 - "reference(전체 데이터 fit 값)가 0에 가까우면 상대
# CV가 통계적으로 의미가 약하다"를 판단하는 진단용 heuristic cutoff. 이
# 값을 바꾼다고 어떤 모델의 stability 점수가 달라지지 않는다 - near-zero
# diagnostic 플래그를 붙일지 말지에만 쓰인다. distortion 계수(k1~k4 등)는
# 보통 이 크기 이상이므로("아주 작은 왜곡"이라도 보통 1e-3보다는 크다)
# 1e-3을 "0에 가깝다"의 기준으로 삼는다 - 특정 계수를 겨냥해 고른 값이 아니다.
_NEAR_ZERO_REFERENCE_THRESHOLD = 1e-3


def _run_bootstrap_sample(args: tuple) -> tuple[float, float, float, float, list[float]] | None:
    (
        object_points,
        image_points,
        image_size,
        idx,
        is_fisheye,
        K_init,
        D_init,
        flags,
        distortion_count,
    ) = args
    obj_sample = [object_points[int(i)] for i in idx]
    img_sample = [image_points[int(i)] for i in idx]
    try:
        if is_fisheye:
            _, K_i, D_i, _, _ = cv2.fisheye.calibrate(
                obj_sample, img_sample, image_size,
                K_init.copy(), D_init.copy(), flags=flags,
            )
        else:
            _, K_i, D_i, _, _ = cv2.calibrateCamera(
                obj_sample, img_sample, image_size,
                K_init.copy(), D_init.copy(), flags=flags,
            )
    except cv2.error:
        return None

    return (
        float(K_i[0, 0]),
        float(K_i[1, 1]),
        float(K_i[0, 2]),
        float(K_i[1, 2]),
        [float(v) for v in D_i.ravel().tolist()[:distortion_count]],
    )


def _stability_score(samples: list[float], reference: float | None = None) -> float | None:
    if len(samples) < 2:
        return None
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    return _stability_from_std(mean, std, reference)


def _stability_from_std(mean: float, std: float, reference: float | None = None) -> float:
    scale = max(abs(mean), abs(reference or 0.0), _STABILITY_EPS)
    cv = std / scale
    return float(max(0.0, min(100.0, 100.0 * (1.0 - cv))))


def _pure_relative_cv(mean: float | None, std: float | None) -> float | None:
    """논문 수식 그대로의 CV_p = sigma_p / mu_p (mu_p = bootstrap 표본 평균).

    기존 stability_score가 쓰는 _stability_from_std()는 scale에 reference
    (전체 데이터 fit 값)도 섞어서(max(|mean|,|reference|,eps)) 안정성 점수
    자체를 계산하는 데 쓰이므로 그 계산은 그대로 둔다 - 이 함수는 "논문
    수식과 정확히 같은 CV"를 진단용으로 별도 노출하기 위한 것이다.
    """
    if mean is None or std is None:
        return None
    denom = max(abs(mean), _STABILITY_EPS)
    return float(std / denom)


def _paper_stability_fields(
    fx: dict, fy: dict, cx: dict, cy: dict,
    d_stats: list[DistortionCoeffStat],
    fx_ref: float | None, fy_ref: float | None, cx_ref: float | None, cy_ref: float | None,
) -> dict:
    """paper_intrinsic_stability 등 Paper Evidence 전용 필드를 한 곳에서 계산.

    compute_parameter_bootstrap()(method="bootstrap")과
    add_normal_approximation_ci()(method="covariance") 둘 다 fx/fy/cx/cy
    각각의 _distribution() 결과(dict: mean/std/median/... /stability)와
    distortion_stats를 이미 갖고 있으므로, 이 함수는 그 값들로부터
    paper_intrinsic_stability/distortion_stability_summary/lowest-stability
    parameter/근-영 진단 플래그만 추가로 뽑아낸다 - 어떤 기존 값도 다시
    계산하거나 바꾸지 않는다(순수 파생값).
    """
    intrinsic_stabilities = [
        v for v in (fx["stability"], fy["stability"], cx["stability"], cy["stability"]) if v is not None
    ]
    paper_intrinsic_stability = float(np.mean(intrinsic_stabilities)) if intrinsic_stabilities else None

    distortion_stabilities = [s.stability_score for s in d_stats if s.stability_score is not None]
    distortion_stability_summary = float(np.mean(distortion_stabilities)) if distortion_stabilities else None

    # 각 distortion coefficient에 reference/relative_cv/near-zero 진단을 채운다
    # (in-place - d_stats는 호출부가 막 만든 리스트라 공유 걱정이 없다).
    for stat in d_stats:
        stat.relative_cv = _pure_relative_cv(stat.mean, stat.std)
        if stat.reference is not None and abs(stat.reference) < _NEAR_ZERO_REFERENCE_THRESHOLD:
            stat.near_zero_reference = True
            stat.diagnostic = "near-zero coefficient; relative CV unstable"

    # 전체(intrinsic 4개 + distortion 전부)에서 stability_score가 가장 낮은
    # 파라미터 하나를 찾는다 - "Fisheye 62%가 어디서 왔는지" UI가 바로
    # 짚어줄 수 있게.
    candidates: list[tuple[str, float]] = []
    for label, value in (("fx", fx["stability"]), ("fy", fy["stability"]), ("cx", cx["stability"]), ("cy", cy["stability"])):
        if value is not None:
            candidates.append((label, value))
    for stat in d_stats:
        if stat.stability_score is not None:
            candidates.append((stat.label or f"d{stat.index}", stat.stability_score))
    lowest_label, lowest_value = (None, None)
    if candidates:
        lowest_label, lowest_value = min(candidates, key=lambda item: item[1])

    return {
        "paper_intrinsic_stability": paper_intrinsic_stability,
        "distortion_stability_summary": distortion_stability_summary,
        "fx_reference": fx_ref, "fy_reference": fy_ref, "cx_reference": cx_ref, "cy_reference": cy_ref,
        "fx_relative_cv": _pure_relative_cv(fx["mean"], fx["std"]),
        "fy_relative_cv": _pure_relative_cv(fy["mean"], fy["std"]),
        "cx_relative_cv": _pure_relative_cv(cx["mean"], cx["std"]),
        "cy_relative_cv": _pure_relative_cv(cy["mean"], cy["std"]),
        "lowest_stability_parameter": lowest_label,
        "lowest_stability_value": lowest_value,
    }


def _distribution(samples: list[float], reference: float | None = None) -> dict[str, float | None]:
    if not samples:
        return {
            "mean": None, "std": None, "median": None, "min": None, "max": None,
            "ci_low": None, "ci_high": None, "stability": None,
        }
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "median": float(np.median(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "stability": _stability_score(samples, reference),
    }


def _distortion_stats(
    distortion_samples: list[list[float]],
    model: CameraModelType,
    D_ref: np.ndarray,
) -> list[DistortionCoeffStat]:
    if not distortion_samples:
        return []
    max_len = max(len(sample) for sample in distortion_samples)
    labels = distortion_coeff_labels(model, max_len)
    ref = [float(v) for v in D_ref.ravel().tolist()]
    stats: list[DistortionCoeffStat] = []
    for i in range(max_len):
        values = [sample[i] for sample in distortion_samples if i < len(sample)]
        ref_i = ref[i] if i < len(ref) else None
        d = _distribution(values, reference=ref_i)
        stats.append(
            DistortionCoeffStat(
                index=i,
                label=labels[i],
                mean=d["mean"],
                std=d["std"],
                median=d["median"],
                min=d["min"],
                max=d["max"],
                ci_low=d["ci_low"],
                ci_high=d["ci_high"],
                stability_score=d["stability"],
                reference=ref_i,
            )
        )
    return stats


def compute_parameter_bootstrap(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    model: CameraModelType,
    K_ref: np.ndarray,
    D_ref: np.ndarray,
    flags: int,
    n_bootstrap: int = 20,
    rng_seed: int = 42,
    n_jobs: int = 1,
) -> ParameterUncertainty | None:
    """Standard 4모델 공용 bootstrap 불확실성 추정.

    flags: 이미 CALIB_USE_INTRINSIC_GUESS까지 포함해 완성된 최종 플래그 값을
    받는다 - 이 함수는 그 값을 그대로 cv2 호출에 넘기기만 한다. fisheye의
    CALIB_USE_INTRINSIC_GUESS는 OpenCV 빌드에 따라 없을 수 있어(모듈
    models/fisheye.py의 _fisheye_flag() 지연 조회 패턴 참고) 안전한 조회는
    호출부(각 모델 파일)의 책임으로 남긴다 - 여기서 cv2.fisheye.* 속성에
    직접 접근하면 그 안전장치가 무의미해진다.

    성공한 재표본이 너무 적으면(기본 5개 미만) None을 반환한다 - 호출부는
    이걸 "계산 안 됨"으로 표시해야지, 0으로 표시하면 안 된다.
    """
    n_frames = len(object_points)
    if n_frames == 0:
        return None

    rng = np.random.default_rng(rng_seed)
    is_fisheye = model == CameraModelType.FISHEYE
    distortion_count = expected_distortion_coeff_count(model)

    fx_samples: list[float] = []
    fy_samples: list[float] = []
    cx_samples: list[float] = []
    cy_samples: list[float] = []
    distortion_samples: list[list[float]] = []

    K_init = K_ref.copy().astype(np.float64)
    D_init = D_ref.copy().astype(np.float64)

    indices = [rng.integers(0, n_frames, size=n_frames) for _ in range(n_bootstrap)]
    tasks = [
        (object_points, image_points, image_size, idx, is_fisheye, K_init, D_init, flags, distortion_count)
        for idx in indices
    ]
    workers = resolve_worker_count(n_jobs, len(tasks))
    if workers > 1:
        logger.info("%s bootstrap 병렬 실행: %d samples, %d workers", model.value, n_bootstrap, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            sample_results = list(executor.map(_run_bootstrap_sample, tasks))
    else:
        sample_results = [_run_bootstrap_sample(task) for task in tasks]

    for sample in sample_results:
        if sample is None:
            continue
        fx, fy, cx, cy, distortion = sample
        fx_samples.append(fx)
        fy_samples.append(fy)
        cx_samples.append(cx)
        cy_samples.append(cy)
        distortion_samples.append(distortion)

    if len(fx_samples) < _MIN_SUCCESSFUL_SAMPLES:
        logger.warning(
            "%s bootstrap 재표본 %d/%d개만 성공해 불확실성 추정을 건너뜁니다.",
            model.value, len(fx_samples), n_bootstrap,
        )
        return None

    fx = _distribution(fx_samples, reference=float(K_ref[0, 0]))
    fy = _distribution(fy_samples, reference=float(K_ref[1, 1]))
    cx = _distribution(cx_samples, reference=float(K_ref[0, 2]))
    cy = _distribution(cy_samples, reference=float(K_ref[1, 2]))
    d_stats = _distortion_stats(distortion_samples, model, D_ref)
    stability_values = [
        value for value in (fx["stability"], fy["stability"], cx["stability"], cy["stability"])
        if value is not None
    ]
    stability_values.extend(
        stat.stability_score for stat in d_stats if stat.stability_score is not None
    )
    overall_stability = float(np.mean(stability_values)) if stability_values else None

    paper_fields = _paper_stability_fields(
        fx, fy, cx, cy, d_stats,
        fx_ref=float(K_ref[0, 0]), fy_ref=float(K_ref[1, 1]),
        cx_ref=float(K_ref[0, 2]), cy_ref=float(K_ref[1, 2]),
    )

    logger.info(
        "%s bootstrap 불확실성 추정 완료: %d/%d개 재표본 성공",
        model.value, len(fx_samples), n_bootstrap,
    )
    return ParameterUncertainty(
        fx_std=fx["std"],
        fy_std=fy["std"],
        cx_std=cx["std"],
        cy_std=cy["std"],
        method="bootstrap",
        n_bootstrap_success=len(fx_samples),
        n_bootstrap_total=n_bootstrap,
        fx_ci_low=fx["ci_low"], fx_ci_high=fx["ci_high"],
        fy_ci_low=fy["ci_low"], fy_ci_high=fy["ci_high"],
        cx_ci_low=cx["ci_low"], cx_ci_high=cx["ci_high"],
        cy_ci_low=cy["ci_low"], cy_ci_high=cy["ci_high"],
        fx_mean=fx["mean"], fy_mean=fy["mean"], cx_mean=cx["mean"], cy_mean=cy["mean"],
        fx_median=fx["median"], fy_median=fy["median"], cx_median=cx["median"], cy_median=cy["median"],
        fx_min=fx["min"], fx_max=fx["max"], fy_min=fy["min"], fy_max=fy["max"],
        cx_min=cx["min"], cx_max=cx["max"], cy_min=cy["min"], cy_max=cy["max"],
        fx_stability=fx["stability"], fy_stability=fy["stability"],
        cx_stability=cx["stability"], cy_stability=cy["stability"],
        overall_stability=overall_stability,
        distortion_stats=d_stats,
        **paper_fields,
    )


def add_normal_approximation_ci(uncertainty: ParameterUncertainty, camera_matrix: np.ndarray) -> ParameterUncertainty:
    """설계 문서 22번 - covariance 기반(method="covariance") 표준편차만 있는
    경우, 정규분포를 가정한 근사 95% CI(mean ± 1.96*std)를 채워 넣는다.

    Ideal Pinhole/Brown-Conrady/Rational은 cv2.calibrateCameraExtended가 stdDeviations를
    바로 주므로 bootstrap 없이도 std는 이미 있다 - 여기서는 그 std로부터
    "95% CI 표시"라는 문서 요구사항만 추가로 채운다. bootstrap 결과(percentile
    기반)에는 이 함수를 쓰지 않는다 - 이미 실측 분포에서 CI를 뽑았으므로
    정규근사를 덧씌우면 오히려 부정확해진다.
    """
    if uncertainty.method != "covariance":
        return uncertainty

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    z = 1.96  # 95% 양측 정규분포 임계값

    if uncertainty.fx_std is not None:
        uncertainty.fx_ci_low, uncertainty.fx_ci_high = fx - z * uncertainty.fx_std, fx + z * uncertainty.fx_std
    if uncertainty.fy_std is not None:
        uncertainty.fy_ci_low, uncertainty.fy_ci_high = fy - z * uncertainty.fy_std, fy + z * uncertainty.fy_std
    if uncertainty.cx_std is not None:
        uncertainty.cx_ci_low, uncertainty.cx_ci_high = cx - z * uncertainty.cx_std, cx + z * uncertainty.cx_std
    if uncertainty.cy_std is not None:
        uncertainty.cy_ci_low, uncertainty.cy_ci_high = cy - z * uncertainty.cy_std, cy + z * uncertainty.cy_std

    uncertainty.fx_mean = fx
    uncertainty.fy_mean = fy
    uncertainty.cx_mean = cx
    uncertainty.cy_mean = cy
    uncertainty.fx_stability = _stability_from_std(fx, uncertainty.fx_std, fx) if uncertainty.fx_std is not None else None
    uncertainty.fy_stability = _stability_from_std(fy, uncertainty.fy_std, fy) if uncertainty.fy_std is not None else None
    uncertainty.cx_stability = _stability_from_std(cx, uncertainty.cx_std, cx) if uncertainty.cx_std is not None else None
    uncertainty.cy_stability = _stability_from_std(cy, uncertainty.cy_std, cy) if uncertainty.cy_std is not None else None
    stability_values = [
        v for v in (
            uncertainty.fx_stability, uncertainty.fy_stability,
            uncertainty.cx_stability, uncertainty.cy_stability,
        )
        if v is not None
    ]
    uncertainty.overall_stability = float(np.mean(stability_values)) if stability_values else None
    # method="covariance"는 distortion coefficient 통계가 애초에 없으므로
    # (cv2.calibrateCameraExtended의 stdDeviationsIntrinsics만 여기서 다룬다)
    # overall_stability와 paper_intrinsic_stability가 이 경로에서는 항상
    # 같은 값이 된다 - 그래도 두 메서드(bootstrap/covariance) 모두에서
    # paper_intrinsic_stability가 존재하도록 명시적으로 채운다.
    uncertainty.paper_intrinsic_stability = uncertainty.overall_stability
    uncertainty.distortion_stability_summary = None
    uncertainty.fx_reference, uncertainty.fy_reference = fx, fy
    uncertainty.cx_reference, uncertainty.cy_reference = cx, cy
    uncertainty.fx_relative_cv = _pure_relative_cv(uncertainty.fx_mean, uncertainty.fx_std)
    uncertainty.fy_relative_cv = _pure_relative_cv(uncertainty.fy_mean, uncertainty.fy_std)
    uncertainty.cx_relative_cv = _pure_relative_cv(uncertainty.cx_mean, uncertainty.cx_std)
    uncertainty.cy_relative_cv = _pure_relative_cv(uncertainty.cy_mean, uncertainty.cy_std)
    candidates = [
        (label, value) for label, value in (
            ("fx", uncertainty.fx_stability), ("fy", uncertainty.fy_stability),
            ("cx", uncertainty.cx_stability), ("cy", uncertainty.cy_stability),
        )
        if value is not None
    ]
    if candidates:
        uncertainty.lowest_stability_parameter, uncertainty.lowest_stability_value = min(
            candidates, key=lambda item: item[1]
        )

    return uncertainty


def format_parameter_uncertainty(uncertainty: ParameterUncertainty | None) -> str:
    """설계 문서 22번 출력 형식.

        fx std = 2.100  (95% CI: 808.2 ~ 816.4)
        fy std = 2.400  (95% CI: 806.1 ~ 815.5)
        cx std = 1.200  (95% CI: 957.8 ~ 962.6)
        cy std = 1.500  (95% CI: 537.2 ~ 543.0)
    """
    if uncertainty is None:
        return "Parameter Uncertainty: 계산되지 않았습니다."

    def fmt_line(name: str, std: float | None, lo: float | None, hi: float | None) -> str:
        if std is None:
            return f"{name} = N/A"
        ci = f"  (95% CI: {lo:.1f} ~ {hi:.1f})" if lo is not None and hi is not None else ""
        return f"{name} std = {std:.3f}{ci}"

    # calibration/repeatability.py의 order-shuffle/initial-condition
    # perturbation("Solver Repeatability")과 절대 혼동되면 안 되므로,
    # provenance(어떤 실험에서 나온 숫자인지)를 문장으로 명시한다 -
    # method 필드 자체("bootstrap"/"covariance")는 바꾸지 않는다.
    provenance = {
        "bootstrap": "Bootstrap Parameter Stability - frame resampling with replacement",
        "covariance": "Covariance-based Parameter Uncertainty (cv2 stdDeviationsIntrinsics, normal approximation)",
    }.get(uncertainty.method, f"method={uncertainty.method}")
    lines = [f"Parameter Uncertainty ({provenance})"]
    lines.append(fmt_line("fx", uncertainty.fx_std, uncertainty.fx_ci_low, uncertainty.fx_ci_high))
    lines.append(fmt_line("fy", uncertainty.fy_std, uncertainty.fy_ci_low, uncertainty.fy_ci_high))
    lines.append(fmt_line("cx", uncertainty.cx_std, uncertainty.cx_ci_low, uncertainty.cx_ci_high))
    lines.append(fmt_line("cy", uncertainty.cy_std, uncertainty.cy_ci_low, uncertainty.cy_ci_high))
    if uncertainty.method == "bootstrap" and uncertainty.n_bootstrap_success is not None:
        lines.append(f"(성공한 재표본 {uncertainty.n_bootstrap_success}개 기준)")
    if uncertainty.method == "bootstrap":
        def fmt_dist(name: str, mean, median, lo, hi, stability) -> str:
            if mean is None:
                return f"{name}: N/A"
            ci = f", 95% CI {lo:.1f} ~ {hi:.1f}" if lo is not None and hi is not None else ""
            stable = f", stability {stability:.1f}/100" if stability is not None else ""
            med = f", median {median:.1f}" if median is not None else ""
            return f"{name}: mean {mean:.1f}{med}{ci}{stable}"

        lines.append("Bootstrap parameter distribution:")
        lines.append(fmt_dist("fx", uncertainty.fx_mean, uncertainty.fx_median, uncertainty.fx_ci_low, uncertainty.fx_ci_high, uncertainty.fx_stability))
        lines.append(fmt_dist("fy", uncertainty.fy_mean, uncertainty.fy_median, uncertainty.fy_ci_low, uncertainty.fy_ci_high, uncertainty.fy_stability))
        lines.append(fmt_dist("cx", uncertainty.cx_mean, uncertainty.cx_median, uncertainty.cx_ci_low, uncertainty.cx_ci_high, uncertainty.cx_stability))
        lines.append(fmt_dist("cy", uncertainty.cy_mean, uncertainty.cy_median, uncertainty.cy_ci_low, uncertainty.cy_ci_high, uncertainty.cy_stability))
    if uncertainty.overall_stability is not None:
        lines.append(f"All-Parameter Stability (legacy overall_stability) = {uncertainty.overall_stability:.1f}/100")
    if uncertainty.paper_intrinsic_stability is not None:
        lines.append(
            f"Paper Intrinsic Stability (fx/fy/cx/cy only, CV_p=sigma_p/mu_p) = "
            f"{uncertainty.paper_intrinsic_stability:.1f}/100"
        )
    if uncertainty.distortion_stability_summary is not None:
        lines.append(f"Distortion Stability Summary (diagnostic only) = {uncertainty.distortion_stability_summary:.1f}/100")
    if uncertainty.lowest_stability_parameter is not None:
        lines.append(
            f"Lowest-stability parameter: {uncertainty.lowest_stability_parameter} "
            f"({uncertainty.lowest_stability_value:.1f}/100)"
        )
    if uncertainty.distortion_stats:
        lines.append("Distortion coefficient stability:")
        for stat in uncertainty.distortion_stats:
            std = f"{stat.std:.4g}" if stat.std is not None else "N/A"
            median = f"{stat.median:.4g}" if stat.median is not None else "N/A"
            stability = f"{stat.stability_score:.1f}/100" if stat.stability_score is not None else "N/A"
            ci = (
                f", 95% CI {stat.ci_low:.4g} ~ {stat.ci_high:.4g}"
                if stat.ci_low is not None and stat.ci_high is not None else ""
            )
            ref = f", reference={stat.reference:.4g}" if stat.reference is not None else ""
            rel_cv = f", relative_cv={stat.relative_cv:.3g}" if stat.relative_cv is not None else ""
            warn = f"  [{stat.diagnostic}]" if stat.diagnostic else ""
            lines.append(
                f"  {stat.label or f'd{stat.index}'}: std={std} "
                f"median={median}{ref} stability={stability}{ci}{rel_cv}{warn}"
            )
    return "\n".join(lines)
