"""
camera_calibrator.calibration.bootstrap
===========================================

설계 문서 20/21/22번 - Bootstrap Stability / Parameter Confidence Interval을
Fisheye 전용이 아니라 세 모델 전부에서 쓸 수 있게 일반화한 모듈.

원래 이 로직은 calibration/models/fisheye.py의 _bootstrap_fisheye_uncertainty()
안에만 있었다 - fisheye는 cv2.fisheye.calibrate가 stdDeviations를 안 줘서
(Pinhole/Extended처럼 "공짜로" 못 얻어서) 어쩔 수 없이 bootstrap을 썼던 것.
하지만 "이 데이터셋으로 추정한 파라미터가 얼마나 안정적인가"는 Pinhole/Extended
에도 똑같이 유용한 질문이다 - covariance 기반 표준편차는 선형화된 근사치일
뿐이고, bootstrap은 실제 재표본화로 얻은 경험적 분포라 서로 다른 관점의
교차검증 역할을 한다. 그래서 이 함수를 모델 무관하게 만들어 세 모델 다
(선택적으로) bootstrap 불확실성을 계산할 수 있게 했다.

방법론(기존 fisheye 전용 버전과 동일, 정직하게 한계도 그대로 명시):
    전체 데이터로 얻은 K_ref/D_ref를 초기값 삼아 프레임을 복원추출(bootstrap)로
    재표본화해 n_bootstrap번 재캘리브레이션하고, fx/fy/cx/cy의 표준편차와
    95% CI(2.5/97.5 percentile)를 구한다.

    한계: 각 재표본이 전체 데이터의 K_ref를 초기값(CALIB_USE_INTRINSIC_GUESS)
    으로 재사용하므로, "전체 데이터 추정치 근방에서의 국소적 분산"을 재는
    셈이라 완전히 독립적인 붓스트랩보다 분산을 다소 과소평가할 수 있다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from calibration.types import CameraModelType, ParameterUncertainty

logger = logging.getLogger(__name__)

_MIN_SUCCESSFUL_SAMPLES = 5


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
) -> ParameterUncertainty | None:
    """세 모델 공용 bootstrap 불확실성 추정.

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

    fx_samples: list[float] = []
    fy_samples: list[float] = []
    cx_samples: list[float] = []
    cy_samples: list[float] = []

    K_init = K_ref.copy().astype(np.float64)
    D_init = D_ref.copy().astype(np.float64)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_frames, size=n_frames)
        obj_sample = [object_points[i] for i in idx]
        img_sample = [image_points[i] for i in idx]
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
            continue  # 이 재표본은 발산 - 조용히 건너뛴다

        fx_samples.append(float(K_i[0, 0]))
        fy_samples.append(float(K_i[1, 1]))
        cx_samples.append(float(K_i[0, 2]))
        cy_samples.append(float(K_i[1, 2]))

    if len(fx_samples) < _MIN_SUCCESSFUL_SAMPLES:
        logger.warning(
            "%s bootstrap 재표본 %d/%d개만 성공해 불확실성 추정을 건너뜁니다.",
            model.value, len(fx_samples), n_bootstrap,
        )
        return None

    def _ci(samples: list[float]) -> tuple[float, float]:
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    fx_lo, fx_hi = _ci(fx_samples)
    fy_lo, fy_hi = _ci(fy_samples)
    cx_lo, cx_hi = _ci(cx_samples)
    cy_lo, cy_hi = _ci(cy_samples)

    logger.info(
        "%s bootstrap 불확실성 추정 완료: %d/%d개 재표본 성공",
        model.value, len(fx_samples), n_bootstrap,
    )
    return ParameterUncertainty(
        fx_std=float(np.std(fx_samples, ddof=1)),
        fy_std=float(np.std(fy_samples, ddof=1)),
        cx_std=float(np.std(cx_samples, ddof=1)),
        cy_std=float(np.std(cy_samples, ddof=1)),
        method="bootstrap",
        n_bootstrap_success=len(fx_samples),
        fx_ci_low=fx_lo, fx_ci_high=fx_hi,
        fy_ci_low=fy_lo, fy_ci_high=fy_hi,
        cx_ci_low=cx_lo, cx_ci_high=cx_hi,
        cy_ci_low=cy_lo, cy_ci_high=cy_hi,
    )


def add_normal_approximation_ci(uncertainty: ParameterUncertainty, camera_matrix: np.ndarray) -> ParameterUncertainty:
    """설계 문서 22번 - covariance 기반(method="covariance") 표준편차만 있는
    경우, 정규분포를 가정한 근사 95% CI(mean ± 1.96*std)를 채워 넣는다.

    Pinhole/Extended Pinhole은 cv2.calibrateCameraExtended가 stdDeviations를
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

    lines = [f"Parameter Uncertainty (method={uncertainty.method})"]
    lines.append(fmt_line("fx", uncertainty.fx_std, uncertainty.fx_ci_low, uncertainty.fx_ci_high))
    lines.append(fmt_line("fy", uncertainty.fy_std, uncertainty.fy_ci_low, uncertainty.fy_ci_high))
    lines.append(fmt_line("cx", uncertainty.cx_std, uncertainty.cx_ci_low, uncertainty.cx_ci_high))
    lines.append(fmt_line("cy", uncertainty.cy_std, uncertainty.cy_ci_low, uncertainty.cy_ci_high))
    if uncertainty.method == "bootstrap" and uncertainty.n_bootstrap_success is not None:
        lines.append(f"(성공한 재표본 {uncertainty.n_bootstrap_success}개 기준)")
    return "\n".join(lines)
