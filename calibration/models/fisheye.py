"""
camera_calibrator.calibration.models.fisheye
================================================

설계 문서 1번, 2번, 17번 Step4 - OpenCV Fisheye (Kannala-Brandt), k1~k4.

cv2.fisheye 모듈은 cv2.calibrateCameraExtended()에 대응하는 Extended 버전이
없다. 즉 perViewErrors(프레임별 오차)와 stdDeviations(파라미터 불확실성)를
공짜로 주지 않는다. 그래서 이 모듈만 다음 세 가지를 직접 구현한다:

1. 초기값 안전장치 (설계 문서 2번):
   피쉬아이는 초기 fx, fy 추정이 나쁘면 최적화가 발산한다.
   Pinhole 캘리브레이션 결과의 fx, fy, cx, cy를 CALIB_USE_INTRINSIC_GUESS로
   넘겨서 시작점을 잡아준다.

       Pinhole calibration -> fx fy cx cy -> Fisheye initial K -> Fisheye calibration

2. 프레임별 재투영 오차 수동 계산:
   cv2.fisheye.projectPoints()를 프레임마다 호출해 pinhole.py가
   perViewErrors로 공짜로 받던 것을 직접 채운다.

3. 파라미터 불확실성(fx_std/fy_std/cx_std/cy_std) - Bootstrap Resampling:
   Pinhole/Extended Pinhole은 calibrateCameraExtended()가 stdDeviationsIntrinsics를
   공짜로 주지만 cv2.fisheye에는 그런 API가 없다. 대신 통계학의 표준 기법인
   bootstrap resampling으로 우회한다 - 자세한 설명은
   `_bootstrap_fisheye_uncertainty()` docstring 참고. `calibrate_fisheye()`의
   `estimate_uncertainty=True`(기본값 False - 비용이 있으므로 호출부가 명시적으로
   켜야 함)로 활성화한다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    ParameterUncertainty,
)
from calibration.models.common import (
    MIN_FRAMES_REQUIRED,
    collect_calibration_inputs,
    infer_image_size,
    compute_regional_error,
)
from calibration.radial_profile import compute_radial_error_profile

logger = logging.getLogger(__name__)

# Bootstrap 재표본화 시 이 장수를 넘는 데이터셋은 건너뛴다 (n_bootstrap번 전체
# 재캘리브레이션을 반복하는 구조라 프레임이 아주 많으면 비용이 선형으로 커진다 -
# 데이터셋이 클수록 애초에 각 파라미터 추정 자체가 이미 안정적이라 불확실성
# 추정의 한계효용도 낮다).
_BOOTSTRAP_MAX_FRAMES = 150
_BOOTSTRAP_MIN_SUCCESSFUL_SAMPLES = 5

# fisheye는 파라미터가 적어도(k1~k4) Brown-Conrady보다 화각 전역에서 비선형성이
# 강해, 뷰 개수가 부족하면 발산 위험이 pinhole/extended보다 크다.
# 그래도 "최소 조건"은 공통 모듈과 동일하게 두고, 실패 시 이유를 명확히 준다.
#
# 주의: cv2.fisheye.CALIB_* 플래그들은 함수 안에서 지연 평가(lazy)해야 한다.
# 예전엔 이 값을 모듈 최상단에서 바로 계산했는데, 그러면 OpenCV 빌드/버전에
# 따라 일부 플래그가 없을 경우(예: CALIB_RECOMPUTE_EXTRINSIC 누락) import 시점에
# AttributeError가 나서 fisheye를 아예 안 쓰는 사용자도 앱 전체가 못 뜨는
# 문제가 생긴다. getattr로 안전하게 조회하고, 없는 플래그는 조용히 건너뛴다
# (계산 자체는 그 플래그 없이도 대체로 동작하고, 완전히 안 되면 아래
# cv2.error 재시도 경로가 이미 있어 처리된다).
def _fisheye_flag(name: str, camera_calibrator_default: int = 0) -> int:
    """cv2.fisheye.CALIB_* 값을 찾는다.

    OpenCV 5.0부터 이 플래그들이 cv2.fisheye 네임스페이스에서 최상위
    cv2로 옮겨갔다 (예: cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC ->
    cv2.CALIB_RECOMPUTE_EXTRINSIC, 이름은 동일). 실제로 OpenCV 5.0.0
    환경에서 재현해서 확인한 사실이다 - 예전엔 getattr 실패시 그냥 0(플래그
    없음) 취급했는데, 그러면 OpenCV 5.x에서는 세 플래그가 전부 조용히
    사라져서(CALIB_FIX_SKEW 없이 skew까지 추정하려 들면서) 캘리브레이션
    자체가 다른 에러로 깨졌다. cv2.fisheye에 먼저 있는지 보고, 없으면
    최상위 cv2에서 다시 찾는다.
    """
    if hasattr(cv2.fisheye, name):
        return getattr(cv2.fisheye, name)
    if hasattr(cv2, name):
        return getattr(cv2, name)
    return camera_calibrator_default


def _fisheye_base_flags() -> int:
    flags = 0
    for name in ("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW", "CALIB_CHECK_COND"):
        flags |= _fisheye_flag(name)
    return flags


def _to_fisheye_points(
    object_points: list[np.ndarray], image_points: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """cv2.fisheye.calibrate()에 맞는 shape/dtype으로 변환.

    dtype: cv2.fisheye.* 함수군은 float64를 기대한다 (버전에 따라 float32는
    silent하게 잘못된 결과를 내거나 예외를 던질 수 있음).

    shape: collect_calibration_inputs()가 주는 배열은 (N,1,3)/(N,1,2)
    형태다 (cv2.calibrateCamera가 쓰는 pinhole 계열 관례) - 그런데
    cv2.fisheye.calibrate()는 이 shape을 안 받아들이고 (1,N,3)/(1,N,2)를
    요구한다. 실제로 OpenCV 5.0.0에서 (N,1,3)으로 넘기면
    "Sizes of input arguments do not match" 에러가 났고, (1,N,3)으로
    바꾸니 해결됐다 (OpenCV 4.13.0에서도 동일하게 잘 동작 - 하위 호환 확인됨).
    cv2.fisheye.projectPoints/undistortPoints 등 다른 함수들은 (N,1,3)도
    받아주므로, 이 reshape은 calibrate() 입력에만 적용한다.
    """
    obj64 = [o.reshape(1, -1, 3).astype(np.float64) for o in object_points]
    img64 = [i.reshape(1, -1, 2).astype(np.float64) for i in image_points]
    return obj64, img64


def _per_frame_errors_fisheye(
    frames,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> dict[str, float]:
    """cv2.fisheye에는 perViewErrors가 없으므로 projectPoints로 직접 계산.
    프레임별 RMS(px) = 검출된 코너와 재투영된 점 사이 거리의 RMS.
    """
    errors: dict[str, float] = {}
    for i, frame in enumerate(frames):
        projected, _ = cv2.fisheye.projectPoints(
            object_points[i], rvecs[i], tvecs[i], K, D
        )
        detected = image_points[i].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        diff = detected - projected
        rms = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        errors[frame.image_info.image_id] = rms
    return errors


def _bootstrap_fisheye_uncertainty(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    K_ref: np.ndarray,
    D_ref: np.ndarray,
    flags: int,
    n_bootstrap: int,
    rng_seed: int,
) -> ParameterUncertainty | None:
    """전체 데이터로 얻은 K_ref/D_ref를 초기값 삼아 프레임을 복원추출(bootstrap)로
    재표본화해 n_bootstrap번 재캘리브레이션하고, fx/fy/cx/cy의 표준편차를 구한다.

    cv2.fisheye는 stdDeviations를 안 준다(모듈 docstring 참고) - Pinhole/Extended처럼
    "공짜로" 얻을 수 없다. 대신 통계학의 표준 기법인 bootstrap resampling으로
    우회한다: 원래 N장짜리 데이터셋에서 매번 N장을 복원추출(같은 프레임이 여러 번
    뽑히거나 아예 안 뽑힐 수 있음)해서 별도의 데이터셋을 만들고, 그걸로 다시
    캘리브레이션한다. 이 과정을 여러 번 반복해서 나온 파라미터들이 흩어진 정도
    (표준편차)가 곧 "이 데이터셋으로 추정한 파라미터가 얼마나 안정적인가"의
    척도가 된다 - OpenCV가 못 주는 covariance를 흉내 내는 셈이다.

    한계(정직하게 명시): 각 재표본은 전체 데이터의 K_ref를 초기값
    (CALIB_USE_INTRINSIC_GUESS)으로 재사용한다. Fisheye는 초기값이 나쁘면
    발산하기 쉬운데(모듈 docstring 1번), 매번 처음부터 추정하게 두면 실패율이
    너무 높아 재표본 대부분을 못 쓰게 된다. 대신 이러면 "전체 데이터 추정치
    근방에서의 국소적 분산"을 재는 셈이라, 완전히 독립적인 (초기값 없이 매번
    새로 추정하는) 붓스트랩보다 분산을 다소 과소평가할 수 있다 - 그래도 "전혀
    모른다"보다는 훨씬 유용한 근사치이며, Pinhole/Extended의 표준편차와 나란히
    놓고 볼 상대적 비교 지표로는 충분하다.

    일부 재표본은 그래도 발산하거나 예외를 던질 수 있다 - 조용히 건너뛰고 성공한
    것만으로 표준편차를 계산한다. 성공한 재표본이 너무 적으면(기본 5개 미만)
    통계적으로 의미가 없으므로 None을 반환한다(호출부는 이걸 "계산 안 됨"으로
    표시해야지, 0으로 표시하면 안 된다).
    """
    n_frames = len(object_points)
    rng = np.random.default_rng(rng_seed)

    fx_samples: list[float] = []
    fy_samples: list[float] = []
    cx_samples: list[float] = []
    cy_samples: list[float] = []

    bootstrap_flags = flags | _fisheye_flag("CALIB_USE_INTRINSIC_GUESS")
    K_init = K_ref.copy().astype(np.float64)
    D_init = D_ref.copy().astype(np.float64)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_frames, size=n_frames)
        obj_sample = [object_points[i] for i in idx]
        img_sample = [image_points[i] for i in idx]
        try:
            _, K_i, D_i, _, _ = cv2.fisheye.calibrate(
                obj_sample, img_sample, image_size,
                K_init.copy(), D_init.copy(),
                flags=bootstrap_flags,
            )
        except cv2.error:
            continue  # 이 재표본은 발산 - 조용히 건너뛴다 (아래에서 성공 개수로 판단)
        fx_samples.append(float(K_i[0, 0]))
        fy_samples.append(float(K_i[1, 1]))
        cx_samples.append(float(K_i[0, 2]))
        cy_samples.append(float(K_i[1, 2]))

    if len(fx_samples) < _BOOTSTRAP_MIN_SUCCESSFUL_SAMPLES:
        logger.warning(
            "Fisheye bootstrap 재표본 %d/%d개만 성공해 불확실성 추정을 건너뜁니다.",
            len(fx_samples), n_bootstrap,
        )
        return None

    logger.info(
        "Fisheye bootstrap 불확실성 추정 완료: %d/%d개 재표본 성공",
        len(fx_samples), n_bootstrap,
    )
    return ParameterUncertainty(
        fx_std=float(np.std(fx_samples, ddof=1)),
        fy_std=float(np.std(fy_samples, ddof=1)),
        cx_std=float(np.std(cx_samples, ddof=1)),
        cy_std=float(np.std(cy_samples, ddof=1)),
    )


def calibrate_fisheye(
    dataset: Dataset,
    camera_config: CameraConfig,
    initial_guess: CalibrationResult | None = None,
    estimate_uncertainty: bool = False,
    n_bootstrap: int = 20,
    bootstrap_seed: int = 42,
) -> CalibrationResult:
    """Fisheye(Kannala-Brandt) 캘리브레이션 실행.

    Args:
        initial_guess: Pinhole 모델의 CalibrationResult를 넘기면
            fx, fy, cx, cy를 초기값으로 사용해 발산을 방지한다 (설계 문서 2번).
            없으면 OpenCV 기본 추정 로직(least-squares)에 맡긴다.
        estimate_uncertainty: True면 bootstrap resampling으로 fx_std/fy_std/
            cx_std/cy_std를 추정한다 (`_bootstrap_fisheye_uncertainty()` 참고).
            기본값 False인 이유: 이 추정은 전체 재캘리브레이션을 n_bootstrap번
            반복하는 구조라 비용이 있다 - validation.py의 hold-out 교차검증이나
            outlier.py의 반복 재계산처럼 fisheye 캘리브레이션 자체를 이미 여러 번
            돌리는 경로에서까지 기본으로 켜면 비용이 곱연산으로 불어난다. 최종
            사용자에게 보여줄 "1차 실행" 결과(compare.py의 run_all_models())에서만
            명시적으로 켠다.
        n_bootstrap: 재표본화 반복 횟수. 데이터셋이 150장을 넘으면(비용이 커서)
            자동으로 건너뛴다.
    """
    frames, object_points, image_points = collect_calibration_inputs(dataset)

    if len(frames) < MIN_FRAMES_REQUIRED:
        return CalibrationResult(
            model_name=CameraModelType.FISHEYE,
            success=False,
            error_message=(
                f"사용 가능한 프레임이 {len(frames)}장뿐입니다. "
                f"최소 {MIN_FRAMES_REQUIRED}장 이상 필요합니다."
            ),
        )

    image_size = infer_image_size(dataset, camera_config)
    object_points, image_points = _to_fisheye_points(object_points, image_points)

    flags = _fisheye_base_flags()
    K_init, D_init = None, None
    if initial_guess is not None and initial_guess.success and initial_guess.camera_matrix is not None:
        K_init = initial_guess.camera_matrix.copy().astype(np.float64)
        # CALIB_USE_INTRINSIC_GUESS는 K와 D 둘 다 비어있지 않아야 한다
        # (K만 채우고 D=None으로 두면 OpenCV가 assertion 에러를 던짐).
        # 왜곡은 0에서 출발해 최적화가 채워나가게 한다.
        D_init = np.zeros((4, 1), dtype=np.float64)
        flags |= _fisheye_flag("CALIB_USE_INTRINSIC_GUESS")

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K_init,
            D_init,
            flags=flags,
        )
    except cv2.error as e:
        # CALIB_CHECK_COND는 조건수가 나쁜(발산 위험) 프레임이 있으면 예외를 던진다.
        # 이 경우 조건 체크를 빼고 한 번 더 시도해, "완전 실패"보다는
        # "경고와 함께 결과라도 준다" 쪽을 택한다. 최종 채택 여부는 검증(hold-out)
        # 단계에서 사용자가 판단하게 한다.
        try:
            relaxed_flags = flags & ~_fisheye_flag("CALIB_CHECK_COND")
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K_init,
                D_init,
                flags=relaxed_flags,
            )
        except cv2.error as e2:
            return CalibrationResult(
                model_name=CameraModelType.FISHEYE,
                success=False,
                error_message=(
                    f"cv2.fisheye.calibrate 실패 (완화된 조건으로 재시도도 실패): {e2}\n"
                    f"(최초 시도 실패 원인: {e})"
                ),
            )

    per_frame_error = _per_frame_errors_fisheye(
        frames, object_points, image_points, rvecs, tvecs, K, D
    )
    for frame in frames:
        frame.reprojection_error = per_frame_error[frame.image_info.image_id]

    regional_error = compute_regional_error(frames, per_frame_error, image_size)
    radial_profile = compute_radial_error_profile(
        frames, list(rvecs), list(tvecs), K, D, image_size, CameraModelType.FISHEYE
    )

    param_uncertainty: ParameterUncertainty | None = None
    if estimate_uncertainty:
        if len(object_points) > _BOOTSTRAP_MAX_FRAMES:
            logger.info(
                "Fisheye bootstrap 불확실성 추정 건너뜀: 프레임 %d장 > 상한 %d장",
                len(object_points), _BOOTSTRAP_MAX_FRAMES,
            )
        else:
            param_uncertainty = _bootstrap_fisheye_uncertainty(
                object_points, image_points, image_size, K, D, flags,
                n_bootstrap=n_bootstrap, rng_seed=bootstrap_seed,
            )

    return CalibrationResult(
        model_name=CameraModelType.FISHEYE,
        camera_matrix=K,
        distortion=D,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=float(rms),
        per_frame_error=per_frame_error,
        regional_error=regional_error,
        radial_profile=radial_profile,
        param_uncertainty=param_uncertainty,
        success=True,
    )
