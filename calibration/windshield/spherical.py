"""
camera_calibrator.calibration.windshield.spherical
=======================================================

Phase 2 - Windshield를 구면(sphere)으로 근사하고 Snell의 법칙으로 굴절을
계산하는 모델.

    3D Point -> Windshield 안쪽 표면(구면) -> Snell 굴절(공기->유리)
             -> Windshield 바깥쪽 표면(같은 중심, radius+thickness인 동심구)
             -> Snell 굴절(유리->공기) -> Camera Ray -> 고정된 Base K,D -> Pixel

두 겹(thin shell) 모델을 쓴다 - 단일 표면만 굴절시키면 광선이 "유리 속을
목표점까지 계속 진행"하는 셈이 되어 실제로 존재하지 않는 큰 굴절을 영구히
남긴다. thickness는 WindshieldConfig.glass_thickness_m에서 고정값으로만
받고 최적화하지 않는다(n_air/n_glass와 동일한 취급).

Calibration은 2단계로 진행한다:

    STAGE A (ray-space initial fit)
        관측 픽셀을 Base K,D로 역투영한 광선과, 그 광선이 windshield sphere를
        통과해 굴절된 뒤의 방향이 실제 목표점을 가리키는지 정렬시킨다. 이
        단계는 프레임 pose를 Standard solvePnP 결과로 고정한 채 sphere
        (Cx,Cy,Cz,R) 4개만 추정하는 값싸고 안정적인 초기 fit이다.

    STAGE B (joint refinement - sphere + per-frame pose)
        Standard solvePnP는 windshield가 있는 상태에서 central-camera 가정으로
        구한 근사 pose라 굴절 오차 일부를 pose가 흡수했을 수 있다. STAGE B는
        이 pose를 "초기값일 뿐"으로 취급하고, sphere와 각 프레임의 pose를
        번갈아(alternating/block-coordinate) 재추정한다.

        중요한 구현 선택: 이 재추정의 residual도 STAGE A와 동일한 "ray-alignment"
        형태를 쓴다(문자 그대로 project_point()를 호출해 픽셀을 역산하는 형태가
        아니다). 이유는 실측으로 확인됨 - project_point()를 outer optimizer
        residual 안에 중첩 호출하면(코너마다 자체 root-solve) 8프레임/240코너
        규모에서도 sphere 4-파라미터 refinement에 24초가 걸리고, 게다가 안
        좋은 후보 sphere에서 project_point()의 root-solve가 실패하는 지점들이
        생겨 목적함수가 울퉁불퉁해져서 24초를 써도 제대로 수렴하지 않았다
        (cost가 0.00012 -> 85로 오히려 나빠짐). 반면 ray-alignment residual은
        "그 코너의 목표점이 관측 픽셀의 굴절 광선 위에 정확히 있는가"를 직접
        묻는 식으로, 이 값이 0이 되는 조건은 project_point(목표점) == 관측
        픽셀이 되는 조건과 수학적으로 완전히 동치다(같은 기하학적 조건을
        암묵적으로 표현한 것뿐, 근사가 아니다) - 그러면서도 코너마다 중첩
        최적화가 전혀 필요 없어 STAGE A와 똑같이 빠르고 매끄럽게 수렴한다.
        사용자에게 최종 보고되는 모든 지표(Hold-out RMS/P95/... 등)는 여전히
        _evaluate_spherical()의 진짜 project_point() 기반 픽셀 계산에서
        나온다 - 이 설계는 "최적화 목적함수의 내부 표현"만 바꾼 것이고,
        "무엇을 최소화하려는 것인가"(픽셀 재투영 오차)는 그대로다.

    STAGE B 결과가 STAGE A보다 실제 픽셀 RMS를 개선하지 못하면 STAGE A
    결과로 되돌아간다(성공/경고는 유지하되 그 사실을 warning_message에
    명시한다 - 조용히 무시하지 않는다).

Optimization 대상: sphere_center(x,y,z) + sphere_radius(안쪽 표면 반지름) +
train 프레임별 (rvec, tvec). Base Camera Model의 K,D는 이 모듈의 어떤
함수도 절대 재추정하지 않는다 - WindshieldConfig.base_camera_matrix/
base_distortion을 그대로 읽기만 한다.

좌표계는 다른 windshield 모듈과 동일: OpenCV 카메라 좌표계(+x 오른쪽, +y
아래쪽, +z 전방). sphere_center와 3D target point 모두 이 좌표계 기준이다.
현재 convention에서는 카메라(원점)가 안쪽 sphere "내부"에 있고(그래야
전방으로 나가는 광선이 안쪽 표면을 "탈출점"으로 만난다 - intersect_ray_sphere의
origin-inside-sphere 분기), 그 표면이 카메라 정면(+z 쪽)에 있어야 물리적으로
말이 된다 - is_valid_spherical_windshield()가 이 두 조건을 검사한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibration.models.common import MIN_FRAMES_REQUIRED, compute_regional_error, infer_image_size
from calibration.radial_profile import bin_radial_error_bands, bin_radial_errors
from calibration.residual_stats import compute_residual_stats
from calibration.spatial_error_map import bin_spatial_errors
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    RadialErrorProfile,
    RegionalError,
    ResidualStats,
    SpatialErrorMap,
)
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.refraction import intersect_ray_sphere, normalize, refract_ray

# ---------------------------------------------------------------------------
# 고정 상수 - 전부 설정으로 덮어쓸 수 있고, 여기 값은 "적당한 기본값"이지
# 실차 windshield 스펙을 하드코딩한 것이 아니다.
# ---------------------------------------------------------------------------
DEFAULT_AIR_REFRACTIVE_INDEX = 1.0
DEFAULT_GLASS_REFRACTIVE_INDEX = 1.52   # 일반적인 laminated 자동차 유리 근사치
DEFAULT_GLASS_THICKNESS_M = 0.005       # ~5mm, 일반적인 windshield 두께 근사치
DEFAULT_INITIAL_RADIUS_M = 5.0          # 일반적인 대곡률 근사치(특정 차종 아님)
DEFAULT_INITIAL_STANDOFF_M = 1.0        # 카메라-windshield 대략적 거리 근사치

MIN_CORNERS_FOR_FIT = 20
MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10

# 물리적 유효성 검사(섹션 10/11/12) - is_valid_spherical_windshield 참고.
MIN_SPHERE_MARGIN_M = 0.01   # 카메라가 안쪽 표면으로부터 최소 이만큼은 안쪽에 있어야 함
MIN_STANDOFF_M = 0.01        # windshield가 카메라 정면(+z)으로 최소 이 거리 이상 있어야 함

_SPHERE_BOUNDS = (
    [-100.0, -100.0, -100.0, 0.05],
    [100.0, 100.0, 100.0, 100.0],
)
_RAY_PENALTY = 5.0  # ray-alignment residual 실패(교차 없음/전반사/비물리적 sphere) 시 채우는 값
_ORIGIN = np.zeros(3, dtype=np.float64)

# STAGE B - pose가 initial solvePnP에서 너무 멀리 벗어나지 않도록 하는 약한
# prior의 가중치. residual 단위가 ray-alignment 단위(대략 라디안 스케일)이므로,
# 이 가중치는 "정확한 pixel 환산"이 아니라 "그쪽으로 크게 끌리지 않게 하는"
# 정도의 weak prior 세기다 - 필요하면 쉽게 조정할 수 있도록 상수로 뺐다.
POSE_ROTATION_REG_WEIGHT = 2.0    # per radian
POSE_TRANSLATION_REG_WEIGHT = 2.0  # per meter

STAGE_B_NUM_ROUNDS = 2

# project_point() 내부 root-solve가 수렴한 뒤 "실제로 광선이 목표점과
# 충분히 정렬됐는가"를 판정하는 각도 임계값(도 단위). 예전에는 단순
# cost < 1.0 같은 단위 없는 임계값을 썼는데, residual이 두 단위벡터의 차이라는
# 걸 이용해 실제 각도(도)로 환산한 뒤 판정한다 - 물리적 의미가 명확하다.
MAX_PROJECT_POINT_ANGULAR_ERROR_DEG = 2.0


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


def is_valid_spherical_windshield(center, radius: float, thickness: float) -> bool:
    """이 (center, radius, thickness) 조합이 현재 모듈의 sphere convention
    에서 물리적으로 말이 되는지 검사한다.

    현재 convention(다른 함수들이 이미 그렇게 만들어져 있음 - 예:
    _initial_sphere_guess가 center_z = standoff - radius로 두는 것):
      1. 카메라(원점)는 안쪽 표면의 "내부"에 있어야 한다
         (||center|| < radius - margin) - 그래야 카메라에서 전방으로 나가는
         모든 광선이 intersect_ray_sphere의 "origin-inside" 분기(탈출점,
         t2)로 안쪽 표면과 만난다.
      2. 카메라가 바라보는 방향의 가장 가까운 표면 점(카메라에서 구 중심을
         지나 반대편 표면까지의 방향에 있는 점)의 z가 카메라 앞(+z, 최소
         MIN_STANDOFF_M 이상)에 있어야 한다 - windshield가 카메라 뒤에 있는
         해는 무효.
      3. radius/thickness는 유한하고 양수여야 한다.
    """
    center = np.asarray(center, dtype=np.float64)
    if not np.all(np.isfinite(center)) or not np.isfinite(radius) or not np.isfinite(thickness):
        return False
    if radius <= 0.0 or thickness <= 0.0:
        return False

    dist_to_center = float(np.linalg.norm(center))
    if dist_to_center < 1e-9:
        return False  # 카메라가 정확히 구 중심 - 전방 방향을 정의할 수 없음
    if dist_to_center >= radius - MIN_SPHERE_MARGIN_M:
        return False  # 카메라가 안쪽 표면 내부에 충분히 있지 않음

    direction_to_camera = -center / dist_to_center
    near_surface_point = center + radius * direction_to_camera
    if near_surface_point[2] < MIN_STANDOFF_M:
        return False  # windshield가 카메라 앞에 충분히 있지 않음(뒤쪽/너무 가까움)

    return True


def _refract_through_shell(
    d_cam: np.ndarray,
    origin: np.ndarray,
    center: np.ndarray,
    radius: float,
    thickness: float,
    n_air: float,
    n_glass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """origin에서 d_cam 방향으로 나가는 광선이 동심구 두 겹(안쪽 radius,
    바깥쪽 radius+thickness)을 공기->유리->공기 순서로 통과한 뒤의
    (최종 exit point, 최종 방향)을 반환한다.

    실패(교차 없음/전반사)하면 ValueError를 던진다 - 조용히 잘못된 값을
    반환하지 않는다. 물리적 유효성(is_valid_spherical_windshield)은 호출부
    책임이다 - 이 함수 자체는 순수하게 "주어진 sphere로 광선을 통과시키는"
    계산만 한다.
    """
    inner_hit = intersect_ray_sphere(origin, d_cam, center, radius)
    if inner_hit is None:
        raise ValueError("Ray does not intersect the windshield's inner surface.")
    p1, _ = inner_hit

    n1 = normalize(p1 - center)
    d_glass = refract_ray(d_cam, n1, n_air, n_glass)
    if d_glass is None:
        raise ValueError("Total internal reflection at the windshield's inner surface.")

    outer_radius = radius + thickness
    outer_hit = intersect_ray_sphere(p1, d_glass, center, outer_radius)
    if outer_hit is None:
        raise ValueError("Ray does not intersect the windshield's outer surface.")
    p2, _ = outer_hit

    n2 = normalize(p2 - center)
    d_out = refract_ray(d_glass, n2, n_glass, n_air)
    if d_out is None:
        raise ValueError("Total internal reflection at the windshield's outer surface.")

    return p2, d_out


def _ray_alignment_residual(
    d_obs: np.ndarray,
    p_cam: np.ndarray,
    center: np.ndarray,
    radius: float,
    thickness: float,
    n_air: float,
    n_glass: float,
) -> np.ndarray:
    """한 코너의 (관측 방향, 목표점)에 대한 ray-alignment residual(3-vector).

    이 값이 [0,0,0]이면 "목표점이 관측 픽셀의 굴절 광선 위에 정확히 있다"는
    뜻이고, 이는 project_point(목표점)이 정확히 관측 픽셀을 돌려준다는 것과
    수학적으로 동치다(모듈 docstring 참고) - STAGE A의 초기 sphere fit과
    STAGE B의 sphere/pose 공동 refinement가 전부 이 함수 하나를 재사용한다.

    실패(sphere가 물리적으로 무효/교차 없음/전반사)하면 예외를 던지지 않고
    페널티 벡터를 반환한다 - optimizer 루프에서 반복 호출되므로.
    """
    if not is_valid_spherical_windshield(center, radius, thickness):
        return np.full(3, _RAY_PENALTY)
    try:
        point, direction = _refract_through_shell(d_obs, _ORIGIN, center, radius, thickness, n_air, n_glass)
    except ValueError:
        return np.full(3, _RAY_PENALTY)
    to_target = p_cam - point
    norm = np.linalg.norm(to_target)
    if norm < 1e-9:
        return np.zeros(3)
    return to_target / norm - direction


def _is_penalized(residual_row: np.ndarray) -> bool:
    return bool(np.all(np.isclose(residual_row, _RAY_PENALTY)))


class SphericalWindshieldModel(WindshieldModel):
    """Windshield를 얇은 동심구 셸로 근사하고 Snell 굴절을 실제로 계산하는
    모델. project_point()/unproject_pixel() 모두 이 굴절 geometry를 거친다 -
    BaselineWindshieldModel.project_point()를 그대로 호출하는 식의 항등
    구현이 아니다.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        sphere_center: np.ndarray,
        sphere_radius: float,
        n_air: float = DEFAULT_AIR_REFRACTIVE_INDEX,
        n_glass: float = DEFAULT_GLASS_REFRACTIVE_INDEX,
        glass_thickness_m: float = DEFAULT_GLASS_THICKNESS_M,
    ):
        self._center = np.asarray(sphere_center, dtype=np.float64)
        self._radius = float(sphere_radius)
        self._n_air = float(n_air)
        self._n_glass = float(n_glass)
        self._thickness = float(glass_thickness_m)
        # Base K,D 전용 두 primitive(픽셀<->카메라 광선, 항등 투영)만 재사용한다 -
        # BaselineWindshieldModel 자체는 절대 K,D를 수정하지 않으므로 안전하게
        # 내부 헬퍼로 감싸 쓸 수 있다.
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def _refract_camera_ray(self, d_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _refract_through_shell(
            d_cam, _ORIGIN, self._center, self._radius, self._thickness, self._n_air, self._n_glass
        )

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        """픽셀 -> 굴절을 반영한 외부 광선 방향(단위 벡터).

        근사: 실제로는 광선의 원점이 windshield 바깥 표면(exit point)으로
        옮겨가지만, WindshieldModel ABC의 반환 형태가 방향뿐이라 원점은
        카메라 중심에 있다고 가정한 채 방향만 보고한다. project_point()는
        이 근사를 쓰지 않고 실제 exit point를 명시적으로 사용한다 - 그쪽이
        정확도가 중요한 forward projection이기 때문이다.
        """
        d_cam = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        _, d_out = self._refract_camera_ray(d_cam)
        return float(d_out[0]), float(d_out[1]), float(d_out[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다 - Base K,D 투영을
        초기값으로 삼아 작은 2변수 root-solve로 푼다(전체 이미지를 뒤지는
        brute-force가 아니다).
        """
        target = np.array([x, y, z], dtype=np.float64)
        initial_uv = np.asarray(self._baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            d_cam = np.asarray(self._baseline.unproject_pixel(float(uv[0]), float(uv[1])), dtype=np.float64)
            try:
                point, direction = self._refract_camera_ray(d_cam)
            except ValueError:
                return np.full(3, _RAY_PENALTY)
            to_target = target - point
            norm = np.linalg.norm(to_target)
            if norm < 1e-9:
                return np.zeros(3)
            return to_target / norm - direction

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        if not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.cost):
            raise ValueError("project_point(): local root-solve did not converge to a finite result.")

        # |a-b|는 두 단위벡터 사이 각도 theta에 대해 2*sin(theta/2)이다 -
        # 최종 residual norm을 실제 각도(도)로 환산해서, 단위 없는 cost
        # 대신 물리적으로 의미 있는 기준으로 성공/실패를 가른다.
        residual_norm = float(np.linalg.norm(result.fun))
        angle_rad = 2.0 * math.asin(min(1.0, residual_norm / 2.0))
        if math.degrees(angle_rad) > MAX_PROJECT_POINT_ANGULAR_ERROR_DEG:
            raise ValueError(
                "Could not find a valid refracted projection for this point "
                "(likely outside the windshield sphere's coverage)."
            )
        return float(result.x[0]), float(result.x[1])

    def ray_angular_error_deg(self, u: float, v: float, target_point_cam: np.ndarray) -> Optional[float]:
        """관측 픽셀(u,v)의 굴절 광선과, 카메라 좌표계의 실제 목표점 방향
        사이의 각도(도) - 학습(fitting)에 쓰인 것과 동일한 ray-alignment
        잔차를 각도로 표현한 것. 교차/굴절이 불가능하면 None(값을 억지로
        만들지 않는다)."""
        d_cam = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        try:
            point, direction = self._refract_camera_ray(d_cam)
        except ValueError:
            return None
        to_target = np.asarray(target_point_cam, dtype=np.float64) - point
        norm = np.linalg.norm(to_target)
        if norm < 1e-9:
            return 0.0
        cos_angle = float(np.clip(np.dot(to_target / norm, direction), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# STAGE A - Ray-space initial sphere fit (pose 고정)
# ---------------------------------------------------------------------------

def _initial_sphere_guess(config: WindshieldConfig, median_depth: float) -> tuple[np.ndarray, float]:
    """windshield_position_hint의 각 키를 개별적으로 참고하고, 없는 키는
    데이터 기반/일반적인 기본값으로 채운다. hint 키 이름은 fitted_params가
    저장하는 키 이름과 동일하게 맞췄다(대칭성 - 한 번 fit한 결과를 다음
    실행의 hint로 그대로 재사용할 수 있게).
    """
    hint = config.windshield_position_hint or {}
    radius = float(hint.get("sphere_radius", DEFAULT_INITIAL_RADIUS_M))
    standoff = float(hint.get("standoff_m", min(DEFAULT_INITIAL_STANDOFF_M, max(median_depth * 0.3, 0.3))))
    cx = float(hint.get("sphere_center_x", 0.0))
    cy = float(hint.get("sphere_center_y", 0.0))
    cz = float(hint.get("sphere_center_z", standoff - radius))
    return np.array([cx, cy, cz]), radius


def _fit_sphere(
    d_obs: np.ndarray,
    p_cam: np.ndarray,
    n_air: float,
    n_glass: float,
    thickness: float,
    initial_center: np.ndarray,
    initial_radius: float,
):
    """Sphere(Cx,Cy,Cz,R) 4개만 ray-alignment residual로 피팅한다.
    (d_obs, p_cam)은 호출부가 고정한 pose로부터 이미 계산해 넘긴다 -
    이 함수는 pose를 전혀 모른다(STAGE A에서는 최초 solvePnP pose,
    STAGE B의 sphere sub-step에서는 그 라운드의 최신 refined pose)."""

    def residual(params: np.ndarray) -> np.ndarray:
        center, radius = params[:3], params[3]
        out = np.empty((len(d_obs), 3))
        for i in range(len(d_obs)):
            out[i] = _ray_alignment_residual(d_obs[i], p_cam[i], center, radius, thickness, n_air, n_glass)
        return out.ravel()

    x0 = np.array([initial_center[0], initial_center[1], initial_center[2], initial_radius])
    # windshield_position_hint(사용자 입력)가 _SPHERE_BOUNDS 밖일 수 있다 -
    # scipy는 초기값이 bounds를 벗어나면 예외를 던지므로(x0 is infeasible),
    # 여기서 미리 clip해서 "이상한 hint를 줘도 crash하지 않는다"는 원칙을 지킨다.
    lower, upper = np.asarray(_SPHERE_BOUNDS[0]), np.asarray(_SPHERE_BOUNDS[1])
    x0 = np.clip(x0, lower, upper)
    return least_squares(
        residual, x0=x0, bounds=_SPHERE_BOUNDS, method="trf", loss="soft_l1", f_scale=0.05
    )


def _sphere_fit_valid_ray_fraction(fit_result) -> float:
    """fit_result.fun(최종 파라미터에서의 residual)을 다시 훑어서, 페널티가
    아닌(=실제로 교차/굴절에 성공한) 코너의 비율을 계산한다."""
    rows = fit_result.fun.reshape(-1, 3)
    if rows.shape[0] == 0:
        return 0.0
    num_penalized = sum(1 for row in rows if _is_penalized(row))
    return 1.0 - num_penalized / rows.shape[0]


def _sphere_fit_is_acceptable(fit_result, thickness: float) -> bool:
    """STAGE A/B의 sphere sub-fit이 "성공"이라고 부를 수 있는 최소 조건 -
    optimizer 자체의 성공 플래그, 값의 유한성, 물리적 유효성, 유효 광선
    비율을 모두 검사한다(섹션 16)."""
    if not fit_result.success:
        return False
    if not np.all(np.isfinite(fit_result.x)) or not np.isfinite(fit_result.cost):
        return False
    center, radius = fit_result.x[:3], float(fit_result.x[3])
    if not is_valid_spherical_windshield(center, radius, thickness):
        return False
    if _sphere_fit_valid_ray_fraction(fit_result) < (1.0 - MAX_ACCEPTABLE_CORNER_FAILURE_RATE):
        return False
    return True


# ---------------------------------------------------------------------------
# STAGE B - Sphere + per-frame pose joint refinement (ray-domain, alternating)
# ---------------------------------------------------------------------------

def refine_frame_pose_ray_domain(
    frame: Frame,
    d_obs_per_corner: np.ndarray,
    center: np.ndarray,
    radius: float,
    thickness: float,
    n_air: float,
    n_glass: float,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    *,
    regularize: bool = True,
):
    """한 프레임의 pose(rvec,tvec)만 ray-alignment residual로 refine한다.
    Sphere/K,D는 고정(호출부가 넘긴 값 그대로) - 이 함수 자체는 절대
    수정하지 않는다.

    Train 프레임의 STAGE B alternating refinement와 Test 프레임의 pose-only
    hold-out refinement(섹션 19) 양쪽이 이 함수 하나를 재사용한다 - "pose만
    ray-alignment로 refine"하는 로직이 두 곳에 따로 있으면 하나가 바뀔 때
    다른 쪽을 깜빡 놓치기 쉽다.

    initial_rvec/initial_tvec은 항상 "이 프레임의 원래 Standard solvePnP
    추정값"이다(라운드가 반복돼도 계속 같은 기준점) - 매 라운드의 이전 결과가
    아니라 고정된 최초 추정값을 향한 weak prior여야 "너무 멀리 도망가지
    마라"는 의미가 유지된다(섹션 9).
    """
    det = frame.detection
    obj = det.object_points.reshape(-1, 3).astype(np.float64)
    initial_rvec = np.asarray(initial_rvec, dtype=np.float64).ravel()
    initial_tvec = np.asarray(initial_tvec, dtype=np.float64).ravel()

    def residual(params: np.ndarray) -> np.ndarray:
        rvec, tvec = params[:3], params[3:6]
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        out = np.empty((len(cam_pts), 3))
        for i in range(len(cam_pts)):
            out[i] = _ray_alignment_residual(d_obs_per_corner[i], cam_pts[i], center, radius, thickness, n_air, n_glass)
        flat = out.ravel()
        if regularize:
            reg = np.concatenate([
                POSE_ROTATION_REG_WEIGHT * (rvec - initial_rvec),
                POSE_TRANSLATION_REG_WEIGHT * (tvec - initial_tvec),
            ])
            flat = np.concatenate([flat, reg])
        return flat

    x0 = np.concatenate([initial_rvec, initial_tvec])
    return least_squares(residual, x0=x0, method="trf", loss="soft_l1", f_scale=0.05, max_nfev=100)


@dataclass
class _JointRefinementOutcome:
    center: np.ndarray
    radius: float
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    converged_cleanly: bool  # 매 라운드의 sphere/pose sub-fit이 전부 acceptable했는지


def _joint_refine_sphere_and_poses(
    ok_frames: list[Frame],
    d_obs_per_frame: list[np.ndarray],
    initial_rvecs: list[np.ndarray],
    initial_tvecs: list[np.ndarray],
    n_air: float,
    n_glass: float,
    thickness: float,
    initial_center: np.ndarray,
    initial_radius: float,
    num_rounds: int = STAGE_B_NUM_ROUNDS,
) -> _JointRefinementOutcome:
    """STAGE B - alternating(block-coordinate) 방식으로 sphere와 프레임별
    pose를 번갈아 refine한다. 한 라운드 = (모든 프레임 pose refine) ->
    (sphere refine). 각 sub-fit이 실패/비물리적이면 그 sub-step만 이전 값을
    유지하고(crash하지 않음) 다음 라운드로 넘어간다."""
    rvecs = [np.asarray(r, dtype=np.float64).copy() for r in initial_rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).copy() for t in initial_tvecs]
    center = np.asarray(initial_center, dtype=np.float64).copy()
    radius = float(initial_radius)
    converged_cleanly = True

    for _ in range(num_rounds):
        for i, frame in enumerate(ok_frames):
            pose_fit = refine_frame_pose_ray_domain(
                frame, d_obs_per_frame[i], center, radius, thickness, n_air, n_glass,
                initial_rvecs[i], initial_tvecs[i],
            )
            if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                rvecs[i] = pose_fit.x[:3].reshape(3, 1)
                tvecs[i] = pose_fit.x[3:6].reshape(3, 1)
            else:
                converged_cleanly = False  # 이 프레임은 이전 pose 유지

        p_cam_list = []
        d_obs_list = []
        for i, frame in enumerate(ok_frames):
            R, _ = cv2.Rodrigues(rvecs[i])
            obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
            cam_pts = (R @ obj.T).T + tvecs[i].reshape(1, 3)
            p_cam_list.append(cam_pts)
            d_obs_list.append(d_obs_per_frame[i])
        p_cam_arr = np.concatenate(p_cam_list, axis=0)
        d_obs_arr = np.concatenate(d_obs_list, axis=0)

        sphere_fit = _fit_sphere(d_obs_arr, p_cam_arr, n_air, n_glass, thickness, center, radius)
        if _sphere_fit_is_acceptable(sphere_fit, thickness):
            center, radius = sphere_fit.x[:3].copy(), float(sphere_fit.x[3])
        else:
            converged_cleanly = False  # sphere는 이전 라운드 값 유지

    return _JointRefinementOutcome(
        center=center, radius=radius, rvecs=rvecs, tvecs=tvecs, converged_cleanly=converged_cleanly
    )


# ---------------------------------------------------------------------------
# Evaluation (진짜 pixel 기반 - project_point() 사용)
# ---------------------------------------------------------------------------

@dataclass
class _SphericalEvalOutcome:
    per_frame_error: dict[str, float]
    residual_stats: ResidualStats
    regional_error: RegionalError
    radial_profile: RadialErrorProfile
    radial_bands: RadialErrorProfile
    spatial_error_map: SpatialErrorMap
    mean_dx: Optional[float]
    mean_dy: Optional[float]
    ray_angular_error_deg: Optional[float]
    num_points_ok: int
    num_points_failed: int


def _evaluate_spherical(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    model: SphericalWindshieldModel,
    image_size: tuple[int, int],
) -> _SphericalEvalOutcome:
    """Spherical 모델로 프레임들을 평가한다 - cv2.projectPoints 기반 central
    투영(calibration.radial_profile.collect_per_point_vectors 등)을 쓰지
    않는다. 대신 SphericalWindshieldModel.project_point()로 직접 예측
    픽셀을 구하고(진짜 pixel 도메인 평가), 결과(x,y,dx,dy) 배열을 기존의
    투영-무관 집계 함수(compute_residual_stats/compute_regional_error/
    bin_radial_errors/bin_radial_error_bands/bin_spatial_errors)에 그대로
    넘긴다. STAGE A/B의 최적화가 ray-domain residual을 쓰더라도, 최종
    보고되는 모든 지표는 항상 이 함수를 거친 진짜 픽셀 값이다.
    """
    per_frame_error: dict[str, float] = {}
    all_x: list[float] = []
    all_y: list[float] = []
    all_dx: list[float] = []
    all_dy: list[float] = []
    angles: list[float] = []
    num_failed = 0

    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        det = frame.detection
        R, _ = cv2.Rodrigues(rvec)
        obj = det.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = det.corners.reshape(-1, 2)

        per_point_errors: list[float] = []
        for (ox, oy), p_cam in zip(corners, cam_pts):
            ox, oy = float(ox), float(oy)
            try:
                pu, pv = model.project_point(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]))
            except ValueError:
                num_failed += 1
                continue
            dx, dy = ox - pu, oy - pv
            all_x.append(ox)
            all_y.append(oy)
            all_dx.append(dx)
            all_dy.append(dy)
            per_point_errors.append(math.hypot(dx, dy))

            angle = model.ray_angular_error_deg(ox, oy, p_cam)
            if angle is not None:
                angles.append(angle)

        if per_point_errors:
            per_frame_error[frame.image_info.image_id] = float(
                np.sqrt(np.mean(np.square(per_point_errors)))
            )

    xs, ys = np.array(all_x), np.array(all_y)
    dxs, dys = np.array(all_dx), np.array(all_dy)
    errors = np.hypot(dxs, dys)

    residual_stats = compute_residual_stats(errors)

    w, h = image_size
    max_radius = float(math.hypot(w / 2.0, h / 2.0))
    radii = np.hypot(xs - w / 2.0, ys - h / 2.0) if xs.size else np.array([])
    radial_profile = bin_radial_errors(radii, errors, max_radius, num_bins=8)
    radial_bands = bin_radial_error_bands(radii, errors, max_radius)
    spatial_map = bin_spatial_errors(xs, ys, dxs, dys, image_size)

    frames_with_error = [f for f in frames if f.image_info.image_id in per_frame_error]
    regional_error = compute_regional_error(frames_with_error, per_frame_error, image_size)

    return _SphericalEvalOutcome(
        per_frame_error=per_frame_error,
        residual_stats=residual_stats,
        regional_error=regional_error,
        radial_profile=radial_profile,
        radial_bands=radial_bands,
        spatial_error_map=spatial_map,
        mean_dx=float(dxs.mean()) if dxs.size else None,
        mean_dy=float(dys.mean()) if dys.size else None,
        ray_angular_error_deg=float(np.mean(angles)) if angles else None,
        num_points_ok=len(all_x),
        num_points_failed=num_failed,
    )


def _failure_result(config: WindshieldConfig, train_ids: list[str], test_ids: list[str], message: str) -> WindshieldCalibrationResult:
    return WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPHERICAL,
        base_model_name=config.base_model_name,
        base_camera_matrix=config.base_camera_matrix,
        base_distortion=config.base_distortion,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        success=False,
        error_message=message,
    )


def _outcome_to_result_fields(outcome: _SphericalEvalOutcome) -> dict:
    return dict(
        per_frame_error=outcome.per_frame_error,
        residual_stats=outcome.residual_stats,
        regional_error=outcome.regional_error,
        radial_profile=outcome.radial_profile,
        radial_bands=outcome.radial_bands,
        spatial_error_map=outcome.spatial_error_map,
        mean_dx=outcome.mean_dx,
        mean_dy=outcome.mean_dy,
        ray_angular_error_deg=outcome.ray_angular_error_deg,
    )


def calibrate_spherical(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Spherical windshield 모델을 fitting한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않는다 - 이 함수 안
    어디에서도 K,D를 쓰는 곳은 읽기(BaselineWindshieldModel에 전달)뿐이다.

    흐름: STAGE A(ray-space, pose 고정) -> STAGE B(sphere+pose joint,
    ray-domain alternating refinement) -> 두 stage의 실제 pixel RMS를
    비교해 더 나은 쪽을 최종으로 채택 -> Train 평가 -> Test는 최종 sphere를
    완전히 고정한 채 자기 pose만 별도로 refine한 뒤 평가(leakage 없음).
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)

    n_air = DEFAULT_AIR_REFRACTIVE_INDEX
    n_glass = config.glass_refractive_index if config.glass_refractive_index is not None else DEFAULT_GLASS_REFRACTIVE_INDEX
    thickness = config.glass_thickness_m if config.glass_thickness_m is not None else DEFAULT_GLASS_THICKNESS_M

    train_frames = _subset_frames(windshield_dataset, train_ids)
    if len(train_frames) < MIN_FRAMES_REQUIRED:
        return _failure_result(
            config, train_ids, test_ids,
            f"Train 프레임이 {len(train_frames)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요).",
        )

    ok_frames, rvecs, tvecs, failed_ids = solve_poses_fixed_intrinsics(train_frames, K, D, model)
    if not ok_frames:
        return _failure_result(config, train_ids, test_ids, "Train 프레임에서 pose를 하나도 구하지 못했습니다.")

    baseline_model = BaselineWindshieldModel(K, D, model)

    # 코너별 관측 방향(d_obs)은 pose/sphere 어느 쪽에도 의존하지 않는다
    # (base K,D + 관측 픽셀만의 함수) - 한 번만 계산해서 STAGE A/B 내내 재사용.
    d_obs_per_frame: list[np.ndarray] = []
    p_cam_per_frame: list[np.ndarray] = []
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        det = frame.detection
        R, _ = cv2.Rodrigues(rvec)
        obj = det.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = det.corners.reshape(-1, 2)
        d_obs_frame = np.array([baseline_model.unproject_pixel(float(px), float(py)) for px, py in corners])
        d_obs_per_frame.append(d_obs_frame)
        p_cam_per_frame.append(cam_pts)

    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    if len(d_obs_arr) < MIN_CORNERS_FOR_FIT:
        return _failure_result(config, train_ids, test_ids, "Sphere를 추정하기에 코너 수가 부족합니다.")

    # --- STAGE A: ray-space initial fit, pose 고정 ---
    median_depth = float(np.median(p_cam_arr[:, 2]))
    initial_center, initial_radius = _initial_sphere_guess(config, median_depth)
    stage_a_fit = _fit_sphere(d_obs_arr, p_cam_arr, n_air, n_glass, thickness, initial_center, initial_radius)

    if not _sphere_fit_is_acceptable(stage_a_fit, thickness):
        return _failure_result(
            config, train_ids, test_ids,
            "Sphere initial fit(STAGE A)이 실패했습니다 - optimizer가 수렴하지 않았거나, "
            "물리적으로 유효하지 않은 sphere로 발산했거나, 유효한 광선 비율이 너무 낮습니다.",
        )

    stage_a_center, stage_a_radius = stage_a_fit.x[:3].copy(), float(stage_a_fit.x[3])
    stage_a_model = SphericalWindshieldModel(K, D, model, stage_a_center, stage_a_radius, n_air, n_glass, thickness)
    stage_a_outcome = _evaluate_spherical(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: sphere + per-frame pose joint refinement (ray-domain) ---
    joint = _joint_refine_sphere_and_poses(
        ok_frames, d_obs_per_frame, rvecs, tvecs, n_air, n_glass, thickness, stage_a_center, stage_a_radius,
    )

    stage_used = "stage_a_ray_only"
    final_center, final_radius = stage_a_center, stage_a_radius
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    refinement_note = ""

    if is_valid_spherical_windshield(joint.center, joint.radius, thickness):
        stage_b_model = SphericalWindshieldModel(K, D, model, joint.center, joint.radius, n_air, n_glass, thickness)
        stage_b_outcome = _evaluate_spherical(ok_frames, joint.rvecs, joint.tvecs, stage_b_model, image_size)

        stage_a_rmse = stage_a_outcome.residual_stats.rmse
        stage_b_rmse = stage_b_outcome.residual_stats.rmse
        improved = (
            stage_b_outcome.residual_stats.n > 0
            and stage_a_rmse is not None
            and stage_b_rmse is not None
            and stage_b_rmse <= stage_a_rmse
        )
        if improved:
            stage_used = "stage_b_joint_refined"
            final_center, final_radius = joint.center, joint.radius
            final_rvecs, final_tvecs = joint.rvecs, joint.tvecs
            final_outcome = stage_b_outcome
            if not joint.converged_cleanly:
                refinement_note = "STAGE B 일부 sub-fit이 수렴하지 않아 해당 프레임/라운드는 이전 값을 유지했습니다. "
        else:
            refinement_note = (
                "STAGE B(ray-domain alternating sphere/pose refinement)가 STAGE A(ray-based initial fit)보다 "
                "실제 pixel RMS를 개선하지 못해 STAGE A 결과를 최종으로 사용했습니다. "
                "(참고: 최적화 자체의 residual은 ray-domain이고, 이 STAGE A/B 채택 여부 판단 기준만 "
                "실제 pixel-domain RMS를 사용합니다.) "
            )
    else:
        refinement_note = (
            "STAGE B가 물리적으로 유효하지 않은 sphere로 수렴해 STAGE A 결과를 최종으로 사용했습니다. "
        )

    final_model = SphericalWindshieldModel(K, D, model, final_center, final_radius, n_air, n_glass, thickness)

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 sphere로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다 (initial guess/hint를 조정해보세요).",
        )

    fitted_params = {
        "sphere_center_x": float(final_center[0]),
        "sphere_center_y": float(final_center[1]),
        "sphere_center_z": float(final_center[2]),
        "sphere_radius": float(final_radius),
        "glass_refractive_index": float(n_glass),
        "air_refractive_index": float(n_air),
        "glass_thickness_m": float(thickness),
        "initial_center_x": float(initial_center[0]),
        "initial_center_y": float(initial_center[1]),
        "initial_center_z": float(initial_center[2]),
        "initial_radius": float(initial_radius),
        "stage_a_optimizer_cost": float(stage_a_fit.cost),
        "num_fit_points": float(len(d_obs_arr)),
    }

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPHERICAL,
        base_model_name=model,
        base_camera_matrix=K,
        base_distortion=D,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        failed_frame_ids=list(failed_ids),
        fitted_params=fitted_params,
        success=True,
        warning_message=(refinement_note or None),
        **_outcome_to_result_fields(final_outcome),
    )
    # fitted_params에 어느 stage를 최종으로 썼는지도 기록(진단용, float dict라
    # 문자열은 별도 필드가 없으므로 warning_message/로그로 대신 남긴다).
    result.fitted_params["stage_used_is_joint_refined"] = 1.0 if stage_used == "stage_b_joint_refined" else 0.0

    if test_ids:
        test_frames = _subset_frames(windshield_dataset, test_ids)
        if test_frames:
            t_ok_frames, t_init_rvecs, t_init_tvecs, t_failed = solve_poses_fixed_intrinsics(test_frames, K, D, model)
            if t_ok_frames:
                # Test pose는 Standard solvePnP를 초기값으로 삼아, 최종(고정된) sphere
                # 기준으로 pose만 다시 refine한다 - sphere/K/D는 여기서 절대 건드리지
                # 않는다(leakage 없음, 섹션 19).
                t_rvecs, t_tvecs = [], []
                for frame, init_rvec, init_tvec in zip(t_ok_frames, t_init_rvecs, t_init_tvecs):
                    corners = frame.detection.corners.reshape(-1, 2)
                    d_obs_test = np.array(
                        [baseline_model.unproject_pixel(float(px), float(py)) for px, py in corners]
                    )
                    pose_fit = refine_frame_pose_ray_domain(
                        frame, d_obs_test, final_center, final_radius, thickness, n_air, n_glass,
                        init_rvec, init_tvec, regularize=True,
                    )
                    if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                        t_rvecs.append(pose_fit.x[:3].reshape(3, 1))
                        t_tvecs.append(pose_fit.x[3:6].reshape(3, 1))
                    else:
                        t_rvecs.append(init_rvec)
                        t_tvecs.append(init_tvec)

                test_outcome = _evaluate_spherical(t_ok_frames, t_rvecs, t_tvecs, final_model, image_size)
                result.test_residual_stats = test_outcome.residual_stats
                result.test_regional_error = test_outcome.regional_error
                result.test_radial_profile = test_outcome.radial_profile
                result.test_radial_bands = test_outcome.radial_bands
                result.test_spatial_error_map = test_outcome.spatial_error_map
                result.test_mean_dx = test_outcome.mean_dx
                result.test_mean_dy = test_outcome.mean_dy
                result.test_ray_angular_error_deg = test_outcome.ray_angular_error_deg

                total_test_points = test_outcome.num_points_ok + test_outcome.num_points_failed
                test_failure_rate = (
                    test_outcome.num_points_failed / total_test_points if total_test_points else 1.0
                )
                if test_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
                    result.warning_message = (
                        (result.warning_message or "")
                        + f"Test 코너의 {test_failure_rate*100:.0f}%에서 유효한 굴절을 계산하지 "
                        "못했습니다 (Test 결과의 신뢰도가 낮을 수 있습니다)."
                    )
            for fid in t_failed:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = (result.warning_message or "") + "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result
