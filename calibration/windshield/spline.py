"""
camera_calibrator.calibration.windshield.spline
====================================================

Phase 4 - Spline Windshield Model. Residual Grid/RBF(Phase 3)와 근본적으로
다른 종류의 모델이다 - 픽셀에서 직접 ΔRay(u,v)를 학습하는 것이 아니라,
Spherical(Phase 2)보다 더 유연한 **물리적 표면(surface) 모델**로 실제
Windshield의 국부적인 곡률 차이를 표현한다.

    Camera
      ↓
    Base K,D 🔒
      ↓
    Camera Ray
      ↓
    Spherical Base Surface (calibrate_spherical()로 먼저 얻어 고정)
      +
    Spline Local Deformation (1-DoF, surface normal 방향으로만)
      ↓
    Refined Windshield Surface
      ↓
    Surface Normal (실제 deformed surface normal, finite difference)
      ↓
    Snell Refraction (공기->유리, calibration.windshield.refraction.refract_ray 재사용)
      ↓
    Outer Surface (같은 local radius-offset을 thickness만큼 더 늘린 concentric 근사)
      ↓
    Snell Refraction (유리->공기)
      ↓
    Exterior Ray

수학적으로:

    S(u,v) = C + (R + Delta_s(u,v)) * n_sphere(u,v)

(u,v)는 그 광선이 나온 픽셀이다 - "Base spherical intersection
parameterization"을 pixel로 인덱싱한 것과 동치다: 고정된 카메라 pose/K,D
아래에서 각 픽셀은 base sphere 위 정확히 한 점(그 점의 각도 좌표)에 대응하는
전단사(bijective) 관계이므로, pixel을 그 각도 좌표의 실용적인 대리
(proxy)로 쓴다 - 같은 pixel/ray는 항상 같은 surface parameter로 매핑된다
(결정론적, 사용자 스펙 8번 핵심 요구사항).

핵심 구현 통찰 - "normal 방향 displacement는 sphere에서 radius 변경과
동치": 구의 표면에서 바깥쪽 법선은 항상 중심에서 뻗어나가는 반지름 방향과
같다. 그래서 "base sphere 표면 점을 그 지점의 법선 방향으로 Delta_s만큼
옮긴다"는 것은 정확히 "그 방향으로는 반지름이 R+Delta_s인 구와 광선의
교차점을 구한다"는 것과 수학적으로 동일하다 - 광선을 다시 광선으로
투영하는 무거운 반복 surface-intersection solve 없이(사용자 스펙 11번
"Full-image brute-force 금지" 정신과 일치), intersect_ray_sphere()를 그대로
재사용하는 완전한 closed-form이 된다. 사용자 스펙 10번이 요구하는 "Base
sphere intersection -> Spline displacement -> local numerical refinement"
파이프라인은 이 항등식 덕분에 수치적 반복(iteration) 없이 정확히
재현된다 - 근사가 아니라 엄밀한 등가 변형이다.

Outer 표면은 같은 논리를 thickness만큼 더 확장해서 근사한다(사용자 스펙
14번 "local-normal thin-shell approximation" - 정확히 그렇게 명시함):
반지름 R+Delta_s(u,v)+thickness인, 같은 중심 C의 구에 굴절된 광선
(d_glass)을 다시 교차시킨다. thickness(~5mm)가 base sphere 곡률 반경(수
미터)에 비해 훨씬 작으므로, outer 표면 자체의 접선 방향 곡률 기여는
무시할 만하다고 보고 outer normal은 순수 반경 방향(normalize(P-C))으로
근사한다 - Spherical 모델이 이미 쓰고 있는 것과 동일한 근사다.

Surface normal(사용자 스펙 12번, 가장 중요): 반드시 실제 deformed surface의
normal이어야 한다 - Spherical normal을 그대로 쓰지 않는다. Delta_s(u,v)의
analytic gradient는 (undistortion + base sphere intersection 전체를 통과해야
하므로) 구현이 복잡해 첫 버전에서는 안정적인 central finite difference로
구한다(사용자 스펙 12번이 명시적으로 허용).

Base Sphere는 이 모듈의 어떤 함수도 재추정하지 않는다 - 항상 Outer Train만
사용한 calibrate_spherical() 결과를 그대로 얼려서(frozen) 쓴다(사용자 스펙
2/5/20번). Base K,D는 물론 절대 재최적화하지 않는다(사용자 스펙 4번).

Optimization 대상: Spline control grid의 Delta_s 값(1-DoF/node) + train
프레임별 (rvec, tvec)뿐이다. Sphere/K/D/굴절률/두께는 전부 고정.

STAGE A(spline만, pose 고정) / STAGE B(spline+pose alternating, ray-domain)
패턴은 Spherical/Residual Grid/RBF와 완전히 동일한 철학이다 - 다만 pose
refinement 자체는 residual_common.refine_frame_pose_ray_domain(Grid/RBF
전용, "d_base+delta" 덧셈 모델을 가정)을 재사용할 수 없다 - Spline은 굴절
physics 전체(표면 교차 -> normal -> Snell)를 거치는 non-additive 모델이라
spherical.py의 refine_frame_pose_ray_domain과 같은 구조(전체 ray-alignment
residual)가 필요하다. 그래서 이 모듈은 spherical.py의 pose-refinement
패턴을 그대로 재현하되 delta_s가 있는 굴절 체인으로 바꿔 쓴다(residual_common
과는 POSE_ROTATION_REG_WEIGHT/POSE_TRANSLATION_REG_WEIGHT 값만 공유해서
pose prior 정책 자체는 Grid/RBF/Spline 전부 동일하게 유지한다).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibration.models.common import MIN_FRAMES_REQUIRED, infer_image_size
from calibration.types import CameraConfig, CameraModelType, Dataset, Frame
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.refraction import intersect_ray_sphere, normalize, refract_ray
from calibration.windshield.residual_common import (
    DEFAULT_REPEATED_HOLDOUT_SEEDS,
    POSE_ROTATION_REG_WEIGHT,
    POSE_TRANSLATION_REG_WEIGHT,
    MAX_PROJECT_POINT_ANGULAR_ERROR_DEG,
    RepeatedHoldoutSummary,
    collect_corner_arrays,
    compute_ray_stability_deg,
    evaluate_residual_ray_model,
    fixed_evaluation_pixels,
    normalize_pixel_coordinates,
    populate_pose_diagnostics,
    populate_repeated_holdout_diagnostics,
)
from calibration.windshield.spherical import calibrate_spherical, is_valid_spherical_windshield

# ---------------------------------------------------------------------------
# 고정 상수 - spline_hint로 덮어쓸 수 있고, 코드 여러 곳에 hard-code하지 않는다.
# ---------------------------------------------------------------------------
DEFAULT_SPLINE_ROWS = 4
DEFAULT_SPLINE_COLS = 6
DEFAULT_LAMBDA_MAG = 1e-2
DEFAULT_LAMBDA_SMOOTH = 1e-1
DEFAULT_LAMBDA_CURVE = 1e-1
# ±10mm - 실제 windshield 국부 변형 스케일보다 넉넉하되, surface folding이
# 일어날 만큼 크지는 않은 값(사용자 스펙 15/31번 UI mockup 기본값과 동일).
DEFAULT_MAX_DISPLACEMENT_M = 0.010

SPLINE_GRID_CANDIDATES: list[tuple[int, int]] = [(3, 4), (4, 6), (6, 8)]
SPLINE_SELECTION_TIE_TOLERANCE = 0.05
SPLINE_STAGE_B_NUM_ROUNDS = 2

MIN_CORNERS_PER_NODE = 3  # spline 1-DoF는 grid+bilinear의 3-DoF보다 자유도가 작으므로 완화
MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10

# Surface normal 계산용 central finite-difference step(픽셀 단위) - 너무
# 작으면 수치 잡음, 너무 크면 국부 곡률을 놓친다. 이미지 크기와 무관하게
# 안정적으로 동작하도록 절대 픽셀 값으로 고정한다.
_NORMAL_FD_STEP_PX = 2.0

_RAY_PENALTY = 5.0
_ORIGIN = np.zeros(3, dtype=np.float64)


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


# ---------------------------------------------------------------------------
# Spline scalar bilinear interpolation (normalized [-1,1] domain)
# ---------------------------------------------------------------------------

def bilinear_interpolate_scalar_grid(grid: np.ndarray, un: float, vn: float) -> float:
    """(rows, cols) 스칼라 grid에서 normalized 좌표 (un,vn) in [-1,1]의 값을
    bilinear interpolation으로 구한다. Node (r,c)는 [-1,1]을 (rows-1)/(cols-1)
    등분한 좌표에 있다 - Residual Grid의 bilinear_interpolate_grid와 같은
    convention(경계 포함, node (0,0)=(-1,-1), node(rows-1,cols-1)=(1,1))을
    normalized 좌표계에 맞춰 재사용한다.

    범위 밖 좌표는 clamp한다(외삽 대신 가장 가까운 경계 셀) - crash 방지.
    """
    rows, cols = grid.shape[0], grid.shape[1]
    if rows < 2 or cols < 2:
        raise ValueError("Spline control grid must have at least 2 rows and 2 columns.")

    fx = (min(max(un, -1.0), 1.0) + 1.0) / 2.0 * (cols - 1)
    fy = (min(max(vn, -1.0), 1.0) + 1.0) / 2.0 * (rows - 1)

    c0 = int(min(max(math.floor(fx), 0), cols - 2))
    r0 = int(min(max(math.floor(fy), 0), rows - 2))
    c1, r1 = c0 + 1, r0 + 1

    tx = fx - c0
    ty = fy - r0

    top = (1 - tx) * grid[r0, c0] + tx * grid[r0, c1]
    bottom = (1 - tx) * grid[r1, c0] + tx * grid[r1, c1]
    return float((1 - ty) * top + ty * bottom)


def _local_delta_s(u: float, v: float, spline_grid: np.ndarray, image_width: float, image_height: float) -> float:
    un, vn = normalize_pixel_coordinates(u, v, image_width, image_height)
    return bilinear_interpolate_scalar_grid(spline_grid, un, vn)


# ---------------------------------------------------------------------------
# Deformed surface: point + real surface normal (finite difference)
# ---------------------------------------------------------------------------

def _deformed_surface_point(
    u: float, v: float, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, spline_grid: np.ndarray,
    image_width: float, image_height: float,
) -> Optional[np.ndarray]:
    """(u,v)에서 나가는 base ray가 "국소적으로 반지름이 R+Delta_s(u,v)로
    바뀐" 구(중심은 base sphere와 동일)와 만나는 점 - normal 방향 displacement와
    수학적으로 동치인 closed-form(모듈 docstring 참고). 교차 실패 시 None."""
    d = np.asarray(baseline_model.unproject_pixel(u, v), dtype=np.float64)
    ds = _local_delta_s(u, v, spline_grid, image_width, image_height)
    local_radius = base_radius + ds
    if local_radius <= 0.0:
        return None
    hit = intersect_ray_sphere(_ORIGIN, d, center, local_radius)
    if hit is None:
        return None
    return hit[0]


def _deformed_surface_point_and_normal(
    u: float, v: float, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, spline_grid: np.ndarray,
    image_width: float, image_height: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """실제 deformed surface의 점과 법선(normal)을 구한다(사용자 스펙 12번
    핵심 요구사항 - Spherical normal을 절대 그대로 쓰지 않는다).

    Analytic gradient 대신 central finite difference로 두 접선(tangent)
    벡터를 구하고 외적(cross product)으로 normal을 얻는다(사용자 스펙
    12번이 명시적으로 허용하는 fallback). 접선이 퇴화(degenerate, surface
    folding/flat spot)되면 None을 반환한다(사용자 스펙 16번 - fold 방지
    검사의 최소 형태)."""
    p = _deformed_surface_point(u, v, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    if p is None:
        return None

    step = _NORMAL_FD_STEP_PX
    p_u_plus = _deformed_surface_point(u + step, v, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    p_u_minus = _deformed_surface_point(u - step, v, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    p_v_plus = _deformed_surface_point(u, v + step, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    p_v_minus = _deformed_surface_point(u, v - step, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    if any(x is None for x in (p_u_plus, p_u_minus, p_v_plus, p_v_minus)):
        return None

    tangent_u = (p_u_plus - p_u_minus) / (2.0 * step)
    tangent_v = (p_v_plus - p_v_minus) / (2.0 * step)
    tu_norm = float(np.linalg.norm(tangent_u))
    tv_norm = float(np.linalg.norm(tangent_v))
    # 퇴화(degenerate) 판정은 절대 크기가 아니라 두 접선 벡터의 정렬(sin
    # angle)로 한다 - 이 표면은 카메라 바로 앞(수 mm~수 cm)에 있어 접선
    # 벡터의 절대 크기 자체가 이미 매우 작다(당연한 스케일이지 결함이
    # 아니다). 두 접선이 거의 평행(sin_angle ≈ 0)할 때만 진짜 fold/flat spot
    # (사용자 스펙 16번)으로 판단한다.
    if tu_norm < 1e-15 or tv_norm < 1e-15:
        return None
    n_raw = np.cross(tangent_u, tangent_v)
    n_norm = float(np.linalg.norm(n_raw))
    sin_angle = n_norm / (tu_norm * tv_norm)
    if not np.isfinite(sin_angle) or sin_angle < 1e-6:
        return None  # 퇴화된 접선(surface folding/flat spot) - 사용자 스펙 16번
    n = n_raw / n_norm
    if np.dot(n, p - center) < 0.0:
        n = -n  # 항상 바깥쪽(카메라 쪽)을 향하도록
    return p, n


def _refract_through_deformed_shell(
    u: float, v: float, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, image_width: float, image_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(u,v) 픽셀의 base ray가 deformed inner surface -> 유리 -> deformed
    outer surface(근사)를 통과한 뒤의 (exit point, exit direction)을 반환한다.
    Spherical의 _refract_through_shell과 완전히 같은 역할이지만 두 표면
    모두 spline으로 국소적으로 변형된 반지름을 쓴다. 실패(교차 없음/전반사/
    퇴화 normal)하면 ValueError."""
    inner = _deformed_surface_point_and_normal(u, v, baseline_model, center, base_radius, spline_grid, image_width, image_height)
    if inner is None:
        raise ValueError("Deformed inner surface intersection/normal failed.")
    p1, n1 = inner

    d_cam = np.asarray(baseline_model.unproject_pixel(u, v), dtype=np.float64)
    d_glass = refract_ray(d_cam, n1, n_air, n_glass)
    if d_glass is None:
        raise ValueError("Total internal reflection at the deformed inner surface.")

    # Outer 표면 근사(사용자 스펙 14번, "local-normal thin-shell
    # approximation") - 같은 local radius offset을 thickness만큼 확장한,
    # 같은 중심의 구에 굴절된 광선(d_glass)을 다시 교차시킨다(정확한 offset
    # surface가 아니라는 점을 명시).
    ds_here = _local_delta_s(u, v, spline_grid, image_width, image_height)
    outer_radius = base_radius + ds_here + thickness
    outer_hit = intersect_ray_sphere(p1, d_glass, center, outer_radius)
    if outer_hit is None:
        raise ValueError("Ray does not intersect the deformed outer surface.")
    p2, _ = outer_hit

    n2 = normalize(p2 - center)  # thickness가 작아 outer 접선 곡률 기여는 무시(Spherical과 동일 근사)
    d_out = refract_ray(d_glass, n2, n_glass, n_air)
    if d_out is None:
        raise ValueError("Total internal reflection at the deformed outer surface.")

    return p2, d_out


def is_valid_spline_shell(center: np.ndarray, base_radius: float, spline_grid: np.ndarray, thickness: float) -> bool:
    """base sphere 자체의 물리적 유효성(is_valid_spherical_windshield)에
    더해, 가장 안쪽으로 파고든(worst-case) deformation을 적용해도 카메라가
    여전히 표면 안쪽에 안전 마진을 두고 있는지 확인한다(사용자 스펙 15/16번)."""
    if not is_valid_spherical_windshield(center, base_radius, thickness):
        return False
    if not np.all(np.isfinite(spline_grid)):
        return False
    worst_case_radius = base_radius + float(np.min(spline_grid))
    return is_valid_spherical_windshield(center, worst_case_radius, thickness)


# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

class SplineWindshieldModel(WindshieldModel):
    """Base Sphere(고정) + Spline local surface deformation(1-DoF/node,
    surface normal 방향)으로 Windshield를 근사하고 실제 Snell 굴절을
    계산하는 모델. project_point()/unproject_pixel() 모두 실제 deformed
    surface 교차 + normal + 굴절을 거친다."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        sphere_center: np.ndarray,
        sphere_radius: float,
        spline_grid: np.ndarray,   # (rows, cols) - Delta_s in meters
        image_width: float,
        image_height: float,
        n_air: float = 1.0,
        n_glass: float = 1.52,
        glass_thickness_m: float = 0.005,
    ):
        self._center = np.asarray(sphere_center, dtype=np.float64)
        self._radius = float(sphere_radius)
        self._grid = np.asarray(spline_grid, dtype=np.float64)
        if self._grid.ndim != 2:
            raise ValueError(f"Spline control grid must have shape (rows, cols), got {self._grid.shape}.")
        self._image_width = float(image_width)
        self._image_height = float(image_height)
        self._n_air = float(n_air)
        self._n_glass = float(n_glass)
        self._thickness = float(glass_thickness_m)
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def local_displacement_m(self, u: float, v: float) -> float:
        """이 픽셀 위치에서의 surface normal 방향 displacement(미터) - 진단용
        Surface Stability 계산(calibrate_spline 모듈의 compute_surface_stability_mm)
        전용 public accessor. 런타임 project_point/unproject_pixel 경로와는
        무관하다."""
        return _local_delta_s(u, v, self._grid, self._image_width, self._image_height)

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        """픽셀 -> 굴절 반영 외부 광선 방향(단위 벡터).

        Spherical과 동일한 근사: 실제로는 광선의 원점이 windshield 바깥
        exit point로 옮겨가지만, WindshieldModel ABC 반환 형태가 방향뿐이라
        방향만 보고한다(non-central ray, 내부 optical computation에서는
        exit point를 버리지 않는다 - project_point()가 그것을 실제로
        사용한다, 사용자 스펙 23번)."""
        _, d_out = _refract_through_deformed_shell(
            u, v, self._baseline, self._center, self._radius, self._thickness,
            self._n_air, self._n_glass, self._grid, self._image_width, self._image_height,
        )
        return float(d_out[0]), float(d_out[1]), float(d_out[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다 - Base K,D 투영을
        초기값으로 삼아 작은 2변수 root-solve로 푼다(Spherical/Grid/RBF와
        동일한 구조)."""
        target = np.array([x, y, z], dtype=np.float64)
        initial_uv = np.asarray(self._baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            try:
                point, direction = _refract_through_deformed_shell(
                    float(uv[0]), float(uv[1]), self._baseline, self._center, self._radius, self._thickness,
                    self._n_air, self._n_glass, self._grid, self._image_width, self._image_height,
                )
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

        residual_norm = float(np.linalg.norm(result.fun))
        angle_rad = 2.0 * math.asin(min(1.0, residual_norm / 2.0))
        if math.degrees(angle_rad) > MAX_PROJECT_POINT_ANGULAR_ERROR_DEG:
            raise ValueError(
                "Could not find a valid refracted projection for this point "
                "(likely outside the spline surface's coverage)."
            )
        return float(result.x[0]), float(result.x[1])

    def ray_angular_error_deg(self, u: float, v: float, target_point_cam: np.ndarray) -> Optional[float]:
        try:
            point, direction = _refract_through_deformed_shell(
                u, v, self._baseline, self._center, self._radius, self._thickness,
                self._n_air, self._n_glass, self._grid, self._image_width, self._image_height,
            )
        except ValueError:
            return None
        to_target = np.asarray(target_point_cam, dtype=np.float64) - point
        norm = np.linalg.norm(to_target)
        if norm < 1e-9:
            return 0.0
        cos_angle = float(np.clip(np.dot(to_target / norm, direction), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))


def build_spline_model_from_fitted_params(
    camera_matrix: np.ndarray, distortion: np.ndarray, model: CameraModelType, fitted_params: dict[str, float],
) -> SplineWindshieldModel:
    """fitted_params(아래 스키마)에서 SplineWindshieldModel을 재구성하는
    유일한 지점(projection.py::build_projector가 재사용) - SciPy 내부
    상태를 저장/역직렬화하지 않는다(spline 자체가 SciPy spline 객체를 쓰지
    않고 순수 bilinear라 애초에 해당 없음), 재구성 가능한 public 값만 쓴다
    (sphere_center_x/y/z, sphere_radius, glass_refractive_index,
    air_refractive_index, glass_thickness_m, spline_rows, spline_cols,
    spline_ds_{r}_{c}, image_width, image_height)."""
    fp = fitted_params
    rows, cols = int(fp["spline_rows"]), int(fp["spline_cols"])
    grid = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            grid[r, c] = fp[f"spline_ds_{r}_{c}"]
    return SplineWindshieldModel(
        camera_matrix, distortion, model,
        sphere_center=np.array([fp["sphere_center_x"], fp["sphere_center_y"], fp["sphere_center_z"]]),
        sphere_radius=fp["sphere_radius"],
        spline_grid=grid,
        image_width=fp["image_width"],
        image_height=fp["image_height"],
        n_air=fp.get("air_refractive_index", 1.0),
        n_glass=fp.get("glass_refractive_index", 1.52),
        glass_thickness_m=fp.get("glass_thickness_m", 0.005),
    )


# ---------------------------------------------------------------------------
# Calibration (fitting)
# ---------------------------------------------------------------------------

def _spline_settings(config: WindshieldConfig) -> tuple[int, int, float, float, float, float]:
    hint = config.spline_hint or {}
    rows = int(hint.get("spline_rows", DEFAULT_SPLINE_ROWS))
    cols = int(hint.get("spline_cols", DEFAULT_SPLINE_COLS))
    lambda_mag = float(hint.get("lambda_mag", DEFAULT_LAMBDA_MAG))
    lambda_smooth = float(hint.get("lambda_smooth", DEFAULT_LAMBDA_SMOOTH))
    lambda_curve = float(hint.get("lambda_curve", DEFAULT_LAMBDA_CURVE))
    max_displacement = float(hint.get("max_displacement_m", DEFAULT_MAX_DISPLACEMENT_M))
    if rows < 2 or cols < 2:
        raise ValueError(f"spline_hint spline_rows/spline_cols must each be >= 2 (got rows={rows}, cols={cols}).")
    if max_displacement <= 0.0:
        raise ValueError(f"spline_hint max_displacement_m must be > 0 (got {max_displacement}).")
    return rows, cols, lambda_mag, lambda_smooth, lambda_curve, max_displacement


def _ray_alignment_residual_spline(
    u: float, v: float, p_cam: np.ndarray, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, image_width: float, image_height: float,
) -> np.ndarray:
    """한 코너의 (관측 픽셀, 목표점)에 대한 ray-alignment residual(3-vector).
    Spherical의 _ray_alignment_residual과 동일한 역할/형태이지만 spline
    deformed surface를 통과한다. STAGE A의 초기 spline fit과 STAGE B의
    spline/pose 공동 refinement가 전부 이 함수 하나를 재사용한다.

    주의: 여기서 is_valid_spline_shell()(grid 전체의 worst-case 검사)을
    호출하지 않는다 - 실제로 겪은 버그: 카메라가 base sphere 안쪽 여유
    마진(MIN_SPHERE_MARGIN_M)에 거의 딱 붙어 있는 흔한 경우, grid의 노드
    "단 하나"가 살짝 음수로 움직이기만 해도 grid 전체의 min()이 그 노드
    값을 반영해 이 코너와 무관한 다른 모든 코너의 residual까지 통째로
    penalty로 만들어버려, optimizer가 gradient 정보를 완전히 잃고 x0=0에서
    전혀 움직이지 못했다. 대신 이 함수가 다루는 "이 코너 하나"에 대해서만
    실제 교차/굴절을 시도하고 실패하면(로컬 radius<=0, 교차 없음, 전반사)
    그 코너만 penalty 처리한다(Grid/RBF의 bilinear/RBF 보간이 애초에 이런
    전역 게이트를 두지 않는 것과 같은 설계). grid 전체의 물리적 유효성은
    fit 완료 후 한 번만(calibrate_spline의 STAGE B 채택 여부 판단, Spherical의
    is_valid_spherical_windshield 사용법과 동일) 검사한다."""
    try:
        point, direction = _refract_through_deformed_shell(
            u, v, baseline_model, center, base_radius, thickness, n_air, n_glass,
            spline_grid, image_width, image_height,
        )
    except ValueError:
        return np.full(3, _RAY_PENALTY)
    to_target = p_cam - point
    norm = np.linalg.norm(to_target)
    if norm < 1e-9:
        return np.zeros(3)
    return to_target / norm - direction


def _fit_spline_stage_a(
    observed_pixels: np.ndarray,   # (N, 2)
    p_cam: np.ndarray,             # (N, 3)
    baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    rows: int, cols: int, image_width: float, image_height: float,
    lambda_mag: float, lambda_smooth: float, lambda_curve: float, max_displacement: float,
):
    """ray-alignment residual + magnitude/smoothness/curvature
    regularization으로 spline grid 전체를 한 번에 피팅한다(사용자 스펙
    17번 - Residual Grid/RBF보다 더 강한 3중 regularization).

        L_mag    = lambda_mag    * sum(Delta_s_i^2)
        L_smooth = lambda_smooth * sum((Delta_s_i - Delta_s_j)^2)   (인접 쌍)
        L_curve  = lambda_curve  * sum((Delta_s_{i-1} - 2*Delta_s_i + Delta_s_{i+1})^2)  (행/열 각각)
    """
    n_points = len(observed_pixels)

    horizontal_pairs = [(r, c, r, c + 1) for r in range(rows) for c in range(cols - 1)]
    vertical_pairs = [(r, c, r + 1, c) for r in range(rows - 1) for c in range(cols)]
    smooth_pairs = horizontal_pairs + vertical_pairs

    horizontal_triples = [(r, c - 1, r, c, r, c + 1) for r in range(rows) for c in range(1, cols - 1)]
    vertical_triples = [(r - 1, c, r, c, r + 1, c) for r in range(1, rows - 1) for c in range(cols)]
    curve_triples = horizontal_triples + vertical_triples

    def residual(params: np.ndarray) -> np.ndarray:
        grid = params.reshape(rows, cols)

        # 코너마다 3-vector residual을 그대로 유지한다(norm()으로 스칼라화하면
        # least_squares가 최소화하는 sum-of-squares 구조가 깨지고 gradient
        # 정보를 잃어 수렴이 사실상 멈춘다 - 실제로 겪은 버그. 파라미터가
        # 노드당 1개(스칼라)라고 해서 residual까지 스칼라일 필요는 없다,
        # Grid/Spherical과 동일하게 코너당 3개 성분을 그대로 펼친다).
        data_res = np.empty((n_points, 3))
        for i in range(n_points):
            u, v = observed_pixels[i]
            data_res[i] = _ray_alignment_residual_spline(
                float(u), float(v), p_cam[i], baseline_model, center, base_radius, thickness, n_air, n_glass,
                grid, image_width, image_height,
            )

        mag_res = math.sqrt(lambda_mag) * grid.ravel()

        smooth_res = np.empty(len(smooth_pairs))
        for i, (r0, c0, r1, c1) in enumerate(smooth_pairs):
            smooth_res[i] = math.sqrt(lambda_smooth) * (grid[r0, c0] - grid[r1, c1])

        curve_res = np.empty(len(curve_triples))
        for i, (r0, c0, r1, c1, r2, c2) in enumerate(curve_triples):
            curve_res[i] = math.sqrt(lambda_curve) * (grid[r0, c0] - 2.0 * grid[r1, c1] + grid[r2, c2])

        return np.concatenate([data_res.ravel(), mag_res, smooth_res, curve_res])

    x0 = np.zeros(rows * cols)  # Delta_s initial = 0 (사용자 스펙 19번)
    bounds = (np.full_like(x0, -max_displacement), np.full_like(x0, max_displacement))
    return least_squares(residual, x0=x0, bounds=bounds, method="trf", loss="soft_l1", f_scale=0.05)


def _refine_frame_pose_ray_domain_spline(
    frame: Frame,
    observed_pixels: np.ndarray,
    baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, image_width: float, image_height: float,
    initial_rvec: np.ndarray, initial_tvec: np.ndarray,
    *, regularize: bool = True,
):
    """한 프레임의 pose(rvec,tvec)만 ray-alignment residual로 refine한다.
    Sphere/Spline/K,D는 고정(호출부가 넘긴 값 그대로).

    residual_common.refine_frame_pose_ray_domain(Grid/RBF 전용, "d_base+delta"
    덧셈 모델을 가정)을 재사용할 수 없다 - Spline은 표면 교차 -> normal ->
    Snell 굴절 전체를 거치는 non-additive 모델이라, spherical.py의
    refine_frame_pose_ray_domain과 같은 구조가 필요하다(사용자 스펙 22번
    "Spherical에서 사용하는 non-central ray logic과 더 가까운 구조" 지침).
    POSE_ROTATION_REG_WEIGHT/POSE_TRANSLATION_REG_WEIGHT는 residual_common
    에서 가져와 Grid/RBF와 동일한 pose prior 정책을 유지한다(사용자 스펙
    18번)."""
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
            u, v = observed_pixels[i]
            out[i] = _ray_alignment_residual_spline(
                float(u), float(v), cam_pts[i], baseline_model, center, base_radius, thickness, n_air, n_glass,
                spline_grid, image_width, image_height,
            )
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
class _JointSplineRefinementOutcome:
    grid: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    converged_cleanly: bool


def _joint_refine_spline_and_poses(
    ok_frames: list[Frame],
    observed_pixels_per_frame: list[np.ndarray],
    baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    initial_rvecs: list[np.ndarray], initial_tvecs: list[np.ndarray],
    rows: int, cols: int, image_width: float, image_height: float,
    lambda_mag: float, lambda_smooth: float, lambda_curve: float, max_displacement: float,
    initial_grid: np.ndarray,
    num_rounds: int = SPLINE_STAGE_B_NUM_ROUNDS,
) -> _JointSplineRefinementOutcome:
    """STAGE B - alternating(block-coordinate) 방식으로 spline과 프레임별
    pose를 번갈아 refine한다. Spherical/Grid의 alternating 구조와 완전히
    동일한 패턴."""
    rvecs = [np.asarray(r, dtype=np.float64).copy() for r in initial_rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).copy() for t in initial_tvecs]
    grid = np.asarray(initial_grid, dtype=np.float64).copy()
    converged_cleanly = True

    for _ in range(num_rounds):
        for i, frame in enumerate(ok_frames):
            pose_fit = _refine_frame_pose_ray_domain_spline(
                frame, observed_pixels_per_frame[i], baseline_model, center, base_radius, thickness, n_air, n_glass,
                grid, image_width, image_height, initial_rvecs[i], initial_tvecs[i],
            )
            if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                rvecs[i] = pose_fit.x[:3].reshape(3, 1)
                tvecs[i] = pose_fit.x[3:6].reshape(3, 1)
            else:
                converged_cleanly = False

        p_cam_list, obs_pixel_list = [], []
        for i, frame in enumerate(ok_frames):
            R, _ = cv2.Rodrigues(rvecs[i])
            obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
            cam_pts = (R @ obj.T).T + tvecs[i].reshape(1, 3)
            p_cam_list.append(cam_pts)
            obs_pixel_list.append(observed_pixels_per_frame[i])
        p_cam_arr = np.concatenate(p_cam_list, axis=0)
        obs_pixel_arr = np.concatenate(obs_pixel_list, axis=0)

        grid_fit = _fit_spline_stage_a(
            obs_pixel_arr, p_cam_arr, baseline_model, center, base_radius, thickness, n_air, n_glass,
            rows, cols, image_width, image_height, lambda_mag, lambda_smooth, lambda_curve, max_displacement,
        )
        if grid_fit.success and np.all(np.isfinite(grid_fit.x)):
            grid = grid_fit.x.reshape(rows, cols).copy()
        else:
            converged_cleanly = False

    return _JointSplineRefinementOutcome(grid=grid, rvecs=rvecs, tvecs=tvecs, converged_cleanly=converged_cleanly)


def _failure_result(config: WindshieldConfig, train_ids: list[str], test_ids: list[str], message: str) -> WindshieldCalibrationResult:
    return WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPLINE,
        base_model_name=config.base_model_name,
        base_camera_matrix=config.base_camera_matrix,
        base_distortion=config.base_distortion,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        success=False,
        error_message=message,
    )


def calibrate_spline(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Spline Windshield 모델을 fitting한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않는다.

    흐름: (Outer Train만으로) Spherical calibration -> Base Sphere 고정 ->
    (AUTO 지정 시 spline grid 해상도 선택, train_ids만 사용) -> STAGE
    A(spline만, pose 고정) -> STAGE B(spline+pose joint, ray-domain
    alternating) -> 두 stage의 실제 pixel RMS를 비교해 더 나은 쪽을 최종으로
    채택 -> Train 평가 -> Test는 최종 sphere+spline을 완전히 고정한 채 자기
    pose만 별도로 refine한 뒤 평가(leakage 없음).
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size

    # --- Base Sphere: Outer Train만 사용(사용자 스펙 20번, leakage 금지) ---
    sphere_result = calibrate_spherical(windshield_dataset, config, camera_config, train_ids, [])
    if not sphere_result.success:
        return _failure_result(
            config, train_ids, test_ids,
            f"Base Sphere(Spherical) calibration이 실패해 Spline을 진행할 수 없습니다: {sphere_result.error_message}",
        )
    fp_sphere = sphere_result.fitted_params
    center = np.array([fp_sphere["sphere_center_x"], fp_sphere["sphere_center_y"], fp_sphere["sphere_center_z"]])
    base_radius = float(fp_sphere["sphere_radius"])
    n_air = float(fp_sphere["air_refractive_index"])
    n_glass = float(fp_sphere["glass_refractive_index"])
    thickness = float(fp_sphere["glass_thickness_m"])

    hint = config.spline_hint or {}
    if hint.get("auto_spline", 0.0) > 0:
        (auto_rows, auto_cols), _candidates = select_best_spline_grid_resolution(
            windshield_dataset, config, camera_config, train_ids,
        )
        config = dataclasses.replace(
            config,
            spline_hint={**hint, "spline_rows": float(auto_rows), "spline_cols": float(auto_cols), "auto_spline": 0.0},
        )

    rows, cols, lambda_mag, lambda_smooth, lambda_curve, max_displacement = _spline_settings(config)
    min_corners = max(20, rows * cols * MIN_CORNERS_PER_NODE)

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
    observed_pixels_per_frame, _d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(
        ok_frames, rvecs, tvecs, baseline_model
    )

    total_corners = sum(len(a) for a in observed_pixels_per_frame)
    if total_corners < min_corners:
        return _failure_result(
            config, train_ids, test_ids,
            f"Spline grid({rows}x{cols})를 추정하기에 코너 수가 부족합니다 "
            f"(코너 {total_corners}개, 최소 {min_corners}개 필요).",
        )

    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    # --- STAGE A: spline만, pose 고정 ---
    stage_a_fit = _fit_spline_stage_a(
        observed_pixels_arr, p_cam_arr, baseline_model, center, base_radius, thickness, n_air, n_glass,
        rows, cols, width, height, lambda_mag, lambda_smooth, lambda_curve, max_displacement,
    )
    if not stage_a_fit.success or not np.all(np.isfinite(stage_a_fit.x)):
        return _failure_result(config, train_ids, test_ids, "Spline grid optimization(STAGE A)이 수렴하지 않았습니다.")

    stage_a_grid = stage_a_fit.x.reshape(rows, cols).copy()
    stage_a_model = SplineWindshieldModel(K, D, model, center, base_radius, stage_a_grid, width, height, n_air, n_glass, thickness)
    stage_a_outcome = evaluate_residual_ray_model(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: spline + per-frame pose joint refinement (ray-domain) ---
    joint = _joint_refine_spline_and_poses(
        ok_frames, observed_pixels_per_frame, baseline_model, center, base_radius, thickness, n_air, n_glass,
        rvecs, tvecs, rows, cols, width, height, lambda_mag, lambda_smooth, lambda_curve, max_displacement, stage_a_grid,
    )

    stage_used_is_joint_refined = False
    final_grid = stage_a_grid
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    refinement_note = ""

    if is_valid_spline_shell(center, base_radius, joint.grid, thickness):
        stage_b_model = SplineWindshieldModel(K, D, model, center, base_radius, joint.grid, width, height, n_air, n_glass, thickness)
        stage_b_outcome = evaluate_residual_ray_model(ok_frames, joint.rvecs, joint.tvecs, stage_b_model, image_size)

        stage_a_rmse = stage_a_outcome.residual_stats.rmse
        stage_b_rmse = stage_b_outcome.residual_stats.rmse
        improved = (
            stage_b_outcome.residual_stats.n > 0
            and stage_a_rmse is not None
            and stage_b_rmse is not None
            and stage_b_rmse <= stage_a_rmse
        )
        if improved:
            stage_used_is_joint_refined = True
            final_grid = joint.grid
            final_rvecs, final_tvecs = joint.rvecs, joint.tvecs
            final_outcome = stage_b_outcome
            if not joint.converged_cleanly:
                refinement_note = "STAGE B 일부 sub-fit이 수렴하지 않아 해당 프레임/라운드는 이전 값을 유지했습니다. "
        else:
            refinement_note = (
                "STAGE B(ray-domain alternating spline/pose refinement)가 STAGE A(spline-only initial fit)보다 "
                "실제 pixel RMS를 개선하지 못해 STAGE A 결과를 최종으로 사용했습니다. "
                "(참고: 최적화 자체의 residual은 ray-domain이고, 이 STAGE A/B 채택 여부 판단 기준만 "
                "실제 pixel-domain RMS를 사용합니다.) "
            )
    else:
        refinement_note = "STAGE B가 물리적으로 유효하지 않은 spline surface로 수렴해 STAGE A 결과를 최종으로 사용했습니다. "

    final_model = SplineWindshieldModel(K, D, model, center, base_radius, final_grid, width, height, n_air, n_glass, thickness)

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 spline surface로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    runtime_param_count = rows * cols  # 1-DoF/node (사용자 스펙 6/30번)
    pose_param_count_train = len(ok_frames) * 6
    abs_grid = np.abs(final_grid)

    fitted_params: dict[str, float] = {
        "sphere_center_x": float(center[0]),
        "sphere_center_y": float(center[1]),
        "sphere_center_z": float(center[2]),
        "sphere_radius": float(base_radius),
        "glass_refractive_index": float(n_glass),
        "air_refractive_index": float(n_air),
        "glass_thickness_m": float(thickness),
        "image_width": float(width),
        "image_height": float(height),
        "spline_rows": float(rows),
        "spline_cols": float(cols),
        "lambda_mag": float(lambda_mag),
        "lambda_smooth": float(lambda_smooth),
        "lambda_curve": float(lambda_curve),
        "max_displacement_m": float(max_displacement),
        "stage_a_optimizer_cost": float(stage_a_fit.cost),
        "num_fit_points": float(total_corners),
        "runtime_param_count": float(runtime_param_count),
        "pose_param_count_train": float(pose_param_count_train),
        "stage_used_is_joint_refined": 1.0 if stage_used_is_joint_refined else 0.0,
        "diag_deformation_mean_abs_m": float(np.mean(abs_grid)),
        "diag_deformation_max_abs_m": float(np.max(abs_grid)),
    }
    populate_pose_diagnostics(fitted_params, rvecs, tvecs, final_rvecs, final_tvecs)
    for r in range(rows):
        for c in range(cols):
            fitted_params[f"spline_ds_{r}_{c}"] = float(final_grid[r, c])

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPLINE,
        base_model_name=model,
        base_camera_matrix=K,
        base_distortion=D,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        failed_frame_ids=list(failed_ids),
        per_frame_error=final_outcome.per_frame_error,
        residual_stats=final_outcome.residual_stats,
        regional_error=final_outcome.regional_error,
        radial_profile=final_outcome.radial_profile,
        radial_bands=final_outcome.radial_bands,
        spatial_error_map=final_outcome.spatial_error_map,
        mean_dx=final_outcome.mean_dx,
        mean_dy=final_outcome.mean_dy,
        ray_angular_error_deg=final_outcome.ray_angular_error_deg,
        fitted_params=fitted_params,
        success=True,
        warning_message=(refinement_note or None),
    )

    if test_ids:
        test_frames = _subset_frames(windshield_dataset, test_ids)
        if test_frames:
            t_ok_frames, t_init_rvecs, t_init_tvecs, t_failed = solve_poses_fixed_intrinsics(test_frames, K, D, model)
            if t_ok_frames:
                # Test pose는 Standard solvePnP를 초기값으로 삼아, 최종(고정된)
                # sphere+spline 기준으로 pose만 다시 refine한다 - sphere/spline/
                # K/D는 여기서 절대 건드리지 않는다(leakage 없음, 사용자 스펙 26번).
                t_rvecs, t_tvecs = [], []
                t_obs_pixels, _t_d_obs, _t_p_cam = collect_corner_arrays(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, baseline_model
                )
                for frame, init_rvec, init_tvec, obs_px in zip(t_ok_frames, t_init_rvecs, t_init_tvecs, t_obs_pixels):
                    pose_fit = _refine_frame_pose_ray_domain_spline(
                        frame, obs_px, baseline_model, center, base_radius, thickness, n_air, n_glass,
                        final_grid, width, height, init_rvec, init_tvec, regularize=True,
                    )
                    if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                        t_rvecs.append(pose_fit.x[:3].reshape(3, 1))
                        t_tvecs.append(pose_fit.x[3:6].reshape(3, 1))
                    else:
                        t_rvecs.append(init_rvec)
                        t_tvecs.append(init_tvec)

                test_outcome = evaluate_residual_ray_model(t_ok_frames, t_rvecs, t_tvecs, final_model, image_size)
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
                        + f"Test 코너의 {test_failure_rate*100:.0f}%에서 유효한 pixel 예측을 계산하지 "
                        "못했습니다 (Test 결과의 신뢰도가 낮을 수 있습니다)."
                    )
            for fid in t_failed:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = (result.warning_message or "") + "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result


# ---------------------------------------------------------------------------
# Surface Stability (사용자 스펙 28번) - Grid/RBF에는 없는 Spline 전용 지표
# ---------------------------------------------------------------------------

def compute_surface_stability_mm(
    models: list[SplineWindshieldModel], image_width: float, image_height: float,
    sample_rows: int = 12, sample_cols: int = 20,
) -> tuple[Optional[float], Optional[float]]:
    """여러 split에서 얻은 fitted spline surface들이 실제로 얼마나 다른
    displacement(mm)를 만들어내는지 측정한다 - Ray Stability와 같은 고정
    샘플 픽셀(residual_common.fixed_evaluation_pixels)을 재사용해 Grid/RBF의
    Ray Stability와 동일한 sampling 철학을 유지한다. Comparison 표에는
    넣지 않고 Spline 전용 Diagnostics에서만 사용한다(사용자 스펙 28번)."""
    if len(models) < 2:
        return None, None
    sample_pixels = fixed_evaluation_pixels(image_width, image_height, sample_rows, sample_cols)
    displacements_per_model = [
        np.array([m.local_displacement_m(float(u), float(v)) for u, v in sample_pixels]) for m in models
    ]
    all_diffs_mm: list[float] = []
    for i in range(len(displacements_per_model)):
        for j in range(i + 1, len(displacements_per_model)):
            diffs = np.abs(displacements_per_model[i] - displacements_per_model[j]) * 1000.0
            all_diffs_mm.extend(diffs.tolist())
    if not all_diffs_mm:
        return None, None
    arr = np.array(all_diffs_mm)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


# ---------------------------------------------------------------------------
# Repeated Hold-out (사용자 스펙 27번)
# ---------------------------------------------------------------------------

def _grid_from_fitted_params(fitted_params: dict[str, float]) -> np.ndarray:
    rows, cols = int(fitted_params["spline_rows"]), int(fitted_params["spline_cols"])
    grid = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            grid[r, c] = fitted_params[f"spline_ds_{r}_{c}"]
    return grid


@dataclass
class SplineRepeatedHoldoutSummary:
    """RepeatedHoldoutSummary(Grid/RBF와 공유하는 필드) + Spline 전용 Surface
    Stability. 별도 dataclass로 감싸는 이유는 surface stability가 Grid/RBF에는
    없는 개념이라 공유 dataclass 자체를 오염시키고 싶지 않기 때문이다
    (사용자 스펙 28번 "Grid/RBF Comparison 표에 억지로 넣지 않아도 된다")."""
    holdout: RepeatedHoldoutSummary
    surface_stability_mean_mm: Optional[float] = None
    surface_stability_p95_mm: Optional[float] = None


def run_repeated_holdout_spline(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    test_ratio: float = 0.25,
) -> SplineRepeatedHoldoutSummary:
    """windshield_dataset 전체를 여러 (다른 seed의) Train/Test로 나눠 반복
    평가한다 - Grid/RBF의 run_repeated_holdout_residual_*와 평행한 구조."""
    from calibration.models.common import regional_edge_average

    test_rmses: list[float] = []
    test_p95s: list[float] = []
    edge_rmses: list[float] = []
    models: list[SplineWindshieldModel] = []
    successful_seeds: list[int] = []

    K, D, model_name = config.base_camera_matrix, config.base_distortion, config.base_model_name
    width, height = infer_image_size(windshield_dataset, camera_config)

    for seed in seeds:
        train_ids, test_ids = split_train_test(windshield_dataset, camera_config, test_ratio, seed)
        result = calibrate_spline(windshield_dataset, config, camera_config, train_ids, test_ids)
        if not result.success or result.test_residual_stats is None:
            continue
        successful_seeds.append(seed)
        test_rmses.append(result.test_residual_stats.rmse)
        if result.test_residual_stats.p95 is not None:
            test_p95s.append(result.test_residual_stats.p95)
        edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
        if edge is not None:
            edge_rmses.append(edge)
        fp = result.fitted_params
        center = np.array([fp["sphere_center_x"], fp["sphere_center_y"], fp["sphere_center_z"]])
        grid = _grid_from_fitted_params(fp)
        models.append(SplineWindshieldModel(
            K, D, model_name, center, fp["sphere_radius"], grid, fp["image_width"], fp["image_height"],
            fp["air_refractive_index"], fp["glass_refractive_index"], fp["glass_thickness_m"],
        ))

    ray_stability_mean_deg, ray_stability_p95_deg = compute_ray_stability_deg(models, width, height)
    surface_stability_mean_mm, surface_stability_p95_mm = compute_surface_stability_mm(models, width, height)

    holdout = RepeatedHoldoutSummary(
        seeds_used=successful_seeds,
        n_successful=len(successful_seeds),
        mean_test_rmse=float(np.mean(test_rmses)) if test_rmses else None,
        std_test_rmse=float(np.std(test_rmses)) if test_rmses else None,
        mean_test_p95=float(np.mean(test_p95s)) if test_p95s else None,
        mean_edge_rms=float(np.mean(edge_rmses)) if edge_rmses else None,
        grid_stability_l2=None,
        ray_stability_mean_deg=ray_stability_mean_deg,
        ray_stability_p95_deg=ray_stability_p95_deg,
    )
    return SplineRepeatedHoldoutSummary(
        holdout=holdout,
        surface_stability_mean_mm=surface_stability_mean_mm,
        surface_stability_p95_mm=surface_stability_p95_mm,
    )


# ---------------------------------------------------------------------------
# AUTO 선택 (사용자 스펙 29/30번)
# ---------------------------------------------------------------------------

@dataclass
class SplineCandidateResult:
    rows: int
    cols: int
    param_count: int
    summary: RepeatedHoldoutSummary


def select_best_spline_grid_resolution(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    candidate_frame_ids: list[str],
    candidates: Optional[list[tuple[int, int]]] = None,
    seeds: Optional[tuple[int, ...]] = None,
    inner_test_ratio: float = 0.3,
    tie_tolerance: float = SPLINE_SELECTION_TIE_TOLERANCE,
) -> tuple[tuple[int, int], list[SplineCandidateResult]]:
    """candidate_frame_ids(바깥쪽 호출부의 Train만) 안에서 다시 나눠서
    (내부 train/내부 test) 여러 spline grid 해상도를 비교한다 - 바깥쪽 실제
    Test는 이 함수 어디에도 등장하지 않는다(사용자 스펙 29번, leakage 없음).

    선택 규칙: Hold-out RMS가 가장 좋은 후보를 기준으로, tie_tolerance
    이내로 비슷한 후보들 중 parameter(=rows*cols)가 가장 적은 것을 최종
    선택한다(사용자 스펙 30번).
    """
    if candidates is None:
        candidates = SPLINE_GRID_CANDIDATES
    if seeds is None:
        seeds = DEFAULT_REPEATED_HOLDOUT_SEEDS[:3]
    inner_dataset = Dataset(frames=_subset_frames(windshield_dataset, candidate_frame_ids))
    candidate_results: list[SplineCandidateResult] = []

    for rows, cols in candidates:
        cfg = dataclasses.replace(
            config,
            spline_hint={
                **(config.spline_hint or {}),
                "spline_rows": float(rows), "spline_cols": float(cols), "auto_spline": 0.0,
            },
        )
        summary = run_repeated_holdout_spline(inner_dataset, cfg, camera_config, seeds=seeds, test_ratio=inner_test_ratio)
        candidate_results.append(SplineCandidateResult(rows=rows, cols=cols, param_count=rows * cols, summary=summary.holdout))

    valid = [c for c in candidate_results if c.summary.mean_test_rmse is not None]
    if not valid:
        return candidates[0], candidate_results

    best_rmse = min(c.summary.mean_test_rmse for c in valid)
    close_enough = [c for c in valid if c.summary.mean_test_rmse <= best_rmse * (1.0 + tie_tolerance)]
    chosen = min(close_enough, key=lambda c: c.param_count)
    return (chosen.rows, chosen.cols), candidate_results


# ---------------------------------------------------------------------------
# Diagnostics orchestrator - UI/worker가 호출하는 단일 진입점
# ---------------------------------------------------------------------------

def run_spline_calibration_with_diagnostics(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
    *,
    compute_repeated_holdout: bool = True,
    repeated_holdout_seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    repeated_holdout_test_ratio: float = 0.25,
) -> WindshieldCalibrationResult:
    """calibrate_spline()에 Repeated Hold-out + Ray/Surface Stability 진단을
    더한 결과를 반환한다 - Grid/RBF의 run_residual_*_calibration_with_
    diagnostics와 평행 구조. UI(및 그 뒤의 worker)가 Spline 모델을 실행할 때
    호출하는 단일 진입점."""
    hint = config.spline_hint or {}
    was_auto = hint.get("auto_spline", 0.0) > 0

    result = calibrate_spline(windshield_dataset, config, camera_config, train_ids, test_ids)
    if not result.success:
        return result

    result.fitted_params["diag_selection_mode_is_auto"] = 1.0 if was_auto else 0.0

    if compute_repeated_holdout:
        resolved_hint = {
            **hint,
            "auto_spline": 0.0,
            "spline_rows": result.fitted_params["spline_rows"],
            "spline_cols": result.fitted_params["spline_cols"],
        }
        resolved_config = dataclasses.replace(config, spline_hint=resolved_hint)
        summary = run_repeated_holdout_spline(
            windshield_dataset, resolved_config, camera_config,
            seeds=repeated_holdout_seeds, test_ratio=repeated_holdout_test_ratio,
        )
        populate_repeated_holdout_diagnostics(result.fitted_params, summary.holdout, len(repeated_holdout_seeds))
        if summary.surface_stability_mean_mm is not None:
            result.fitted_params["diag_surface_stability_mean_mm"] = summary.surface_stability_mean_mm
        if summary.surface_stability_p95_mm is not None:
            result.fitted_params["diag_surface_stability_p95_mm"] = summary.surface_stability_p95_mm

    return result
