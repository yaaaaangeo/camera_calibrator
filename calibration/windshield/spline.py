"""
camera_calibrator.calibration.windshield.spline
====================================================

Phase 4 - Spline Windshield Model.실제 **Bicubic tensor-product B-Spline
normal-offset parametric surface**로 Base Sphere(Phase 2)보다 더 유연하게
Windshield의 국부적인 곡률 차이를 표현한다.

    Camera
      ↓
    Base K,D 🔒
      ↓
    Camera Ray
      ↓
    Base Sphere S0(p,q) (calibrate_spherical()로 먼저 얻어 고정, Outer Train만 사용)
      +
    Bicubic B-Spline Delta_s(p,q) (1-DoF/node, surface normal 방향으로만)
      ↓
    Inner Surface S(p,q) = S0(p,q) + Delta_s(p,q) * N0(p,q)
      ↓
    실제 deformed surface normal N = normalize(dS/dp x dS/dq)
      ↓
    Snell 굴절(공기->유리, refraction.refract_ray 재사용)
      ↓
    Outer Surface(같은 (p,q)에서 thickness만큼 normal 방향 offset한
    local-normal thin-shell 근사, N_outer ≈ N_inner)
      ↓
    Snell 굴절(유리->공기)
      ↓
    Exterior Ray

## STEP 4 물리 모델 핵심 보완(이번 라운드) - 이전 구현의 문제

이전 구현은 "base sphere 표면 점을 그 지점의 법선 방향으로 Delta_s만큼
옮기는 것"이 "반지름이 R+Delta_s(pixel)인 구와 camera ray의 교차점을 구하는
것"과 수학적으로 동치라고 주장했다. **이 등가성은 일반적으로 성립하지
않는다** - camera origin이 sphere center와 정확히 같지 않은 이상
`camera ray direction != sphere normal direction`이기 때문에, "카메라
광선과 반지름이 변한 구의 교차점"은 "그 광선이 base sphere와 만나는 점을
normal 방향으로 옮긴 점"과 다른 점이다. 이번 라운드에서 이 잘못된 근사
(`intersect_ray_sphere(origin, ray, center, R+ds(pixel))`)를 완전히
제거하고, 아래처럼 진짜 parametric surface + 3-unknown(t,p,q) ray-surface
intersection으로 바꿨다.

## Surface Parameterization (p,q)

`(p,q) in [-1,1] x [-1,1]`은 **Base Sphere 위의 surface parameter**다(픽셀
좌표가 아니다). Base sphere 중심 C 기준 local 방향:

    n0 = (P0 - C) / R           (P0 = base ray와 base sphere의 교차점)
    theta = atan2(n0_x, n0_z)   (방위각)
    phi   = asin(n0_y)          (고도각)
    p = clip(theta / theta_scale, -1, 1)
    q = clip(phi   / phi_scale,   -1, 1)

`theta_scale`/`phi_scale`은 calibration 시점에 실제 이미지 FOV(모서리 +
변 중점 픽셀들을 base sphere에 투영해 관측되는 최대 |theta|,|phi|에 여유
margin을 곱한 값)로 한 번 계산해 `fitted_params`에 저장한다(재구성 시
동일 convention을 보장하기 위해 - 사용자 스펙 4/8번 "동일 pixel/ray -> 동일
surface parameter가 deterministic해야 한다").

`(p,q) -> (theta,phi) -> n0(p,q)`의 역변환과 analytic derivative
(dn0/dp, dn0/dq)를 닫힌 형태로 계산한다(아래 `_pq_to_direction_and_derivatives`).

## Bicubic B-Spline

Control grid `grid[r,c]`(각 노드 1개 스칼라, surface normal 방향
displacement)는 그대로 유지한다(사용자 스펙 2번 - 3-DoF로 확장하지
않는다). 이 control coefficient들을 **tensor-product B-Spline(차수
3, clamped uniform knot vector, domain [-1,1])**의 계수로 직접 사용한다
(사용자 스펙 5번 - "control value를 interpolation data point와 혼동하지
마라": `scipy.interpolate.RectBivariateSpline`처럼 데이터를 스무딩
피팅하는 게 아니라, `scipy.interpolate.BSpline(knots, coefficients,
degree)`로 계수를 직접 스플라인 기저와 결합한다). Tensor-product 평가는
"행별로 p방향 스플라인을 계산해 q방향 계수로 접는" 2단계로 구현한다
(`_tensor_bspline_eval`) - `BSpline`의 벡터 계수(coefficients에 trailing
dimension을 붙이면 여러 채널을 한 번에 평가하는 기능)를 이용해 row 개수만큼
개별 스플라인 객체를 만드는 대신 2개의 BSpline 객체만으로 tensor product
전체와 analytic gradient를 함께 얻는다. Degree=3(bicubic)이므로 grid는
최소 4x4 이상이어야 한다(사용자 스펙 6번, `MIN_SPLINE_GRID_SIZE`).

## Inner Surface

    S(p,q)  = C + (R + Delta_s(p,q)) * n0(p,q)
    dS/dp   = dDelta_s/dp * n0(p,q) + (R+Delta_s(p,q)) * dn0/dp
    dS/dq   = dDelta_s/dq * n0(p,q) + (R+Delta_s(p,q)) * dn0/dq
    N(p,q)  = normalize(dS/dp x dS/dq)   (바깥쪽으로 향하도록 부호 정렬)

`_evaluate_inner_surface_at_pq`가 이 계산의 유일한 지점(single source of
truth, 사용자 스펙 11번)이다 - Delta_s의 analytic derivative(B-spline
자체 미분)와 n0의 analytic derivative(닫힌 형태 삼각함수 미분)를 결합해
**analytic normal**을 쓴다(사용자 스펙 13번 "가능하면 analytic derivative
우선" - finite difference로 낮추지 않는다).

## Camera Ray <-> Inner Surface Intersection

일반적인 parametric surface라 closed-form intersection이 없다(사용자
스펙 14번). Unknown `(t,p,q)` 3개에 대해:

    F(t,p,q) = O + t*D - S(p,q)  (3-vector)

를 `scipy.optimize.least_squares(method="trf")` + bounds(`t>0`,
`p,q in [-1,1]`)로 local solve한다. 초기값은 base sphere 교차점에서 얻은
`(t0,p0,q0)`을 그대로 쓴다(강한 initial guess, 사용자 스펙 15번 - 전체
surface search가 아니다). 성공 판정은 optimizer 성공 플래그뿐 아니라
finite/`t>0`/`p,q` 유효 범위/residual norm 허용오차/유효 normal을 모두
확인한다(사용자 스펙 17번, `_intersect_ray_with_surface`).

## Outer Surface (local-normal thin-shell 근사)

    S_outer(p,q) = S_inner(p,q) + t_g * N_inner(p,q)
    N_outer(p,q) ≈ N_inner(p,q)   (fallback 근사 - 아래 설명)

정확한 offset-surface normal(진짜 S_outer의 접선 미분)은 표면의 곡률
(2차 미분/shape operator)까지 필요해 상당히 복잡하다 - 이번 라운드는
사용자 스펙 21번이 명시적으로 허용한 fallback(`N_outer ≈ N_inner`)을
쓴다. **금지된 것**은 이전 구현의 `normalize(p_outer - sphere_center)`
(순수 반경 방향 normal)이다 - 이건 국소 변형을 완전히 무시하므로 절대
쓰지 않는다. 굴절된 광선(`d_glass`, inner 광선과 다른 방향)이 이 outer
offset surface와 만나는 지점도 같은 3-unknown solve로 다시 찾는다(사용자
스펙 22번) - inner의 `(p,q)`를 초기값으로 재사용한다.

## Repeated Hold-out = Outer Train only (이번 라운드에 고친 두 번째 버그)

`run_spline_calibration_with_diagnostics()`가 이전에는 `windshield_dataset`
(전체 데이터셋)을 그대로 `run_repeated_holdout_spline()`에 넘겨서, 내부
`split_train_test()`가 Outer Test 프레임을 다시 골라갈 수 있는 leakage가
있었다(Residual Grid/RBF의 동일 오케스트레이터는 이미 `outer_train_dataset
= Dataset(frames=_subset_frames(windshield_dataset, train_ids))`로 이
문제를 막고 있었는데 Spline만 빠져 있었다). 이번 라운드에 Grid/RBF와 동일한
패턴으로 고쳤다.

Base Sphere는 이 모듈의 어떤 함수도 재추정하지 않는다 - 항상 Outer Train만
사용한 calibrate_spherical() 결과를 그대로 얼려서(frozen) 쓴다. Base K,D는
물론 절대 재최적화하지 않는다.

STAGE A(spline만, pose 고정) / STAGE B(spline+pose alternating, ray-domain)
패턴, regularization(magnitude/smoothness/curvature), Repeated Hold-out,
AUTO grid selection, Ray/Surface Stability, 진단 오케스트레이터 구조 자체는
이전 라운드와 동일하게 유지한다 - 이번 라운드는 물리 surface 정의만
바꾼다.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from calibration.models.common import MIN_FRAMES_REQUIRED, infer_image_size
from calibration.types import CameraConfig, CameraModelType, Dataset, Frame
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.refraction import intersect_ray_sphere, refract_ray
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
    populate_pose_diagnostics,
    populate_repeated_holdout_diagnostics,
)
from calibration.windshield.spherical import calibrate_spherical, is_valid_spherical_windshield

# ---------------------------------------------------------------------------
# 고정 상수 - spline_hint로 덮어쓸 수 있고, 코드 여러 곳에 hard-code하지 않는다.
# ---------------------------------------------------------------------------
SPLINE_DEGREE = 3  # Bicubic (사용자 스펙 6번)
MIN_SPLINE_GRID_SIZE = SPLINE_DEGREE + 1  # 최소 4x4 - degree=3 clamped B-spline의 최소 control point 수

DEFAULT_SPLINE_ROWS = 4
DEFAULT_SPLINE_COLS = 6
DEFAULT_LAMBDA_MAG = 1e-2
DEFAULT_LAMBDA_SMOOTH = 1e-1
DEFAULT_LAMBDA_CURVE = 1e-1
# ±10mm - 실제 windshield 국부 변형 스케일보다 넉넉하되, surface folding이
# 일어날 만큼 크지는 않은 값(사용자 스펙 15/31번 UI mockup 기본값과 동일).
DEFAULT_MAX_DISPLACEMENT_M = 0.010

# 4x4 미만 후보는 bicubic degree를 지원하지 못하므로 전부 제거했다(사용자
# 스펙 6번 "2x2 같은 경우는 테스트 fixture를 4x4 이상으로 수정한다").
SPLINE_GRID_CANDIDATES: list[tuple[int, int]] = [(4, 4), (4, 6), (6, 8)]
SPLINE_SELECTION_TIE_TOLERANCE = 0.05
SPLINE_STAGE_B_NUM_ROUNDS = 2

MIN_CORNERS_PER_NODE = 3
MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10

# Camera ray <-> parametric surface intersection 허용 오차(미터) - 이
# 좌표계는 windshield가 카메라에서 수mm~수cm 거리에 있으므로 상당히 타이트한
# 절대값이어도 무리 없다(사용자 스펙 17번).
_INTERSECTION_TOL_M = 1e-6
_RAY_PENALTY = 5.0
_ORIGIN = np.zeros(3, dtype=np.float64)

# 실제 이미지 FOV로 theta/phi normalization scale을 구할 때, 관측된 최대
# 각도에 곱하는 여유 margin - 경계 픽셀이 정확히 |p|=1/|q|=1에 딱 붙지
# 않도록 약간의 여유를 둔다.
_FOV_SCALE_MARGIN = 1.15
_MIN_ANGULAR_SCALE_RAD = 1e-6


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


# ---------------------------------------------------------------------------
# Bicubic tensor-product B-Spline (control coefficients -> Delta_s(p,q))
# ---------------------------------------------------------------------------

def _clamped_uniform_knots(n_control: int, degree: int, domain: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Clamped(양 끝 고정) uniform open knot vector - domain 양 끝에서
    degree+1번 knot이 반복되고, 안쪽은 균등 간격이다(사용자 스펙 7번).
    길이는 n_control + degree + 1(scipy.interpolate.BSpline의 요구사항과
    동일)."""
    if n_control < degree + 1:
        raise ValueError(f"n_control must be >= degree+1 (got n_control={n_control}, degree={degree}).")
    a, b = domain
    num_interior = n_control - degree - 1
    interior = np.linspace(a, b, num_interior + 2)[1:-1]
    return np.concatenate([np.full(degree + 1, a), interior, np.full(degree + 1, b)])


@dataclass
class _SplineBasis:
    """(rows,cols,degree) 조합 하나에 대한 재사용 가능한 B-spline basis
    evaluator - knot vector에만 의존하고 control 값(grid)에는 의존하지
    않으므로, grid 값이 바뀌는 매 optimizer iteration/코너마다 다시 만들
    필요가 전혀 없다.

    성능이 핵심인 이유: `evaluate_inner_surface()`가 ray-surface
    intersection(3-unknown least_squares) 안에서, 그리고 그 intersection이
    다시 STAGE A/B의 outer least_squares(코너마다 반복) 안에서 반복
    호출되므로, "매 호출마다 새 BSpline 객체를 만드는" 방식은 실측으로
    확인된 심각한 성능 문제(단일 calibrate_spline 호출이 5분을 넘겨도
    끝나지 않음)를 일으켰다. `BSpline(knots, identity_matrix, degree)`를
    **한 번만** 만들어두면, 그 객체를 서로 다른 p/q에서 반복 평가하는 것은
    순수 컴파일된 de Boor 평가 호출일 뿐이라 매우 빠르다."""
    col_spline: BSpline   # cols개 basis function, identity 계수
    col_deriv: BSpline
    row_spline: BSpline   # rows개 basis function, identity 계수
    row_deriv: BSpline
    degree: int


def _build_spline_basis(rows: int, cols: int, degree: int) -> _SplineBasis:
    t_row = _clamped_uniform_knots(rows, degree)
    t_col = _clamped_uniform_knots(cols, degree)
    col_spline = BSpline(t_col, np.eye(cols), degree)
    row_spline = BSpline(t_row, np.eye(rows), degree)
    return _SplineBasis(
        col_spline=col_spline, col_deriv=col_spline.derivative(),
        row_spline=row_spline, row_deriv=row_spline.derivative(),
        degree=degree,
    )


def _tensor_bspline_eval(grid: np.ndarray, p: float, q: float, basis: _SplineBasis) -> tuple[float, float, float]:
    """Tensor-product B-spline 평가: 값 + p/q 방향 analytic 미분을 한 번에
    반환한다. `grid`는 (rows, cols) control coefficient 행렬이다.

    구현: 미리 만들어둔 basis function 값 벡터(`basis.col_spline(p)`가
    p에서의 cols개 basis function 값을 한 번에 준다, `basis.row_spline(q)`도
    동일)를 얻은 뒤 `grid`와의 행렬곱만으로 tensor product 값/미분을
    계산한다(사용자 스펙 5/8번 - control coefficient와 B-spline basis의
    결합) - 매 호출마다 BSpline 객체를 새로 만들지 않는다."""
    basis_p = basis.col_spline(p)      # (cols,)
    dbasis_p = basis.col_deriv(p)      # (cols,)
    basis_q = basis.row_spline(q)      # (rows,)
    dbasis_q = basis.row_deriv(q)      # (rows,)

    value = float(basis_q @ grid @ basis_p)
    d_value_dp = float(basis_q @ grid @ dbasis_p)
    d_value_dq = float(dbasis_q @ grid @ basis_p)
    return value, d_value_dp, d_value_dq


# ---------------------------------------------------------------------------
# Base Sphere 각도 parameterization (p,q) <-> 3D 방향
# ---------------------------------------------------------------------------

def compute_angular_fov_scale(
    baseline_model: BaselineWindshieldModel, center: np.ndarray, radius: float,
    image_width: float, image_height: float, margin: float = _FOV_SCALE_MARGIN,
) -> tuple[float, float]:
    """실제 calibration에 쓰이는 이미지 FOV에서 관측되는 최대 |theta|,|phi|
    (base sphere 각도 좌표)를 측정해 (p,q) in [-1,1] 정규화 scale을 정한다
    (사용자 스펙 4번 "실제 FOV 범위를 [-1,1]x[-1,1]로 normalize"). 이미지
    모서리/변 중점/중심 픽셀을 base sphere(변형 전, 반지름 R)에 투영해서
    구한다 - calibration 시점에 한 번만 계산하고 fitted_params에 저장해서
    (spline_theta_scale_rad/spline_phi_scale_rad) runtime 재구성 시 동일
    convention을 보장한다."""
    w, h = image_width, image_height
    sample_pixels = [
        (0.0, 0.0), (w, 0.0), (0.0, h), (w, h),
        (w / 2.0, 0.0), (w / 2.0, h), (0.0, h / 2.0), (w, h / 2.0),
        (w / 2.0, h / 2.0),
    ]
    max_theta, max_phi = 0.0, 0.0
    for u, v in sample_pixels:
        d = np.asarray(baseline_model.unproject_pixel(u, v), dtype=np.float64)
        hit = intersect_ray_sphere(_ORIGIN, d, center, radius)
        if hit is None:
            continue
        n0 = (hit[0] - center) / radius
        phi = math.asin(float(np.clip(n0[1], -1.0, 1.0)))
        theta = math.atan2(float(n0[0]), float(n0[2]))
        max_theta = max(max_theta, abs(theta))
        max_phi = max(max_phi, abs(phi))
    theta_scale = max(max_theta * margin, _MIN_ANGULAR_SCALE_RAD)
    phi_scale = max(max_phi * margin, _MIN_ANGULAR_SCALE_RAD)
    return theta_scale, phi_scale


def _direction_to_pq(n0: np.ndarray, theta_scale: float, phi_scale: float) -> tuple[float, float]:
    phi = math.asin(float(np.clip(n0[1], -1.0, 1.0)))
    theta = math.atan2(float(n0[0]), float(n0[2]))
    p = float(np.clip(theta / theta_scale, -1.0, 1.0))
    q = float(np.clip(phi / phi_scale, -1.0, 1.0))
    return p, q


def _pq_to_direction_and_derivatives(
    p: float, q: float, theta_scale: float, phi_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p,q) -> base sphere 위 방향 n0(p,q) + analytic 미분(dn0/dp, dn0/dq).
    n0 = (cos(phi)*sin(theta), sin(phi), cos(phi)*cos(theta))
    (atan2(x,z) 방위각 convention과 asin(y) 고도각 convention에 대응)."""
    theta = p * theta_scale
    phi = q * phi_scale
    cphi, sphi = math.cos(phi), math.sin(phi)
    ctheta, stheta = math.cos(theta), math.sin(theta)

    n0 = np.array([cphi * stheta, sphi, cphi * ctheta])
    dn0_dtheta = np.array([cphi * ctheta, 0.0, -cphi * stheta])
    dn0_dphi = np.array([-sphi * stheta, cphi, -sphi * ctheta])

    dn0_dp = dn0_dtheta * theta_scale
    dn0_dq = dn0_dphi * phi_scale
    return n0, dn0_dp, dn0_dq


# ---------------------------------------------------------------------------
# Surface Evaluator - single source of truth (사용자 스펙 11번)
# ---------------------------------------------------------------------------

@dataclass
class SurfaceEvaluation:
    point: np.ndarray
    normal: np.ndarray
    dS_dp: np.ndarray
    dS_dq: np.ndarray


def evaluate_inner_surface(
    p: float, q: float, center: np.ndarray, radius: float, spline_grid: np.ndarray,
    theta_scale: float, phi_scale: float, basis: _SplineBasis,
) -> Optional[SurfaceEvaluation]:
    """S(p,q) = C + (R + Delta_s(p,q)) * n0(p,q). Analytic normal:
    N = normalize(dS/dp x dS/dq). 접선이 퇴화(fold/flat spot)되면 None."""
    n0, dn0_dp, dn0_dq = _pq_to_direction_and_derivatives(p, q, theta_scale, phi_scale)
    ds, dds_dp, dds_dq = _tensor_bspline_eval(spline_grid, p, q, basis)
    if not np.isfinite(ds) or not np.isfinite(dds_dp) or not np.isfinite(dds_dq):
        return None

    local_radius = radius + ds
    point = center + local_radius * n0
    dS_dp = dds_dp * n0 + local_radius * dn0_dp
    dS_dq = dds_dq * n0 + local_radius * dn0_dq

    normal_raw = np.cross(dS_dp, dS_dq)
    norm = float(np.linalg.norm(normal_raw))
    tu_norm, tv_norm = float(np.linalg.norm(dS_dp)), float(np.linalg.norm(dS_dq))
    if tu_norm < 1e-15 or tv_norm < 1e-15:
        return None
    sin_angle = norm / (tu_norm * tv_norm)
    if not np.isfinite(sin_angle) or sin_angle < 1e-6:
        return None  # 퇴화된 접선(surface folding/flat spot) - 사용자 스펙 16/27번
    normal = normal_raw / norm
    if np.dot(normal, point - center) < 0.0:
        normal = -normal  # 항상 바깥쪽(카메라 쪽)을 향하도록
    return SurfaceEvaluation(point=point, normal=normal, dS_dp=dS_dp, dS_dq=dS_dq)


def evaluate_outer_surface(inner: SurfaceEvaluation, thickness: float) -> SurfaceEvaluation:
    """Outer Surface(local-normal thin-shell 근사, 사용자 스펙 19번):

        S_outer(p,q) = S_inner(p,q) + t_g * N_inner(p,q)
        N_outer(p,q) ~= N_inner(p,q)   (fallback, 사용자 스펙 21번 - 정확한
            offset-surface normal은 표면 곡률(2차 미분)까지 필요해 복잡하므로
            첫 구현에서는 명시적으로 이 근사를 쓴다. 절대 쓰지 않는 것은
            "normalize(point - sphere_center)" 같은 순수 반경 방향 normal
            이다(사용자 스펙 20/38번 금지 사항)."""
    outer_point = inner.point + thickness * inner.normal
    return SurfaceEvaluation(point=outer_point, normal=inner.normal, dS_dp=inner.dS_dp, dS_dq=inner.dS_dq)


# ---------------------------------------------------------------------------
# Camera Ray <-> Parametric Surface Intersection (사용자 스펙 14-17번)
# ---------------------------------------------------------------------------

def _intersect_ray_with_surface(
    origin: np.ndarray, direction: np.ndarray,
    surface_fn: Callable[[float, float], Optional[SurfaceEvaluation]],
    initial_t: float, initial_p: float, initial_q: float,
    t_upper_bound: float,
) -> Optional[tuple[float, float, float, SurfaceEvaluation]]:
    """일반적인 parametric surface에는 closed-form intersection이 없다 -
    unknown (t,p,q) 3개에 대한 residual F(t,p,q) = O+tD-S(p,q)를 local
    least_squares로 푼다(사용자 스펙 14번). 초기값은 base sphere 교차점에서
    구한 강한 initial guess를 그대로 쓴다(전체 surface search가 아니다,
    사용자 스펙 15/18번).

    성공 판정(사용자 스펙 17번): optimizer 성공 플래그뿐 아니라 finite,
    t>0, p/q가 유효 범위 안, residual norm이 허용오차 이내, 그리고 surface_fn
    이 유효한 normal을 반환했는지(퇴화 아님)까지 전부 확인한다.

    성능: analytic Jacobian을 제공한다(사용자 스펙 13번 "가능하면 analytic
    derivative 우선"과도 일치) - `dF/dt=D`, `dF/dp=-dS/dp`, `dF/dq=-dS/dq`는
    `surface_fn`이 이미 계산해 둔 `dS_dp`/`dS_dq`에서 바로 나온다. 이걸
    안 주면 scipy가 3-unknown마다 유한차분으로 근사하면서 매 iteration마다
    surface_fn을 4번씩 호출하는데(값 1번 + 변수 3개 perturbation 3번),
    이 solve가 STAGE A/B의 코너마다/pose refine마다 반복 호출되므로
    실측으로 확인된 심각한 성능 문제(analytic Jacobian 없이 단일
    calibrate_spline() 호출이 5분을 넘겨도 끝나지 않음)의 핵심 원인이었다.
    `fun`과 `jac`가 같은 x에서 연속으로 호출되는 scipy의 표준 동작을
    이용해 마지막 surface_fn 결과를 캐싱, `jac`가 다시 계산하지 않게 한다."""
    lower = np.array([1e-9, -1.0, -1.0])
    upper = np.array([t_upper_bound, 1.0, 1.0])
    x0 = np.clip(np.array([initial_t, initial_p, initial_q]), lower, upper)

    cache: dict[tuple[float, float], Optional[SurfaceEvaluation]] = {}

    def _cached_eval(p: float, q: float) -> Optional[SurfaceEvaluation]:
        key = (p, q)
        if key not in cache:
            cache.clear()  # 마지막 하나만 유지 - fun/jac가 같은 x를 바로 이어서 부르는 패턴에 최적화
            cache[key] = surface_fn(p, q)
        return cache[key]

    def residual(x: np.ndarray) -> np.ndarray:
        t, p, q = float(x[0]), float(x[1]), float(x[2])
        ev = _cached_eval(p, q)
        if ev is None:
            return np.full(3, _RAY_PENALTY)
        ray_point = origin + t * direction
        return ray_point - ev.point

    def jac(x: np.ndarray) -> np.ndarray:
        t, p, q = float(x[0]), float(x[1]), float(x[2])
        ev = _cached_eval(p, q)
        if ev is None:
            return np.eye(3)  # residual이 이미 penalty이므로 값 자체는 중요하지 않음(유한하기만 하면 됨)
        return np.column_stack([direction, -ev.dS_dp, -ev.dS_dq])

    result = least_squares(
        residual, x0=x0, jac=jac, bounds=(lower, upper), method="trf", xtol=1e-10, ftol=1e-12, max_nfev=50,
    )
    if not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.cost):
        return None
    t, p, q = float(result.x[0]), float(result.x[1]), float(result.x[2])
    if t <= 0.0 or abs(p) > 1.0 + 1e-6 or abs(q) > 1.0 + 1e-6:
        return None
    if float(np.linalg.norm(result.fun)) > _INTERSECTION_TOL_M:
        return None
    ev = surface_fn(p, q)
    if ev is None:
        return None
    return t, p, q, ev


def _initial_guess_from_base_sphere(
    u: float, v: float, baseline_model: BaselineWindshieldModel, center: np.ndarray, radius: float,
    theta_scale: float, phi_scale: float,
) -> Optional[tuple[np.ndarray, float, float, float]]:
    """(u,v) 픽셀의 base ray가 변형 전 base sphere와 만나는 점에서 강한
    초기값 (t0,p0,q0)을 구한다(사용자 스펙 15번)."""
    d_cam = np.asarray(baseline_model.unproject_pixel(u, v), dtype=np.float64)
    hit = intersect_ray_sphere(_ORIGIN, d_cam, center, radius)
    if hit is None:
        return None
    point0, t0 = hit
    n0 = (point0 - center) / radius
    p0, q0 = _direction_to_pq(n0, theta_scale, phi_scale)
    return d_cam, float(t0), p0, q0


def _refract_through_spline_shell(
    u: float, v: float, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, theta_scale: float, phi_scale: float, basis: _SplineBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """(u,v) 픽셀의 base ray가 실제 inner spline surface -> 유리 -> outer
    offset surface(근사)를 통과한 뒤의 (exit point, exit direction)을
    반환한다(사용자 스펙 23번 Optical Chain). 실패(교차 없음/전반사/퇴화
    normal)하면 ValueError."""
    guess = _initial_guess_from_base_sphere(u, v, baseline_model, center, radius, theta_scale, phi_scale)
    if guess is None:
        raise ValueError("Base sphere intersection (initial guess) failed.")
    d_cam, t0, p0, q0 = guess
    t_bound = max(t0 * 5.0, radius * 4.0)

    def inner_fn(p: float, q: float) -> Optional[SurfaceEvaluation]:
        return evaluate_inner_surface(p, q, center, radius, spline_grid, theta_scale, phi_scale, basis)

    inner_hit = _intersect_ray_with_surface(_ORIGIN, d_cam, inner_fn, t0, p0, q0, t_bound)
    if inner_hit is None:
        raise ValueError("Inner spline surface intersection failed.")
    _t1, p1, q1, inner_ev = inner_hit

    d_glass = refract_ray(d_cam, inner_ev.normal, n_air, n_glass)
    if d_glass is None:
        raise ValueError("Total internal reflection at the inner spline surface.")

    def outer_fn(p: float, q: float) -> Optional[SurfaceEvaluation]:
        inner = inner_fn(p, q)
        if inner is None:
            return None
        return evaluate_outer_surface(inner, thickness)

    cos_incidence = max(0.2, abs(float(np.dot(d_glass, inner_ev.normal))))
    tau0 = thickness / cos_incidence
    outer_hit = _intersect_ray_with_surface(inner_ev.point, d_glass, outer_fn, tau0, p1, q1, max(tau0 * 5.0, thickness * 20.0))
    if outer_hit is None:
        raise ValueError("Outer offset surface intersection failed.")
    _t2, _p2, _q2, outer_ev = outer_hit

    d_out = refract_ray(d_glass, outer_ev.normal, n_glass, n_air)
    if d_out is None:
        raise ValueError("Total internal reflection at the outer offset surface.")

    return outer_ev.point, d_out


# ---------------------------------------------------------------------------
# Surface validity (사용자 스펙 15/16/27번)
# ---------------------------------------------------------------------------

def _check_normal_continuity(
    center: np.ndarray, radius: float, spline_grid: np.ndarray, theta_scale: float, phi_scale: float,
    basis: _SplineBasis, lattice: int = 6,
) -> bool:
    """고정된 (p,q) lattice에서 인접 normal이 갑자기 거의 반대 방향으로
    튀지 않는지 확인한다(사용자 스펙 27/37번 - continuity 검사의 최소
    형태). 인접 normal의 내적이 0 이하(각도 90도 이상)면 급격한 fold로
    보고 invalid 처리한다."""
    coords = np.linspace(-0.9, 0.9, lattice)
    normals = np.full((lattice, lattice, 3), np.nan)
    for i, p in enumerate(coords):
        for j, q in enumerate(coords):
            ev = evaluate_inner_surface(p, q, center, radius, spline_grid, theta_scale, phi_scale, basis)
            if ev is None:
                return False
            normals[i, j] = ev.normal
    for i in range(lattice):
        for j in range(lattice):
            if i + 1 < lattice and np.dot(normals[i, j], normals[i + 1, j]) <= 0.0:
                return False
            if j + 1 < lattice and np.dot(normals[i, j], normals[i, j + 1]) <= 0.0:
                return False
    return True


def is_valid_spline_shell(
    center: np.ndarray, base_radius: float, spline_grid: np.ndarray, thickness: float,
    theta_scale: Optional[float] = None, phi_scale: Optional[float] = None, degree: int = SPLINE_DEGREE,
    check_normal_continuity: bool = True,
) -> bool:
    """base sphere 자체의 물리적 유효성(is_valid_spherical_windshield)에
    더해:
      1. control 값이 전부 finite한지
      2. 가장 안쪽으로 파고든(worst-case) deformation을 적용해도 카메라가
         안전 마진을 두고 있는지(근사적 필요조건 - S(p,q)는 항상 정확히
         center에서 거리 R+Delta_s(p,q)에 있으므로, 이 worst-case 반경
         검사는 여전히 유효한 하한 검사다)
      3. (theta_scale/phi_scale이 주어지면) 고정 (p,q) lattice에서 normal
         continuity가 깨지지 않는지(사용자 스펙 27번)
    를 확인한다."""
    if not is_valid_spherical_windshield(center, base_radius, thickness):
        return False
    if not np.all(np.isfinite(spline_grid)):
        return False
    worst_case_radius = base_radius + float(np.min(spline_grid))
    if not is_valid_spherical_windshield(center, worst_case_radius, thickness):
        return False
    if check_normal_continuity and theta_scale is not None and phi_scale is not None:
        rows, cols = spline_grid.shape
        basis = _build_spline_basis(rows, cols, degree)
        if not _check_normal_continuity(center, base_radius, spline_grid, theta_scale, phi_scale, basis):
            return False
    return True


# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

class SplineWindshieldModel(WindshieldModel):
    """Base Sphere(고정) + Bicubic B-Spline normal-offset local deformation
    으로 Windshield를 근사하고 실제 Snell 굴절을 계산하는 모델.
    project_point()/unproject_pixel() 모두 실제 parametric surface
    교차 + analytic normal + 굴절을 거친다."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        sphere_center: np.ndarray,
        sphere_radius: float,
        spline_grid: np.ndarray,   # (rows, cols) - Delta_s control coefficients, meters
        theta_scale: float,
        phi_scale: float,
        n_air: float = 1.0,
        n_glass: float = 1.52,
        glass_thickness_m: float = 0.005,
        degree: int = SPLINE_DEGREE,
    ):
        self._center = np.asarray(sphere_center, dtype=np.float64)
        self._radius = float(sphere_radius)
        self._grid = np.asarray(spline_grid, dtype=np.float64)
        if self._grid.ndim != 2:
            raise ValueError(f"Spline control grid must have shape (rows, cols), got {self._grid.shape}.")
        rows, cols = self._grid.shape
        if rows < degree + 1 or cols < degree + 1:
            raise ValueError(
                f"Spline control grid must be at least {degree + 1}x{degree + 1} for degree={degree}, "
                f"got {rows}x{cols}."
            )
        self._degree = degree
        self._basis = _build_spline_basis(rows, cols, degree)
        self._theta_scale = float(theta_scale)
        self._phi_scale = float(phi_scale)
        self._n_air = float(n_air)
        self._n_glass = float(n_glass)
        self._thickness = float(glass_thickness_m)
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def local_displacement_m(self, u: float, v: float) -> Optional[float]:
        """이 픽셀 방향의 surface normal 방향 displacement(미터) - 진단용
        Surface Stability 계산 전용 public accessor. base ray를 base
        sphere에 교차시켜 (p,q)를 얻은 뒤 B-spline 값을 직접 평가한다(런타임
        project_point/unproject_pixel 경로와는 무관 - 그쪽은 deformed
        surface 교차점 자체를 다시 푼다)."""
        guess = _initial_guess_from_base_sphere(u, v, self._baseline, self._center, self._radius, self._theta_scale, self._phi_scale)
        if guess is None:
            return None
        _d, _t0, p0, q0 = guess
        ds, _dp, _dq = _tensor_bspline_eval(self._grid, p0, q0, self._basis)
        return ds

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        """픽셀 -> 굴절 반영 외부 광선 방향(단위 벡터). Spherical과 동일한
        근사: 실제로는 광선의 원점이 windshield 바깥 exit point로 옮겨가지만,
        WindshieldModel ABC 반환 형태가 방향뿐이라 방향만 보고한다
        (non-central ray, 내부 optical computation에서는 exit point를
        버리지 않는다 - project_point()가 그것을 실제로 사용한다, 사용자
        스펙 24번)."""
        _, d_out = _refract_through_spline_shell(
            u, v, self._baseline, self._center, self._radius, self._thickness,
            self._n_air, self._n_glass, self._grid, self._theta_scale, self._phi_scale, self._basis,
        )
        return float(d_out[0]), float(d_out[1]), float(d_out[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다 - Base K,D 투영을
        초기값으로 삼아 작은 2변수 root-solve로 푼다(사용자 스펙 25번,
        Spherical/Grid/RBF와 동일한 구조). 내부에서 unproject through real
        spline optical surface(exit point + exterior direction)를 그대로
        사용한다."""
        target = np.array([x, y, z], dtype=np.float64)
        initial_uv = np.asarray(self._baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            try:
                point, direction = _refract_through_spline_shell(
                    float(uv[0]), float(uv[1]), self._baseline, self._center, self._radius, self._thickness,
                    self._n_air, self._n_glass, self._grid, self._theta_scale, self._phi_scale, self._basis,
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
            point, direction = _refract_through_spline_shell(
                u, v, self._baseline, self._center, self._radius, self._thickness,
                self._n_air, self._n_glass, self._grid, self._theta_scale, self._phi_scale, self._basis,
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
    BSpline 객체를 직렬화하지 않는다, 재구성 가능한 public 값만 쓴다
    (sphere_center_x/y/z, sphere_radius, glass_refractive_index,
    air_refractive_index, glass_thickness_m, spline_rows, spline_cols,
    spline_ds_{r}_{c}, spline_degree, spline_theta_scale_rad,
    spline_phi_scale_rad)."""
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
        theta_scale=fp["spline_theta_scale_rad"],
        phi_scale=fp["spline_phi_scale_rad"],
        n_air=fp.get("air_refractive_index", 1.0),
        n_glass=fp.get("glass_refractive_index", 1.52),
        glass_thickness_m=fp.get("glass_thickness_m", 0.005),
        degree=int(fp.get("spline_degree", SPLINE_DEGREE)),
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
    if rows < MIN_SPLINE_GRID_SIZE or cols < MIN_SPLINE_GRID_SIZE:
        raise ValueError(
            f"spline_hint spline_rows/spline_cols must each be >= {MIN_SPLINE_GRID_SIZE} "
            f"(bicubic B-spline degree={SPLINE_DEGREE}); got rows={rows}, cols={cols}."
        )
    if max_displacement <= 0.0:
        raise ValueError(f"spline_hint max_displacement_m must be > 0 (got {max_displacement}).")
    return rows, cols, lambda_mag, lambda_smooth, lambda_curve, max_displacement


def _ray_alignment_residual_spline(
    u: float, v: float, p_cam: np.ndarray, baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, theta_scale: float, phi_scale: float, basis: _SplineBasis,
) -> np.ndarray:
    """한 코너의 (관측 픽셀, 목표점)에 대한 ray-alignment residual(3-vector).
    STAGE A의 초기 spline fit과 STAGE B의 spline/pose 공동 refinement가
    전부 이 함수 하나를 재사용한다. 실패(교차 없음/전반사/퇴화 normal)하면
    코너 하나만 penalty 처리한다(다른 코너에 영향 주지 않는 로컬 실패)."""
    try:
        point, direction = _refract_through_spline_shell(
            u, v, baseline_model, center, base_radius, thickness, n_air, n_glass,
            spline_grid, theta_scale, phi_scale, basis,
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
    rows: int, cols: int, theta_scale: float, phi_scale: float, degree: int,
    lambda_mag: float, lambda_smooth: float, lambda_curve: float, max_displacement: float,
):
    """ray-alignment residual + magnitude/smoothness/curvature
    regularization으로 spline grid 전체를 한 번에 피팅한다(사용자 스펙
    26번 - 기존 3중 regularization 유지).

        L_mag    = lambda_mag    * sum(Delta_s_i^2)
        L_smooth = lambda_smooth * sum((Delta_s_i - Delta_s_j)^2)   (인접 쌍)
        L_curve  = lambda_curve  * sum((Delta_s_{i-1} - 2*Delta_s_i + Delta_s_{i+1})^2)  (행/열 각각)
    """
    n_points = len(observed_pixels)
    basis = _build_spline_basis(rows, cols, degree)

    horizontal_pairs = [(r, c, r, c + 1) for r in range(rows) for c in range(cols - 1)]
    vertical_pairs = [(r, c, r + 1, c) for r in range(rows - 1) for c in range(cols)]
    smooth_pairs = horizontal_pairs + vertical_pairs

    horizontal_triples = [(r, c - 1, r, c, r, c + 1) for r in range(rows) for c in range(1, cols - 1)]
    vertical_triples = [(r - 1, c, r, c, r + 1, c) for r in range(1, rows - 1) for c in range(cols)]
    curve_triples = horizontal_triples + vertical_triples

    def residual(params: np.ndarray) -> np.ndarray:
        grid = params.reshape(rows, cols)

        data_res = np.empty((n_points, 3))
        for i in range(n_points):
            u, v = observed_pixels[i]
            data_res[i] = _ray_alignment_residual_spline(
                float(u), float(v), p_cam[i], baseline_model, center, base_radius, thickness, n_air, n_glass,
                grid, theta_scale, phi_scale, basis,
            )

        mag_res = math.sqrt(lambda_mag) * grid.ravel()

        smooth_res = np.empty(len(smooth_pairs))
        for i, (r0, c0, r1, c1) in enumerate(smooth_pairs):
            smooth_res[i] = math.sqrt(lambda_smooth) * (grid[r0, c0] - grid[r1, c1])

        curve_res = np.empty(len(curve_triples))
        for i, (r0, c0, r1, c1, r2, c2) in enumerate(curve_triples):
            curve_res[i] = math.sqrt(lambda_curve) * (grid[r0, c0] - 2.0 * grid[r1, c1] + grid[r2, c2])

        return np.concatenate([data_res.ravel(), mag_res, smooth_res, curve_res])

    x0 = np.zeros(rows * cols)  # Delta_s initial = 0
    bounds = (np.full_like(x0, -max_displacement), np.full_like(x0, max_displacement))
    return least_squares(residual, x0=x0, bounds=bounds, method="trf", loss="soft_l1", f_scale=0.05)


def _refine_frame_pose_ray_domain_spline(
    frame: Frame,
    observed_pixels: np.ndarray,
    baseline_model: BaselineWindshieldModel,
    center: np.ndarray, base_radius: float, thickness: float, n_air: float, n_glass: float,
    spline_grid: np.ndarray, theta_scale: float, phi_scale: float, basis: _SplineBasis,
    initial_rvec: np.ndarray, initial_tvec: np.ndarray,
    *, regularize: bool = True,
):
    """한 프레임의 pose(rvec,tvec)만 ray-alignment residual로 refine한다.
    Sphere/Spline/K,D는 고정(호출부가 넘긴 값 그대로). Spline은 표면 교차
    -> normal -> Snell 굴절 전체를 거치는 non-additive 모델이라
    residual_common.refine_frame_pose_ray_domain(Grid/RBF 전용, "d_base+delta"
    덧셈 모델 가정)을 재사용할 수 없다 - spherical.py의
    refine_frame_pose_ray_domain과 같은 구조를 재현한다. POSE_ROTATION_REG_
    WEIGHT/POSE_TRANSLATION_REG_WEIGHT는 residual_common에서 가져와 Grid/
    RBF와 동일한 pose prior 정책을 유지한다."""
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
                spline_grid, theta_scale, phi_scale, basis,
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
    rows: int, cols: int, theta_scale: float, phi_scale: float, degree: int,
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
    basis = _build_spline_basis(rows, cols, degree)

    for _ in range(num_rounds):
        for i, frame in enumerate(ok_frames):
            pose_fit = _refine_frame_pose_ray_domain_spline(
                frame, observed_pixels_per_frame[i], baseline_model, center, base_radius, thickness, n_air, n_glass,
                grid, theta_scale, phi_scale, basis, initial_rvecs[i], initial_tvecs[i],
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
            rows, cols, theta_scale, phi_scale, degree, lambda_mag, lambda_smooth, lambda_curve, max_displacement,
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
    실제 이미지 FOV로 (p,q) 정규화 scale 결정 -> (AUTO 지정 시 spline grid
    해상도 선택, train_ids만 사용) -> STAGE A(spline만, pose 고정) -> STAGE
    B(spline+pose joint, ray-domain alternating) -> 두 stage의 실제 pixel
    RMS를 비교해 더 나은 쪽을 최종으로 채택 -> Train 평가 -> Test는 최종
    sphere+spline을 완전히 고정한 채 자기 pose만 별도로 refine한 뒤
    평가(leakage 없음).
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size

    # --- Base Sphere: Outer Train만 사용(leakage 금지) ---
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

    baseline_model = BaselineWindshieldModel(K, D, model)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline_model, center, base_radius, width, height)
    degree = SPLINE_DEGREE

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
        rows, cols, theta_scale, phi_scale, degree, lambda_mag, lambda_smooth, lambda_curve, max_displacement,
    )
    if not stage_a_fit.success or not np.all(np.isfinite(stage_a_fit.x)):
        return _failure_result(config, train_ids, test_ids, "Spline grid optimization(STAGE A)이 수렴하지 않았습니다.")

    stage_a_grid = stage_a_fit.x.reshape(rows, cols).copy()
    stage_a_model = SplineWindshieldModel(K, D, model, center, base_radius, stage_a_grid, theta_scale, phi_scale, n_air, n_glass, thickness, degree)
    stage_a_outcome = evaluate_residual_ray_model(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: spline + per-frame pose joint refinement (ray-domain) ---
    joint = _joint_refine_spline_and_poses(
        ok_frames, observed_pixels_per_frame, baseline_model, center, base_radius, thickness, n_air, n_glass,
        rvecs, tvecs, rows, cols, theta_scale, phi_scale, degree, lambda_mag, lambda_smooth, lambda_curve, max_displacement, stage_a_grid,
    )

    stage_used_is_joint_refined = False
    final_grid = stage_a_grid
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    refinement_note = ""

    if is_valid_spline_shell(center, base_radius, joint.grid, thickness, theta_scale, phi_scale, degree):
        stage_b_model = SplineWindshieldModel(K, D, model, center, base_radius, joint.grid, theta_scale, phi_scale, n_air, n_glass, thickness, degree)
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

    final_model = SplineWindshieldModel(K, D, model, center, base_radius, final_grid, theta_scale, phi_scale, n_air, n_glass, thickness, degree)

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 spline surface로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    runtime_param_count = rows * cols  # 1-DoF/node
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
        "spline_degree": float(degree),
        "spline_theta_scale_rad": float(theta_scale),
        "spline_phi_scale_rad": float(phi_scale),
        "spline_surface_representation_is_bicubic_bspline": 1.0,
        "spline_outer_normal_is_inner_approx": 1.0,  # 항상 1.0 - N_outer≈N_inner fallback을 쓴다는 진단 표식(사용자 스펙 21/40번)
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
                # K/D는 여기서 절대 건드리지 않는다(leakage 없음).
                t_rvecs, t_tvecs = [], []
                t_obs_pixels, _t_d_obs, _t_p_cam = collect_corner_arrays(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, baseline_model
                )
                final_basis = _build_spline_basis(rows, cols, degree)
                for frame, init_rvec, init_tvec, obs_px in zip(t_ok_frames, t_init_rvecs, t_init_tvecs, t_obs_pixels):
                    pose_fit = _refine_frame_pose_ray_domain_spline(
                        frame, obs_px, baseline_model, center, base_radius, thickness, n_air, n_glass,
                        final_grid, theta_scale, phi_scale, final_basis, init_rvec, init_tvec, regularize=True,
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
# Surface Stability - Grid/RBF에는 없는 Spline 전용 지표
# ---------------------------------------------------------------------------

def compute_surface_stability_mm(
    models: list[SplineWindshieldModel], image_width: float, image_height: float,
    sample_rows: int = 12, sample_cols: int = 20,
) -> tuple[Optional[float], Optional[float]]:
    """여러 split에서 얻은 fitted spline surface들이 실제로 얼마나 다른
    displacement(mm)를 만들어내는지 측정한다 - Ray Stability와 같은 고정
    샘플 픽셀(residual_common.fixed_evaluation_pixels)을 재사용해 Grid/RBF의
    Ray Stability와 동일한 sampling 철학을 유지한다. Comparison 표에는
    넣지 않고 Spline 전용 Diagnostics에서만 사용한다."""
    if len(models) < 2:
        return None, None
    sample_pixels = fixed_evaluation_pixels(image_width, image_height, sample_rows, sample_cols)
    displacements_per_model = []
    for m in models:
        vals = [m.local_displacement_m(float(u), float(v)) for u, v in sample_pixels]
        displacements_per_model.append(np.array([v if v is not None else np.nan for v in vals]))
    all_diffs_mm: list[float] = []
    for i in range(len(displacements_per_model)):
        for j in range(i + 1, len(displacements_per_model)):
            diffs = np.abs(displacements_per_model[i] - displacements_per_model[j]) * 1000.0
            all_diffs_mm.extend(diffs[np.isfinite(diffs)].tolist())
    if not all_diffs_mm:
        return None, None
    arr = np.array(all_diffs_mm)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


# ---------------------------------------------------------------------------
# Repeated Hold-out
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
    없는 개념이라 공유 dataclass 자체를 오염시키고 싶지 않기 때문이다."""
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
    """windshield_dataset을 여러 (다른 seed의) Train/Test로 나눠 반복
    평가한다 - Grid/RBF의 run_repeated_holdout_residual_*와 평행한 구조.

    호출부(run_spline_calibration_with_diagnostics/select_best_spline_grid_
    resolution) 책임: 여기 넘기는 `windshield_dataset`은 항상 **Outer Train만
    포함하는 subset**이어야 한다(Outer Test가 이 함수의 내부 split_train_test
    에 다시 섞여 들어가면 leakage가 된다) - 이 함수 자체는 자신이 받은
    dataset 전체를 자유롭게 나눠 쓴다는 점만 보장한다."""
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
        models.append(build_spline_model_from_fitted_params(K, D, model_name, result.fitted_params))

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
# AUTO 선택
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
    Test는 이 함수 어디에도 등장하지 않는다(leakage 없음).

    선택 규칙: Hold-out RMS가 가장 좋은 후보를 기준으로, tie_tolerance
    이내로 비슷한 후보들 중 parameter(=rows*cols)가 가장 적은 것을 최종
    선택한다.
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
    호출하는 단일 진입점.

    이번 라운드에 고친 leakage 버그: repeated hold-out에 넘기는 dataset을
    반드시 **Outer Train만 포함하는 subset**으로 만든다(`outer_train_dataset`)
    - 예전에는 `windshield_dataset`(전체)을 그대로 넘겨서, 내부
    `run_repeated_holdout_spline`의 `split_train_test`가 Outer Test 프레임을
    다시 골라갈 수 있었다(Residual Grid/RBF의 동일 오케스트레이터는 이미
    이렇게 하고 있었다 - Spline만 빠져 있던 버그)."""
    hint = config.spline_hint or {}
    was_auto = hint.get("auto_spline", 0.0) > 0

    result = calibrate_spline(windshield_dataset, config, camera_config, train_ids, test_ids)
    if not result.success:
        return result

    result.fitted_params["diag_selection_mode_is_auto"] = 1.0 if was_auto else 0.0

    if compute_repeated_holdout:
        outer_train_dataset = Dataset(frames=_subset_frames(windshield_dataset, train_ids))
        resolved_hint = {
            **hint,
            "auto_spline": 0.0,
            "spline_rows": result.fitted_params["spline_rows"],
            "spline_cols": result.fitted_params["spline_cols"],
        }
        resolved_config = dataclasses.replace(config, spline_hint=resolved_hint)
        summary = run_repeated_holdout_spline(
            outer_train_dataset, resolved_config, camera_config,
            seeds=repeated_holdout_seeds, test_ratio=repeated_holdout_test_ratio,
        )
        populate_repeated_holdout_diagnostics(result.fitted_params, summary.holdout, len(repeated_holdout_seeds))
        if summary.surface_stability_mean_mm is not None:
            result.fitted_params["diag_surface_stability_mean_mm"] = summary.surface_stability_mean_mm
        if summary.surface_stability_p95_mm is not None:
            result.fitted_params["diag_surface_stability_p95_mm"] = summary.surface_stability_p95_mm

    return result
