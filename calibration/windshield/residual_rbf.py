"""
camera_calibrator.calibration.windshield.residual_rbf
==========================================================

STEP 3-B - Residual Ray의 두 번째 variant: `scipy.interpolate.RBFInterpolator`
기반의 smooth `(u,v) -> ΔRay` 모델. `calibration/windshield/residual_ray.py`
(Grid + Bilinear)와 완전히 같은 최상위 개념(residual-ray correction)을 쓴다:

    Pixel -> 고정된 Base K,D -> Base Ray -> RBF ΔRay(u,v) -> Corrected Ray

    d_corrected = normalize(d_base + Delta_d(u, v))

차이는 Delta_d(u,v)를 만드는 방법뿐이다 - Grid는 성긴 control grid +
bilinear interpolation을, 이 모듈은 대표 center 집합 + RBF(thin plate
spline)를 쓴다:

    Delta_d(u,v) = sum_i w_i * phi(||(u,v) - c_i||)

`WindshieldModelType`에 새 멤버(RESIDUAL_GRID/RESIDUAL_RBF)를 추가하지
않는다 - 여전히 `WindshieldModelType.RESIDUAL_RAY` 하나이고,
`WindshieldConfig.residual_ray_hint["method"]`("grid" 기본값/"rbf")로만
Grid와 구분된다(dispatch는 calibration/windshield/validation.py와
ui/windshield_worker.py에 있다). YAML round-trip 이후에도 구분할 수 있도록
`fitted_params["residual_ray_method"]`에 숫자 코드(0.0=grid, 1.0=rbf)를
남긴다 - 기존 export/windshield.py의 fitted_params 스키마가 flat한
name->float dict만 허용하기 때문에 문자열 대신 숫자 코드를 쓴다
(build_projector()가 이 값으로 재구성 시점에 분기한다).

Pose 정책(ray-domain residual + weak initial-pose prior), corrected-ray
angular stability, 실제 pixel-domain evaluation, Repeated Hold-out 요약
형태는 Grid와 완전히 동일해야 한다는 요구사항에 따라
`calibration/windshield/residual_common.py`의 공유 함수를 그대로 재사용한다
- 이 모듈은 RBF 고유의 fitting 로직(center 선택 + RBFInterpolator 생성)만
갖고 있다.

RBF center 정규화 좌표계(사용자 스펙 5번):

    u_n = 2*u/W - 1
    v_n = 2*v/H - 1

Center 선택(사용자 스펙 10/11번) - "Spatial Binning":
    모든 관측 corner를 RBF center로 쓰면 runtime 모델이 지나치게 커질 수
    있으므로, normalized 이미지 평면을 결정론적인 cell 격자로 나누고 cell
    "중심 좌표"를 RBF center 위치로 쓴다(데이터 위치가 아니라 결정론적
    위치이므로 중복 center가 생기지 않는다). 각 cell의 representative
    residual 값은 그 cell에 속한 코너들의 raw Δd(= target_dir - d_base)의
    성분별 median이다. 샘플이 없는 cell은 제외되므로 최종 center 수가
    요청보다 작아질 수 있다(clean fallback).

STAGE A는 Grid처럼 least_squares 최적화를 쓰지 않는다 - `RBFInterpolator`
자체가 scattered-data 보간 + `smoothing` regularization을 내장하므로,
cell representative 샘플로 직접 `RBFInterpolator`를 생성하는 것으로
충분하다(사용자 스펙 6-9번). STAGE B는 Grid와 동일한 alternating 구조
(pose refine <-> RBF refit)를 반복한다. 최종 채택은 두 STAGE의 실제 pixel
RMS를 비교해서 정한다(ray-domain 최적화 / pixel-domain 최종 평가 구분은
Grid와 동일).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from calibration.models.common import MIN_FRAMES_REQUIRED, infer_image_size
from calibration.types import CameraConfig, CameraModelType, Dataset, Frame
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.refraction import normalize
from calibration.windshield.residual_common import (
    DEFAULT_REPEATED_HOLDOUT_SEEDS,
    MAX_PROJECT_POINT_ANGULAR_ERROR_DEG,
    RepeatedHoldoutSummary,
    collect_corner_arrays,
    compute_ray_stability_deg,
    evaluate_residual_ray_model,
    normalize_pixel_coordinates,
    populate_pose_diagnostics,
    populate_repeated_holdout_diagnostics,
    refine_frame_pose_ray_domain,
    residual_ray_failure_result,
)

# ---------------------------------------------------------------------------
# 고정 상수 - residual_ray_hint로 덮어쓸 수 있고, 코드 여러 곳에 하드코딩하지
# 않는다(사용자 스펙 12번과 동일한 원칙을 RBF에도 적용).
# ---------------------------------------------------------------------------
DEFAULT_RBF_KERNEL = "thin_plate_spline"
DEFAULT_RBF_NUM_CENTERS = 64
DEFAULT_RBF_SMOOTHING = 1e-4

# AUTO selection 기본 후보(사용자 스펙 13번 예시) - 코드 여러 곳에
# hard-code하지 않는다.
RBF_CENTER_CANDIDATES: list[int] = [32, 64, 128, 256]
RBF_SMOOTHING_CANDIDATES: list[float] = [1e-5, 1e-4, 1e-3]

# Hold-out RMS가 이 비율 이내로 비슷하면 parameter(=num_centers)가 더 적은
# 후보를 선택한다 - Grid의 GRID_SELECTION_TIE_TOLERANCE와 동일한 철학/값.
RBF_SELECTION_TIE_TOLERANCE = 0.05

RBF_STAGE_B_NUM_ROUNDS = 2  # Grid의 GRID_STAGE_B_NUM_ROUNDS와 동일 값

# center 하나당 최소 이만큼의 유효 corner 샘플이 있어야 안정적인 fit으로
# 인정한다 - Grid의 MIN_CORNERS_PER_NODE와 동일한 철학.
MIN_SAMPLES_PER_CENTER = 2

MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10

# Edge extrapolation guard(사용자 스펙 35번) - RBF가 학습 영역 밖에서 극단적인
# 값을 내는 것을 막는다. 실제 synthetic GT의 보정 스케일(~0.02)보다 훨씬
# 크게 잡아서, 정상 동작 범위에서는 전혀 개입하지 않고 진짜 병적인
# extrapolation/NaN만 잡아낸다.
MAX_CORRECTION_MAGNITUDE = 1.0

# kernel <-> 숫자 코드 매핑(fitted_params가 flat float dict라 문자열을 직접
# 저장할 수 없음) - 향후 multiquadric/gaussian을 추가할 여지를 남긴다.
_KERNEL_TO_CODE: dict[str, float] = {"thin_plate_spline": 0.0}
_CODE_TO_KERNEL: dict[float, str] = {0.0: "thin_plate_spline"}


def evaluate_rbf_delta(
    rbf: RBFInterpolator,
    u: float,
    v: float,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    un, vn = normalize_pixel_coordinates(u, v, image_width, image_height)
    raw = np.asarray(rbf(np.array([[un, vn]]))[0], dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        return np.zeros(3)
    norm = float(np.linalg.norm(raw))
    if norm > MAX_CORRECTION_MAGNITUDE:
        raw = raw * (MAX_CORRECTION_MAGNITUDE / norm)
    return raw


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

class ResidualRBFWindshieldModel(WindshieldModel):
    """이미지 위 성긴 center 집합에 저장된 3D ray-correction을
    `RBFInterpolator`(thin plate spline)로 보간해 Base Ray에 더하는 모델.
    project_point()/unproject_pixel() 모두 이 보정을 실제로 반영한다.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        centers_normalized: np.ndarray,   # (N, 2), [-1,1] normalized (u,v)
        residual_values: np.ndarray,      # (N, 3)
        image_width: float,
        image_height: float,
        kernel: str = DEFAULT_RBF_KERNEL,
        smoothing: float = DEFAULT_RBF_SMOOTHING,
        epsilon: Optional[float] = None,
    ):
        self._centers = np.asarray(centers_normalized, dtype=np.float64)
        self._residual_values = np.asarray(residual_values, dtype=np.float64)
        if self._centers.ndim != 2 or self._centers.shape[1] != 2:
            raise ValueError(f"RBF centers must have shape (N, 2), got {self._centers.shape}.")
        if self._residual_values.shape != (self._centers.shape[0], 3):
            raise ValueError(
                f"RBF residual_values must have shape (N, 3) matching centers, "
                f"got {self._residual_values.shape} vs {self._centers.shape[0]} centers."
            )
        self._image_width = float(image_width)
        self._image_height = float(image_height)
        self._kernel = kernel
        self._smoothing = float(smoothing)
        self._epsilon = epsilon
        # thin_plate_spline은 scale-invariant kernel이라 epsilon이 필요
        # 없다(scipy가 자동으로 1.0을 쓴다) - None이면 아예 넘기지 않는다
        # (사용자 스펙 8번, "epsilon이 의미 있는 kernel에서만 사용").
        kwargs = {} if epsilon is None else {"epsilon": float(epsilon)}
        self._rbf = RBFInterpolator(self._centers, self._residual_values, kernel=kernel, smoothing=self._smoothing, **kwargs)
        # Base K,D 전용 두 primitive(픽셀<->카메라 광선, 항등 투영)만 재사용한다 -
        # BaselineWindshieldModel 자체는 절대 K,D를 수정하지 않으므로 안전하게
        # 내부 헬퍼로 감싸 쓸 수 있다(Grid와 동일한 재사용 패턴).
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def _delta(self, u: float, v: float) -> np.ndarray:
        return evaluate_rbf_delta(self._rbf, u, v, self._image_width, self._image_height)

    def evaluate_delta(self, u: float, v: float) -> np.ndarray:
        return self._delta(u, v)

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다(RBF 보정이 픽셀에
        따라 달라지므로) - Base K,D 투영을 초기값으로 삼아 작은 2변수
        root-solve로 푼다. ResidualRayWindshieldModel.project_point()와
        동일한 구조."""
        from scipy.optimize import least_squares

        target_dir = normalize(np.array([x, y, z], dtype=np.float64))
        initial_uv = np.asarray(self._baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            d_base = np.asarray(self._baseline.unproject_pixel(float(uv[0]), float(uv[1])), dtype=np.float64)
            corrected = normalize(d_base + self._delta(float(uv[0]), float(uv[1])))
            return target_dir - corrected

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        if not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.cost):
            raise ValueError("project_point(): local root-solve did not converge to a finite result.")

        residual_norm = float(np.linalg.norm(result.fun))
        angle_rad = 2.0 * math.asin(min(1.0, residual_norm / 2.0))
        if math.degrees(angle_rad) > MAX_PROJECT_POINT_ANGULAR_ERROR_DEG:
            raise ValueError(
                "Could not find a valid corrected projection for this point "
                "(RBF residual correction may not cover this region well)."
            )
        return float(result.x[0]), float(result.x[1])

    def ray_angular_error_deg(self, u: float, v: float, target_point_cam: np.ndarray) -> Optional[float]:
        """관측 픽셀(u,v)의 보정된 광선과, 카메라 좌표계의 실제 목표점 방향
        사이의 각도(도) - Grid와 동일한 개념의 보조 metric."""
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        target = np.asarray(target_point_cam, dtype=np.float64)
        norm = np.linalg.norm(target)
        if norm < 1e-9:
            return None
        cos_angle = float(np.clip(np.dot(target / norm, corrected), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))


def build_residual_rbf_model_from_fitted_params(
    camera_matrix: np.ndarray, distortion: np.ndarray, model: CameraModelType, fitted_params: dict[str, float],
) -> ResidualRBFWindshieldModel:
    """fitted_params(아래 스키마)에서 RBF runtime 모델을 재구성하는 유일한
    지점 - projection.py::build_projector와 repeated-holdout stability
    계산이 둘 다 이 함수 하나만 쓴다(사용자 스펙 26번, "재구성 가능한 public
    입력만 저장하고 SciPy 내부 private coefficient에 의존하지 않는다").

    스키마: rbf_num_centers, rbf_center_u_{i}/rbf_center_v_{i},
    rbf_residual_dx_{i}/_dy_{i}/_dz_{i}, rbf_kernel_code, rbf_smoothing,
    image_width, image_height.
    """
    fp = fitted_params
    n = int(fp["rbf_num_centers"])
    centers = np.zeros((n, 2), dtype=np.float64)
    values = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        centers[i, 0] = fp[f"rbf_center_u_{i}"]
        centers[i, 1] = fp[f"rbf_center_v_{i}"]
        values[i, 0] = fp[f"rbf_residual_dx_{i}"]
        values[i, 1] = fp[f"rbf_residual_dy_{i}"]
        values[i, 2] = fp[f"rbf_residual_dz_{i}"]
    kernel = _CODE_TO_KERNEL.get(fp.get("rbf_kernel_code", 0.0), DEFAULT_RBF_KERNEL)
    smoothing = float(fp.get("rbf_smoothing", DEFAULT_RBF_SMOOTHING))
    return ResidualRBFWindshieldModel(
        camera_matrix, distortion, model, centers, values,
        fp["image_width"], fp["image_height"], kernel=kernel, smoothing=smoothing,
    )


# ---------------------------------------------------------------------------
# Center 선택 (Spatial Binning, 사용자 스펙 10/11번)
# ---------------------------------------------------------------------------

def _rbf_settings(config: WindshieldConfig) -> tuple[int, float]:
    hint = config.residual_ray_hint or {}
    num_centers = int(hint.get("rbf_num_centers", DEFAULT_RBF_NUM_CENTERS))
    smoothing = float(hint.get("rbf_smoothing", DEFAULT_RBF_SMOOTHING))
    if num_centers < 1:
        raise ValueError(f"residual_ray_hint rbf_num_centers must be >= 1 (got {num_centers}).")
    return num_centers, smoothing


def _select_rbf_centers(
    observed_pixels: np.ndarray,   # (N, 2) raw pixel coords
    raw_deltas: np.ndarray,        # (N, 3) target_dir - d_base per corner
    image_width: float,
    image_height: float,
    num_centers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized 이미지 평면을 결정론적 cell 격자로 나누고, cell 중심
    좌표를 RBF center로, 그 cell에 속한 코너들의 성분별 median Δd를
    representative 값으로 쓴다.

    Center **위치**는 데이터와 무관하게 고정(cell 중심)이라 중복 center가
    생길 수 없다(사용자 스펙 11번). 샘플이 하나도 없는 cell은 제외되므로
    최종 center 수는 요청(`num_centers`)보다 작을 수 있다(clean fallback).
    """
    aspect = float(image_width) / float(image_height)
    cell_cols = max(1, int(round(math.sqrt(num_centers * aspect))))
    cell_rows = max(1, int(round(num_centers / cell_cols)))

    normalized = np.array([
        normalize_pixel_coordinates(u, v, image_width, image_height) for u, v in observed_pixels
    ])
    # cell (r,c)는 [-1,1]을 cell_rows x cell_cols로 균등 분할한 영역이다.
    col_idx = np.clip(((normalized[:, 0] + 1.0) / 2.0 * cell_cols).astype(int), 0, cell_cols - 1)
    row_idx = np.clip(((normalized[:, 1] + 1.0) / 2.0 * cell_rows).astype(int), 0, cell_rows - 1)

    centers: list[list[float]] = []
    values: list[list[float]] = []
    for r in range(cell_rows):
        for c in range(cell_cols):
            mask = (row_idx == r) & (col_idx == c)
            if not np.any(mask):
                continue
            # cell 중심 좌표(결정론적, 데이터 위치 무관) - [-1,1] 범위를
            # cell_cols/cell_rows 등분한 각 셀의 기하학적 중심.
            center_u = -1.0 + (c + 0.5) * (2.0 / cell_cols)
            center_v = -1.0 + (r + 0.5) * (2.0 / cell_rows)
            centers.append([center_u, center_v])
            values.append(np.median(raw_deltas[mask], axis=0).tolist())

    return np.array(centers, dtype=np.float64), np.array(values, dtype=np.float64)


def _raw_deltas(observed_pixels: np.ndarray, d_obs: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    """코너별 raw ray residual: Δd_i = normalize(p_cam_i) - d_obs_i.
    d_obs_i/target_dir 모두 단위벡터이므로 이 값을 그대로 RBF에 학습시키면
    (smoothing=0 가정 시) 관측 코너 위치에서 정확히
    normalize(d_base+Δd) == target_dir가 성립한다(사용자 스펙 9번)."""
    target_dirs = np.array([normalize(p) for p in p_cam])
    return target_dirs - d_obs


def _fit_rbf_stage_a(
    observed_pixels: np.ndarray, d_obs: np.ndarray, p_cam: np.ndarray,
    image_width: float, image_height: float, num_centers: int, smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """STAGE A - pose 고정 상태에서 RBF center/값을 직접 만든다. Grid와
    달리 least_squares 최적화가 필요 없다 - RBFInterpolator 생성 자체가
    scattered-data 보간 + smoothing regularization을 수행한다."""
    raw_deltas = _raw_deltas(observed_pixels, d_obs, p_cam)
    return _select_rbf_centers(observed_pixels, raw_deltas, image_width, image_height, num_centers)


# ---------------------------------------------------------------------------
# STAGE B - RBF + per-frame pose joint refinement (ray-domain, alternating)
# ---------------------------------------------------------------------------

@dataclass
class _JointRBFRefinementOutcome:
    centers: np.ndarray
    values: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    converged_cleanly: bool


def _joint_refine_rbf_and_poses(
    ok_frames: list[Frame],
    observed_pixels_per_frame: list[np.ndarray],
    d_obs_per_frame: list[np.ndarray],
    initial_rvecs: list[np.ndarray],
    initial_tvecs: list[np.ndarray],
    image_width: float,
    image_height: float,
    num_centers: int,
    smoothing: float,
    kernel: str,
    initial_centers: np.ndarray,
    initial_values: np.ndarray,
    num_rounds: int = RBF_STAGE_B_NUM_ROUNDS,
) -> _JointRBFRefinementOutcome:
    """STAGE B - alternating(block-coordinate) 방식으로 RBF와 프레임별
    pose를 번갈아 refine한다. Grid의 `_joint_refine_grid_and_poses`와
    완전히 동일한 alternating 구조 - pose refine은 공유
    `refine_frame_pose_ray_domain`에 위임한다."""
    rvecs = [np.asarray(r, dtype=np.float64).copy() for r in initial_rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).copy() for t in initial_tvecs]
    centers = np.asarray(initial_centers, dtype=np.float64).copy()
    values = np.asarray(initial_values, dtype=np.float64).copy()
    converged_cleanly = True

    for _ in range(num_rounds):
        try:
            current_rbf = RBFInterpolator(centers, values, kernel=kernel, smoothing=smoothing)
        except (ValueError, np.linalg.LinAlgError):
            converged_cleanly = False
            break

        def delta_fn(u: float, v: float, _rbf=current_rbf) -> np.ndarray:
            return evaluate_rbf_delta(_rbf, u, v, image_width, image_height)

        for i, frame in enumerate(ok_frames):
            pose_fit = refine_frame_pose_ray_domain(
                frame, observed_pixels_per_frame[i], d_obs_per_frame[i], delta_fn,
                initial_rvecs[i], initial_tvecs[i],
            )
            if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                rvecs[i] = pose_fit.x[:3].reshape(3, 1)
                tvecs[i] = pose_fit.x[3:6].reshape(3, 1)
            else:
                converged_cleanly = False

        obs_pixel_list, p_cam_list = [], []
        for i, frame in enumerate(ok_frames):
            R, _ = cv2.Rodrigues(rvecs[i])
            obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
            cam_pts = (R @ obj.T).T + tvecs[i].reshape(1, 3)
            p_cam_list.append(cam_pts)
            obs_pixel_list.append(observed_pixels_per_frame[i])
        p_cam_arr = np.concatenate(p_cam_list, axis=0)
        obs_pixel_arr = np.concatenate(obs_pixel_list, axis=0)
        d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)

        raw_deltas = _raw_deltas(obs_pixel_arr, d_obs_arr, p_cam_arr)
        new_centers, new_values = _select_rbf_centers(obs_pixel_arr, raw_deltas, image_width, image_height, num_centers)
        if new_centers.shape[0] >= 1:
            centers, values = new_centers, new_values
        else:
            converged_cleanly = False

    return _JointRBFRefinementOutcome(centers=centers, values=values, rvecs=rvecs, tvecs=tvecs, converged_cleanly=converged_cleanly)


def _failure_result(config: WindshieldConfig, train_ids: list[str], test_ids: list[str], message: str) -> WindshieldCalibrationResult:
    return residual_ray_failure_result(config, train_ids, test_ids, message)


def calibrate_residual_rbf(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Residual Ray RBF 모델을 fitting한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않는다.

    흐름: (AUTO 지정 시 center 수/smoothing 선택, train_ids만 사용) ->
    STAGE A(RBF만, pose 고정) -> STAGE B(RBF+pose joint, ray-domain
    alternating) -> 두 stage의 실제 pixel RMS를 비교해 더 나은 쪽을
    최종으로 채택 -> Train 평가 -> Test는 최종 RBF를 완전히 고정한 채
    자기 pose만 별도로 refine한 뒤 평가(leakage 없음).
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size

    hint = config.residual_ray_hint or {}
    if hint.get("auto_rbf", 0.0) > 0:
        (auto_centers, auto_smoothing), _candidates = select_best_rbf_hyperparams(
            windshield_dataset, config, camera_config, train_ids,
        )
        config = dataclasses.replace(
            config,
            residual_ray_hint={
                **hint, "rbf_num_centers": float(auto_centers), "rbf_smoothing": float(auto_smoothing), "auto_rbf": 0.0,
            },
        )

    num_centers, smoothing = _rbf_settings(config)
    kernel = DEFAULT_RBF_KERNEL
    min_corners = max(20, num_centers * MIN_SAMPLES_PER_CENTER)

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
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(
        ok_frames, rvecs, tvecs, baseline_model
    )

    total_corners = sum(len(a) for a in d_obs_per_frame)
    if total_corners < min_corners:
        return _failure_result(
            config, train_ids, test_ids,
            f"RBF({num_centers} centers)를 추정하기에 코너 수가 부족합니다 "
            f"(코너 {total_corners}개, 최소 {min_corners}개 필요).",
        )

    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    # --- STAGE A: RBF만, pose 고정 ---
    try:
        stage_a_centers, stage_a_values = _fit_rbf_stage_a(
            observed_pixels_arr, d_obs_arr, p_cam_arr, width, height, num_centers, smoothing,
        )
    except ValueError as e:
        return _failure_result(config, train_ids, test_ids, f"RBF center 선택이 실패했습니다: {e}")

    if stage_a_centers.shape[0] < 3:
        return _failure_result(config, train_ids, test_ids, "RBF requires at least three non-collinear centers for thin_plate_spline.")

    try:
        stage_a_model = ResidualRBFWindshieldModel(
            K, D, model, stage_a_centers, stage_a_values, width, height, kernel=kernel, smoothing=smoothing,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        return _failure_result(config, train_ids, test_ids, f"RBFInterpolator 생성에 실패했습니다: {e}")

    stage_a_outcome = evaluate_residual_ray_model(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: RBF + per-frame pose joint refinement (ray-domain) ---
    joint = _joint_refine_rbf_and_poses(
        ok_frames, observed_pixels_per_frame, d_obs_per_frame, rvecs, tvecs,
        width, height, num_centers, smoothing, kernel, stage_a_centers, stage_a_values,
    )

    stage_used_is_joint_refined = False
    final_centers, final_values = stage_a_centers, stage_a_values
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    refinement_note = ""

    try:
        stage_b_model = ResidualRBFWindshieldModel(
            K, D, model, joint.centers, joint.values, width, height, kernel=kernel, smoothing=smoothing,
        )
        stage_b_outcome = evaluate_residual_ray_model(ok_frames, joint.rvecs, joint.tvecs, stage_b_model, image_size)

        stage_a_rmse = stage_a_outcome.residual_stats.rmse
        stage_b_rmse = stage_b_outcome.residual_stats.rmse
        improved = (
            stage_b_outcome.residual_stats.n > 0
            and stage_a_rmse is not None
            and stage_b_rmse is not None
            and stage_b_rmse <= stage_a_rmse
        )
    except (ValueError, np.linalg.LinAlgError):
        improved = False

    if improved:
        stage_used_is_joint_refined = True
        final_centers, final_values = joint.centers, joint.values
        final_rvecs, final_tvecs = joint.rvecs, joint.tvecs
        final_outcome = stage_b_outcome
        if not joint.converged_cleanly:
            refinement_note = "STAGE B 일부 sub-fit이 수렴하지 않아 해당 프레임/라운드는 이전 값을 유지했습니다. "
    else:
        refinement_note = (
            "STAGE B(ray-domain alternating RBF/pose refinement)가 STAGE A(RBF-only initial fit)보다 "
            "실제 pixel RMS를 개선하지 못해 STAGE A 결과를 최종으로 사용했습니다. "
            "(참고: 최적화 자체의 residual은 ray-domain이고, 이 STAGE A/B 채택 여부 판단 기준만 "
            "실제 pixel-domain RMS를 사용합니다.) "
        )

    final_model = ResidualRBFWindshieldModel(
        K, D, model, final_centers, final_values, width, height, kernel=kernel, smoothing=smoothing,
    )

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 RBF로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    final_num_centers = final_centers.shape[0]
    runtime_param_count = final_num_centers * 3
    pose_param_count_train = len(ok_frames) * 6

    fitted_params: dict[str, float] = {
        "residual_ray_method": 1.0,  # 0.0 = Grid + Bilinear, 1.0 = RBF
        "image_width": float(width),
        "image_height": float(height),
        "rbf_kernel_code": _KERNEL_TO_CODE[kernel],
        "rbf_smoothing": float(smoothing),
        "rbf_num_centers": float(final_num_centers),
        "num_fit_points": float(total_corners),
        "runtime_param_count": float(runtime_param_count),
        "residual_value_param_count": float(final_num_centers * 3),
        "rbf_center_count": float(final_num_centers),
        "serialized_numeric_value_count": float(final_num_centers * 5),
        "pose_param_count_train": float(pose_param_count_train),
        "stage_used_is_joint_refined": 1.0 if stage_used_is_joint_refined else 0.0,
    }
    populate_pose_diagnostics(fitted_params, rvecs, tvecs, final_rvecs, final_tvecs)
    for i in range(final_num_centers):
        fitted_params[f"rbf_center_u_{i}"] = float(final_centers[i, 0])
        fitted_params[f"rbf_center_v_{i}"] = float(final_centers[i, 1])
        fitted_params[f"rbf_residual_dx_{i}"] = float(final_values[i, 0])
        fitted_params[f"rbf_residual_dy_{i}"] = float(final_values[i, 1])
        fitted_params[f"rbf_residual_dz_{i}"] = float(final_values[i, 2])

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
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
                # RBF 기준으로 pose만 다시 refine한다 - RBF/K/D는 여기서 절대
                # 건드리지 않는다(leakage 없음).
                t_rvecs, t_tvecs = [], []
                t_obs_pixels, t_d_obs, _t_p_cam = collect_corner_arrays(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, baseline_model
                )

                for frame, init_rvec, init_tvec, obs_px, d_obs in zip(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, t_obs_pixels, t_d_obs
                ):
                    pose_fit = refine_frame_pose_ray_domain(
                        frame, obs_px, d_obs, final_model.evaluate_delta, init_rvec, init_tvec, regularize=True,
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
# Repeated Hold-out (사용자 스펙 22번)
# ---------------------------------------------------------------------------

def run_repeated_holdout_residual_rbf(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    test_ratio: float = 0.25,
) -> RepeatedHoldoutSummary:
    """windshield_dataset 전체를 여러 (다른 seed의) Train/Test로 나눠 반복
    평가한다 - Grid의 run_repeated_holdout_residual_ray와 평행한 구조.
    DEFAULT_REPEATED_HOLDOUT_SEEDS를 residual_common에서 공유하므로 Grid와
    항상 같은 기본 seed 집합을 쓴다(사용자 스펙 22번)."""
    from calibration.models.common import regional_edge_average

    test_rmses: list[float] = []
    test_p95s: list[float] = []
    edge_rmses: list[float] = []
    models: list[WindshieldModel] = []
    successful_seeds: list[int] = []

    K, D, model_name = config.base_camera_matrix, config.base_distortion, config.base_model_name
    width, height = infer_image_size(windshield_dataset, camera_config)

    for seed in seeds:
        train_ids, test_ids = split_train_test(windshield_dataset, camera_config, test_ratio, seed)
        result = calibrate_residual_rbf(windshield_dataset, config, camera_config, train_ids, test_ids)
        if not result.success or result.test_residual_stats is None:
            continue
        successful_seeds.append(seed)
        test_rmses.append(result.test_residual_stats.rmse)
        if result.test_residual_stats.p95 is not None:
            test_p95s.append(result.test_residual_stats.p95)
        edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
        if edge is not None:
            edge_rmses.append(edge)
        models.append(build_residual_rbf_model_from_fitted_params(K, D, model_name, result.fitted_params))

    ray_stability_mean_deg, ray_stability_p95_deg = compute_ray_stability_deg(models, width, height)

    return RepeatedHoldoutSummary(
        seeds_used=successful_seeds,
        n_successful=len(successful_seeds),
        mean_test_rmse=float(np.mean(test_rmses)) if test_rmses else None,
        std_test_rmse=float(np.std(test_rmses)) if test_rmses else None,
        mean_test_p95=float(np.mean(test_p95s)) if test_p95s else None,
        mean_edge_rms=float(np.mean(edge_rmses)) if edge_rmses else None,
        grid_stability_l2=None,  # Grid 전용 legacy metric - RBF에는 적용되지 않음
        ray_stability_mean_deg=ray_stability_mean_deg,
        ray_stability_p95_deg=ray_stability_p95_deg,
    )


# ---------------------------------------------------------------------------
# AUTO 선택 (사용자 스펙 13/14/15번)
# ---------------------------------------------------------------------------

@dataclass
class RBFCandidateResult:
    num_centers: int
    smoothing: float
    param_count: int
    summary: RepeatedHoldoutSummary


def select_best_rbf_hyperparams(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    candidate_frame_ids: list[str],
    center_candidates: Optional[list[int]] = None,
    smoothing_candidates: Optional[list[float]] = None,
    seeds: Optional[tuple[int, ...]] = None,
    inner_test_ratio: float = 0.3,
    tie_tolerance: float = RBF_SELECTION_TIE_TOLERANCE,
) -> tuple[tuple[int, float], list[RBFCandidateResult]]:
    """candidate_frame_ids(바깥쪽 호출부의 Train만) 안에서 다시 나눠서
    (내부 train/내부 test) center 수 x smoothing 조합을 비교한다 - 바깥쪽
    실제 Test는 이 함수 어디에도 등장하지 않는다(모델 선택 자체도 leakage
    없이 이뤄짐, 사용자 스펙 14번).

    선택 규칙: Hold-out RMS가 가장 좋은 후보를 기준으로, tie_tolerance
    이내로 비슷한 후보들 중 parameter(=num_centers*3)가 가장 적은 것을
    최종 선택한다(사용자 스펙 15번).

    center_candidates/smoothing_candidates/seeds가 None이면 모듈 상수를
    함수 호출 시점에 읽는다 - Grid의 select_best_grid_resolution과 동일한
    이유(default 인자로 바인딩하면 monkeypatch가 반영되지 않음)."""
    if center_candidates is None:
        center_candidates = RBF_CENTER_CANDIDATES
    if smoothing_candidates is None:
        smoothing_candidates = RBF_SMOOTHING_CANDIDATES
    if seeds is None:
        seeds = DEFAULT_REPEATED_HOLDOUT_SEEDS[:3]
    inner_dataset = Dataset(frames=_subset_frames(windshield_dataset, candidate_frame_ids))
    candidate_results: list[RBFCandidateResult] = []

    for num_centers in center_candidates:
        for smoothing in smoothing_candidates:
            cfg = dataclasses.replace(
                config,
                residual_ray_hint={
                    **(config.residual_ray_hint or {}),
                    "rbf_num_centers": float(num_centers), "rbf_smoothing": float(smoothing), "auto_rbf": 0.0,
                },
            )
            summary = run_repeated_holdout_residual_rbf(
                inner_dataset, cfg, camera_config, seeds=seeds, test_ratio=inner_test_ratio,
            )
            candidate_results.append(
                RBFCandidateResult(num_centers=num_centers, smoothing=smoothing, param_count=num_centers * 3, summary=summary)
            )

    valid = [c for c in candidate_results if c.summary.mean_test_rmse is not None]
    if not valid:
        # 아무 후보도 평가할 수 없으면(데이터 부족 등) 가장 단순한 후보로
        # fallback한다 - calibrate_residual_rbf의 본 fitting이 그 이유를
        # 다시 명확한 실패 메시지로 보고할 것이다.
        return (center_candidates[0], smoothing_candidates[0]), candidate_results

    best_rmse = min(c.summary.mean_test_rmse for c in valid)
    close_enough = [c for c in valid if c.summary.mean_test_rmse <= best_rmse * (1.0 + tie_tolerance)]
    chosen = min(close_enough, key=lambda c: c.param_count)
    return (chosen.num_centers, chosen.smoothing), candidate_results


# ---------------------------------------------------------------------------
# Diagnostics orchestrator - UI/worker가 호출하는 단일 진입점
# ---------------------------------------------------------------------------

def run_residual_rbf_calibration_with_diagnostics(
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
    """calibrate_residual_rbf()에 Repeated Hold-out + Ray Stability 진단을
    더한 결과를 반환한다 - Grid의 run_residual_ray_calibration_with_
    diagnostics와 평행 구조(UI/worker가 Residual Ray RBF 모델을 실행할 때
    호출하는 단일 진입점)."""
    hint = config.residual_ray_hint or {}
    was_auto = hint.get("auto_rbf", 0.0) > 0

    result = calibrate_residual_rbf(windshield_dataset, config, camera_config, train_ids, test_ids)
    if not result.success:
        return result

    result.fitted_params["diag_selection_mode_is_auto"] = 1.0 if was_auto else 0.0

    if compute_repeated_holdout:
        outer_train_dataset = Dataset(frames=_subset_frames(windshield_dataset, train_ids))
        resolved_hint = {
            **hint,
            "auto_rbf": 0.0,
            "rbf_num_centers": result.fitted_params["rbf_num_centers"],
            "rbf_smoothing": result.fitted_params["rbf_smoothing"],
        }
        resolved_config = dataclasses.replace(config, residual_ray_hint=resolved_hint)
        summary = run_repeated_holdout_residual_rbf(
            outer_train_dataset, resolved_config, camera_config,
            seeds=repeated_holdout_seeds, test_ratio=repeated_holdout_test_ratio,
        )
        populate_repeated_holdout_diagnostics(result.fitted_params, summary, len(repeated_holdout_seeds))

    return result
