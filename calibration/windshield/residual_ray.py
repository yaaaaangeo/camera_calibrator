"""
camera_calibrator.calibration.windshield.residual_ray
==========================================================

Phase 3-A - "Base Ray + Residual Correction" 모델. 정확한 Windshield CAD나
곡률을 모르는 실차에서도 쓸 수 있도록, Windshield의 물리적 형태(구/스플라인)를
가정하지 않고 이미지 위 각 위치에서 카메라 광선이 얼마나 휘는지를 직접
추정한다.

    Pixel -> 고정된 Base K,D -> Base Ray -> Residual Ray Correction -> Corrected Ray

수학적으로:

    d_corrected = normalize(d_base + Delta_d(u, v))

Delta_d(u,v)는 이미지 위에 성긴 control grid(기본 6행 x 8열 "node")를 두고
bilinear interpolation으로 임의의 픽셀에서 값을 구한다 - Neural Network나
RBF는 이번 단계(3-A)에서 다루지 않는다(3-B/5에서 별도 구현).

설계 선택 - Delta_d를 3D 자유 벡터로 둔 이유:
    Delta_d를 "d_base에 수직인 tangent-plane 위의 2-DoF 보정"으로 제한하는
    방식도 검토했다. 이론적으로는 자유도 하나(= d_base 방향 성분)가
    normalize() 이후 아무 효과가 없으므로 "낭비되는" 자유도이긴 하다 - 하지만
    d_base + k*d_base = (1+k)*d_base이고 normalize()가 스칼라배를 지워버리므로
    이 여분 자유도는 항등 함수 결과에 전혀 영향을 주지 않고, 아래 magnitude
    regularization이 어차피 이 값을 0으로 밀어붙인다. 반면 노드마다 접평면
    기저(tangent basis)를 만들려면 d_base가 축과 거의 평행한 극단적인 경우의
    특이점을 처리해야 하는 등 구현이 복잡해진다. 그 복잡도 대비 얻는 이득이
    작다고 판단해 첫 구현은 3D 자유 벡터로 한다.

Fitting 목적함수 - Spherical STEP 2에서 검증된 "ray-alignment residual"
재사용:
    각 코너에 대해 residual = normalize(p_cam) - normalize(d_base + Delta_d)
    (3-vector)를 쓴다. 이 값이 0이 되는 조건은 project_point(p_cam)이
    정확히 관측 픽셀을 돌려주는 것과 수학적으로 동치다(Spherical의
    calibrate_spherical/spherical.py 모듈 docstring에서 이미 증명/실측함) -
    코너마다 project_point()의 내부 root-solve를 중첩 호출할 필요가 없어서
    grid 전체(최대 6*8*3=144개 파라미터)를 단일 forward-only least_squares로
    빠르게 피팅할 수 있다.

Pose 처리: Spherical STAGE A와 동일한 단순화를 쓴다 - Standard
solve_poses_fixed_intrinsics()로 구한 pose를 이번 라운드에서는 고정값으로
쓰고, grid와 공동 최적화(joint refinement)하지 않는다. 이는 누락이 아니라
의도적 범위 제한이다(추후 확장 지점).
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
from calibration.windshield.refraction import normalize

# ---------------------------------------------------------------------------
# 고정 상수 - residual_ray_hint로 덮어쓸 수 있고, 코드 여러 곳에 하드코딩하지
# 않는다(사용자 스펙 5번).
# ---------------------------------------------------------------------------
DEFAULT_GRID_ROWS = 6
DEFAULT_GRID_COLS = 8
DEFAULT_LAMBDA_MAG = 1e-3
DEFAULT_LAMBDA_SMOOTH = 1e-2

# Grid는 Sphere(파라미터 4개)보다 자유도가 훨씬 크므로(기본 6*8*3=144개),
# 안정적인 피팅을 위해 Spherical의 MIN_CORNERS_FOR_FIT(20)보다 넉넉한
# 최소 코너 수를 요구한다 - 대략 "grid node 수의 2배" 정도를 기준으로 삼는다.
MIN_CORNERS_PER_NODE = 2

MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10
MAX_PROJECT_POINT_ANGULAR_ERROR_DEG = 2.0
_PROJECT_PENALTY = 5.0


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------

def bilinear_interpolate_grid(
    grid: np.ndarray,
    u: float,
    v: float,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """(rows, cols, 3) grid에서 임의의 픽셀 (u,v)의 보정 벡터를 bilinear
    interpolation으로 구한다.

    Grid의 node (r,c)는 이미지 전체(경계 포함)에 걸쳐
        u_node(c) = c/(cols-1) * image_width
        v_node(r) = r/(rows-1) * image_height
    위치에 있다 - 그래서 node (0,0)은 (0,0), node (rows-1,cols-1)은
    (image_width,image_height)에 정확히 대응한다.

    이미지 밖의 픽셀이 들어와도 gridxt 범위 안으로 clamp해서(외삽 대신 가장
    가까운 경계 셀의 보간값을 사용) crash 없이 안정적으로 동작한다.
    """
    rows, cols = grid.shape[0], grid.shape[1]
    if rows < 2 or cols < 2:
        raise ValueError("Residual ray grid must have at least 2 rows and 2 columns.")

    cell_w = image_width / (cols - 1)
    cell_h = image_height / (rows - 1)

    fx = min(max(u, 0.0), image_width) / cell_w
    fy = min(max(v, 0.0), image_height) / cell_h

    c0 = int(min(max(math.floor(fx), 0), cols - 2))
    r0 = int(min(max(math.floor(fy), 0), rows - 2))
    c1, r1 = c0 + 1, r0 + 1

    tx = fx - c0
    ty = fy - r0

    top = (1 - tx) * grid[r0, c0] + tx * grid[r0, c1]
    bottom = (1 - tx) * grid[r1, c0] + tx * grid[r1, c1]
    return (1 - ty) * top + ty * bottom


# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

class ResidualRayWindshieldModel(WindshieldModel):
    """Windshield를 물리적 표면(구/스플라인)으로 모델링하지 않고, 이미지 위
    성긴 grid에 저장된 3D ray-correction을 bilinear interpolation으로 읽어
    Base Ray에 더하는 모델. project_point()/unproject_pixel() 모두 이
    보정을 실제로 반영한다.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        grid: np.ndarray,
        image_width: float,
        image_height: float,
    ):
        self._grid = np.asarray(grid, dtype=np.float64)
        if self._grid.ndim != 3 or self._grid.shape[2] != 3:
            raise ValueError(f"Residual ray grid must have shape (rows, cols, 3), got {self._grid.shape}.")
        self._image_width = float(image_width)
        self._image_height = float(image_height)
        # Base K,D 전용 두 primitive(픽셀<->카메라 광선, 항등 투영)만 재사용한다 -
        # BaselineWindshieldModel 자체는 절대 K,D를 수정하지 않으므로 안전하게
        # 내부 헬퍼로 감싸 쓸 수 있다(Spherical과 동일한 재사용 패턴).
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def _delta(self, u: float, v: float) -> np.ndarray:
        return bilinear_interpolate_grid(self._grid, u, v, self._image_width, self._image_height)

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다(grid 보정이 픽셀에
        따라 달라지므로) - Base K,D 투영을 초기값으로 삼아 작은 2변수
        root-solve로 푼다(전체 이미지를 뒤지는 brute-force가 아니다) -
        SphericalWindshieldModel.project_point()와 동일한 구조."""
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
                "(residual ray correction may not cover this region well)."
            )
        return float(result.x[0]), float(result.x[1])

    def ray_angular_error_deg(self, u: float, v: float, target_point_cam: np.ndarray) -> Optional[float]:
        """관측 픽셀(u,v)의 보정된 광선과, 카메라 좌표계의 실제 목표점 방향
        사이의 각도(도) - Spherical과 동일한 개념의 보조 metric."""
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        target = np.asarray(target_point_cam, dtype=np.float64)
        norm = np.linalg.norm(target)
        if norm < 1e-9:
            return None
        cos_angle = float(np.clip(np.dot(target / norm, corrected), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Calibration (fitting)
# ---------------------------------------------------------------------------

def _grid_settings(config: WindshieldConfig) -> tuple[int, int, float, float]:
    hint = config.residual_ray_hint or {}
    rows = int(hint.get("grid_rows", DEFAULT_GRID_ROWS))
    cols = int(hint.get("grid_cols", DEFAULT_GRID_COLS))
    lambda_mag = float(hint.get("lambda_mag", DEFAULT_LAMBDA_MAG))
    lambda_smooth = float(hint.get("lambda_smooth", DEFAULT_LAMBDA_SMOOTH))
    if rows < 2 or cols < 2:
        raise ValueError(f"residual_ray_hint grid_rows/grid_cols must each be >= 2 (got rows={rows}, cols={cols}).")
    return rows, cols, lambda_mag, lambda_smooth


def _fit_residual_grid(
    observed_pixels: np.ndarray,     # (N, 2)
    d_obs: np.ndarray,               # (N, 3) - Base K,D로 구한 관측 픽셀의 광선(고정)
    p_cam: np.ndarray,                # (N, 3) - 각 코너의 목표점(카메라 좌표)
    rows: int,
    cols: int,
    image_width: float,
    image_height: float,
    lambda_mag: float,
    lambda_smooth: float,
):
    """ray-alignment residual + magnitude/smoothness regularization으로
    grid 전체를 한 번에 피팅한다. project_point()의 중첩 root-solve가 전혀
    필요 없다(모듈 docstring의 "Fitting 목적함수" 참고) - 매 residual 평가가
    순수 forward 계산이라 grid가 커도(최대 144 파라미터) 수 초 안에 수렴한다.
    """
    n_points = len(d_obs)
    target_dirs = np.array([normalize(p) for p in p_cam])

    # 인접 node 쌍(가로/세로) - smoothness residual에 재사용.
    horizontal_pairs = [(r, c, r, c + 1) for r in range(rows) for c in range(cols - 1)]
    vertical_pairs = [(r, c, r + 1, c) for r in range(rows - 1) for c in range(cols)]
    smooth_pairs = horizontal_pairs + vertical_pairs

    def residual(params: np.ndarray) -> np.ndarray:
        grid = params.reshape(rows, cols, 3)

        data_res = np.empty((n_points, 3))
        for i in range(n_points):
            u, v = observed_pixels[i]
            delta = bilinear_interpolate_grid(grid, u, v, image_width, image_height)
            corrected = normalize(d_obs[i] + delta)
            data_res[i] = target_dirs[i] - corrected

        mag_res = math.sqrt(lambda_mag) * grid.ravel()

        smooth_res = np.empty((len(smooth_pairs), 3))
        for i, (r0, c0, r1, c1) in enumerate(smooth_pairs):
            smooth_res[i] = math.sqrt(lambda_smooth) * (grid[r0, c0] - grid[r1, c1])

        return np.concatenate([data_res.ravel(), mag_res, smooth_res.ravel()])

    x0 = np.zeros(rows * cols * 3)  # "보정 없음"에서 시작 - hint 의존적이지 않은 중립적 초기값
    bounds = (np.full_like(x0, -5.0), np.full_like(x0, 5.0))
    return least_squares(residual, x0=x0, bounds=bounds, method="trf", loss="soft_l1", f_scale=0.05)


@dataclass
class _ResidualRayEvalOutcome:
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


def _evaluate_residual_ray(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    model: ResidualRayWindshieldModel,
    image_size: tuple[int, int],
) -> _ResidualRayEvalOutcome:
    """Residual Ray 모델로 프레임들을 평가한다 - Spherical의
    _evaluate_spherical()과 동일한 패턴(진짜 project_point() 기반 pixel 평가,
    투영-무관 집계 함수 재사용)."""
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

    return _ResidualRayEvalOutcome(
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
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=config.base_model_name,
        base_camera_matrix=config.base_camera_matrix,
        base_distortion=config.base_distortion,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        success=False,
        error_message=message,
    )


def calibrate_residual_ray(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Residual Ray Grid 모델을 fitting한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않는다.
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size
    rows, cols, lambda_mag, lambda_smooth = _grid_settings(config)
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

    observed_pixels: list[np.ndarray] = []
    d_obs_list: list[np.ndarray] = []
    p_cam_list: list[np.ndarray] = []
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        det = frame.detection
        R, _ = cv2.Rodrigues(rvec)
        obj = det.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = det.corners.reshape(-1, 2)
        for (px, py), p_cam in zip(corners, cam_pts):
            try:
                d = baseline_model.unproject_pixel(float(px), float(py))
            except Exception:  # noqa: BLE001
                continue
            observed_pixels.append(np.array([px, py], dtype=np.float64))
            d_obs_list.append(np.asarray(d, dtype=np.float64))
            p_cam_list.append(p_cam)

    if len(d_obs_list) < min_corners:
        return _failure_result(
            config, train_ids, test_ids,
            f"Residual grid({rows}x{cols})를 추정하기에 코너 수가 부족합니다 "
            f"(코너 {len(d_obs_list)}개, 최소 {min_corners}개 필요).",
        )

    observed_pixels_arr = np.asarray(observed_pixels)
    d_obs_arr = np.asarray(d_obs_list)
    p_cam_arr = np.asarray(p_cam_list)

    fit_result = _fit_residual_grid(
        observed_pixels_arr, d_obs_arr, p_cam_arr, rows, cols, width, height, lambda_mag, lambda_smooth,
    )
    if not fit_result.success or not np.all(np.isfinite(fit_result.x)):
        return _failure_result(config, train_ids, test_ids, "Residual grid optimization이 수렴하지 않았습니다.")

    grid = fit_result.x.reshape(rows, cols, 3)
    residual_model = ResidualRayWindshieldModel(K, D, model, grid, width, height)

    train_outcome = _evaluate_residual_ray(ok_frames, rvecs, tvecs, residual_model, image_size)
    total_train_points = train_outcome.num_points_ok + train_outcome.num_points_failed
    train_failure_rate = (train_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 grid로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    fitted_params: dict[str, float] = {
        "grid_rows": float(rows),
        "grid_cols": float(cols),
        "image_width": float(width),
        "image_height": float(height),
        "lambda_mag": float(lambda_mag),
        "lambda_smooth": float(lambda_smooth),
        "optimizer_cost": float(fit_result.cost),
        "num_fit_points": float(len(d_obs_list)),
    }
    for r in range(rows):
        for c in range(cols):
            fitted_params[f"grid_dx_{r}_{c}"] = float(grid[r, c, 0])
            fitted_params[f"grid_dy_{r}_{c}"] = float(grid[r, c, 1])
            fitted_params[f"grid_dz_{r}_{c}"] = float(grid[r, c, 2])

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=model,
        base_camera_matrix=K,
        base_distortion=D,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        failed_frame_ids=list(failed_ids),
        per_frame_error=train_outcome.per_frame_error,
        residual_stats=train_outcome.residual_stats,
        regional_error=train_outcome.regional_error,
        radial_profile=train_outcome.radial_profile,
        radial_bands=train_outcome.radial_bands,
        spatial_error_map=train_outcome.spatial_error_map,
        mean_dx=train_outcome.mean_dx,
        mean_dy=train_outcome.mean_dy,
        ray_angular_error_deg=train_outcome.ray_angular_error_deg,
        fitted_params=fitted_params,
        success=True,
    )

    if test_ids:
        test_frames = _subset_frames(windshield_dataset, test_ids)
        if test_frames:
            t_ok_frames, t_rvecs, t_tvecs, t_failed = solve_poses_fixed_intrinsics(test_frames, K, D, model)
            if t_ok_frames:
                test_outcome = _evaluate_residual_ray(t_ok_frames, t_rvecs, t_tvecs, residual_model, image_size)
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
                        f"Test 코너의 {test_failure_rate*100:.0f}%에서 유효한 pixel 예측을 계산하지 "
                        "못했습니다 (Test 결과의 신뢰도가 낮을 수 있습니다)."
                    )
            for fid in t_failed:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result
