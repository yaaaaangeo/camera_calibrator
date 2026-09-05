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
bilinear interpolation으로 임의의 픽셀에서 값을 구한다 - 이 모듈은 그 중
"Grid + Bilinear" variant만 구현한다. RBF 기반 variant(STEP 3-B)는
`calibration/windshield/residual_rbf.py`에 별도로 구현돼 있고, 두 variant
모두 같은 `WindshieldModelType.RESIDUAL_RAY` 아래에서
`WindshieldConfig.residual_ray_hint["method"]`("grid" 기본값 / "rbf")로만
구분된다(WindshieldModelType에 새 멤버를 추가하지 않는다) - dispatch는
`calibration/windshield/validation.py::run_windshield_calibration`과
`ui/windshield_worker.py`에 있다.

Grid/RBF가 공유해야 하는 부분(pose refinement 정책, corrected-ray angular
stability, pixel-domain evaluation, Repeated Hold-out 요약 형태)은
`calibration/windshield/residual_common.py`에 있다 - 이 파일은 그 공유
유틸을 가져다 쓰고, Grid 고유의 fitting 로직(bilinear interpolation +
least_squares)만 갖고 있다.

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

STEP 3-A 안정화(2차 라운드) - 이번에 추가된 것:
    STAGE A(grid만, pose 고정) 다음에 STAGE B(grid + per-frame pose를
    alternating으로 공동 refine)를 추가했다 - Spherical STEP 2의
    calibrate_spherical()과 완전히 동일한 이유/구조다: Standard solvePnP가
    windshield distortion 일부를 pose로 흡수할 수 있으므로, 이 pose를
    "초기값일 뿐"으로 취급하고 grid와 번갈아 재추정한다. STAGE B도 STAGE A와
    같은 ray-alignment residual을 쓰므로(project_point()의 중첩 root-solve
    없이) 계산이 여전히 저렴하다. STAGE B가 실제 pixel RMS를 개선하지
    못하면 조용히 무시하지 않고 STAGE A로 되돌아가며 warning_message에
    남긴다(Spherical과 동일한 정책). 최적화 자체의 residual은 ray-domain,
    STAGE A/B 최종 채택 여부 판단 기준은 실제 pixel-domain RMS다.

    추가로 Repeated Hold-out(여러 split 평균/표준편차), Grid Resolution
    Comparison + AUTO 선택(Hold-out RMS가 비슷하면 parameter가 적은 쪽
    선택), corrected-ray angular stability(split 간 fitted grid 변동을
    물리적 각도로 측정), 그리고 runtime/calibration parameter 수 구분
    표기를 추가했다.
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
from calibration.windshield.refraction import normalize
from calibration.windshield.residual_common import (
    DEFAULT_REPEATED_HOLDOUT_SEEDS,
    POSE_ROTATION_REG_WEIGHT,
    POSE_TRANSLATION_REG_WEIGHT,
    MAX_PROJECT_POINT_ANGULAR_ERROR_DEG,
    RepeatedHoldoutSummary,
    collect_corner_arrays,
    compute_ray_stability_deg,
    evaluate_residual_ray_model,
    populate_pose_diagnostics,
    populate_repeated_holdout_diagnostics,
    refine_frame_pose_ray_domain,
)

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
_PROJECT_PENALTY = 5.0

GRID_STAGE_B_NUM_ROUNDS = 2

# Grid Resolution Comparison / AUTO 선택(사용자 스펙 7/8번) 기본 후보 -
# (rows, cols) 순서는 기존 convention(bilinear_interpolate_grid의 grid.shape
# == (rows, cols, 3))과 맞춘다.
DEFAULT_GRID_CANDIDATES: list[tuple[int, int]] = [(3, 4), (4, 6), (6, 8), (8, 12)]
# Hold-out RMS가 이 비율 이내로 비슷하면(사용자 스펙 8번 "거의 같은 경우")
# parameter가 더 적은 후보를 선택한다.
GRID_SELECTION_TIE_TOLERANCE = 0.05


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

    이미지 밖의 픽셀이 들어와도 grid 범위 안으로 clamp해서(외삽 대신 가장
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


# ---------------------------------------------------------------------------
# STAGE B - Grid + per-frame pose joint refinement (ray-domain, alternating)
# ---------------------------------------------------------------------------

@dataclass
class _JointGridRefinementOutcome:
    grid: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    converged_cleanly: bool


def _joint_refine_grid_and_poses(
    ok_frames: list[Frame],
    observed_pixels_per_frame: list[np.ndarray],
    d_obs_per_frame: list[np.ndarray],
    initial_rvecs: list[np.ndarray],
    initial_tvecs: list[np.ndarray],
    rows: int,
    cols: int,
    image_width: float,
    image_height: float,
    lambda_mag: float,
    lambda_smooth: float,
    initial_grid: np.ndarray,
    num_rounds: int = GRID_STAGE_B_NUM_ROUNDS,
) -> _JointGridRefinementOutcome:
    """STAGE B - alternating(block-coordinate) 방식으로 grid와 프레임별
    pose를 번갈아 refine한다. 한 라운드 = (모든 프레임 pose refine) ->
    (grid refine, 최신 pose로 재계산한 p_cam 사용). calibration.windshield.
    spherical._joint_refine_sphere_and_poses와 완전히 동일한 패턴.

    pose refine 자체는 residual_common.refine_frame_pose_ray_domain(공유)에
    위임한다 - Grid는 그 콜백으로 bilinear_interpolate_grid를 넘긴다."""
    rvecs = [np.asarray(r, dtype=np.float64).copy() for r in initial_rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).copy() for t in initial_tvecs]
    grid = np.asarray(initial_grid, dtype=np.float64).copy()
    converged_cleanly = True

    for _ in range(num_rounds):
        for i, frame in enumerate(ok_frames):
            delta_fn = lambda u, v, g=grid: bilinear_interpolate_grid(g, u, v, image_width, image_height)  # noqa: E731
            pose_fit = refine_frame_pose_ray_domain(
                frame, observed_pixels_per_frame[i], d_obs_per_frame[i], delta_fn,
                initial_rvecs[i], initial_tvecs[i],
            )
            if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                rvecs[i] = pose_fit.x[:3].reshape(3, 1)
                tvecs[i] = pose_fit.x[3:6].reshape(3, 1)
            else:
                converged_cleanly = False

        p_cam_list = []
        obs_pixel_list = []
        d_obs_list = []
        for i, frame in enumerate(ok_frames):
            R, _ = cv2.Rodrigues(rvecs[i])
            obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
            cam_pts = (R @ obj.T).T + tvecs[i].reshape(1, 3)
            p_cam_list.append(cam_pts)
            obs_pixel_list.append(observed_pixels_per_frame[i])
            d_obs_list.append(d_obs_per_frame[i])
        p_cam_arr = np.concatenate(p_cam_list, axis=0)
        obs_pixel_arr = np.concatenate(obs_pixel_list, axis=0)
        d_obs_arr = np.concatenate(d_obs_list, axis=0)

        grid_fit = _fit_residual_grid(
            obs_pixel_arr, d_obs_arr, p_cam_arr, rows, cols, image_width, image_height, lambda_mag, lambda_smooth,
        )
        if grid_fit.success and np.all(np.isfinite(grid_fit.x)):
            grid = grid_fit.x.reshape(rows, cols, 3).copy()
        else:
            converged_cleanly = False

    return _JointGridRefinementOutcome(grid=grid, rvecs=rvecs, tvecs=tvecs, converged_cleanly=converged_cleanly)


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

    흐름: (AUTO 지정 시 grid 해상도 선택, train_ids만 사용) -> STAGE A(grid만,
    pose 고정) -> STAGE B(grid+pose joint, ray-domain alternating) -> 두
    stage의 실제 pixel RMS를 비교해 더 나은 쪽을 최종으로 채택 -> Train 평가
    -> Test는 최종 grid를 완전히 고정한 채 자기 pose만 별도로 refine한 뒤
    평가(leakage 없음).
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size

    hint = config.residual_ray_hint or {}
    if hint.get("auto_grid", 0.0) > 0:
        (auto_rows, auto_cols), _candidates = select_best_grid_resolution(
            windshield_dataset, config, camera_config, train_ids,
        )
        config = dataclasses.replace(
            config,
            residual_ray_hint={**hint, "grid_rows": float(auto_rows), "grid_cols": float(auto_cols), "auto_grid": 0.0},
        )

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
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(
        ok_frames, rvecs, tvecs, baseline_model
    )

    total_corners = sum(len(a) for a in d_obs_per_frame)
    if total_corners < min_corners:
        return _failure_result(
            config, train_ids, test_ids,
            f"Residual grid({rows}x{cols})를 추정하기에 코너 수가 부족합니다 "
            f"(코너 {total_corners}개, 최소 {min_corners}개 필요).",
        )

    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    # --- STAGE A: grid만, pose 고정 ---
    stage_a_fit = _fit_residual_grid(
        observed_pixels_arr, d_obs_arr, p_cam_arr, rows, cols, width, height, lambda_mag, lambda_smooth,
    )
    if not stage_a_fit.success or not np.all(np.isfinite(stage_a_fit.x)):
        return _failure_result(config, train_ids, test_ids, "Residual grid optimization(STAGE A)이 수렴하지 않았습니다.")

    stage_a_grid = stage_a_fit.x.reshape(rows, cols, 3).copy()
    stage_a_model = ResidualRayWindshieldModel(K, D, model, stage_a_grid, width, height)
    stage_a_outcome = evaluate_residual_ray_model(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: grid + per-frame pose joint refinement (ray-domain) ---
    joint = _joint_refine_grid_and_poses(
        ok_frames, observed_pixels_per_frame, d_obs_per_frame, rvecs, tvecs,
        rows, cols, width, height, lambda_mag, lambda_smooth, stage_a_grid,
    )

    stage_used_is_joint_refined = False
    final_grid = stage_a_grid
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    refinement_note = ""

    stage_b_model = ResidualRayWindshieldModel(K, D, model, joint.grid, width, height)
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
            "STAGE B(ray-domain alternating grid/pose refinement)가 STAGE A(grid-only initial fit)보다 "
            "실제 pixel RMS를 개선하지 못해 STAGE A 결과를 최종으로 사용했습니다. "
            "(참고: 최적화 자체의 residual은 ray-domain이고, 이 STAGE A/B 채택 여부 판단 기준만 "
            "실제 pixel-domain RMS를 사용합니다.) "
        )

    final_model = ResidualRayWindshieldModel(K, D, model, final_grid, width, height)

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 grid로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    # Parameter count 구분(사용자 스펙 12번) - runtime model이 실제로 들고
    # 다니는 파라미터(grid)와, 이번 calibration 과정에서만 쓰인 nuisance
    # parameter(프레임별 pose)를 분리해서 기록한다.
    runtime_param_count = rows * cols * 3
    pose_param_count_train = len(ok_frames) * 6

    fitted_params: dict[str, float] = {
        "residual_ray_method": 0.0,  # 0.0 = Grid + Bilinear, 1.0 = RBF(residual_rbf.py) - build_projector가 이 값으로 분기
        "grid_rows": float(rows),
        "grid_cols": float(cols),
        "image_width": float(width),
        "image_height": float(height),
        "lambda_mag": float(lambda_mag),
        "lambda_smooth": float(lambda_smooth),
        "stage_a_optimizer_cost": float(stage_a_fit.cost),
        "num_fit_points": float(total_corners),
        "runtime_param_count": float(runtime_param_count),
        "pose_param_count_train": float(pose_param_count_train),
        "stage_used_is_joint_refined": 1.0 if stage_used_is_joint_refined else 0.0,
    }
    # Pose 진단(사용자 스펙 4번) - STAGE B가 initial solvePnP pose에서 실제로
    # 얼마나 움직였는지를 직접 계산한다(STAGE A가 최종으로 채택됐다면
    # final_rvecs/final_tvecs가 rvecs/tvecs와 동일한 객체라 델타는 항상 0).
    populate_pose_diagnostics(fitted_params, rvecs, tvecs, final_rvecs, final_tvecs)
    for r in range(rows):
        for c in range(cols):
            fitted_params[f"grid_dx_{r}_{c}"] = float(final_grid[r, c, 0])
            fitted_params[f"grid_dy_{r}_{c}"] = float(final_grid[r, c, 1])
            fitted_params[f"grid_dz_{r}_{c}"] = float(final_grid[r, c, 2])

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
                # grid 기준으로 pose만 다시 refine한다 - grid/K/D는 여기서 절대
                # 건드리지 않는다(leakage 없음).
                t_rvecs, t_tvecs = [], []
                t_obs_pixels, t_d_obs, _t_p_cam = collect_corner_arrays(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, baseline_model
                )
                grid_delta_fn = lambda u, v: bilinear_interpolate_grid(final_grid, u, v, width, height)  # noqa: E731
                for frame, init_rvec, init_tvec, obs_px, d_obs in zip(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, t_obs_pixels, t_d_obs
                ):
                    pose_fit = refine_frame_pose_ray_domain(
                        frame, obs_px, d_obs, grid_delta_fn, init_rvec, init_tvec, regularize=True,
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
# Repeated Hold-out (사용자 스펙 9/10번)
# ---------------------------------------------------------------------------

def _grid_from_fitted_params(fitted_params: dict[str, float]) -> np.ndarray:
    rows, cols = int(fitted_params["grid_rows"]), int(fitted_params["grid_cols"])
    grid = np.zeros((rows, cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            grid[r, c, 0] = fitted_params[f"grid_dx_{r}_{c}"]
            grid[r, c, 1] = fitted_params[f"grid_dy_{r}_{c}"]
            grid[r, c, 2] = fitted_params[f"grid_dz_{r}_{c}"]
    return grid


def run_repeated_holdout_residual_ray(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    test_ratio: float = 0.25,
) -> RepeatedHoldoutSummary:
    """windshield_dataset 전체를 여러 (다른 seed의) Train/Test로 나눠 반복
    평가한다 - 단일 split 하나의 우연에 결과가 흔들리는지 확인하기 위함."""
    from calibration.models.common import regional_edge_average

    test_rmses: list[float] = []
    test_p95s: list[float] = []
    edge_rmses: list[float] = []
    grids: list[np.ndarray] = []
    models: list[WindshieldModel] = []
    successful_seeds: list[int] = []

    K, D, model_name = config.base_camera_matrix, config.base_distortion, config.base_model_name
    width, height = infer_image_size(windshield_dataset, camera_config)

    for seed in seeds:
        train_ids, test_ids = split_train_test(windshield_dataset, camera_config, test_ratio, seed)
        result = calibrate_residual_ray(windshield_dataset, config, camera_config, train_ids, test_ids)
        if not result.success or result.test_residual_stats is None:
            continue
        successful_seeds.append(seed)
        test_rmses.append(result.test_residual_stats.rmse)
        if result.test_residual_stats.p95 is not None:
            test_p95s.append(result.test_residual_stats.p95)
        edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
        if edge is not None:
            edge_rmses.append(edge)
        grid = _grid_from_fitted_params(result.fitted_params)
        grids.append(grid)
        models.append(ResidualRayWindshieldModel(
            K, D, model_name, grid,
            result.fitted_params["image_width"], result.fitted_params["image_height"],
        ))

    grid_stability_l2 = None
    if len(grids) >= 2:
        distances = []
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                if grids[i].shape == grids[j].shape:
                    distances.append(float(np.linalg.norm(grids[i] - grids[j])))
        grid_stability_l2 = float(np.mean(distances)) if distances else None

    ray_stability_mean_deg, ray_stability_p95_deg = compute_ray_stability_deg(models, width, height)

    return RepeatedHoldoutSummary(
        seeds_used=successful_seeds,
        n_successful=len(successful_seeds),
        mean_test_rmse=float(np.mean(test_rmses)) if test_rmses else None,
        std_test_rmse=float(np.std(test_rmses)) if test_rmses else None,
        mean_test_p95=float(np.mean(test_p95s)) if test_p95s else None,
        mean_edge_rms=float(np.mean(edge_rmses)) if edge_rmses else None,
        grid_stability_l2=grid_stability_l2,
        ray_stability_mean_deg=ray_stability_mean_deg,
        ray_stability_p95_deg=ray_stability_p95_deg,
    )


# ---------------------------------------------------------------------------
# Grid Resolution Comparison + AUTO 선택 (사용자 스펙 7/8번)
# ---------------------------------------------------------------------------

@dataclass
class GridCandidateResult:
    rows: int
    cols: int
    param_count: int
    summary: RepeatedHoldoutSummary


def select_best_grid_resolution(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    candidate_frame_ids: list[str],
    candidates: Optional[list[tuple[int, int]]] = None,
    seeds: Optional[tuple[int, ...]] = None,
    inner_test_ratio: float = 0.3,
    tie_tolerance: float = GRID_SELECTION_TIE_TOLERANCE,
) -> tuple[tuple[int, int], list[GridCandidateResult]]:
    """candidate_frame_ids(바깥쪽 호출부의 Train만) 안에서 다시 나눠서
    (내부 train/내부 test) 여러 grid 해상도를 비교한다 - 바깥쪽 실제 Test는
    이 함수 어디에도 등장하지 않는다(모델 선택 자체도 leakage 없이 이뤄짐).

    선택 규칙: Hold-out RMS가 가장 좋은 후보를 기준으로, tie_tolerance
    이내로 비슷한 후보들 중 parameter가 가장 적은 것을 최종 선택한다
    (사용자 스펙 8번 "RMS가 거의 같으면 parameter 적은 쪽").

    candidates/seeds가 None이면 모듈 상수(DEFAULT_GRID_CANDIDATES/
    DEFAULT_REPEATED_HOLDOUT_SEEDS의 앞 3개)를 함수 호출 시점에 읽는다 -
    default 인자로 직접 바인딩하지 않는 이유는 그러면 모듈 상수를 나중에
    바꿔도(예: 테스트에서 monkeypatch) 이미 정의된 함수의 default 값은 정의
    시점에 고정돼버려 반영되지 않기 때문이다.
    """
    if candidates is None:
        candidates = DEFAULT_GRID_CANDIDATES
    if seeds is None:
        seeds = DEFAULT_REPEATED_HOLDOUT_SEEDS[:3]
    inner_dataset = Dataset(frames=_subset_frames(windshield_dataset, candidate_frame_ids))
    candidate_results: list[GridCandidateResult] = []

    for rows, cols in candidates:
        cfg = dataclasses.replace(
            config,
            residual_ray_hint={
                **(config.residual_ray_hint or {}),
                "grid_rows": float(rows), "grid_cols": float(cols), "auto_grid": 0.0,
            },
        )
        summary = run_repeated_holdout_residual_ray(inner_dataset, cfg, camera_config, seeds=seeds, test_ratio=inner_test_ratio)
        candidate_results.append(GridCandidateResult(rows=rows, cols=cols, param_count=rows * cols * 3, summary=summary))

    valid = [c for c in candidate_results if c.summary.mean_test_rmse is not None]
    if not valid:
        # 아무 후보도 평가할 수 없으면(데이터 부족 등) 가장 단순한 후보로
        # fallback한다 - calibrate_residual_ray의 본 fitting이 그 이유를
        # 다시 명확한 실패 메시지로 보고할 것이다.
        return candidates[0], candidate_results

    best_rmse = min(c.summary.mean_test_rmse for c in valid)
    close_enough = [c for c in valid if c.summary.mean_test_rmse <= best_rmse * (1.0 + tie_tolerance)]
    chosen = min(close_enough, key=lambda c: c.param_count)
    return (chosen.rows, chosen.cols), candidate_results


# ---------------------------------------------------------------------------
# Diagnostics orchestrator - UI/worker가 호출하는 단일 진입점
# ---------------------------------------------------------------------------

def run_residual_ray_calibration_with_diagnostics(
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
    """calibrate_residual_ray()에 Repeated Hold-out + Ray Stability 진단을
    더한 결과를 반환한다 - UI(및 그 뒤의 worker)가 Residual Ray(Grid) 모델을
    실행할 때 호출하는 단일 진입점(사용자 스펙 2번, "UI는 backend가 계산해
    둔 값만 표시한다"는 요구사항을 만족시키는 지점이 바로 여기다).

    본 계산(calibrate_residual_ray)이 AUTO 모드로 grid 해상도를 스스로
    고른 경우, 여기서 다시 반복 계산할 때는 그 선택된 해상도를 "고정"한
    resolved config로 Repeated Hold-out을 돌린다 - 그렇지 않으면 (a) 매
    반복 seed마다 AUTO의 select_best_grid_resolution이 다시 실행돼 비용이
    기하급수적으로 커지고, (b) split마다 grid 크기 자체가 달라져서 Ray
    Stability 비교가 "같은 해상도끼리"가 아니게 되어 의미가 없어진다.
    """
    hint = config.residual_ray_hint or {}
    was_auto = hint.get("auto_grid", 0.0) > 0

    result = calibrate_residual_ray(windshield_dataset, config, camera_config, train_ids, test_ids)
    if not result.success:
        return result

    result.fitted_params["diag_selection_mode_is_auto"] = 1.0 if was_auto else 0.0

    if compute_repeated_holdout:
        outer_train_dataset = Dataset(frames=_subset_frames(windshield_dataset, train_ids))
        resolved_hint = {
            **hint,
            "auto_grid": 0.0,
            "grid_rows": result.fitted_params["grid_rows"],
            "grid_cols": result.fitted_params["grid_cols"],
        }
        resolved_config = dataclasses.replace(config, residual_ray_hint=resolved_hint)
        summary = run_repeated_holdout_residual_ray(
            outer_train_dataset, resolved_config, camera_config,
            seeds=repeated_holdout_seeds, test_ratio=repeated_holdout_test_ratio,
        )
        populate_repeated_holdout_diagnostics(result.fitted_params, summary, len(repeated_holdout_seeds))

    return result
