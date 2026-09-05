"""
camera_calibrator.calibration.windshield.residual_common
==============================================================

Residual Ray의 두 variant(Grid+Bilinear, RBF)가 공유해야 하는 부분만 모은
모듈. "공유해야 하는 부분"은 사용자 스펙이 명시적으로 강조한 것들이다:

    - pose refinement 정책(ray-domain residual + weak initial-pose prior)
    - corrected-ray angular stability의 sampling 위치
    - 실제 pixel-domain evaluation
    - Repeated Hold-out 결과 요약 형태 + 기본 seed 집합

반대로 "각 variant 고유의 fitting 로직"(Grid의 bilinear interpolation +
least_squares, RBF의 RBFInterpolator)은 이 모듈에 넣지 않는다 - 억지로 하나의
상위 추상화로 합치면 두 variant 모두의 코드가 오히려 읽기 어려워지고, 이미
검증된 Grid 코드를 건드리는 리스크만 커진다.

이 모듈의 함수들은 전부 `WindshieldModel` ABC(project_point/unproject_pixel)
API에만 의존한다 - Grid든 RBF든, 심지어 향후 다른 variant든 이 API만
만족하면 그대로 재사용할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibration.models.common import compute_regional_error
from calibration.radial_profile import bin_radial_error_bands, bin_radial_errors
from calibration.residual_stats import compute_residual_stats
from calibration.spatial_error_map import bin_spatial_errors
from calibration.types import Frame, RadialErrorProfile, RegionalError, ResidualStats, SpatialErrorMap
from calibration.windshield.base import WindshieldModel
from calibration.windshield.baseline import BaselineWindshieldModel

# ---------------------------------------------------------------------------
# 공유 상수 - Grid/RBF의 pose prior 정책이 불필요하게 달라지지 않도록 한 곳만
# 수정하면 양쪽에 반영되게 한다(사용자 스펙 18번).
# ---------------------------------------------------------------------------
POSE_ROTATION_REG_WEIGHT = 2.0    # per radian
POSE_TRANSLATION_REG_WEIGHT = 2.0  # per meter
MAX_PROJECT_POINT_ANGULAR_ERROR_DEG = 2.0
DEFAULT_REPEATED_HOLDOUT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)


def normalize_pixel_coordinates(u: float, v: float, image_width: float, image_height: float) -> tuple[float, float]:
    """픽셀 좌표를 [-1,1]x[-1,1] normalized 좌표로 바꾼다(사용자 스펙 5번).

        u_n = 2*u/W - 1
        v_n = 2*v/H - 1

    RBF 입력은 항상 이 convention을 쓴다 - Train/runtime/YAML load 이후
    전부 동일해야 하므로, 픽셀->normalized 변환이 필요한 모든 곳(fitting,
    runtime 조회, 재구성)이 이 함수 하나만 호출한다.
    """
    un = 2.0 * float(u) / float(image_width) - 1.0
    vn = 2.0 * float(v) / float(image_height) - 1.0
    return un, vn


def _rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """두 rotation vector(rvec) 사이의 회전각(도) - trace(R_b @ R_a^T) 공식."""
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).ravel())
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).ravel())
    cos_theta = float(np.clip((np.trace(Rb @ Ra.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def _translation_delta_mm(tvec_a: np.ndarray, tvec_b: np.ndarray) -> float:
    """두 translation vector(tvec, 미터 단위) 사이 거리를 mm로 환산."""
    a = np.asarray(tvec_a, dtype=np.float64).ravel()
    b = np.asarray(tvec_b, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) * 1000.0)


def populate_pose_diagnostics(
    fitted_params: dict[str, float],
    initial_rvecs: list[np.ndarray], initial_tvecs: list[np.ndarray],
    final_rvecs: list[np.ndarray], final_tvecs: list[np.ndarray],
) -> None:
    """STAGE B가 initial solvePnP pose에서 실제로 얼마나 움직였는지를
    fitted_params에 직접 기록한다(Grid/RBF 공유 - 완전히 동일한 계산이라
    한 곳에만 둔다). STAGE A가 최종으로 채택됐다면 final==initial pose이므로
    델타는 항상 정확히 0이 된다."""
    delta_r_degs = [_rotation_angle_deg(r0, r1) for r0, r1 in zip(initial_rvecs, final_rvecs)]
    delta_t_mms = [_translation_delta_mm(t0, t1) for t0, t1 in zip(initial_tvecs, final_tvecs)]
    fitted_params["diag_pose_delta_r_median_deg"] = float(np.median(delta_r_degs)) if delta_r_degs else 0.0
    fitted_params["diag_pose_delta_r_p95_deg"] = float(np.percentile(delta_r_degs, 95)) if delta_r_degs else 0.0
    fitted_params["diag_pose_delta_t_median_mm"] = float(np.median(delta_t_mms)) if delta_t_mms else 0.0
    fitted_params["diag_pose_delta_t_p95_mm"] = float(np.percentile(delta_t_mms, 95)) if delta_t_mms else 0.0


def collect_corner_arrays(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    baseline_model: BaselineWindshieldModel,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """프레임마다 (관측 픽셀, base ray, 목표점) per-frame 배열을 만든다 -
    STAGE A는 이걸 펼쳐서(concat) 쓰고, STAGE B는 프레임 단위 그대로
    (pose refine이 프레임별이므로) 쓴다. Grid/RBF 완전히 동일한 코드."""
    observed_pixels_per_frame: list[np.ndarray] = []
    d_obs_per_frame: list[np.ndarray] = []
    p_cam_per_frame: list[np.ndarray] = []
    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        det = frame.detection
        R, _ = cv2.Rodrigues(rvec)
        obj = det.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = det.corners.reshape(-1, 2)

        pixels, rays, targets = [], [], []
        for (px, py), p_cam in zip(corners, cam_pts):
            try:
                d = baseline_model.unproject_pixel(float(px), float(py))
            except Exception:  # noqa: BLE001
                continue
            pixels.append([px, py])
            rays.append(d)
            targets.append(p_cam)
        observed_pixels_per_frame.append(np.array(pixels, dtype=np.float64))
        d_obs_per_frame.append(np.array(rays, dtype=np.float64))
        p_cam_per_frame.append(np.array(targets, dtype=np.float64))

    return observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame


def refine_frame_pose_ray_domain(
    frame: Frame,
    observed_pixels: np.ndarray,
    d_obs: np.ndarray,
    delta_fn: Callable[[float, float], np.ndarray],
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    *,
    regularize: bool = True,
):
    """한 프레임의 pose(rvec,tvec)만 ray-alignment residual로 refine한다.
    보정 메커니즘(Grid bilinear/RBF)은 고정된 `delta_fn(u,v)->Δd(3,)` 콜백
    하나로 추상화된다 - 이 함수 자체는 절대 delta_fn을 수정하지 않는다.

    Train의 STAGE B alternating refinement와 Test의 pose-only hold-out
    refinement 양쪽이(Grid든 RBF든) 이 함수 하나를 재사용한다.

    observed_pixels: (N,2) - 이 프레임 코너들의 관측 픽셀(u,v). delta_fn을
        읽어오는 위치는 항상 "관측 픽셀"이지 목표점을 투영한 위치가
        아니다(project_point()의 정의와 일치시키기 위함).
    d_obs: (N,3) - 각 코너의 observed_pixels에서 Base K,D로 구한 광선(고정,
        pose/보정 모델 어느 쪽에도 의존하지 않음 - 미리 한 번만 계산해 재사용).

    initial_rvec/initial_tvec은 항상 "이 프레임의 원래 Standard solvePnP
    추정값"이다(라운드가 반복돼도 계속 같은 기준점) - 매 라운드의 이전
    결과가 아니라 고정된 최초 추정값을 향한 weak prior여야 "너무 멀리
    도망가지 마라"는 의미가 유지된다.
    """
    from calibration.windshield.refraction import normalize

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
            delta = delta_fn(float(u), float(v))
            corrected = normalize(d_obs[i] + delta)
            out[i] = normalize(cam_pts[i]) - corrected
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
class ResidualEvalOutcome:
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


def evaluate_residual_ray_model(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    model: WindshieldModel,
    image_size: tuple[int, int],
) -> ResidualEvalOutcome:
    """어떤 Residual Ray variant든(Grid/RBF) 진짜 project_point() 기반 pixel
    평가를 한다 - `model.project_point()`/`model.ray_angular_error_deg()`만
    호출하므로 grid-specific/rbf-specific 코드가 전혀 없다. STAGE A/B의
    최적화가 ray-domain residual을 쓰더라도, 최종 보고되는 모든 지표는
    항상 이 함수를 거친 진짜 픽셀 값이다.
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

            if hasattr(model, "ray_angular_error_deg"):
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

    return ResidualEvalOutcome(
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


def fixed_evaluation_pixels(
    image_width: float, image_height: float, sample_rows: int = 12, sample_cols: int = 20,
) -> np.ndarray:
    """Ray Stability 계산 전용 - 비교 대상 모델의 내부 표현(Grid rows/cols,
    RBF center 수 등)과 무관하게 항상 동일한 물리적 픽셀 위치를 샘플링하기
    위한 고정 격자(사용자 스펙 23/24번 "Grid와 RBF가 서로 다른 sampling
    위치를 사용하면 안 된다"). 이미지 경계를 포함해 image_width/image_height
    전체에 고르게 퍼진 (sample_rows*sample_cols, 2) 배열을 반환한다."""
    us = np.linspace(0.0, float(image_width), sample_cols)
    vs = np.linspace(0.0, float(image_height), sample_rows)
    grid_u, grid_v = np.meshgrid(us, vs)
    return np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)


def compute_ray_stability_deg(
    models: list[WindshieldModel],
    image_width: float,
    image_height: float,
    sample_rows: int = 12,
    sample_cols: int = 20,
) -> tuple[Optional[float], Optional[float]]:
    """서로 다른 split에서 얻은 여러 fitted 모델(Grid든 RBF든, 심지어 섞여
    있어도)이 실제로 얼마나 다른 광선을 만들어내는지를 물리적으로 의미
    있는 각도(도)로 측정한다. 고정 샘플 픽셀(fixed_evaluation_pixels)에서
    각 모델의 `unproject_pixel()`을 직접 호출해 "보정된 광선"을 얻는다 -
    Grid의 grid 배열이나 RBF의 center/RBFInterpolator 내부를 전혀 몰라도
    되므로, 새 variant가 추가돼도 이 함수는 그대로 재사용 가능하다.

    모델이 2개 미만이면 비교 대상이 없으므로 (None, None)을 반환한다(억지로
    0을 만들지 않는다) - Stability Score 같은 0~100 인위적 지표로 바꾸지
    말라는 요구사항에 따라, 반환값은 항상 실제 각도(도) 단위다.
    """
    if len(models) < 2:
        return None, None

    sample_pixels = fixed_evaluation_pixels(image_width, image_height, sample_rows, sample_cols)

    corrected_rays_per_model = []
    for model in models:
        rays = np.array([model.unproject_pixel(float(u), float(v)) for u, v in sample_pixels])
        corrected_rays_per_model.append(rays)

    all_angles: list[float] = []
    for i in range(len(corrected_rays_per_model)):
        for j in range(i + 1, len(corrected_rays_per_model)):
            dots = np.clip(np.sum(corrected_rays_per_model[i] * corrected_rays_per_model[j], axis=1), -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            all_angles.extend(angles.tolist())

    if not all_angles:
        return None, None
    arr = np.array(all_angles)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


@dataclass
class RepeatedHoldoutSummary:
    """여러 Train/Test split에서 반복 평가한 결과 요약 + split마다 fitted
    모델이 얼마나 달라지는지(안정성). Grid/RBF 공유 - 필드 이름을 동일하게
    맞춰서 UI/Comparison 쪽 표시 로직도 그대로 재사용할 수 있게 한다."""
    seeds_used: list[int]
    n_successful: int
    mean_test_rmse: Optional[float] = None
    std_test_rmse: Optional[float] = None
    mean_test_p95: Optional[float] = None
    mean_edge_rms: Optional[float] = None
    grid_stability_l2: Optional[float] = None  # legacy/debug, Grid 전용(RBF는 항상 None)
    ray_stability_mean_deg: Optional[float] = None
    ray_stability_p95_deg: Optional[float] = None


def populate_repeated_holdout_diagnostics(
    fitted_params: dict[str, float],
    summary: RepeatedHoldoutSummary,
    n_requested: int,
) -> None:
    """RepeatedHoldoutSummary를 fitted_params의 diag_* 키로 펼쳐 담는다
    (Grid/RBF 공유 - 완전히 동일한 매핑)."""
    fitted_params["diag_repeated_n_requested"] = float(n_requested)
    fitted_params["diag_repeated_n_successful"] = float(summary.n_successful)
    if summary.mean_test_rmse is not None:
        fitted_params["diag_repeated_mean_test_rmse"] = summary.mean_test_rmse
    if summary.std_test_rmse is not None:
        fitted_params["diag_repeated_std_test_rmse"] = summary.std_test_rmse
    if summary.mean_test_p95 is not None:
        fitted_params["diag_repeated_mean_test_p95"] = summary.mean_test_p95
    if summary.mean_edge_rms is not None:
        fitted_params["diag_repeated_mean_edge_rms"] = summary.mean_edge_rms
    if summary.ray_stability_mean_deg is not None:
        fitted_params["diag_ray_stability_mean_deg"] = summary.ray_stability_mean_deg
    if summary.ray_stability_p95_deg is not None:
        fitted_params["diag_ray_stability_p95_deg"] = summary.ray_stability_p95_deg


def residual_ray_failure_result(config, train_ids: list[str], test_ids: list[str], message: str):
    """Grid/RBF가 공유하는 실패 결과 생성 보일러플레이트."""
    from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModelType

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
