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
남긴다. 실제 windshield는 얇은 곡면 셸이라 두 표면의 법선이 거의 평행해서
각도 변화는 대부분 서로 상쇄되고 주로 약간의 lateral shift만 남는다(평면
유리창을 통과할 때와 같은 현상의 곡면 버전) - 이게 실제 windshield 굴절의
지배적 효과이므로, 첫 구현부터 이 형태로 만든다. thickness는
WindshieldConfig.glass_thickness_m에서 고정값으로만 받고 최적화하지 않는다
(n_air/n_glass와 동일한 취급 - 사용자 스펙 5/9번).

Optimization 대상은 sphere_center(x,y,z) + sphere_radius(안쪽 표면 반지름)
4개뿐이다. Base Camera Model의 K,D는 이 모듈의 어떤 함수도 절대 재추정하지
않는다 - WindshieldConfig.base_camera_matrix/base_distortion을 그대로 읽기만
한다.

좌표계는 다른 windshield 모듈과 동일: OpenCV 카메라 좌표계(+x 오른쪽, +y
아래쪽, +z 전방). sphere_center와 3D target point 모두 이 좌표계 기준이다.
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

_SPHERE_BOUNDS = (
    [-100.0, -100.0, -100.0, 0.05],
    [100.0, 100.0, 100.0, 100.0],
)
_PROJECT_PENALTY = 5.0
_PROJECT_FAILURE_COST_THRESHOLD = 1.0
_ORIGIN = np.zeros(3, dtype=np.float64)


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


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

    SphericalWindshieldModel._refract_camera_ray()가 이 자유 함수를 그대로
    감싸고, sphere 최적화의 residual 계산도 매 후보 파라미터마다 객체를
    새로 만들지 않고 이 함수를 직접 호출한다(불필요한 오브젝트 생성 없이
    빠르게 반복 평가하기 위함).

    실패(교차 없음/전반사)하면 ValueError를 던진다 - 조용히 잘못된 값을
    반환하지 않는다.
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
                return np.full(3, _PROJECT_PENALTY)
            to_target = target - point
            norm = np.linalg.norm(to_target)
            if norm < 1e-9:
                return np.zeros(3)
            return to_target / norm - direction

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        if result.cost > _PROJECT_FAILURE_COST_THRESHOLD:
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
# Calibration (fitting)
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
    def residual(params: np.ndarray) -> np.ndarray:
        center = params[:3]
        radius = params[3]
        out = np.empty((len(d_obs), 3))
        if radius <= 0:
            out[:] = _PROJECT_PENALTY
            return out.ravel()
        for i in range(len(d_obs)):
            try:
                point, direction = _refract_through_shell(
                    d_obs[i], _ORIGIN, center, radius, thickness, n_air, n_glass
                )
            except ValueError:
                out[i] = _PROJECT_PENALTY
                continue
            to_target = p_cam[i] - point
            norm = np.linalg.norm(to_target)
            if norm < 1e-9:
                out[i] = 0.0
                continue
            out[i] = to_target / norm - direction
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
    않는다(사용자 스펙 19번 - Spherical Hold-out은 그 함수들을 직접 쓰면
    이론적으로 틀림). 대신 SphericalWindshieldModel.project_point()로 직접
    예측 픽셀을 구하고, 결과(x,y,dx,dy) 배열을 기존의 투영-무관 집계 함수
    (compute_residual_stats/compute_regional_error/bin_radial_errors/
    bin_radial_error_bands/bin_spatial_errors)에 그대로 넘긴다.
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
            except Exception:  # noqa: BLE001 - cv2가 던질 수 있는 예외까지 방어
                continue
            d_obs_list.append(np.asarray(d, dtype=np.float64))
            p_cam_list.append(p_cam)

    if len(d_obs_list) < MIN_CORNERS_FOR_FIT:
        return _failure_result(config, train_ids, test_ids, "Sphere를 추정하기에 코너 수가 부족합니다.")

    d_obs_arr = np.asarray(d_obs_list)
    p_cam_arr = np.asarray(p_cam_list)
    median_depth = float(np.median(p_cam_arr[:, 2]))
    initial_center, initial_radius = _initial_sphere_guess(config, median_depth)

    fit_result = _fit_sphere(d_obs_arr, p_cam_arr, n_air, n_glass, thickness, initial_center, initial_radius)

    if not np.all(np.isfinite(fit_result.x)) or fit_result.x[3] <= 0:
        return _failure_result(config, train_ids, test_ids, "Sphere optimization이 유효하지 않은 값으로 발산했습니다.")

    center = fit_result.x[:3]
    radius = float(fit_result.x[3])
    spherical_model = SphericalWindshieldModel(K, D, model, center, radius, n_air, n_glass, thickness)

    train_outcome = _evaluate_spherical(ok_frames, rvecs, tvecs, spherical_model, image_size)
    total_train_points = train_outcome.num_points_ok + train_outcome.num_points_failed
    train_failure_rate = (train_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"Fitted sphere로 코너의 {train_failure_rate*100:.0f}%에서 유효한 굴절을 계산하지 "
            "못했습니다 (initial guess/hint를 조정해보세요).",
        )

    fitted_params = {
        "sphere_center_x": float(center[0]),
        "sphere_center_y": float(center[1]),
        "sphere_center_z": float(center[2]),
        "sphere_radius": radius,
        "glass_refractive_index": float(n_glass),
        "air_refractive_index": float(n_air),
        "glass_thickness_m": float(thickness),
        "initial_center_x": float(initial_center[0]),
        "initial_center_y": float(initial_center[1]),
        "initial_center_z": float(initial_center[2]),
        "initial_radius": float(initial_radius),
        "optimizer_cost": float(fit_result.cost),
        "num_fit_points": float(len(d_obs_list)),
    }

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPHERICAL,
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
                test_outcome = _evaluate_spherical(t_ok_frames, t_rvecs, t_tvecs, spherical_model, image_size)
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
                        f"Test 코너의 {test_failure_rate*100:.0f}%에서 유효한 굴절을 계산하지 "
                        "못했습니다 (Test 결과의 신뢰도가 낮을 수 있습니다)."
                    )
            for fid in t_failed:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result
