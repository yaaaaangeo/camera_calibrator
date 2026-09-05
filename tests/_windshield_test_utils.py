"""
tests._windshield_test_utils
=================================

Windshield 테스트 전용 헬퍼 - 실제 ChArUco 이미지 렌더링/검출 없이, 알려진
K,D,pose로 noiseless 코너를 직접 생성한 최소 Dataset을 만든다. Baseline은
"관측 픽셀 - 예측 픽셀"만 비교하므로 실제 이미지/검출기가 전혀 필요 없다 -
이 방식이 tests/conftest.py::synthetic_distorted_dataset_dir(실제 이미지
렌더링 + 검출)보다 훨씬 빠르고, ground-truth 오차를 정확히 통제할 수 있다.

이 파일은 pytest 테스트 파일이 아니다(test_*.py 아님) - pytest가 수집하지
않고, 다른 test_windshield_*.py들이 import해서 쓰는 순수 헬퍼 모듈이다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

IMG_W, IMG_H = 1280, 800

# 6x5 평면 그리드(체스보드 코너 배치와 동일한 형태) - object_points로 사용.
_GRID_COLS, _GRID_ROWS = 6, 5
_SQUARE = 0.04

# 보드가 카메라 앞 다양한 위치/자세에 놓인 경우들 (rvec, tvec).
_POSES = [
    (np.array([0.05, -0.03, 0.0]), np.array([-0.15, -0.05, 0.6])),
    (np.array([0.02, 0.10, 0.05]), np.array([0.12, -0.08, 0.5])),
    (np.array([-0.08, 0.05, 0.0]), np.array([-0.05, 0.10, 0.7])),
    (np.array([0.0, -0.12, 0.02]), np.array([0.10, 0.05, 0.55])),
    (np.array([0.10, 0.02, -0.02]), np.array([-0.10, -0.10, 0.65])),
    (np.array([-0.05, -0.08, 0.03]), np.array([0.05, 0.12, 0.60])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.8])),
    (np.array([0.06, 0.06, -0.03]), np.array([-0.08, 0.08, 0.5])),
]


def _object_grid() -> np.ndarray:
    pts = []
    for r in range(_GRID_ROWS):
        for c in range(_GRID_COLS):
            pts.append([c * _SQUARE, r * _SQUARE, 0.0])
    return np.array(pts, dtype=np.float64).reshape(-1, 1, 3)


def default_camera_matrix_distortion() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[900.0, 0.0, IMG_W / 2], [0.0, 900.0, IMG_H / 2], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    return K, D


def default_camera_config() -> CameraConfig:
    return CameraConfig(width=IMG_W, height=IMG_H, sensor_name="windshield-test")


def build_synthetic_windshield_dataset(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    shear_k: float = 0.0,
) -> Dataset:
    """camera_matrix/distortion으로 noiseless 투영한 코너를 가진 프레임들을
    만든다.

    shear_k가 0이 아니면, 각 코너의 y좌표에 `shear_k * (x - cx)` 만큼 추가
    변위를 더한다 - 이미지 중심 기준 좌/우로 반대 방향 변위가 생기는 비강체
    (non-rigid) 패턴이다. 단일 rvec/tvec(강체 변환) + 카메라 모델로는 이
    패턴을 설명할 수 없으므로, solvePnP로 포즈를 다시 맞춰도 체계적인 잔차가
    남는다 - 실제 Windshield 굴절이 남기는 것과 같은 종류의 "설명 안 되는
    잔차"를 흉내낸다(균일한 픽셀 평행이동과 달리, PnP가 흡수해서 없애버릴 수
    없다).
    """
    obj = _object_grid()
    cx = float(camera_matrix[0, 2])
    frames: list[Frame] = []
    for i, (rvec, tvec) in enumerate(_POSES):
        frame_id = f"synth_{i:02d}"
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        pts = projected.reshape(-1, 2).astype(np.float64)
        if shear_k:
            pts[:, 1] += shear_k * (pts[:, 0] - cx)
        corners = pts.reshape(-1, 1, 2).astype(np.float32)
        center = pts.mean(axis=0)
        detection = DetectionResult(
            image_id=frame_id,
            success=True,
            corners=corners,
            object_points=obj.astype(np.float32),
            num_corners=corners.shape[0],
            board_area_ratio=0.2,
            board_center_px=(float(center[0]), float(center[1])),
        )
        frame = Frame(
            image_info=ImageInfo(image_id=frame_id, path=f"synthetic://{frame_id}", width=IMG_W, height=IMG_H),
            detection=detection,
            status=FrameStatus.DETECTED,
        )
        frames.append(frame)
    return Dataset(frames=frames)


# 기본 windshield sphere - _POSES의 보드 깊이(0.5~0.8m)보다 windshield가 카메라
# 쪽에 훨씬 가까이 있고(안쪽 표면 z ~= 0.3m), 곡률(radius)이 커서(10m) 실제
# 자동차 windshield처럼 완만하다. calibration/windshield/spherical.py의
# refraction.py 기반 forward model로 직접 생성하므로 "알려진 정답"이 명확하다.
DEFAULT_SPHERE_CENTER = np.array([0.0, 0.0, -9.7])
DEFAULT_SPHERE_RADIUS = 10.0
DEFAULT_GLASS_THICKNESS_M = 0.005
DEFAULT_AIR_INDEX = 1.0
DEFAULT_GLASS_INDEX = 1.52


def build_synthetic_spherical_windshield_dataset(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    sphere_center: np.ndarray = DEFAULT_SPHERE_CENTER,
    sphere_radius: float = DEFAULT_SPHERE_RADIUS,
    n_air: float = DEFAULT_AIR_INDEX,
    n_glass: float = DEFAULT_GLASS_INDEX,
    thickness: float = DEFAULT_GLASS_THICKNESS_M,
    model=None,
) -> Dataset:
    """known sphere+두 표면 Snell 굴절을 통해 "관측" 코너를 만든다(모두
    calibration/windshield/refraction.py와 spherical.py의 실제 forward model
    (SphericalWindshieldModel.project_point)을 그대로 사용) - Baseline용
    build_synthetic_windshield_dataset()과 달리 진짜 굴절 geometry를 통과시킨다.

    자기 자신의 구현(refraction.py/spherical.py)으로 정답 데이터를 만드는
    self-consistency 테스트라는 한계가 있다 - Snell 굴절 공식 자체의 정오는
    tests/test_windshield_refraction.py의 손으로 계산한 독립적인 각도
    검증이 담당한다.
    """
    from calibration.types import CameraModelType
    from calibration.windshield.spherical import SphericalWindshieldModel

    model = model or CameraModelType.BROWN_CONRADY
    true_model = SphericalWindshieldModel(
        camera_matrix, distortion, model, sphere_center, sphere_radius, n_air, n_glass, thickness
    )

    obj = _object_grid()
    frames: list[Frame] = []
    for i, (rvec, tvec) in enumerate(_POSES):
        frame_id = f"sph_{i:02d}"
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.reshape(-1, 3).T).T + tvec.reshape(1, 3)

        pixels = []
        ok = True
        for p in cam_pts:
            try:
                u, v = true_model.project_point(float(p[0]), float(p[1]), float(p[2]))
            except ValueError:
                ok = False
                break
            pixels.append([u, v])
        if not ok:
            continue

        corners = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
        center = corners.reshape(-1, 2).mean(axis=0)
        detection = DetectionResult(
            image_id=frame_id,
            success=True,
            corners=corners,
            object_points=obj.astype(np.float32),
            num_corners=corners.shape[0],
            board_area_ratio=0.2,
            board_center_px=(float(center[0]), float(center[1])),
        )
        frame = Frame(
            image_info=ImageInfo(image_id=frame_id, path=f"synthetic://{frame_id}", width=IMG_W, height=IMG_H),
            detection=detection,
            status=FrameStatus.DETECTED,
        )
        frames.append(frame)
    return Dataset(frames=frames)


# ---------------------------------------------------------------------------
# Residual Ray (STEP 3-A) 합성 ground-truth
# ---------------------------------------------------------------------------

def default_residual_delta_fn(camera_matrix, image_width: float = IMG_W, image_height: float = IMG_H, scale: float = 0.02):
    """사용자 스펙 25번 형태의 smooth ray-correction GT field.

        dx = a*xn^2 + b*yn
        dy = c*yn^3 + d*xn*yn

    (xn, yn)은 이미지 중심 기준 정규화 좌표(대략 [-1,1]). scale은 ray-direction
    단위로의 크기 조절 - 초점거리(~900px) 기준으로 몇 픽셀급 효과가 나도록
    작게 잡는다. Grid 구현(calibration/windshield/residual_ray.py)과 완전히
    독립적인 closed-form 함수다 - 이 함수 자체가 grid를 전혀 참조하지 않는다.
    """
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    half_w, half_h = image_width / 2.0, image_height / 2.0
    a, b, c, d = 0.6, 0.3, 0.5, 0.4

    def delta_fn(u: float, v: float) -> np.ndarray:
        xn = (u - cx) / half_w
        yn = (v - cy) / half_h
        dx = a * xn**2 + b * yn
        dy = c * yn**3 + d * xn * yn
        return np.array([scale * dx, scale * dy, 0.0])

    return delta_fn


def default_spline_bump_fn(amplitude: float = 0.006, theta0: float = 0.010, phi0: float = -0.006, sigma_rad: float = 0.012):
    """STEP 4 물리 모델 보완 라운드 - smooth Gaussian bump Δs(n0) GT field
    (미터 단위), **3D 방향(n0, base sphere 중심 기준 unit vector)의 각도
    거리** 함수로 정의한다 - production(calibration/windshield/spline.py)의
    (p,q) pixel-FOV 정규화 convention과 전혀 무관한 독립적 표현이다(사용자
    스펙 31/33번 - "Spline interpolation 자체를 그대로 호출하지 않는다").

        Delta_s(n0) = amplitude * exp(-angle(n0, n0_target)^2 / (2*sigma_rad^2))

    `theta0`/`phi0`/`sigma_rad`는 기본 테스트 sphere(DEFAULT_SPHERE_CENTER/
    DEFAULT_SPHERE_RADIUS, 카메라와 표준 렌즈 기준)에서 실제 관측되는 각도
    범위(~±0.02-0.03 rad, 별도 스크립트로 한 번 확인한 값 - production
    함수를 호출해 구한 게 아니라 상수로 박아둔 값이다)에 맞춰 고른 기본값
    이다 - 그래야 bump가 실제 코너들이 덮는 각도 범위 안에서 의미 있는
    변화를 만든다. amplitude=6mm는 기본 max_displacement_m bound(10mm)보다
    작아 optimizer bound에 걸리지 않는다.
    """
    import math

    theta_t, phi_t = theta0, phi0
    target = np.array([
        math.cos(phi_t) * math.sin(theta_t),
        math.sin(phi_t),
        math.cos(phi_t) * math.cos(theta_t),
    ])
    target = target / np.linalg.norm(target)

    def bump_fn(n0: np.ndarray) -> float:
        cos_angle = float(np.clip(np.dot(n0, target), -1.0, 1.0))
        angle = math.acos(cos_angle)
        return amplitude * math.exp(-(angle ** 2) / (2.0 * sigma_rad ** 2))

    return bump_fn


def build_synthetic_spline_windshield_dataset(
    camera_matrix,
    distortion,
    *,
    sphere_center: np.ndarray = DEFAULT_SPHERE_CENTER,
    sphere_radius: float = DEFAULT_SPHERE_RADIUS,
    n_air: float = DEFAULT_AIR_INDEX,
    n_glass: float = DEFAULT_GLASS_INDEX,
    thickness: float = DEFAULT_GLASS_THICKNESS_M,
    bump_fn=None,
    model=None,
) -> Dataset:
    """알려진 base sphere + smooth Gaussian bump(3D 방향 각도 거리 함수)
    형태의 **진짜 normal-offset surface deformation**을 통해 "관측" 코너를
    만든다. calibration/windshield/spline.py의 B-spline interpolation이나
    SplineWindshieldModel, `compute_angular_fov_scale`/(p,q) 매핑을 전혀
    호출하지 않는 완전히 독립적인 forward model이다(사용자 스펙 31/32/33번 -
    self-consistency 회피). 저수준 primitive(refraction.py의
    intersect_ray_sphere/refract_ray/normalize, BaselineWindshieldModel)만
    재사용한다 - 그 primitive들의 정오는 tests/test_windshield_refraction.py가
    독립적으로 검증한다.

    핵심: 이전 라운드의 버그(등가성이 성립하지 않는 `sphere(R+ds)` 교차)를
    GT 생성에서도 제거했다 - 여기서는 **base sphere(반지름 R, 변형 없음)와
    ray를 먼저 교차**시켜 그 지점의 방향 n0을 얻고, `S = P0 + Delta_s(n0)*n0`
    로 진짜 normal 방향 offset을 적용한다. Normal은 근처 픽셀들에서 이
    "진짜" deformed point(S)를 다시 계산해 finite difference로 구한다 -
    tangential 변화를 올바르게 반영한다."""
    from calibration.types import CameraModelType
    from calibration.windshield.baseline import BaselineWindshieldModel
    from calibration.windshield.refraction import intersect_ray_sphere, refract_ray
    from scipy.optimize import least_squares

    model = model or CameraModelType.BROWN_CONRADY
    baseline = BaselineWindshieldModel(camera_matrix, distortion, model)
    bump_fn = bump_fn or default_spline_bump_fn()
    center = np.asarray(sphere_center, dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64)

    def base_hit_and_normal(u: float, v: float):
        d = np.asarray(baseline.unproject_pixel(u, v), dtype=np.float64)
        hit = intersect_ray_sphere(origin, d, center, sphere_radius)
        if hit is None:
            return None
        p0, _t0 = hit
        n0 = (p0 - center) / sphere_radius
        return d, p0, n0

    def deformed_point(u: float, v: float):
        """S = P0(방향 n0) + Delta_s(n0) * n0 - 진짜 normal-offset 정의
        그대로(사용자 스펙 9번), sphere(R+ds) intersection이 아니다."""
        res = base_hit_and_normal(u, v)
        if res is None:
            return None
        _d, p0, n0 = res
        ds = bump_fn(n0)
        return p0 + ds * n0

    def normal_at(u: float, v: float, p: np.ndarray, step: float = 2.0):
        p_u1, p_u2 = deformed_point(u + step, v), deformed_point(u - step, v)
        p_v1, p_v2 = deformed_point(u, v + step), deformed_point(u, v - step)
        if any(x is None for x in (p_u1, p_u2, p_v1, p_v2)):
            return None
        tu = (p_u1 - p_u2) / (2.0 * step)
        tv = (p_v1 - p_v2) / (2.0 * step)
        n = np.cross(tu, tv)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            return None
        n = n / norm
        if np.dot(n, p - center) < 0.0:
            n = -n
        return n

    def refract_through_deformed(u: float, v: float):
        p = deformed_point(u, v)
        if p is None:
            raise ValueError("no inner hit")
        n1 = normal_at(u, v, p)
        if n1 is None:
            raise ValueError("degenerate normal")
        d_cam = np.asarray(baseline.unproject_pixel(u, v), dtype=np.float64)
        d_glass = refract_ray(d_cam, n1, n_air, n_glass)
        if d_glass is None:
            raise ValueError("TIR at inner surface")

        # Outer surface: local-normal thin-shell 근사(사용자 스펙 19/21번과
        # 같은 철학이지만, production의 3-unknown iterative solve를 재사용
        # 하지 않는 독립적이고 단순한 구현 - 굴절된 광선을, 예상 outer
        # point(p + thickness*n1)를 지나고 법선이 n1인 평면과 closed-form
        # ray-plane intersection으로 교차시킨다. thickness가 곡률반경보다
        # 훨씬 작으므로 국소 평면 근사가 합리적이다).
        outer_point_guess = p + thickness * n1
        denom = float(np.dot(d_glass, n1))
        if abs(denom) < 1e-9:
            raise ValueError("grazing ray at outer surface")
        tau = float(np.dot(outer_point_guess - p, n1)) / denom
        if tau <= 0.0:
            raise ValueError("outer surface behind ray")
        p2 = p + tau * d_glass
        n2 = n1  # N_outer ≈ N_inner (사용자 스펙 21번 fallback, 독립 재구현)

        d_out = refract_ray(d_glass, n2, n_glass, n_air)
        if d_out is None:
            raise ValueError("TIR at outer surface")
        return p2, d_out

    def project_with_bump(x: float, y: float, z: float):
        target = np.array([x, y, z], dtype=np.float64)
        initial_uv = np.asarray(baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv):
            try:
                point, direction = refract_through_deformed(float(uv[0]), float(uv[1]))
            except ValueError:
                return np.full(3, 5.0)
            to_target = target - point
            norm = np.linalg.norm(to_target)
            if norm < 1e-9:
                return np.zeros(3)
            return to_target / norm - direction

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        return float(result.x[0]), float(result.x[1])

    obj = _object_grid()
    frames: list[Frame] = []
    for i, (rvec, tvec) in enumerate(_POSES):
        frame_id = f"spl_{i:02d}"
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.reshape(-1, 3).T).T + tvec.reshape(1, 3)

        pixels = []
        ok = True
        for p in cam_pts:
            try:
                u, v = project_with_bump(float(p[0]), float(p[1]), float(p[2]))
            except Exception:  # noqa: BLE001
                ok = False
                break
            pixels.append([u, v])
        if not ok:
            continue

        corners = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
        center_px = corners.reshape(-1, 2).mean(axis=0)
        detection = DetectionResult(
            image_id=frame_id,
            success=True,
            corners=corners,
            object_points=obj.astype(np.float32),
            num_corners=corners.shape[0],
            board_area_ratio=0.2,
            board_center_px=(float(center_px[0]), float(center_px[1])),
        )
        frame = Frame(
            image_info=ImageInfo(image_id=frame_id, path=f"synthetic://{frame_id}", width=IMG_W, height=IMG_H),
            detection=detection,
            status=FrameStatus.DETECTED,
        )
        frames.append(frame)
    return Dataset(frames=frames)


def build_synthetic_residual_ray_dataset(camera_matrix, distortion, delta_fn, model=None) -> Dataset:
    """delta_fn(u,v)->[dx,dy,dz](ray-direction 단위)으로 정의된 임의의
    closed-form ray-correction field를 통해 "관측" 코너를 만든다.

    Grid 구현을 재사용하지 않고 이 함수 자체가 독립적으로 root-solve를
    수행한다(ResidualRayWindshieldModel.project_point()와 같은 계산이지만,
    grid 보간 대신 delta_fn을 직접 호출) - Grid Recovery 테스트가 "자기
    자신의 구현으로 정답을 만드는" 것을 피하기 위함이다.
    """
    from calibration.types import CameraModelType
    from calibration.windshield.baseline import BaselineWindshieldModel
    from calibration.windshield.refraction import normalize
    from scipy.optimize import least_squares

    model = model or CameraModelType.BROWN_CONRADY
    baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def project_with_delta(x: float, y: float, z: float) -> tuple[float, float]:
        target_dir = normalize(np.array([x, y, z], dtype=np.float64))
        initial_uv = np.asarray(baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            d_base = np.asarray(baseline.unproject_pixel(float(uv[0]), float(uv[1])), dtype=np.float64)
            corrected = normalize(d_base + delta_fn(float(uv[0]), float(uv[1])))
            return target_dir - corrected

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        return float(result.x[0]), float(result.x[1])

    obj = _object_grid()
    frames: list[Frame] = []
    for i, (rvec, tvec) in enumerate(_POSES):
        frame_id = f"res_{i:02d}"
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.reshape(-1, 3).T).T + tvec.reshape(1, 3)

        pixels = []
        ok = True
        for p in cam_pts:
            try:
                u, v = project_with_delta(float(p[0]), float(p[1]), float(p[2]))
            except Exception:  # noqa: BLE001
                ok = False
                break
            pixels.append([u, v])
        if not ok:
            continue

        corners = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
        center = corners.reshape(-1, 2).mean(axis=0)
        detection = DetectionResult(
            image_id=frame_id,
            success=True,
            corners=corners,
            object_points=obj.astype(np.float32),
            num_corners=corners.shape[0],
            board_area_ratio=0.2,
            board_center_px=(float(center[0]), float(center[1])),
        )
        frame = Frame(
            image_info=ImageInfo(image_id=frame_id, path=f"synthetic://{frame_id}", width=IMG_W, height=IMG_H),
            detection=detection,
            status=FrameStatus.DETECTED,
        )
        frames.append(frame)
    return Dataset(frames=frames)
