"""
camera_calibrator.calibration.radial_profile
=================================================

설계 문서 4번 - Radial Error Profile (Edge/Radial Error Map).

    "전체 RMS 하나만으로는 부족하다. 자율주행용 120°급 카메라에서는
     중앙은 잘 맞아도 외곽에서 2~3px씩 틀릴 수 있다."
    "이미지 중심으로부터의 거리(radius)에 따른 재투영 오차를 그래프로
     표시하면, 렌즈 외곽에서 모델이 잘 동작하는지 바로 확인 가능하다."

RegionalError(center/left/right/top/bottom/corner, models/common.py)는
"프레임 단위"로 하나의 대표 위치(board_center_px)만 보고 영역을 나누지만,
이 모듈은 "코너 포인트 단위"로 모든 코너 각각의 (반지름, 오차)를 모아
구간별 평균을 낸다. 화각 전역의 경향(중심 vs 외곽)을 훨씬 세밀하게 보여준다.

pinhole.py / extended_pinhole.py / fisheye.py 세 모델 함수 모두 이 함수를
동일하게 호출해서 CalibrationResult.radial_profile을 채운다 - regional_error와
동일한 패턴(공용 모듈, 모델마다 같은 기준)을 따른다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import CameraModelType, Frame, RadialBin, RadialErrorProfile


def _project(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> np.ndarray:
    """모델에 맞는 projectPoints를 호출해 (N,2) 재투영 좌표를 반환.

    fisheye는 cv2.fisheye.projectPoints를 써야 하고 float64를 요구한다
    (calibration.models.fisheye와 동일한 주의사항).
    """
    if model == CameraModelType.FISHEYE:
        obj = object_points.astype(np.float64)
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
    else:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return projected.reshape(-1, 2)


def compute_radial_error_profile(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    model: CameraModelType,
    num_bins: int = 8,
) -> RadialErrorProfile:
    """모든 프레임의 모든 코너를 모아 반지름 구간별 평균 재투영 오차를 계산.

    Args:
        frames: object_points/corners를 가진 프레임 리스트. rvecs/tvecs와
            반드시 같은 순서/개수여야 한다 (models/*.py의 collect_calibration_inputs
            결과를 그대로 사용하는 것을 전제로 함 - 호출부에서 순서를 보장해야 함).
        num_bins: 0 ~ max_radius를 몇 구간으로 나눌지. 프레임 수가 적으면
            (코너 포인트 총량도 적으므로) 구간을 너무 잘게 쪼개면 구간마다
            포인트가 1~2개뿐이라 노이즈가 심해진다 - 호출부(UI)에서 필요시
            줄여서 재호출할 수 있게 파라미터로 노출.
    """
    w, h = image_size
    center_x, center_y = w / 2.0, h / 2.0
    max_radius = float(np.hypot(center_x, center_y))  # 이미지 대각선의 절반 (코너까지 거리)

    if max_radius <= 0 or len(frames) != len(rvecs) or len(frames) != len(tvecs):
        return RadialErrorProfile(bins=[], max_radius=max_radius)

    all_radii: list[float] = []
    all_errors: list[float] = []

    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        det = frame.detection
        if not det or det.object_points is None or det.corners is None:
            continue

        try:
            projected = _project(det.object_points, rvec, tvec, camera_matrix, distortion, model)
        except cv2.error:
            # 한 프레임의 pose/투영이 잘못돼도 전체 profile 계산이 죽지 않게 건너뛴다.
            continue

        detected = det.corners.reshape(-1, 2)
        if detected.shape[0] != projected.shape[0]:
            continue

        radii = np.hypot(detected[:, 0] - center_x, detected[:, 1] - center_y)
        errors = np.hypot(detected[:, 0] - projected[:, 0], detected[:, 1] - projected[:, 1])

        all_radii.extend(radii.tolist())
        all_errors.extend(errors.tolist())

    if not all_radii:
        return RadialErrorProfile(bins=[], max_radius=max_radius)

    radii_arr = np.array(all_radii)
    errors_arr = np.array(all_errors)

    bin_edges = np.linspace(0.0, max_radius, num_bins + 1)
    bins: list[RadialBin] = []
    for i in range(num_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        # 마지막 구간만 상한 포함 (딱 코너에 걸리는 포인트 포함시키기 위함)
        if i < num_bins - 1:
            mask = (radii_arr >= lo) & (radii_arr < hi)
        else:
            mask = (radii_arr >= lo) & (radii_arr <= hi)

        count = int(mask.sum())
        mean_err = float(errors_arr[mask].mean()) if count > 0 else None
        bins.append(RadialBin(radius_min=lo, radius_max=hi, mean_error=mean_err, num_points=count))

    return RadialErrorProfile(bins=bins, max_radius=max_radius)


# ---------------------------------------------------------------------------
# 출력용 요약
# ---------------------------------------------------------------------------

def format_radial_profile(profile: RadialErrorProfile) -> str:
    """터미널/CLI 확인용 ASCII 표.

        반지름(px)        평균오차(px)   포인트 수
        0 - 135              0.31          142
        135 - 270             0.34          201
        ...
        945 - 1080(외곽)      1.82           88
    """
    if not profile.bins:
        return "Radial Error Profile을 계산할 데이터가 없습니다."

    lines = [f"{'반지름(px)':<22}{'평균오차(px)':>12}{'포인트 수':>10}"]
    for i, b in enumerate(profile.bins):
        label = f"{b.radius_min:.0f} - {b.radius_max:.0f}"
        if i == len(profile.bins) - 1:
            label += "(외곽)"
        err = f"{b.mean_error:.3f}" if b.mean_error is not None else "N/A"
        lines.append(f"{label:<22}{err:>12}{b.num_points:>10}")
    return "\n".join(lines)
