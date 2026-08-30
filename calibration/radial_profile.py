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

pinhole.py / extended_pinhole.py / fisheye.py(brown_conrady.py는
extended_pinhole.py를 재사용) 모두 이 함수를 동일하게 호출해서
CalibrationResult.radial_profile을 채운다 - regional_error와 동일한 패턴
(공용 모듈, 모델마다 같은 기준)을 따른다.
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
        obj = object_points.astype(np.float64).reshape(1, -1, 3)
        D = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, camera_matrix, D)
    else:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return projected.reshape(-1, 2)


def collect_per_point_vectors(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """모든 프레임의 모든 코너에 대해 (검출 위치 x, 검출 위치 y, dx, dy)를
    모은다. dx/dy = detected - projected, 즉 부호가 있는 재투영 오차 벡터다.

    collect_per_point_residuals()(반지름/스칼라 오차만 필요할 때)와
    spatial_error_map.py(방향까지 필요할 때)가 이 함수 하나를 공유한다 -
    "코너마다 projectPoints를 부르고 detected와 비교하는" 투영 로직 자체가
    두 곳에 따로 있으면 언젠가 한쪽만 고치는 사고가 난다.

    Returns:
        (xs, ys, dxs, dys) - 전부 1차원 배열, 길이가 같다. 계산 가능한
        포인트가 하나도 없으면 빈 배열 네 개를 반환한다(예외를 던지지 않음).
    """
    if len(frames) != len(rvecs) or len(frames) != len(tvecs):
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_x: list[float] = []
    all_y: list[float] = []
    all_dx: list[float] = []
    all_dy: list[float] = []

    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        det = frame.detection
        if not det or det.object_points is None or det.corners is None:
            continue

        try:
            projected = _project(det.object_points, rvec, tvec, camera_matrix, distortion, model)
        except cv2.error:
            # 한 프레임의 pose/투영이 잘못돼도 전체 계산이 죽지 않게 건너뛴다.
            continue

        detected = det.corners.reshape(-1, 2)
        if detected.shape[0] != projected.shape[0]:
            continue

        diff = detected - projected
        all_x.extend(detected[:, 0].tolist())
        all_y.extend(detected[:, 1].tolist())
        all_dx.extend(diff[:, 0].tolist())
        all_dy.extend(diff[:, 1].tolist())

    return np.array(all_x), np.array(all_y), np.array(all_dx), np.array(all_dy)


def collect_per_point_residuals(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    model: CameraModelType,
) -> tuple[np.ndarray, np.ndarray]:
    """모든 프레임의 모든 코너에 대해 (이미지 중심으로부터의 반지름, 재투영 오차)를
    모아 반환한다. compute_radial_error_profile()과 residual_stats.py가 둘 다
    이 함수를 재사용한다 - "코너 포인트 단위로 전부 모으는" 로직이 두 곳에
    따로 있으면 하나가 바뀔 때 다른 쪽을 깜빡 놓치기 쉽다.

    collect_per_point_vectors()의 부호 있는 (dx,dy)에서 반지름/스칼라 크기만
    뽑아내는 얇은 래퍼다 - 투영 계산 자체는 중복하지 않는다.

    Returns:
        (radii, errors) - 둘 다 1차원 배열, 길이가 같다. 계산 가능한 포인트가
        하나도 없으면 빈 배열 두 개를 반환한다(예외를 던지지 않음).
    """
    w, h = image_size
    center_x, center_y = w / 2.0, h / 2.0

    xs, ys, dxs, dys = collect_per_point_vectors(frames, rvecs, tvecs, camera_matrix, distortion, model)
    if xs.size == 0:
        return np.array([]), np.array([])

    radii = np.hypot(xs - center_x, ys - center_y)
    errors = np.hypot(dxs, dys)
    return radii, errors


def _radial_bin_stats(errors_in_bin: np.ndarray) -> dict:
    """한 구간에 속한 오차 배열로부터 문서 14번이 요구하는 5개 통계량을 계산.
    compute_radial_error_profile()과 compute_radial_error_bands()가 공유한다.
    """
    if errors_in_bin.size == 0:
        return dict(mean_error=None, median_error=None, rms_error=None, p95_error=None, max_error=None)
    return dict(
        mean_error=float(errors_in_bin.mean()),
        median_error=float(np.median(errors_in_bin)),
        rms_error=float(np.sqrt(np.mean(errors_in_bin ** 2))),
        p95_error=float(np.percentile(errors_in_bin, 95)),
        max_error=float(np.max(errors_in_bin)),
    )


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
    """모든 프레임의 모든 코너를 모아 반지름 구간별 오차 통계(Mean/Median/RMS/
    P95/Max)를 계산한다 - "Radius -> Error Curve"를 그리기 위한 촘촘한(기본
    8구간) 프로파일이다. 명명된 6단계 대역(Center~Corner)이 필요하면
    compute_radial_error_bands()를 쓴다 - 이 함수는 그대로 두고 별도로 둔
    이유는, 곡선을 매끄럽게 그리려면 구간 수가 자유로워야 하고(그래서 num_bins
    파라미터가 있음), 문서 14번이 요구하는 "표"는 이름 붙은 6구간으로 고정된
    별개의 요구사항이기 때문이다 - 하나로 억지로 합치면 둘 다 어중간해진다.

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
    max_radius = float(np.hypot(w / 2.0, h / 2.0))  # 이미지 대각선의 절반 (코너까지 거리)

    if max_radius <= 0:
        return RadialErrorProfile(bins=[], max_radius=max_radius)

    radii_arr, errors_arr = collect_per_point_residuals(
        frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )

    if radii_arr.size == 0:
        return RadialErrorProfile(bins=[], max_radius=max_radius)

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
        stats = _radial_bin_stats(errors_arr[mask])
        bins.append(RadialBin(radius_min=lo, radius_max=hi, num_points=count, **stats))

    return RadialErrorProfile(bins=bins, max_radius=max_radius)


# ---------------------------------------------------------------------------
# 설계 문서 14번 - 명명된 6단계 대역 (Center/Inner/Middle/Outer/Edge/Corner)
# ---------------------------------------------------------------------------

# max_radius(이미지 대각선의 절반) 대비 각 대역의 경계 비율. 6등분 균등 분할 -
# "Corner"가 마지막 1/6을 차지하는 게 실제 코너 영역과 대략 맞아떨어진다
# (예: 1920x1080에서 max_radius~1100px, Corner 대역은 916~1100px).
_RADIAL_BAND_LABELS = ["Center", "Inner", "Middle", "Outer", "Edge", "Corner"]
_RADIAL_BAND_BOUNDARIES = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]


def compute_radial_error_bands(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    model: CameraModelType,
) -> RadialErrorProfile:
    """설계 문서 14번 - Center/Inner/Middle/Outer/Edge/Corner 6단계 명명된
    대역별로 Mean/Median/RMS/P95/Max를 계산한다. compute_radial_error_profile()
    과 투영/수집 로직을 공유하고(collect_per_point_residuals), 경계와 라벨만
    다르다.
    """
    w, h = image_size
    max_radius = float(np.hypot(w / 2.0, h / 2.0))
    if max_radius <= 0:
        return RadialErrorProfile(bins=[], max_radius=max_radius)

    radii_arr, errors_arr = collect_per_point_residuals(
        frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )
    if radii_arr.size == 0:
        return RadialErrorProfile(bins=[], max_radius=max_radius)

    bins: list[RadialBin] = []
    for i, label in enumerate(_RADIAL_BAND_LABELS):
        lo = _RADIAL_BAND_BOUNDARIES[i] * max_radius
        hi = _RADIAL_BAND_BOUNDARIES[i + 1] * max_radius
        if i < len(_RADIAL_BAND_LABELS) - 1:
            mask = (radii_arr >= lo) & (radii_arr < hi)
        else:
            mask = (radii_arr >= lo) & (radii_arr <= hi)

        count = int(mask.sum())
        stats = _radial_bin_stats(errors_arr[mask])
        bins.append(RadialBin(radius_min=lo, radius_max=hi, num_points=count, label=label, **stats))

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


def format_radial_bands(profile: RadialErrorProfile) -> str:
    """설계 문서 14번 - Center/Inner/Middle/Outer/Edge/Corner 6단계 표.

        Radial Error Bands (Mean/Median/RMS/P95/Max)
        Band      Range(px)     Mean   Median      RMS      P95      Max    N
        Center      0 - 183    0.180    0.165    0.201    0.350    0.512  84
        Inner     183 - 367    0.210    0.195    0.235    0.410    0.598  91
        Middle    367 - 550    0.250    0.228    0.278    0.490    0.701  88
        Outer     550 - 733    0.310    0.285    0.342    0.601    0.855  79
        Edge      733 - 917    0.402    0.370    0.445    0.780    1.050  65
        Corner    917 - 1100   0.521    0.480    0.577    0.990    1.320  42
    """
    if not profile.bins:
        return "Radial Error Bands를 계산할 데이터가 없습니다."

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    header = f"{'Band':<8}{'Range(px)':>16}{'Mean':>9}{'Median':>9}{'RMS':>9}{'P95':>9}{'Max':>9}{'N':>6}"
    lines = ["Radial Error Bands (Mean/Median/RMS/P95/Max)", header]
    for b in profile.bins:
        band_name = b.label or "?"
        range_str = f"{b.radius_min:.0f} - {b.radius_max:.0f}"
        lines.append(
            f"{band_name:<8}{range_str:>16}{fmt(b.mean_error):>9}{fmt(b.median_error):>9}"
            f"{fmt(b.rms_error):>9}{fmt(b.p95_error):>9}{fmt(b.max_error):>9}{b.num_points:>6}"
        )
    return "\n".join(lines)


def radial_error_curve(profile: RadialErrorProfile, metric: str = "mean_error") -> list[tuple[float, float]]:
    """설계 문서 14번 - "Radius -> Error Curve". compute_radial_error_profile()의
    촘촘한 bins에서 (반지름 중심, 오차) 점들을 뽑아 곡선 데이터로 만든다.

    metric: RadialBin의 어느 통계량을 곡선 y값으로 쓸지
        ("mean_error"/"median_error"/"rms_error"/"p95_error"/"max_error").
    포인트가 없는(num_points=0) 구간은 건너뛴다 - 끊어진 곡선보다 존재하는
    점만 잇는 게 노이즈를 덜 만든다.
    """
    points = []
    for b in profile.bins:
        value = getattr(b, metric, None)
        if b.num_points > 0 and value is not None:
            points.append((b.radius_center, value))
    return points


def format_radial_curve(profile: RadialErrorProfile, metric: str = "mean_error", width: int = 40) -> str:
    """Radius -> Error Curve를 터미널용 ASCII 라인 그래프로 표시."""
    points = radial_error_curve(profile, metric=metric)
    if not points:
        return "Radius -> Error Curve를 계산할 데이터가 없습니다."

    max_err = max(v for _, v in points)
    lines = [f"Radius -> Error Curve ({metric})"]
    for radius, value in points:
        bar_len = int(round((value / max_err) * width)) if max_err > 0 else 0
        lines.append(f"r={radius:7.1f}px {'*' * bar_len} {value:.3f}")
    return "\n".join(lines)
