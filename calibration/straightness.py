"""
camera_calibrator.calibration.straightness
================================================

설계 문서 3.4번 - Line Straightness Residual.

    "보정된 이미지에서 직선이어야 할 요소(격자선, 문틀 등) 위 점을 추출해
     직선 ax+by+c=0에 피팅. 점과 직선 사이 평균 거리가 0.5px 이하여야
     방사 왜곡이 완벽히 제거된 것으로 판단."

문서 원문은 "문틀" 같은 이미지 속 임의의 직선 요소를 상정하지만, 이 프로젝트는
ChArUco 패턴을 쓰고 있어 훨씬 안정적인 소스가 이미 손에 있다: **체스보드 격자
자체의 행/열이 실세계에서 정확히 직선**이다. 별도 Hough 변환이나 사용자가
"이 영역이 직선입니다"를 지정하는 UI 없이도, 이미 검출된 ChArUco 코너의
board id만으로 같은 행/열에 속하는 점들을 정확히 골라낼 수 있다.

board id -> (row, col) 매핑 (cv2.aruco.CharucoBoard의 실측 규약, 실험으로 검증됨):
    row = id // (squares_x - 1)
    col = id % (squares_x - 1)

핵심 원리: 왜곡이 완벽히 제거됐다면, 실세계에서 직선인 보드의 한 행(또는 열)은
"왜곡 보정된" 픽셀 좌표에서도 정확히 일직선이어야 한다 (핀홀 카메라는 3D 직선을
2D 직선으로 사영하므로). 왜곡 모델이 안 맞을수록 undistort 후에도 살짝 휘어
있는 자국이 남고, 그 휘어진 정도가 곧 잔차다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import CameraModelType, Frame, PatternConfig

# 한 줄(행 또는 열)을 직선으로 피팅하려면 최소 이 정도 점은 있어야 신뢰할 만하다.
# 3점이면 이미 "무조건 거의 직선"에 가까워 잔차가 과소평가되기 쉽다.
MIN_POINTS_PER_LINE = 4


def _undistort_points_pixel_space(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> np.ndarray:
    """검출된(왜곡 있는) 픽셀 좌표를 "왜곡이 없다고 가정했을 때의" 픽셀 좌표로 변환.

    P=camera_matrix를 넘겨서 정규화 좌표가 아니라 같은 K로 다시 픽셀 공간에
    투영되게 한다 - 그래야 결과 잔차의 단위가 원본과 동일한 "px"가 된다.
    """
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    if model == CameraModelType.FISHEYE:
        undistorted = cv2.fisheye.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix)
    else:
        undistorted = cv2.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix)
    return undistorted.reshape(-1, 2)


def _fit_line_residual(points: np.ndarray) -> float:
    """총최소제곱(Total Least Squares)으로 ax+by+c=0 직선을 피팅하고,
    점들과 직선 사이 평균 수직 거리를 반환.

    일반 최소제곱(y = mx+b)은 수직에 가까운 선(x 변화가 거의 없는 세로줄)에서
    기울기가 발산해 깨지므로, PCA(공분산 행렬의 고유벡터) 기반 TLS를 쓴다 -
    어떤 방향의 선이든 동일하게 안정적으로 처리된다.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순 정렬됨
    normal = eigvecs[:, 0]  # 최소 고유값의 고유벡터 = 직선에 수직인 방향

    a, b = normal
    c = -(a * centroid[0] + b * centroid[1])

    distances = np.abs(a * centered[:, 0] + b * centered[:, 1])  # centered라 c 상쇄됨
    return float(distances.mean())


def compute_straightness_residual(
    frames: list[Frame],
    pattern: PatternConfig,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    min_points_per_line: int = MIN_POINTS_PER_LINE,
) -> tuple[float | None, int]:
    """프레임들의 ChArUco 코너를 행/열로 묶어 각각 직선 피팅하고,
    전체 점-직선 거리의 평균을 반환.

    Returns:
        (straightness_residual_px, 사용된 라인 개수). 피팅 가능한 라인이
        하나도 없으면 (None, 0).
    """
    cols_per_row = pattern.squares_x - 1  # id -> row/col 역산에 필요

    all_distances: list[float] = []
    num_lines = 0

    for frame in frames:
        det = frame.detection
        if not det or not det.success or det.corners is None or det.ids is None:
            continue
        if det.num_corners < min_points_per_line:
            continue

        try:
            undistorted = _undistort_points_pixel_space(det.corners, camera_matrix, distortion, model)
        except cv2.error:
            # 왜곡 계수가 이 프레임의 극단적인 코너에서 수치적으로 불안정할 수 있음 -
            # 해당 프레임만 건너뛰고 나머지는 계속 진행.
            continue

        ids_flat = det.ids.reshape(-1)
        rows: dict[int, list[np.ndarray]] = {}
        cols: dict[int, list[np.ndarray]] = {}
        for idx, cid in enumerate(ids_flat):
            cid = int(cid)
            row_key = cid // cols_per_row
            col_key = cid % cols_per_row
            rows.setdefault(row_key, []).append(undistorted[idx])
            cols.setdefault(col_key, []).append(undistorted[idx])

        for group in list(rows.values()) + list(cols.values()):
            if len(group) < min_points_per_line:
                continue
            pts = np.array(group)
            all_distances.append(_fit_line_residual(pts))
            num_lines += 1

    if not all_distances:
        return None, 0

    return float(np.mean(all_distances)), num_lines


def format_straightness_summary(residual: float | None, num_lines: int) -> str:
    if residual is None:
        return "Line Straightness: 계산할 수 있는 직선(행/열)이 부족합니다."
    grade = (
        "Excellent" if residual < 0.3 else
        "Good" if residual < 0.5 else
        "Warning" if residual < 1.0 else
        "Poor"
    )
    return f"Line Straightness: {residual:.3f}px ({num_lines}개 라인 기준, {grade})"
