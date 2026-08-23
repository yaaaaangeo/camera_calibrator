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

from dataclasses import dataclass

import cv2
import numpy as np

from calibration.types import CameraModelType, Frame, PatternConfig, StraightnessBreakdown

# 한 줄(행 또는 열)을 직선으로 피팅하려면 최소 이 정도 점은 있어야 신뢰할 만하다.
# 3점이면 이미 "무조건 거의 직선"에 가까워 잔차가 과소평가되기 쉽다.
MIN_POINTS_PER_LINE = 4


@dataclass
class StraightnessLine:
    """보드의 한 행 또는 열(또는 대각선) - 시각화(ui/straightness_view.py)를 위한
    프레임 단위 상세 정보. compute_straightness_residual()의 집계용
    스칼라와 달리, "어느 줄이 얼마나 휘었는지"를 그대로 유지한다.
    """
    line_type: str  # "row", "col", "diag_main"(좌상->우하), "diag_anti"(우상->좌하)
    line_index: int  # board 상의 행/열/대각선 번호
    points: np.ndarray  # undistort된 픽셀 좌표, shape (N, 2)
    residual: float  # 이 줄만의 평균 점-직선 거리(px)
    # 설계 문서 15번 - Line Straightness 평가 강화. direction은 "row"/"col"을
    # horizontal/vertical/diagonal이라는 사람이 읽기 쉬운 이름으로, position은
    # 보드 안에서 이 줄이 중앙 쪽인지 가장자리 쪽인지를 나타낸다.
    direction: str = "horizontal"   # "horizontal" | "vertical" | "diagonal"
    position: str = "center"        # "center" | "edge" | "corner"


def _undistort_points_pixel_space(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    target_K: np.ndarray | None = None,
) -> np.ndarray:
    """검출된(왜곡 있는) 픽셀 좌표를 "왜곡이 없다고 가정했을 때의" 픽셀 좌표로 변환.

    target_K(=P)를 넘겨서 정규화 좌표가 아니라 특정 카메라 행렬 기준
    픽셀 공간으로 다시 투영되게 한다 - 기본값은 camera_matrix 그대로
    (숫자 집계용 compute_straightness_residual과 동일하게 동작, 기존 동작
    변경 없음). 화면에 실제 undistort된 이미지 위에 겹쳐 그리려면
    (ui/straightness_view.py), models.common.undistort_image()가 쓰는
    것과 같은 new_K를 넘겨야 좌표계가 이미지와 정확히 맞는다 - fisheye는
    estimateNewCameraMatrixForUndistortRectify로 K가 재추정되기 때문이다.
    """
    target_K = camera_matrix if target_K is None else target_K
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    if model == CameraModelType.FISHEYE:
        undistorted = cv2.fisheye.undistortPoints(pts, camera_matrix, distortion, P=target_K)
    else:
        undistorted = cv2.undistortPoints(pts, camera_matrix, distortion, P=target_K)
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
    # c = -(a*centroid[0] + b*centroid[1])는 직선 방정식(ax+by+c=0)을 완성하려면
    # 필요하지만, 아래 거리 계산은 이미 centroid를 뺀 좌표(centered)를 쓰므로
    # c가 수학적으로 상쇄되어 실제로는 필요 없다 - 계산하지 않고 생략한다.

    distances = np.abs(a * centered[:, 0] + b * centered[:, 1])
    return float(distances.mean())


def sort_points_along_line(points: np.ndarray) -> np.ndarray:
    """시각화(폴리라인으로 그리기)를 위해 점들을 직선 방향을 따라 정렬한다.

    ChArUco 검출 결과의 코너 순서는 board id 순서가 아니라 검출 알고리즘이
    찾은 순서라, 그대로 선을 이으면 지그재그가 된다. 공분산행렬의 가장 큰
    고유값 방향(=직선 방향, _fit_line_residual이 쓰는 것과 같은 PCA)으로
    투영한 값 기준 정렬하면 항상 한쪽 끝에서 반대쪽 끝으로 깔끔하게 이어진다.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1]  # 가장 큰 고유값 = 직선이 뻗은 방향
    projection = centered @ direction
    order = np.argsort(projection)
    return points[order]


def compute_frame_straightness_lines(
    frame: Frame,
    pattern: PatternConfig,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    min_points_per_line: int = MIN_POINTS_PER_LINE,
    target_K: np.ndarray | None = None,
) -> list[StraightnessLine]:
    """이 프레임 하나에서 만들어지는 모든 행/열 라인의 상세 정보를 반환.

    compute_straightness_residual()이 이 함수를 프레임마다 호출해 집계하므로,
    "전체 평균 하나"와 "어느 줄이 문제인지"가 항상 같은 계산에서 나온다
    (숫자가 서로 어긋날 일이 없음).
    """
    det = frame.detection
    if not det or not det.success or det.corners is None or det.ids is None:
        return []
    if det.num_corners < min_points_per_line:
        return []

    try:
        undistorted = _undistort_points_pixel_space(det.corners, camera_matrix, distortion, model, target_K)
    except cv2.error:
        # 왜곡 계수가 이 프레임의 극단적인 코너에서 수치적으로 불안정할 수 있음 -
        # 해당 프레임만 건너뛰고 나머지는 계속 진행.
        return []

    return _lines_from_points(det, pattern, undistorted, min_points_per_line)


def compute_frame_raw_straightness_lines(
    frame: Frame,
    pattern: PatternConfig,
    min_points_per_line: int = MIN_POINTS_PER_LINE,
) -> list[StraightnessLine]:
    """왜곡 보정을 아예 하지 않은, 검출된 그대로의(raw) 픽셀 좌표 기준 행/열 라인.

    `compute_frame_straightness_lines()`의 "보정 후" 잔차와 짝을 이루는
    "보정 전" 기준선이다 - Undistort Preview 탭(ui/preview.py)에서 "이 모델로
    보정하면 얼마나 좋아지는가"를 숫자로 보여주려면 보정 전 잔차가 필요한데,
    지금까지는 육안 비교만 있었다 (undistort_image()로 보정된 이미지를 보여줄
    뿐, "곧아진 정도"를 정량화하지 않음 - 프로젝트 철학 "RMS만 보고 판단 금지"를
    undistort 단계에서도 지키려면 여기도 숫자 근거가 있어야 한다).

    `_undistort_points_pixel_space()`를 아예 거치지 않고 검출된 코너를 그대로
    행/열로 묶어 직선을 피팅한다 - 카메라 모델(camera_matrix/distortion)이
    전혀 필요 없다는 점이 핵심이다: "보정 안 한 원본이 얼마나 휘어 있었는가"는
    어떤 모델을 나중에 적용할지와 무관한, 순수하게 검출 결과만으로 정해지는
    값이기 때문이다.
    """
    det = frame.detection
    if not det or not det.success or det.corners is None or det.ids is None:
        return []
    if det.num_corners < min_points_per_line:
        return []

    raw_points = det.corners.reshape(-1, 2).astype(np.float64)
    return _lines_from_points(det, pattern, raw_points, min_points_per_line)


def _classify_position(index: int, total: int) -> str:
    """설계 문서 15번 - "center line" vs "edge line" 판정.

    common.classify_regions()가 이미지 좌표를 3등분(1/3, 2/3 경계)해서
    center/edge/corner를 나누는 것과 같은 사고방식을 보드의 행/열 인덱스에
    적용한 것 - 인덱스가 양 끝 1/3 구간에 있으면 "edge", 가운데 1/3이면
    "center"로 본다. total이 아주 작으면(1~2줄) 전부 "edge"로 보수적으로
    분류한다 - 그런 보드는 애초에 "중앙"이라 부를 만한 여유가 없다.
    """
    if total <= 2:
        return "edge"
    low, high = total / 3.0, total * 2.0 / 3.0
    return "edge" if (index < low or index >= high) else "center"


def _lines_from_points(
    det, pattern: PatternConfig, points: np.ndarray, min_points_per_line: int,
) -> list[StraightnessLine]:
    """검출 결과(det)의 board id로 점들을 행/열/대각선으로 묶어 직선을 피팅.

    compute_frame_straightness_lines()(보정 후)와
    compute_frame_raw_straightness_lines()(보정 전) 둘 다 이 함수를 공유한다 -
    "points가 어디서 왔는지"만 다르고 그 뒤 grouping/fitting 로직은 완전히
    동일해야, 전/후 숫자를 공정하게 비교할 수 있다 (로직이 갈라지면 차이가
    실제 개선 때문인지 계산 방식 차이 때문인지 알 수 없게 된다).

    설계 문서 15번 - row/col(수평/수직)에 더해 두 방향의 대각선(diag_main:
    좌상->우하, diag_anti: 우상->좌하)도 같은 board id 격자에서 뽑아낸다 -
    대각선은 정의상 보드의 반대편 코너를 잇기 때문에 position="corner"로
    분류한다(코드 내 상세 이유는 _classify_position 근처 주석 참고).
    """
    n_cols = pattern.squares_x - 1  # id -> row/col 역산에 필요
    n_rows = pattern.squares_y - 1
    ids_flat = det.ids.reshape(-1)
    rows: dict[int, list[np.ndarray]] = {}
    cols: dict[int, list[np.ndarray]] = {}
    diag_main: dict[int, list[np.ndarray]] = {}
    diag_anti: dict[int, list[np.ndarray]] = {}

    for idx, cid in enumerate(ids_flat):
        cid = int(cid)
        row_key = cid // n_cols
        col_key = cid % n_cols
        rows.setdefault(row_key, []).append(points[idx])
        cols.setdefault(col_key, []).append(points[idx])
        # 두 종류 모두 "row_key - col_key"류 불변식으로 대각선을 묶는다 -
        # 격자 위의 좌표에서 이 값이 일정한 점들이 정확히 한 대각선을 이룬다.
        diag_main.setdefault(row_key - col_key, []).append(points[idx])
        diag_anti.setdefault(row_key + col_key, []).append(points[idx])

    lines: list[StraightnessLine] = []
    group_specs = (
        ("row", rows, "horizontal", n_rows),
        ("col", cols, "vertical", n_cols),
        ("diag_main", diag_main, "diagonal", None),
        ("diag_anti", diag_anti, "diagonal", None),
    )
    for line_type, groups, direction, total_for_position in group_specs:
        for line_index, group in groups.items():
            if len(group) < min_points_per_line:
                continue
            pts = np.array(group)
            position = "corner" if total_for_position is None else _classify_position(line_index, total_for_position)
            lines.append(
                StraightnessLine(
                    line_type=line_type, line_index=line_index,
                    points=pts, residual=_fit_line_residual(pts),
                    direction=direction, position=position,
                )
            )
    return lines


def compute_straightness_improvement(
    frame: Frame,
    pattern: PatternConfig,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    min_points_per_line: int = MIN_POINTS_PER_LINE,
) -> tuple[float | None, float | None]:
    """단일 프레임에 대해 (보정 전 잔차, 보정 후 잔차)를 함께 반환 (단위: px).

    Undistort Preview 탭이 이미지 한 장 + 모델 하나를 고른 순간 바로 보여줄
    "이 모델로 보정하면 직선이 얼마나 곧아지는가"를 위한 편의 함수 - 두 잔차가
    같은 프레임, 같은 행/열 grouping 기준으로 계산되므로 그 차이가 곧 이
    모델의 실질적인 보정 효과다. 계산할 수 있는 라인이 없으면 (None, None).
    """
    raw_lines = compute_frame_raw_straightness_lines(frame, pattern, min_points_per_line)
    corrected_lines = compute_frame_straightness_lines(
        frame, pattern, camera_matrix, distortion, model, min_points_per_line
    )
    raw_residual = float(np.mean([l.residual for l in raw_lines])) if raw_lines else None
    corrected_residual = float(np.mean([l.residual for l in corrected_lines])) if corrected_lines else None
    return raw_residual, corrected_residual


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
    all_residuals: list[float] = []

    for frame in frames:
        lines = compute_frame_straightness_lines(
            frame, pattern, camera_matrix, distortion, model, min_points_per_line
        )
        all_residuals.extend(line.residual for line in lines)

    if not all_residuals:
        return None, 0

    return float(np.mean(all_residuals)), len(all_residuals)


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


# ---------------------------------------------------------------------------
# 설계 문서 15번 - Line Straightness 평가 강화 (방향/위치별 분해)
# ---------------------------------------------------------------------------

def compute_straightness_breakdown(
    frames: list[Frame],
    pattern: PatternConfig,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    min_points_per_line: int = MIN_POINTS_PER_LINE,
) -> StraightnessBreakdown:
    """모든 프레임의 모든 줄(행/열/대각선)을 모아 방향별(horizontal/vertical/
    diagonal)·위치별(center/edge/corner) 평균 잔차를 계산한다.

    compute_straightness_residual()과 동일한 라인 수집(compute_frame_straightness_lines)을
    재사용하므로, overall_error는 항상 compute_straightness_residual()의 결과와
    정확히 같다 - 두 함수가 서로 다른 숫자를 내는 일이 없도록 보장한다.
    """
    all_lines: list[StraightnessLine] = []
    for frame in frames:
        all_lines.extend(
            compute_frame_straightness_lines(frame, pattern, camera_matrix, distortion, model, min_points_per_line)
        )

    if not all_lines:
        return StraightnessBreakdown(num_lines=0)

    def _mean_or_none(residuals: list[float]) -> float | None:
        return float(np.mean(residuals)) if residuals else None

    by_direction: dict[str, list[float]] = {"horizontal": [], "vertical": [], "diagonal": []}
    by_position: dict[str, list[float]] = {"center": [], "edge": [], "corner": []}
    for line in all_lines:
        by_direction.setdefault(line.direction, []).append(line.residual)
        by_position.setdefault(line.position, []).append(line.residual)

    return StraightnessBreakdown(
        horizontal_error=_mean_or_none(by_direction["horizontal"]),
        vertical_error=_mean_or_none(by_direction["vertical"]),
        diagonal_error=_mean_or_none(by_direction["diagonal"]),
        center_line_error=_mean_or_none(by_position["center"]),
        edge_line_error=_mean_or_none(by_position["edge"]),
        corner_line_error=_mean_or_none(by_position["corner"]),
        overall_error=_mean_or_none([l.residual for l in all_lines]),
        num_lines=len(all_lines),
    )


def format_straightness_breakdown(breakdown: StraightnessBreakdown) -> str:
    """설계 문서 15번 출력 형식.

        Line Straightness Breakdown (n=142 lines)
        Horizontal        0.185 px
        Vertical          0.203 px
        Diagonal          0.241 px
        --------------------------
        Center line       0.171 px
        Edge line         0.235 px
        Corner line       0.241 px
        --------------------------
        Overall           0.198 px
    """
    if breakdown.num_lines == 0:
        return "Line Straightness Breakdown: 계산할 수 있는 직선이 부족합니다."

    def fmt(v: float | None) -> str:
        return f"{v:.3f} px" if v is not None else "N/A"

    lines = [
        f"Line Straightness Breakdown (n={breakdown.num_lines} lines)",
        f"{'Horizontal':<16}{fmt(breakdown.horizontal_error):>10}",
        f"{'Vertical':<16}{fmt(breakdown.vertical_error):>10}",
        f"{'Diagonal':<16}{fmt(breakdown.diagonal_error):>10}",
        "-" * 26,
        f"{'Center line':<16}{fmt(breakdown.center_line_error):>10}",
        f"{'Edge line':<16}{fmt(breakdown.edge_line_error):>10}",
        f"{'Corner line':<16}{fmt(breakdown.corner_line_error):>10}",
        "-" * 26,
        f"{'Overall':<16}{fmt(breakdown.overall_error):>10}",
    ]
    return "\n".join(lines)
