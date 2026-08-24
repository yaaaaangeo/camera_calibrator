"""
camera_calibrator.calibration.detector
========================================

설계 문서 17번 Step2 - ChArUco Detection + 일반 체스보드(Chessboard) +
AprilGrid(AprilTag grid) 검출.

이미지(경로 또는 배열)를 받아 코너를 검출하고 DetectionResult를 채운다.
UI 없이 이 모듈만으로 터미널에서 다음처럼 확인 가능해야 한다:

    img001.jpg -> 42 corners  [DETECTED]
    img003.jpg -> 12 corners  [DETECTED]  (경고: 코너 수 부족)
    img004.jpg ->  0 corners  [DETECTION_FAILED] no charuco corners found

주의:
- OpenCV 4.7 이상의 신규 ArUco API(CharucoDetector, CharucoBoard)를 사용한다.
  구버전(4.6 이하)은 cv2.aruco.detectMarkers + interpolateCornersCharuco
  조합으로 별도 분기가 필요하며, 이 모듈은 그 분기를 포함하지 않는다.
- 코너 검출 실패 이미지는 예외를 던지지 않고 DetectionResult(success=False)로
  반환한다. Frame.status만 DETECTION_FAILED로 바뀔 뿐 파일이나 레코드는
  삭제되지 않는다 (설계 문서 9번, 17번 원칙).

체스보드(PatternType.CHESSBOARD) 지원에 대한 중요한 한계:
    설계 문서 2번이 애초에 ChArUco를 우선한 이유가 두 가지 있다 -
    (1) 부분 가림 시 검출 불가(보드 전체가 다 보여야 함), (2) 대칭 패턴이라
    "어느 쪽이 진짜 첫 번째 코너인지"를 원리적으로 구분할 방법이 없다.
    ChArUco는 마커 ID로 (2)를 해결하지만 일반 체스보드는 못 한다 - 같은
    보드를 정방향으로 찍든 180도 돌려서 찍든(대칭이라 육안으로는 구분도 안 됨)
    검출 알고리즘은 "이미지 안에서 먼저 발견한 쪽"을 코너 0번으로 삼을 뿐이라,
    데이터셋 안에 방향이 뒤섞인 사진이 섞이면 캘리브레이션이 심하게 틀어질
    수 있다. 이건 OpenCV 표준 체스보드 캘리브레이션 자체의 잘 알려진 한계이며
    (거의 모든 OpenCV 체스보드 튜토리얼이 "보드를 항상 같은 방향으로 촬영하라"고
    권장하는 이유), 이 프로젝트가 새로 만든 문제가 아니다. 가능하면 ChArUco를
    쓰고, 체스보드를 꼭 써야 한다면 촬영 내내 보드 방향을 일관되게 유지할 것.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from calibration.image_quality import compute_contrast, compute_motion_blur_score, compute_phash, compute_saturation
from calibration.types import (
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)

logger = logging.getLogger(__name__)

# ChArUco 코너가 이 개수 미만이면 검출 성공이어도 경고 취급 (캘리브레이션에는 사용 가능)
MIN_RECOMMENDED_CORNERS = 6

_preprocess_cache_lock = threading.Lock()
_preprocess_cache: dict[tuple[str, int, int], tuple[int, int, float, float, float, float, float, str]] = {}


def _image_file_fingerprint(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return str(path), 0, -1


def clear_image_preprocess_cache() -> None:
    """테스트/장기 실행 UI에서 이미지 전처리 캐시를 명시적으로 비운다."""
    with _preprocess_cache_lock:
        _preprocess_cache.clear()


def _preprocess_image_metadata(path: Path, img: np.ndarray) -> tuple[int, int, float, float, float, float, float, str]:
    key = _image_file_fingerprint(path)
    with _preprocess_cache_lock:
        cached = _preprocess_cache.get(key)
    if cached is not None:
        return cached

    h, w = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    brightness = float(gray_img.mean())
    contrast = compute_contrast(gray_img)
    saturation = compute_saturation(gray_img)
    motion_blur_score = compute_motion_blur_score(gray_img)
    phash = compute_phash(gray_img)
    value = (w, h, sharpness, brightness, contrast, saturation, motion_blur_score, phash)
    with _preprocess_cache_lock:
        _preprocess_cache[key] = value
    return value


# ---------------------------------------------------------------------------
# Board / Detector 생성
# ---------------------------------------------------------------------------

def build_charuco_board(pattern: PatternConfig) -> cv2.aruco.CharucoBoard:
    """PatternConfig -> cv2.aruco.CharucoBoard

    squares_x, squares_y는 '칸' 개수 기준이며, 코너 개수는
    (squares_x-1) * (squares_y-1) 이 된다.
    """
    if pattern.type != PatternType.CHARUCO:
        raise ValueError(
            f"build_charuco_board는 CHARUCO 패턴만 지원합니다. 입력: {pattern.type}"
        )
    if not pattern.marker_size:
        raise ValueError("ChArUco 패턴은 marker_size가 반드시 필요합니다.")
    if not pattern.dictionary:
        raise ValueError("ChArUco 패턴은 dictionary가 반드시 필요합니다. 예: 'DICT_6X6_250'")

    dict_id = getattr(cv2.aruco, pattern.dictionary, None)
    if dict_id is None:
        raise ValueError(f"알 수 없는 ArUco dictionary: {pattern.dictionary}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y),
        pattern.square_size,
        pattern.marker_size,
        aruco_dict,
    )
    return board


def build_charuco_detector(
    board: cv2.aruco.CharucoBoard,
    refine_markers: bool = True,
) -> cv2.aruco.CharucoDetector:
    """검출 파라미터를 담은 CharucoDetector 생성.

    refine_markers=True는 마커 일부가 가려지거나 화면 밖으로 잘려도
    코너 보간(interpolation)을 시도하게 한다. 피쉬아이/광각 렌즈에서
    특히 중요 (설계 문서 2번).
    """
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    refine_params = cv2.aruco.RefineParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = refine_markers

    return cv2.aruco.CharucoDetector(
        board,
        charucoParams=charuco_params,
        detectorParams=detector_params,
        refineParams=refine_params,
    )


def build_aprilgrid_dictionary(pattern: PatternConfig) -> cv2.aruco.Dictionary:
    """PatternConfig -> AprilGrid용 OpenCV AprilTag dictionary.

    이 프로젝트의 AprilGrid는 Kalibr 계열 보드처럼 marker ID가 row-major
    순서(왼쪽 위 0번, 오른쪽으로 증가, 다음 줄로 이동)로 배치된다는 관례를
    따른다. squares_x/y는 태그 개수, square_size는 태그 origin 사이 간격,
    marker_size는 실제 태그 한 변 길이다.
    """
    if pattern.type != PatternType.APRILGRID:
        raise ValueError(
            f"build_aprilgrid_dictionary는 APRILGRID 패턴만 지원합니다. 입력: {pattern.type}"
        )
    if not pattern.marker_size:
        raise ValueError("AprilGrid 패턴은 marker_size가 반드시 필요합니다.")
    if pattern.marker_size >= pattern.square_size:
        raise ValueError("AprilGrid marker_size는 square_size보다 작아야 합니다.")
    if not pattern.dictionary:
        raise ValueError("AprilGrid 패턴은 dictionary가 반드시 필요합니다. 예: 'DICT_APRILTAG_36h11'")
    if not str(pattern.dictionary).startswith("DICT_APRILTAG_"):
        raise ValueError(
            "AprilGrid 패턴은 OpenCV AprilTag dictionary가 필요합니다 "
            "(예: DICT_APRILTAG_36h11)."
        )

    dict_id = getattr(cv2.aruco, pattern.dictionary, None)
    if dict_id is None:
        raise ValueError(f"알 수 없는 AprilTag dictionary: {pattern.dictionary}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def build_aprilgrid_detector(
    aruco_dict: cv2.aruco.Dictionary,
) -> Optional[cv2.aruco.ArucoDetector]:
    """AprilGrid 마커 검출기 생성.

    OpenCV 4.7+는 ArucoDetector를 제공한다. 혹시 더 오래된 API로 실행되는
    환경에서는 detect_aprilgrid()에서 cv2.aruco.detectMarkers fallback을 쓴다.
    """
    if not hasattr(cv2.aruco, "ArucoDetector"):
        return None
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, detector_params)


# ---------------------------------------------------------------------------
# 단일 이미지 검출
# ---------------------------------------------------------------------------

def _compute_board_geometry(
    corners: np.ndarray, image_shape: tuple[int, int]
) -> tuple[float, tuple[float, float], float, float]:
    """검출된 코너들로부터 보드의 대략적인 기하 정보를 계산한다.

    Returns:
        board_area_ratio: 코너 convex hull 면적 / 전체 이미지 면적
        board_center_px: 코너 중심 (cx, cy)
        board_tilt_deg: minAreaRect 기준 회전각 (절대값, 0=정면에 가까움 판단용은 아님)
        min_edge_margin_px: 코너 중 이미지 경계에 가장 가까운 코너까지의 거리(px)
            (설계 문서 3-2번 - "corner가 이미지 경계에 너무 가까운지" 판정용)
    """
    h, w = image_shape
    pts = corners.reshape(-1, 2).astype(np.float32)

    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    board_area_ratio = float(hull_area / (w * h)) if (w * h) > 0 else 0.0

    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    rect = cv2.minAreaRect(pts)
    tilt_deg = float(rect[-1])

    # 각 코너에서 가장 가까운 이미지 경계(상하좌우 4개)까지의 거리 중 최솟값.
    margins = np.minimum.reduce([pts[:, 0], w - pts[:, 0], pts[:, 1], h - pts[:, 1]])
    min_edge_margin_px = float(np.min(margins)) if margins.size > 0 else 0.0

    return board_area_ratio, (cx, cy), tilt_deg, min_edge_margin_px


def detect_charuco(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    image_id: str,
    detector: Optional[cv2.aruco.CharucoDetector] = None,
) -> DetectionResult:
    """이미 메모리에 로드된 이미지(BGR 또는 grayscale)에 대해 ChArUco 검출 수행.

    파일 I/O를 하지 않는 순수 함수로 두어, 테스트/합성 이미지에도
    바로 재사용할 수 있게 한다.
    """
    if detector is None:
        detector = build_charuco_detector(board)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return DetectionResult(
            image_id=image_id,
            success=False,
            num_corners=0,
            failure_reason=(
                "ArUco 마커가 하나도 검출되지 않음 (조명/초점/각도 확인 필요)"
                if marker_ids is None or len(marker_ids) == 0
                else "마커는 검출됐지만 체스보드 코너 보간 실패 (보드 일부만 보이거나 각도가 너무 큼)"
            ),
        )

    num_corners = int(len(charuco_ids))
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)

    area_ratio, center_px, tilt_deg, min_edge_margin_px = _compute_board_geometry(
        charuco_corners, gray.shape[:2]
    )

    # 설계 문서 3-2번 "corner detection confidence" - ChArUco API는 개별 코너
    # 단위 confidence를 직접 주지 않으므로(대화 중 확인된 OpenCV API 한계),
    # 이론상 검출 가능한 코너 수 대비 실제 검출 수의 비율을 대리 지표로 쓴다.
    # 예: 7x5 보드는 내부 교차점이 6x4=24개인데 18개만 검출됐으면 confidence=0.75.
    squares_x, squares_y = board.getChessboardSize()
    max_possible = max(1, (squares_x - 1) * (squares_y - 1))
    corner_confidence = float(min(1.0, num_corners / max_possible))

    return DetectionResult(
        image_id=image_id,
        success=True,
        corners=charuco_corners,
        object_points=object_points,
        ids=charuco_ids,
        num_corners=num_corners,
        board_area_ratio=area_ratio,
        board_center_px=center_px,
        board_tilt_deg=tilt_deg,
        corner_confidence=corner_confidence,
        min_edge_margin_px=min_edge_margin_px,
    )


def detect_image_file(
    image_path: str,
    detect_fn: Callable[[np.ndarray, str], DetectionResult],
    image_id: Optional[str] = None,
) -> tuple[ImageInfo, DetectionResult]:
    """파일 경로를 받아 ImageInfo + DetectionResult를 함께 반환.

    detect_fn(image_array, image_id) -> DetectionResult 형태의 콜백을 받는다 -
    ChArUco든 체스보드든 AprilGrid든 이 함수는 몰라도
    되게 분리했다. 파일이 없거나 읽을 수 없는 경우에도 예외를 던지지 않고
    success=False DetectionResult를 반환한다 (파이프라인이 한 장 때문에 죽지 않도록).
    """
    path = Path(image_path)
    image_id = image_id or path.stem

    img = cv2.imread(str(path))
    if img is None:
        info = ImageInfo(image_id=image_id, path=str(path), width=0, height=0)
        result = DetectionResult(
            image_id=image_id,
            success=False,
            num_corners=0,
            failure_reason=f"이미지를 읽을 수 없음: {path}",
        )
        return info, result

    # 설계 문서 3-1번 이미지 품질 전처리는 같은 파일을 반복 검출할 때
    # 재사용한다. 검출 자체는 pattern/알고리즘에 따라 달라질 수 있어 캐시하지
    # 않고, 파일 내용에만 의존하는 이미지 메타만 캐시한다.
    w, h, sharpness, brightness, contrast, saturation, motion_blur_score, phash = (
        _preprocess_image_metadata(path, img)
    )

    info = ImageInfo(
        image_id=image_id,
        path=str(path),
        width=w,
        height=h,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        motion_blur_score=motion_blur_score,
        phash=phash,
    )
    result = detect_fn(img, image_id)
    return info, result


# ---------------------------------------------------------------------------
# 체스보드(Chessboard) 검출
# ---------------------------------------------------------------------------

def build_chessboard_object_points(pattern: PatternConfig) -> np.ndarray:
    """PatternConfig -> 체스보드 내부 코너의 3D object point 배열, shape (N,1,3).

    row-major 순서(row가 느리게, col이 빠르게 증가)로 만든다 - 이게
    detect_chessboard()가 코너를 정렬하는 순서와 정확히 일치해야
    calibrateCamera에 넘길 때 대응이 맞는다. calibration/straightness.py의
    row/col 역산 공식(id // cols, id % cols)도 이 순서를 전제로 한다.
    """
    cols = pattern.squares_x - 1
    rows = pattern.squares_y - 1
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= pattern.square_size
    return objp.reshape(-1, 1, 3)


def _normalize_chessboard_corner_order(corners: np.ndarray) -> np.ndarray:
    """findChessboardCornersSB와 findChessboardCorners(고전)는 서로 정확히
    반대 순서로 코너를 반환할 수 있다(실측 확인됨) - 어느 함수로 검출됐든
    "이미지 원점(왼쪽 위)에 더 가까운 쪽이 첫 코너"라는 하나의 관례로
    통일한다. 이렇게 해야 build_chessboard_object_points()가 만든 순서와
    항상 짝이 맞는다.

    한계: 이건 SB와 고전 함수 사이의 순서 불일치만 해결한다 - "물리적으로
    180도 돌려서 찍은 보드"까지 구분해주진 못한다 (모듈 docstring 참고,
    체스보드 자체의 근본적인 한계).
    """
    flat = corners.reshape(-1, 2)
    origin_dist_first = float(np.hypot(*flat[0]))
    origin_dist_last = float(np.hypot(*flat[-1]))
    if origin_dist_first > origin_dist_last:
        return corners[::-1]
    return corners


def detect_chessboard(
    image: np.ndarray,
    pattern: PatternConfig,
    image_id: str,
) -> DetectionResult:
    """이미 메모리에 로드된 이미지에 대해 일반 체스보드 코너 검출 수행.

    findChessboardCornersSB(더 강건, OpenCV 4.x+)를 먼저 시도하고, 실패하면
    고전 방식(findChessboardCorners + cornerSubPix)으로 재시도한다 - fisheye
    모듈의 "엄격한 조건 실패 시 완화해서 재시도" 패턴과 같은 철학이다.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols = pattern.squares_x - 1
    rows = pattern.squares_y - 1
    pattern_size = (cols, rows)

    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
        except cv2.error:
            found = False

    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not found or corners is None:
        return DetectionResult(
            image_id=image_id,
            success=False,
            num_corners=0,
            failure_reason=(
                f"체스보드 코너를 찾지 못함 (내부 코너 {cols}x{rows}개 필요). "
                f"일반 체스보드는 ChArUco와 달리 보드 전체가 이미지 안에 다 보여야 "
                f"검출됩니다 - 모서리가 잘렸거나 각도가 너무 크면 실패하기 쉽습니다."
            ),
        )

    corners = _normalize_chessboard_corner_order(corners.astype(np.float32))
    num_corners = cols * rows
    ids = np.arange(num_corners, dtype=np.int32).reshape(-1, 1)
    object_points = build_chessboard_object_points(pattern)

    area_ratio, center_px, tilt_deg, min_edge_margin_px = _compute_board_geometry(corners, gray.shape[:2])

    return DetectionResult(
        image_id=image_id,
        success=True,
        corners=corners,
        object_points=object_points,
        ids=ids,
        num_corners=num_corners,
        board_area_ratio=area_ratio,
        board_center_px=center_px,
        board_tilt_deg=tilt_deg,
        # 체스보드는 findChessboardCorners*가 "전부 찾거나 아예 실패"만 하므로
        # (ChArUco처럼 일부만 검출되는 경우가 없음) confidence는 항상 1.0.
        corner_confidence=1.0,
        min_edge_margin_px=min_edge_margin_px,
    )


def detect_aprilgrid(
    image: np.ndarray,
    pattern: PatternConfig,
    image_id: str,
    aruco_dict: Optional[cv2.aruco.Dictionary] = None,
    detector: Optional[cv2.aruco.ArucoDetector] = None,
) -> DetectionResult:
    """이미 메모리에 로드된 이미지에 대해 AprilGrid(AprilTag grid) 검출 수행.

    OpenCV의 AprilTag dictionary 기반 ArUco detector를 사용하고, 검출된 marker
    ID를 row-major AprilGrid 좌표로 변환한다. marker ID가 보드 범위를 벗어나면
    해당 태그는 무시한다.
    """
    if aruco_dict is None:
        aruco_dict = build_aprilgrid_dictionary(pattern)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict)

    if marker_ids is None or len(marker_ids) == 0:
        return DetectionResult(
            image_id=image_id,
            success=False,
            num_corners=0,
            failure_reason=(
                "AprilTag 마커가 하나도 검출되지 않음 "
                "(dictionary/조명/초점/보드 크기 확인 필요)"
            ),
        )

    total_tags = pattern.squares_x * pattern.squares_y
    image_points: list[np.ndarray] = []
    object_points: list[np.ndarray] = []
    corner_ids: list[int] = []
    valid_marker_count = 0

    for corners, raw_id in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(raw_id)
        if marker_id < 0 or marker_id >= total_tags:
            continue

        row = marker_id // pattern.squares_x
        col = marker_id % pattern.squares_x
        x0 = col * pattern.square_size
        y0 = row * pattern.square_size
        size = float(pattern.marker_size)
        marker_obj = np.array(
            [
                [x0, y0, 0.0],
                [x0 + size, y0, 0.0],
                [x0 + size, y0 + size, 0.0],
                [x0, y0 + size, 0.0],
            ],
            dtype=np.float32,
        )
        image_points.append(corners.reshape(4, 2).astype(np.float32))
        object_points.append(marker_obj)
        corner_ids.extend(marker_id * 4 + i for i in range(4))
        valid_marker_count += 1

    if not image_points:
        return DetectionResult(
            image_id=image_id,
            success=False,
            num_corners=0,
            failure_reason=(
                f"AprilTag는 검출됐지만 보드 범위 ID(0~{total_tags - 1})에 해당하는 "
                "마커가 없습니다. AprilGrid ID 배치를 확인하세요."
            ),
        )

    corners = np.concatenate(image_points, axis=0).reshape(-1, 1, 2)
    obj = np.concatenate(object_points, axis=0).reshape(-1, 1, 3)
    ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1, 1)
    area_ratio, center_px, tilt_deg, min_edge_margin_px = _compute_board_geometry(corners, gray.shape[:2])
    corner_confidence = float(min(1.0, valid_marker_count / max(1, total_tags)))

    return DetectionResult(
        image_id=image_id,
        success=True,
        corners=corners,
        object_points=obj,
        ids=ids,
        num_corners=int(corners.shape[0]),
        board_area_ratio=area_ratio,
        board_center_px=center_px,
        board_tilt_deg=tilt_deg,
        corner_confidence=corner_confidence,
        min_edge_margin_px=min_edge_margin_px,
    )


# ---------------------------------------------------------------------------
# 데이터셋 전체 검출
# ---------------------------------------------------------------------------

def build_detect_fn(pattern: PatternConfig) -> Callable[[np.ndarray, str], DetectionResult]:
    """PatternConfig에 맞는 detect_fn(image_array, image_id) 콜백을 만든다.

    detect_dataset()과 실시간 캡처(ui/live_capture_dialog.py)가 "패턴 타입별로
    어느 검출 함수를 쓸지" 분기 로직을 중복해서 들고 있으면 하나가 바뀔 때
    다른 쪽을 깜빡 놓치기 쉬워서, 이 분기 자체를 공용 함수로 뺐다.
    """
    if pattern.type == PatternType.CHARUCO:
        board = build_charuco_board(pattern)
        detector = build_charuco_detector(board)
        return lambda img, image_id: detect_charuco(img, board, image_id=image_id, detector=detector)
    if pattern.type == PatternType.CHESSBOARD:
        return lambda img, image_id: detect_chessboard(img, pattern, image_id=image_id)
    if pattern.type == PatternType.APRILGRID:
        aruco_dict = build_aprilgrid_dictionary(pattern)
        detector = build_aprilgrid_detector(aruco_dict)
        return lambda img, image_id: detect_aprilgrid(
            img, pattern, image_id=image_id, aruco_dict=aruco_dict, detector=detector
        )
    supported = (PatternType.CHARUCO, PatternType.CHESSBOARD, PatternType.APRILGRID)
    raise ValueError(
        f"현재 지원하는 패턴 타입은 {', '.join(p.value for p in supported)}뿐입니다 "
        f"(입력: {pattern.type.value}). 검출 로직이 아직 구현되지 않았습니다."
    )


def detect_dataset(
    image_paths: list[str],
    pattern: PatternConfig,
    *,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Dataset:
    """이미지 경로 리스트 전체를 검출해 Dataset(Frame 리스트)으로 반환.

    UI 없이 이 함수 하나로 2단계 목표(검출 파이프라인 안정화)를 검증할 수 있다.
    패턴 타입(ChArUco/체스보드/AprilGrid)에 따라 알맞은 검출 함수로 분기한다 - 이 함수
    호출부(quality.py, compare.py 등)는 어느 패턴이었는지 몰라도 된다.

    기본은 순차 처리다(기존 동작과 완전히 동일, 테스트 결과도 그대로 재현됨).
    `parallel=True`를 주면 `concurrent.futures.ProcessPoolExecutor`로 이미지별
    검출을 여러 프로세스에 나눠 돌린다 - 검출은 이미지 한 장 단위로 완전히
    독립적인 순수 계산(cv2 호출만 함, 공유 상태 없음)이라 프레임 수가 많은
    데이터셋(수백 장)에서 코어 수에 비례해 빨라진다. 결과 순서는 image_paths
    순서와 항상 동일하게 유지된다(ProcessPoolExecutor.map은 완료 순서가 아니라
    제출 순서로 결과를 돌려줌) - 그래야 Dataset.frames의 순서가 순차 실행 때와
    똑같아서 이 함수를 호출하는 다른 코드(quality.py 등)가 병렬 여부를 몰라도 된다.

    Args:
        parallel: True면 프로세스 풀 사용. 이미지 1장뿐이거나 os.cpu_count()가
            1인 환경에서는 parallel=True여도 자동으로 순차 처리로 내려간다
            (프로세스 생성 비용이 이득보다 커서).
        max_workers: 프로세스 개수. None이면 os.cpu_count()-1(코어 하나는 GUI 몫으로
            남겨둠 - 실사용자 버그: 코어를 전부 쓰면 이미지 수백 장 검출 중에
            GUI 프로세스가 OS 스케줄링을 못 받아 "python3 is not responding"
            창이 뜬다. QThread/별도 프로세스로 계산을 분리해도, OS가 CPU 자체를
            GUI에 배분 못 하면 소용없다).
    """
    cpu_count = os.cpu_count() or 1
    if parallel and len(image_paths) > 1 and cpu_count > 1:
        effective_workers = max_workers if max_workers is not None else max(1, cpu_count - 1)
        logger.info(
            "병렬 검출 시작: 이미지 %d장, max_workers=%s",
            len(image_paths), effective_workers,
        )
        pairs = _detect_dataset_parallel(image_paths, pattern, effective_workers)
    else:
        logger.debug("순차 검출: 이미지 %d장", len(image_paths))
        detect_fn = build_detect_fn(pattern)
        pairs = [detect_image_file(p, detect_fn) for p in image_paths]

    dataset = Dataset()
    for info, result in pairs:
        status = FrameStatus.DETECTED if result.success else FrameStatus.DETECTION_FAILED
        dataset.frames.append(Frame(image_info=info, detection=result, status=status))

    return dataset


# ---------------------------------------------------------------------------
# 병렬 검출 (프로세스별 워커)
# ---------------------------------------------------------------------------
#
# cv2.aruco.CharucoDetector 같은 OpenCV 객체는 pickle이 안 돼서 ProcessPoolExecutor로
# 그대로 넘길 수 없다 - 그래서 각 워커 프로세스가 "자기 몫의 이미지를 처리하기
# 시작할 때 딱 한 번" PatternConfig(순수 데이터라 pickle 가능)로부터 자기만의
# detect_fn을 새로 만들게 한다(initializer). 프로세스당 한 번만 만들고, 그
# 프로세스에 배정된 여러 이미지가 재사용하므로 이미지 1장마다 board를 새로 만드는
# 낭비는 없다.

_worker_detect_fn: Optional[Callable[[np.ndarray, str], DetectionResult]] = None


def _init_worker(pattern: PatternConfig) -> None:
    global _worker_detect_fn
    _worker_detect_fn = build_detect_fn(pattern)


def _detect_one_in_worker(image_path: str) -> tuple[ImageInfo, DetectionResult]:
    if _worker_detect_fn is None:  # pragma: no cover - initializer가 항상 먼저 돎
        raise RuntimeError("워커 프로세스 초기화가 안 된 상태에서 호출됨 (internal error)")
    return detect_image_file(image_path, _worker_detect_fn)


def _detect_dataset_parallel(
    image_paths: list[str],
    pattern: PatternConfig,
    max_workers: Optional[int],
) -> list[tuple[ImageInfo, DetectionResult]]:
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(pattern,),
    ) as executor:
        # executor.map은 결과를 제출 순서대로 돌려준다 (완료 순서 아님) -
        # 순차 실행과 동일한 프레임 순서를 보장하기 위해 중요.
        return list(executor.map(_detect_one_in_worker, image_paths))


def summarize_dataset(dataset: Dataset) -> str:
    """터미널 출력용 요약. UI 붙이기 전 단계에서 육안 확인용."""
    lines = []
    for frame in dataset.frames:
        det = frame.detection
        name = frame.image_info.image_id
        if det is None:
            lines.append(f"{name:15s} -> (검출 안 됨)")
            continue

        if det.success:
            warn = ""
            if det.num_corners < MIN_RECOMMENDED_CORNERS:
                warn = f"  (경고: 코너 {MIN_RECOMMENDED_CORNERS}개 미만)"
            lines.append(f"{name:15s} -> {det.num_corners:3d} corners  [DETECTED]{warn}")
        else:
            lines.append(f"{name:15s} ->   0 corners  [DETECTION_FAILED] {det.failure_reason}")

    total = dataset.num_total
    detected = dataset.num_detected
    lines.append("-" * 50)
    lines.append(f"총 {total}장 중 {detected}장 검출 성공 ({detected/total*100:.1f}%)" if total else "이미지 없음")
    return "\n".join(lines)
