"""
camera_calibrator.calibration.detector
========================================

설계 문서 17번 Step2 - ChArUco Detection + 일반 체스보드(Chessboard) 검출.

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

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from calibration.types import (
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)

# ChArUco 코너가 이 개수 미만이면 검출 성공이어도 경고 취급 (캘리브레이션에는 사용 가능)
MIN_RECOMMENDED_CORNERS = 6


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


# ---------------------------------------------------------------------------
# 단일 이미지 검출
# ---------------------------------------------------------------------------

def _compute_board_geometry(
    corners: np.ndarray, image_shape: tuple[int, int]
) -> tuple[float, tuple[float, float], float]:
    """검출된 코너들로부터 보드의 대략적인 기하 정보를 계산한다.

    Returns:
        board_area_ratio: 코너 convex hull 면적 / 전체 이미지 면적
        board_center_px: 코너 중심 (cx, cy)
        board_tilt_deg: minAreaRect 기준 회전각 (절대값, 0=정면에 가까움 판단용은 아님)
    """
    h, w = image_shape
    pts = corners.reshape(-1, 2).astype(np.float32)

    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    board_area_ratio = float(hull_area / (w * h)) if (w * h) > 0 else 0.0

    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    rect = cv2.minAreaRect(pts)
    tilt_deg = float(rect[-1])

    return board_area_ratio, (cx, cy), tilt_deg


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

    area_ratio, center_px, tilt_deg = _compute_board_geometry(charuco_corners, gray.shape[:2])

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
    )


def detect_image_file(
    image_path: str,
    detect_fn: Callable[[np.ndarray, str], DetectionResult],
    image_id: Optional[str] = None,
) -> tuple[ImageInfo, DetectionResult]:
    """파일 경로를 받아 ImageInfo + DetectionResult를 함께 반환.

    detect_fn(image_array, image_id) -> DetectionResult 형태의 콜백을 받는다 -
    ChArUco든 체스보드든(또는 나중에 추가될 다른 패턴이든) 이 함수는 몰라도
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

    h, w = img.shape[:2]
    sharpness = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    brightness = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

    info = ImageInfo(
        image_id=image_id,
        path=str(path),
        width=w,
        height=h,
        sharpness=sharpness,
        brightness=brightness,
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

    area_ratio, center_px, tilt_deg = _compute_board_geometry(corners, gray.shape[:2])

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
    raise ValueError(
        f"현재는 ChArUco, Chessboard 패턴만 지원합니다 (입력: {pattern.type}). "
        f"AprilGrid는 아직 미구현입니다."
    )


def detect_dataset(
    image_paths: list[str],
    pattern: PatternConfig,
) -> Dataset:
    """이미지 경로 리스트 전체를 검출해 Dataset(Frame 리스트)으로 반환.

    UI 없이 이 함수 하나로 2단계 목표(검출 파이프라인 안정화)를 검증할 수 있다.
    패턴 타입(ChArUco/체스보드)에 따라 알맞은 검출 함수로 분기한다 - 이 함수
    호출부(quality.py, compare.py 등)는 어느 패턴이었는지 몰라도 된다.
    """
    detect_fn = build_detect_fn(pattern)

    dataset = Dataset()
    for image_path in image_paths:
        info, result = detect_image_file(image_path, detect_fn)

        if result.success:
            status = (
                FrameStatus.DETECTED
                if result.num_corners >= MIN_RECOMMENDED_CORNERS
                else FrameStatus.DETECTED  # 검출은 성공, quality.py에서 점수로 별도 평가
            )
        else:
            status = FrameStatus.DETECTION_FAILED

        frame = Frame(image_info=info, detection=result, status=status)
        dataset.frames.append(frame)

    return dataset


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
