"""
camera_calibrator.calibration.detector
========================================

설계 문서 17번 Step2 - ChArUco Detection.

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration.types import (
    CameraConfig,
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
            failure_reason="no charuco corners found"
            if marker_ids is None or len(marker_ids) == 0
            else "markers found but charuco interpolation failed",
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
    board: cv2.aruco.CharucoBoard,
    detector: Optional[cv2.aruco.CharucoDetector] = None,
    image_id: Optional[str] = None,
) -> tuple[ImageInfo, DetectionResult]:
    """파일 경로를 받아 ImageInfo + DetectionResult를 함께 반환.

    파일이 없거나 읽을 수 없는 경우에도 예외를 던지지 않고
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
    result = detect_charuco(img, board, image_id=image_id, detector=detector)
    return info, result


# ---------------------------------------------------------------------------
# 데이터셋 전체 검출
# ---------------------------------------------------------------------------

def detect_dataset(
    image_paths: list[str],
    pattern: PatternConfig,
) -> Dataset:
    """이미지 경로 리스트 전체를 검출해 Dataset(Frame 리스트)으로 반환.

    UI 없이 이 함수 하나로 2단계 목표(검출 파이프라인 안정화)를 검증할 수 있다.
    """
    board = build_charuco_board(pattern)
    detector = build_charuco_detector(board)

    dataset = Dataset()
    for image_path in image_paths:
        info, result = detect_image_file(image_path, board, detector=detector)

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
