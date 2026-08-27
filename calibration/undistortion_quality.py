"""
camera_calibrator.calibration.undistortion_quality
==================================================

Undistort 후 실제 영상으로 쓸 수 있는 영역을 평가한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    QualityGrade,
    UndistortionQualityReport,
)


def _grade(score: float) -> QualityGrade:
    if score >= 85.0:
        return QualityGrade.GOOD
    if score >= 70.0:
        return QualityGrade.WARNING
    return QualityGrade.POOR


def _largest_valid_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Binary mask에서 전부 True인 최대 면적 직사각형을 찾는다."""
    if mask.size == 0:
        return 0, 0, 0, 0

    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)

    for y in range(rows):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        for i in range(cols + 1):
            current = int(heights[i]) if i < cols else 0
            while stack and current < heights[stack[-1]]:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = i - left
                area = height * width
                if area > best_area:
                    best_area = area
                    best = (left, y - height + 1, width, height)
            stack.append(i)
    return best


def _undistort_maps(
    result: CalibrationResult,
    image_size: tuple[int, int],
    balance: float,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = image_size
    K = np.asarray(result.camera_matrix, dtype=np.float64)
    D = np.asarray(result.distortion, dtype=np.float64)

    if result.model_name == CameraModelType.FISHEYE:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D.reshape(-1, 1), (w, h), np.eye(3), balance=balance
        )
        return cv2.fisheye.initUndistortRectifyMap(
            K, D.reshape(-1, 1), np.eye(3), new_K, (w, h), cv2.CV_32FC1
        )

    return cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)


def _sample_image(dataset: Dataset) -> tuple[str | None, np.ndarray | None]:
    for frame in dataset.frames:
        path = frame.image_info.path
        if not path:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return frame.image_info.image_id, img
    return None, None


def evaluate_undistortion_quality(
    result: CalibrationResult,
    image_size: tuple[int, int],
    sample_image: np.ndarray | None = None,
    sample_frame_id: str | None = None,
    balance: float = 0.0,
    black_threshold: int = 3,
) -> UndistortionQualityReport | None:
    if (
        not result.success
        or result.camera_matrix is None
        or result.distortion is None
        or image_size[0] <= 0
        or image_size[1] <= 0
    ):
        return None

    w, h = image_size
    map_x, map_y = _undistort_maps(result, image_size, balance)
    valid = (map_x >= 0.0) & (map_x < w) & (map_y >= 0.0) & (map_y < h)
    total = float(w * h)
    valid_pixel_ratio = float(np.count_nonzero(valid) / total)
    black_border_ratio = 1.0 - valid_pixel_ratio

    roi = _largest_valid_rect(valid)
    roi_area = float(roi[2] * roi[3])
    roi_loss_ratio = 1.0 - roi_area / total

    image_black_ratio: float | None = None
    if sample_image is not None:
        if sample_image.shape[1] != w or sample_image.shape[0] != h:
            sample_image = cv2.resize(sample_image, (w, h), interpolation=cv2.INTER_AREA)
        undistorted = cv2.remap(
            sample_image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if undistorted.ndim == 2:
            black = undistorted <= black_threshold
        else:
            black = np.all(undistorted <= black_threshold, axis=2)
        image_black_ratio = float(np.count_nonzero(black) / total)

    border_penalty = image_black_ratio if image_black_ratio is not None else black_border_ratio
    quality_score = max(
        0.0,
        min(
            100.0,
            100.0 * (
                0.50 * valid_pixel_ratio
                + 0.30 * (1.0 - roi_loss_ratio)
                + 0.20 * (1.0 - border_penalty)
            ),
        ),
    )

    warnings: list[str] = []
    if valid_pixel_ratio < 0.90:
        warnings.append(f"Valid pixel ratio is low ({valid_pixel_ratio * 100:.1f}%).")
    if black_border_ratio > 0.10:
        warnings.append(f"Black border ratio is high ({black_border_ratio * 100:.1f}%).")
    if roi_loss_ratio > 0.20:
        warnings.append(f"ROI crop would lose {roi_loss_ratio * 100:.1f}% of the image.")
    if image_black_ratio is not None and image_black_ratio > 0.10:
        warnings.append(f"Undistorted sample image has {image_black_ratio * 100:.1f}% black pixels.")

    return UndistortionQualityReport(
        image_width=w,
        image_height=h,
        valid_pixel_ratio=valid_pixel_ratio,
        black_border_ratio=black_border_ratio,
        roi_loss_ratio=roi_loss_ratio,
        valid_roi=roi,
        undistorted_black_pixel_ratio=image_black_ratio,
        sample_frame_id=sample_frame_id,
        quality_score=quality_score,
        quality_grade=_grade(quality_score),
        warnings=warnings,
    )


def attach_undistortion_quality_report(
    result: CalibrationResult,
    dataset: Dataset,
    camera_config: CameraConfig,
    balance: float = 0.0,
) -> CalibrationResult:
    w = camera_config.width
    h = camera_config.height
    if not w or not h:
        for frame in dataset.frames:
            if frame.image_info.width and frame.image_info.height:
                w, h = frame.image_info.width, frame.image_info.height
                break
    if not w or not h:
        return result

    frame_id, img = _sample_image(dataset)
    result.undistortion_quality = evaluate_undistortion_quality(
        result, (w, h), sample_image=img, sample_frame_id=frame_id, balance=balance
    )
    return result
