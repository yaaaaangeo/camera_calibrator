"""
camera_calibrator.calibration.models.object_releasing
===========================================================

Advanced Calibration - Object-Releasing Brown-Conrady (`cv2.calibrateCameraRO`/
`calibrateCameraROExtended` 기반). Standard 4모델(Ideal Pinhole/Brown-Conrady/
Rational/Fisheye)과는 완전히 분리된 결과이며, 카메라 파라미터(K/D)뿐 아니라
캘리브레이션 타겟 형상(refined_object_points) 자체도 함께 추정한다.

지원 타겟: Checkerboard, Circle Grid만 (SUPPORTED_OBJECT_RELEASING_PATTERN_TYPES).
ChArUco/AprilGrid는 지원하지 않는다 - 부분 검출이 흔해 "매 프레임 동일한 개수/
순서의 포인트 대응"이라는 전제를 보장하기 어렵기 때문이다.

요구 사항: Full-board 검출(타겟 전체가 빠짐없이 보이고, ID가 기대값과 정확히
일치)만 입력으로 쓴다 - collect_object_releasing_inputs()가 이 필터링과 ID
canonicalization(모든 프레임이 동일한 포인트 순서를 갖도록 정렬)을 담당한다.
이 모듈은 DetectionResult.excluded_corner_indices를 의도적으로 무시한다 -
Object-Releasing은 프레임 전체 단위로 정확한 대응이 필요하므로, 나쁜 관측치는
프레임 단위로(collect_object_releasing_inputs에서) 걸러야 한다.

Hold-out Validation과 Standard Brown-Conrady와의 공정 비교는 이 모듈이 아니라
calibration/object_releasing_validation.py에서 수행한다 (Train에서만 이
모듈의 calibrate_object_releasing_brown_conrady()를 호출하고, Test는 그 결과의
K/D/refined_object_points를 고정한 채 pose만 재추정 - "Test 데이터로 다시
캘리브레이션하지 않는다"는 원칙을 지키기 위해 계산 경로를 명확히 분리했다).
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.bootstrap import add_normal_approximation_ci
from calibration.models.common import (
    DEFAULT_TERM_CRITERIA,
    MIN_FRAMES_REQUIRED,
    compute_regional_error,
    infer_image_size,
    validate_finite_calibration_output,
)
from calibration.radial_profile import compute_radial_error_bands, compute_radial_error_profile
from calibration.residual_stats import compute_residual_stats_for_calibration
from calibration.spatial_error_map import compute_spatial_error_map
from calibration.types import (
    CalibrationMethod,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    ParameterUncertainty,
    PatternConfig,
    PatternType,
)


MAX_DIAGNOSTIC_FRAMES_IN_MESSAGE = 25
SUPPORTED_OBJECT_RELEASING_PATTERN_TYPES = (PatternType.CHESSBOARD, PatternType.CIRCLE_GRID)


def is_object_releasing_supported_pattern(pattern_config: PatternConfig) -> bool:
    return _pattern_type(pattern_config) in SUPPORTED_OBJECT_RELEASING_PATTERN_TYPES


def _pattern_type(pattern_config: PatternConfig) -> PatternType:
    return pattern_config.type if isinstance(pattern_config.type, PatternType) else PatternType(str(pattern_config.type))


def _circle_grid_type(pattern_config: PatternConfig):
    value = pattern_config.circle_grid_type
    return value.value if hasattr(value, "value") else str(value)


def expected_object_releasing_ids(pattern_config: PatternConfig) -> np.ndarray:
    ptype = _pattern_type(pattern_config)
    if ptype in (PatternType.CHESSBOARD, PatternType.CHARUCO):
        count = max(0, (pattern_config.squares_x - 1) * (pattern_config.squares_y - 1))
    elif ptype == PatternType.CIRCLE_GRID:
        count = max(0, pattern_config.squares_x * pattern_config.squares_y)
    elif ptype == PatternType.APRILGRID:
        count = max(0, pattern_config.squares_x * pattern_config.squares_y * 4)
    else:
        count = 0
    return np.arange(count, dtype=np.int32)


def expected_object_releasing_object_points(pattern_config: PatternConfig) -> np.ndarray:
    ptype = _pattern_type(pattern_config)
    if ptype == PatternType.CHESSBOARD:
        cols = pattern_config.squares_x - 1
        rows = pattern_config.squares_y - 1
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj *= pattern_config.square_size
        return obj.reshape(-1, 1, 3)
    if ptype == PatternType.CHARUCO:
        cols = pattern_config.squares_x - 1
        rows = pattern_config.squares_y - 1
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj *= pattern_config.square_size
        return obj.reshape(-1, 1, 3)
    if ptype == PatternType.CIRCLE_GRID:
        cols = pattern_config.squares_x
        rows = pattern_config.squares_y
        spacing = float(pattern_config.square_size)
        obj = np.zeros((rows * cols, 3), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if _circle_grid_type(pattern_config) == "asymmetric":
                    obj[idx, 0] = (2 * col + (row % 2)) * spacing
                    obj[idx, 1] = row * spacing
                else:
                    obj[idx, 0] = col * spacing
                    obj[idx, 1] = row * spacing
        return obj.reshape(-1, 1, 3)
    if ptype == PatternType.APRILGRID:
        if pattern_config.marker_size is None:
            return np.zeros((0, 1, 3), dtype=np.float32)
        points: list[list[float]] = []
        for marker_id in range(pattern_config.squares_x * pattern_config.squares_y):
            row = marker_id // pattern_config.squares_x
            col = marker_id % pattern_config.squares_x
            x0 = col * pattern_config.square_size
            y0 = row * pattern_config.square_size
            size = float(pattern_config.marker_size)
            points.extend([
                [x0, y0, 0.0],
                [x0 + size, y0, 0.0],
                [x0 + size, y0 + size, 0.0],
                [x0, y0 + size, 0.0],
            ])
        return np.asarray(points, dtype=np.float32).reshape(-1, 1, 3)
    return np.zeros((0, 1, 3), dtype=np.float32)


def _canonicalize_detection_for_object_releasing(
    frame: Frame,
    pattern_config: PatternConfig,
    expected_ids: np.ndarray,
    expected_object_points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    expected_count = int(len(expected_ids))
    diag = {
        "image_id": frame.image_info.image_id,
        "accepted": False,
        "expected_count": expected_count,
        "detected_count": 0,
        "missing_ids": expected_ids.tolist(),
        "missing_tag_ids": [],
        "reject_reason": "",
    }

    det = frame.detection
    if det is None or not det.success:
        if det is not None:
            diag["detected_count"] = int(det.num_corners)
        diag["reject_reason"] = det.failure_reason if det is not None and det.failure_reason else "detection failed"
        return None, None, diag
    if det.object_points is None or det.corners is None or det.ids is None:
        if det.corners is not None:
            diag["detected_count"] = int(np.asarray(det.corners).reshape(-1, 2).shape[0])
        diag["reject_reason"] = "missing corners/object_points/ids in DetectionResult"
        return None, None, diag

    obj = np.asarray(det.object_points, dtype=np.float32).reshape(-1, 1, 3)
    img = np.asarray(det.corners, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(det.ids, dtype=np.int32).reshape(-1)
    diag["detected_count"] = int(len(ids))
    missing = sorted(set(expected_ids.tolist()) - set(ids.tolist()))
    unexpected = sorted(set(ids.tolist()) - set(expected_ids.tolist()))
    diag["missing_ids"] = missing
    if _pattern_type(pattern_config) == PatternType.APRILGRID:
        expected_tags = set(int(i) // 4 for i in expected_ids.tolist())
        expected_id_set = set(expected_ids.tolist())
        detected_tags = set(int(i) // 4 for i in ids.tolist() if int(i) in expected_id_set)
        diag["missing_tag_ids"] = sorted(expected_tags - detected_tags)

    if len(obj) != len(img) or len(obj) != len(ids):
        diag["reject_reason"] = (
            f"array length mismatch: object_points={len(obj)}, corners={len(img)}, ids={len(ids)}"
        )
        return None, None, diag
    if len(ids) != len(expected_ids):
        diag["reject_reason"] = f"detected {len(ids)} of {len(expected_ids)} expected points"
        return None, None, diag
    if len(np.unique(ids)) != len(ids):
        diag["reject_reason"] = "duplicate point IDs detected"
        return None, None, diag
    if set(ids.tolist()) != set(expected_ids.tolist()):
        if missing and unexpected:
            diag["reject_reason"] = f"ID set mismatch: missing IDs {missing}, unexpected IDs {unexpected}"
        elif missing:
            diag["reject_reason"] = f"missing IDs {missing}"
        else:
            diag["reject_reason"] = f"unexpected IDs {unexpected}"
        return None, None, diag

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if not np.array_equal(sorted_ids, expected_ids):
        diag["reject_reason"] = "canonical ID ordering does not match expected IDs"
        return None, None, diag
    sorted_obj = obj[order]
    if sorted_obj.shape != expected_object_points.shape:
        diag["reject_reason"] = (
            f"object point shape mismatch: got {sorted_obj.shape}, expected {expected_object_points.shape}"
        )
        return None, None, diag
    if not np.allclose(sorted_obj, expected_object_points, rtol=0.0, atol=1e-7):
        diag["reject_reason"] = "nominal object points differ from expected board geometry"
        return None, None, diag
    diag["accepted"] = True
    diag["missing_ids"] = []
    diag["missing_tag_ids"] = []
    diag["reject_reason"] = ""
    return sorted_obj, img[order], diag


def _format_ids(ids: list[int], *, limit: int = 24) -> str:
    if not ids:
        return "-"
    shown = ids[:limit]
    suffix = f", ... +{len(ids) - limit}" if len(ids) > limit else ""
    return ",".join(str(i) for i in shown) + suffix


def format_object_releasing_diagnostics(
    diagnostics: list[dict],
    *,
    accepted_count: int,
    expected_count: int,
    pattern_config: PatternConfig,
    min_frames_required: int = MIN_FRAMES_REQUIRED,
    max_frames: int = MAX_DIAGNOSTIC_FRAMES_IN_MESSAGE,
) -> str:
    total = len(diagnostics)
    ptype = _pattern_type(pattern_config)
    target_label = ptype.value
    lines = [
        (
            f"Object-Releasing full-board validation: "
            f"{accepted_count}/{total} full-board frames accepted "
            f"(target={target_label}, expected_points={expected_count}, min_required={min_frames_required})."
        )
    ]
    for diag in diagnostics[:max_frames]:
        status = "ACCEPT" if diag.get("accepted") else "REJECT"
        line = (
            f"- {diag.get('image_id')}: {status}; "
            f"expected={diag.get('expected_count')}, detected={diag.get('detected_count')}"
        )
        missing = diag.get("missing_ids") or []
        missing_tags = diag.get("missing_tag_ids") or []
        if ptype in (PatternType.CHARUCO, PatternType.APRILGRID):
            line += f"; missing_ids={_format_ids(list(missing))}"
        if ptype == PatternType.APRILGRID:
            line += f"; missing_tag_ids={_format_ids(list(missing_tags))}"
        reason = diag.get("reject_reason")
        if reason:
            line += f"; reason={reason}"
        lines.append(line)
    if len(diagnostics) > max_frames:
        lines.append(f"- ... {len(diagnostics) - max_frames} more frames omitted")
    return "\n".join(lines)


def collect_object_releasing_inputs(
    dataset: Dataset,
    pattern_config: PatternConfig,
) -> tuple[list[Frame], list[np.ndarray], list[np.ndarray], list[dict]]:
    expected_ids = expected_object_releasing_ids(pattern_config)
    expected_object_points = expected_object_releasing_object_points(pattern_config)
    accepted_frames: list[Frame] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    diagnostics: list[dict] = []
    canonical_obj: np.ndarray | None = None

    for frame in dataset.enabled_frames:
        obj, img, diag = _canonicalize_detection_for_object_releasing(
            frame, pattern_config, expected_ids, expected_object_points
        )
        if obj is None or img is None:
            diagnostics.append(diag)
            continue
        if canonical_obj is None:
            canonical_obj = obj
        elif not np.allclose(obj, canonical_obj, rtol=0.0, atol=1e-7):
            diag["accepted"] = False
            diag["reject_reason"] = "nominal object points differ from first accepted full-board frame"
            diagnostics.append(diag)
            continue

        accepted_frames.append(frame)
        object_points.append(obj)
        image_points.append(img)
        diagnostics.append(diag)

    return accepted_frames, object_points, image_points, diagnostics


def _choose_fixed_point_index(object_points: np.ndarray) -> int:
    pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("Object-Releasing requires at least two object points.")
    top_y = float(np.min(pts[:, 1]))
    top = np.where(np.isclose(pts[:, 1], top_y))[0]
    if len(top) == 0:
        fixed_idx = min(1, len(pts) - 1)
    else:
        fixed_idx = int(top[np.argmax(pts[top, 0])])
    if not 0 <= fixed_idx < len(pts):
        raise ValueError(f"Invalid iFixedPoint {fixed_idx} for {len(pts)} object points.")
    return fixed_idx


def _target_geometry_refinement(nominal: np.ndarray, refined: np.ndarray) -> dict[str, float]:
    nominal_pts = np.asarray(nominal, dtype=np.float32).reshape(-1, 3)
    refined_pts = np.asarray(refined, dtype=np.float32).reshape(-1, 3)
    n = min(len(nominal_pts), len(refined_pts))
    if n == 0:
        return {"mean_displacement": 0.0, "median_displacement": 0.0, "p95_displacement": 0.0, "max_displacement": 0.0}
    displacement = np.linalg.norm(refined_pts[:n] - nominal_pts[:n], axis=1)
    return {
        "mean_displacement": float(np.mean(displacement)),
        "median_displacement": float(np.median(displacement)),
        "p95_displacement": float(np.percentile(displacement, 95)),
        "max_displacement": float(np.max(displacement)),
    }


def calibrate_object_releasing_brown_conrady(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    *,
    fix_tangent_dist: bool = False,
) -> CalibrationResult:
    """Run OpenCV object-releasing calibration for full-target planar grids.

    This path intentionally ignores DetectionResult.excluded_corner_indices:
    object-releasing requires identical target point correspondence per frame,
    so bad observations should be rejected at frame level before calling this.
    """
    if not is_object_releasing_supported_pattern(pattern_config):
        ptype = _pattern_type(pattern_config)
        supported = ", ".join(p.value for p in SUPPORTED_OBJECT_RELEASING_PATTERN_TYPES)
        return CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            success=False,
            error_message=(
                f"Object-Releasing is disabled for {ptype.value}. "
                f"Supported targets: {supported}. Use Standard calibration for ChArUco/AprilGrid."
            ),
        )

    frames, object_points, image_points, diagnostics = collect_object_releasing_inputs(dataset, pattern_config)
    expected_count = int(len(expected_object_releasing_ids(pattern_config)))
    diagnostic_summary = format_object_releasing_diagnostics(
        diagnostics,
        accepted_count=len(frames),
        expected_count=expected_count,
        pattern_config=pattern_config,
    )
    if len(frames) < MIN_FRAMES_REQUIRED:
        return CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            success=False,
            error_message=diagnostic_summary,
            object_releasing_diagnostics=diagnostics,
        )

    image_size = infer_image_size(dataset, camera_config)
    flags = cv2.CALIB_ZERO_TANGENT_DIST if fix_tangent_dist else 0
    try:
        fixed_idx = _choose_fixed_point_index(object_points[0])
    except ValueError as e:
        return CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            success=False,
            error_message=str(e),
            object_releasing_diagnostics=diagnostics,
        )

    try:
        if hasattr(cv2, "calibrateCameraROExtended"):
            out = cv2.calibrateCameraROExtended(
                object_points,
                image_points,
                image_size,
                fixed_idx,
                None,
                None,
                flags=flags,
                criteria=DEFAULT_TERM_CRITERIA,
            )
            (
                rms,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs,
                refined_object_points,
                std_intrinsics,
                _std_extrinsics,
                _std_obj,
                per_view_errors,
            ) = out
        else:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs, refined_object_points = cv2.calibrateCameraRO(
                object_points,
                image_points,
                image_size,
                fixed_idx,
                None,
                None,
                flags=flags,
                criteria=DEFAULT_TERM_CRITERIA,
            )
            std_intrinsics = None
            per_view_errors = None
    except cv2.error as e:
        return CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            success=False,
            error_message=f"cv2.calibrateCameraRO failed: {e}",
            object_releasing_diagnostics=diagnostics,
        )

    invalid_reason = validate_finite_calibration_output(camera_matrix, dist_coeffs)
    if invalid_reason:
        return CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            success=False,
            error_message=invalid_reason,
            object_releasing_diagnostics=diagnostics,
        )

    if refined_object_points is None:
        refined_object_points = object_points[0].copy()
    if per_view_errors is None:
        per_frame_error = {}
    else:
        per_frame_error = {
            frame.image_info.image_id: float(per_view_errors[i][0])
            for i, frame in enumerate(frames)
        }
    for frame in frames:
        if frame.image_info.image_id in per_frame_error:
            frame.reprojection_error = per_frame_error[frame.image_info.image_id]

    regional_error = compute_regional_error(frames, per_frame_error, image_size)
    radial_profile = compute_radial_error_profile(
        frames, list(rvecs), list(tvecs), camera_matrix, dist_coeffs, image_size,
        CameraModelType.EXTENDED_PINHOLE,
    )
    radial_bands = compute_radial_error_bands(
        frames, list(rvecs), list(tvecs), camera_matrix, dist_coeffs, image_size,
        CameraModelType.EXTENDED_PINHOLE,
    )
    residual_stats = compute_residual_stats_for_calibration(
        frames, list(rvecs), list(tvecs), camera_matrix, dist_coeffs, image_size,
        CameraModelType.EXTENDED_PINHOLE,
    )
    spatial_error_map = compute_spatial_error_map(
        frames, list(rvecs), list(tvecs), camera_matrix, dist_coeffs, image_size,
        CameraModelType.EXTENDED_PINHOLE,
    )

    param_uncertainty = None
    if std_intrinsics is not None:
        param_uncertainty = ParameterUncertainty(
            fx_std=float(std_intrinsics[0][0]),
            fy_std=float(std_intrinsics[1][0]),
            cx_std=float(std_intrinsics[2][0]),
            cy_std=float(std_intrinsics[3][0]),
            method="covariance",
        )
        add_normal_approximation_ci(param_uncertainty, camera_matrix)

    return CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY,
        camera_matrix=camera_matrix,
        distortion=dist_coeffs,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=float(rms),
        per_frame_error=per_frame_error,
        regional_error=regional_error,
        radial_profile=radial_profile,
        radial_bands=radial_bands,
        param_uncertainty=param_uncertainty,
        residual_stats=residual_stats,
        spatial_error_map=spatial_error_map,
        calibration_method=CalibrationMethod.OBJECT_RELEASING,
        refined_object_points=np.asarray(refined_object_points, dtype=np.float32).reshape(-1, 1, 3),
        target_geometry_refinement=_target_geometry_refinement(object_points[0], refined_object_points),
        object_releasing_diagnostics=diagnostics,
        warning_message=diagnostic_summary,
        success=True,
    )
