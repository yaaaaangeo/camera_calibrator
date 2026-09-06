"""
calibration.windshield.ghost.evaluator
==============================================================

STEP 8A - Ghost Evaluation 최상위 API.

Point-Source/Edge-Target/General(No-Reference) Likelihood 세 모드를 모두
제공하되, 절대 서로 섞지 않는다 - `ReflectionEvaluationResult`와도 완전히
분리된 `GhostEvaluationResult`만 반환한다(사용자 스펙 1/22번).

K, D(그리고 base windshield model)는 여기서 절대 재보정하지 않는다 - 이미
확정된 값을 받아서 main/ghost pixel을 camera ray로 바꾸는 데만 쓴다(사용자
스펙 16-18번, Angular Ghost Separation).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import cv2
import numpy as np

from calibration.types import CameraModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.ghost.config import (
    DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX,
    DEFAULT_LIKELIHOOD_MIN_GRADIENT,
    DEFAULT_MAX_SEARCH_RADIUS_PX,
    DEFAULT_MIN_BLOB_AREA_PX,
    GHOST_METRIC_VERSION,
)
from calibration.windshield.ghost.edge_detector import detect_edge_ghost, extract_edge_profiles
from calibration.windshield.ghost.point_detector import detect_bright_blobs, pair_main_and_ghost_blobs
from calibration.windshield.ghost.spatial_model import build_spatial_map, compute_region_metrics
from calibration.windshield.ghost.types import (
    GhostDatasetResult,
    GhostEvaluationConfig,
    GhostEvaluationResult,
    GhostPointDetection,
)

_GENERAL_LIKELIHOOD_WARNING = (
    "General Image 모드의 결과는 point-source ground truth 없이 얻은 heuristic "
    "'Ghost Likelihood'입니다 - Ghost Ground Truth가 아닙니다. 차선, 건물 외곽선, "
    "창문/텍스처, motion blur, defocus 등이 모두 double-edge처럼 보일 수 있으므로 "
    "이 값은 낮은 신뢰도로 취급해야 합니다."
)


def _angular_separations(
    detections: list[GhostPointDetection],
    *,
    camera_matrix: Optional[np.ndarray],
    distortion: Optional[np.ndarray],
    camera_model: Optional[CameraModelType],
) -> list[float]:
    if camera_matrix is None or distortion is None or camera_model is None:
        return []
    base_model = BaselineWindshieldModel(camera_matrix, distortion, camera_model)
    angles: list[float] = []
    for det in detections:
        if not det.detected or det.ghost_x is None or det.ghost_y is None:
            continue
        try:
            r_main = np.array(base_model.unproject_pixel(det.main_x, det.main_y), dtype=np.float64)
            r_ghost = np.array(base_model.unproject_pixel(det.ghost_x, det.ghost_y), dtype=np.float64)
        except Exception:
            continue
        dot = float(np.clip(np.dot(r_main, r_ghost), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        det.angular_separation_deg = angle_deg
        angles.append(angle_deg)
    return angles


def _percentile_or_none(values: list[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def evaluate_ghost_point_source(
    image_bgr: np.ndarray,
    config: Optional[GhostEvaluationConfig] = None,
    *,
    camera_matrix: Optional[np.ndarray] = None,
    distortion: Optional[np.ndarray] = None,
    camera_model: Optional[CameraModelType] = None,
    pair_id: str = "",
) -> GhostEvaluationResult:
    """LED dot/night light 같은 point-source 이미지에서 Ghost를 평가한다."""
    cfg = config or GhostEvaluationConfig(mode="point_source")
    try:
        blobs = detect_bright_blobs(
            image_bgr,
            threshold=cfg.bright_source_threshold,
            min_area_px=cfg.min_blob_area_px,
            gaussian_sigma=cfg.gaussian_sigma,
            min_peak_distance_px=cfg.min_peak_distance_px,
        )
        detections = pair_main_and_ghost_blobs(
            blobs,
            max_search_radius_px=cfg.max_search_radius_px,
            consensus_radius_px=cfg.pairing_consensus_radius_px,
            min_consensus_candidates=cfg.min_consensus_candidates,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return GhostEvaluationResult(
            success=False,
            mode="point_source",
            pair_id=pair_id,
            error_message=f"point-source ghost 검출 중 예외 발생: {exc}",
        )

    h, w = image_bgr.shape[:2]
    _angular_separations(detections, camera_matrix=camera_matrix, distortion=distortion, camera_model=camera_model)

    detected = [d for d in detections if d.detected]
    offsets_x = [d.offset_x_px for d in detected if d.offset_x_px is not None]
    offsets_y = [d.offset_y_px for d in detected if d.offset_y_px is not None]
    distances = [d.distance_px for d in detected if d.distance_px is not None]
    strengths = [d.strength_ratio for d in detected if d.strength_ratio is not None]
    angles = [d.angular_separation_deg for d in detected if d.angular_separation_deg is not None]

    detection_rate = (len(detected) / len(detections)) if detections else None

    spatial_map = build_spatial_map(detections, image_width=w, image_height=h, rows=cfg.spatial_rows, cols=cfg.spatial_cols)
    regional_metrics = compute_region_metrics(detections, image_width=w, image_height=h)

    warning = None
    if not detections:
        warning = "밝은 point-source 후보를 찾지 못했습니다 - threshold/데이터셋을 확인하세요."

    return GhostEvaluationResult(
        success=True,
        mode="point_source",
        metric_version=GHOST_METRIC_VERSION,
        pair_id=pair_id,
        detection_count=len(detected),
        candidate_count=len(detections),
        detection_rate=detection_rate,
        mean_offset_x_px=_mean_or_none(offsets_x),
        mean_offset_y_px=_mean_or_none(offsets_y),
        median_distance_px=_median_or_none(distances),
        p95_distance_px=_percentile_or_none(distances, 95),
        mean_angular_separation_deg=_mean_or_none(angles),
        p95_angular_separation_deg=_percentile_or_none(angles, 95),
        mean_strength_ratio=_mean_or_none(strengths),
        p95_strength_ratio=_percentile_or_none(strengths, 95),
        regional_metrics=regional_metrics,
        spatial_map=spatial_map,
        detections=detections,
        warning_message=warning,
    )


def evaluate_ghost_edge_target(
    intensity_profiles: list[np.ndarray],
    config: Optional[GhostEvaluationConfig] = None,
    *,
    pair_id: str = "",
) -> GhostEvaluationResult:
    """고대비 직선(edge) target의 1D intensity profile 리스트에서 Ghost를
    평가한다. 각 profile은 edge에 수직인 방향으로 스캔한 밝기 값이다."""
    cfg = config or GhostEvaluationConfig(mode="edge_target")
    try:
        results = [
            detect_edge_ghost(
                profile,
                min_gradient=cfg.edge_min_gradient,
                max_search_radius_px=cfg.edge_max_search_radius_px,
                min_secondary_ratio=cfg.edge_min_secondary_ratio,
            )
            for profile in intensity_profiles
        ]
    except Exception as exc:  # pragma: no cover - defensive
        return GhostEvaluationResult(
            success=False,
            mode="edge_target",
            pair_id=pair_id,
            error_message=f"edge ghost 검출 중 예외 발생: {exc}",
        )

    detected = [r for r in results if r.detected]
    offsets = [r.offset_px for r in detected if r.offset_px is not None]
    strengths = [r.strength_ratio for r in detected if r.strength_ratio is not None]
    detection_rate = (len(detected) / len(results)) if results else None

    warning = None
    if not results:
        warning = "edge intensity profile이 제공되지 않았습니다."

    abs_offsets = [abs(o) for o in offsets]
    return GhostEvaluationResult(
        success=True,
        mode="edge_target",
        metric_version=GHOST_METRIC_VERSION,
        pair_id=pair_id,
        detection_count=len(detected),
        candidate_count=len(results),
        detection_rate=detection_rate,
        # Point Source의 2D dx/dy(mean_offset_x_px/median_distance_px)와
        # Edge의 1D scalar offset을 절대 같은 필드로 재사용하지 않는다
        # (STEP 8 stabilization 2-F번) - edge 전용 필드에만 채운다.
        edge_offset_median_px=_median_or_none(abs_offsets) if abs_offsets else None,
        edge_offset_p95_px=_percentile_or_none(abs_offsets, 95) if abs_offsets else None,
        mean_strength_ratio=_mean_or_none(strengths),
        p95_strength_ratio=_percentile_or_none(strengths, 95),
        warning_message=warning,
    )


def evaluate_ghost_general_likelihood(
    image_bgr: np.ndarray,
    config: Optional[GhostEvaluationConfig] = None,
    *,
    pair_id: str = "",
) -> GhostEvaluationResult:
    """Point-source ground truth가 없는 일반 도로 영상에서 heuristic하게
    ghost 가능성을 추정한다(사용자 스펙 24-26번). 결과는 절대 "Ground
    Truth"라고 부르지 않고 `ghost_likelihood`/`is_likelihood=True`로만
    노출한다."""
    cfg = config or GhostEvaluationConfig(mode="general_likelihood")
    try:
        lum = image_bgr.astype(np.float32)
        if lum.ndim == 3:
            lum = lum.mean(axis=2)
        # 수평 방향 double-edge 탐색: 각 행을 1D profile로 보고 edge 검출기를 재사용한다.
        row_results = [
            detect_edge_ghost(
                lum[row, :],
                min_gradient=DEFAULT_LIKELIHOOD_MIN_GRADIENT if cfg.likelihood_min_gradient is None else cfg.likelihood_min_gradient,
                max_search_radius_px=cfg.likelihood_max_search_radius_px,
                min_secondary_ratio=0.05,
            )
            for row in range(0, lum.shape[0], max(1, lum.shape[0] // 64))
        ]
    except Exception as exc:  # pragma: no cover - defensive
        return GhostEvaluationResult(
            success=False,
            mode="general_likelihood",
            pair_id=pair_id,
            is_likelihood=True,
            error_message=f"general likelihood 평가 중 예외 발생: {exc}",
        )

    detected = [r for r in row_results if r.detected]
    likelihood = (len(detected) / len(row_results)) if row_results else 0.0
    strengths = [r.strength_ratio for r in detected if r.strength_ratio is not None]

    return GhostEvaluationResult(
        success=True,
        mode="general_likelihood",
        metric_version=GHOST_METRIC_VERSION,
        pair_id=pair_id,
        detection_count=len(detected),
        candidate_count=len(row_results),
        detection_rate=float(likelihood) if row_results else None,
        mean_strength_ratio=_mean_or_none(strengths),
        p95_strength_ratio=_percentile_or_none(strengths, 95),
        ghost_likelihood=float(likelihood),
        is_likelihood=True,
        warning_message=_GENERAL_LIKELIHOOD_WARNING,
    )


def evaluate_ghost_dataset(
    per_frame: list[GhostEvaluationResult],
    *,
    mode: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    num_input_frames: Optional[int] = None,
) -> GhostDatasetResult:
    """여러 프레임의 `GhostEvaluationResult`를 dataset 단위로 집계한다.

    Edge-target 모드는 point-source의 `median_distance_px`가 아니라 전용
    `edge_offset_median_px`를 "distance-like" 집계 소스로 쓴다(사용자
    스펙 - Point/Edge 필드를 절대 섞지 않는다).

    General(No-Reference) Likelihood 모드는 dataset-level primary summary가
    `mean_strength`/`p95_strength`(2차 edge의 상대 강도)가 아니라
    `mean_ghost_likelihood`/`median_ghost_likelihood`/`p95_ghost_likelihood`
    (double-edge 패턴이 나타난 profile 비율)여야 한다(STEP 8 semantic fix
    1번, "Ghost Likelihood != Ghost Strength") - 이 두 값을 절대 같은
    필드에 넣지 않는다."""
    successful = [r for r in per_frame if r.success]
    is_general = mode == "general_likelihood"

    def _distance_like(r: GhostEvaluationResult) -> Optional[float]:
        return r.edge_offset_median_px if mode == "edge_target" else r.median_distance_px

    distances = [d for r in successful if (d := _distance_like(r)) is not None]
    strengths = [r.mean_strength_ratio for r in successful if r.mean_strength_ratio is not None]
    likelihoods = [r.ghost_likelihood for r in successful if r.ghost_likelihood is not None]

    worst_frame_id = None
    worst_val = -1.0
    for r in successful:
        # General 모드는 "가장 심한 프레임"을 likelihood 기준으로 고른다 -
        # distance/strength는 이 모드에서 정의되지 않는다.
        val = (r.ghost_likelihood if is_general else _distance_like(r))
        val = val if val is not None else -1.0
        if val > worst_val:
            worst_val = val
            worst_frame_id = r.pair_id or None

    warning = None
    if not per_frame:
        warning = "평가할 프레임이 없습니다."
    elif not successful:
        warning = "성공적으로 평가된 프레임이 없습니다."

    return GhostDatasetResult(
        mode=mode,
        metric_version=GHOST_METRIC_VERSION,
        per_frame=per_frame,
        # mean_distance_px/mean_strength는 그대로 유지한다(하위 호환 -
        # 사용자 스펙 1-B번). General mode에서도 "검출된 secondary edge가
        # 얼마나 강한지"(Strength)는 여전히 의미가 있는 별개의 통계이고,
        # Likelihood("double-edge 패턴이 나타난 비율")를 대체하지 않는다
        # (사용자 스펙 1-A번, "Likelihood=0.70, Strength=0.12는 충분히
        # 가능하다"). General 모드 per-frame 결과는 애초에 distance를
        # 채우지 않으므로 mean_distance_px는 자연히 None으로 남는다.
        mean_distance_px=_mean_or_none(distances),
        p95_distance_px=_percentile_or_none(distances, 95),
        mean_strength=_mean_or_none(strengths),
        p95_strength=_percentile_or_none(strengths, 95),
        mean_ghost_likelihood=_mean_or_none(likelihoods) if is_general else None,
        median_ghost_likelihood=_median_or_none(likelihoods) if is_general else None,
        p95_ghost_likelihood=_percentile_or_none(likelihoods, 95) if is_general else None,
        worst_frame_id=worst_frame_id,
        image_width=image_width,
        image_height=image_height,
        num_input_frames=num_input_frames if num_input_frames is not None else len(per_frame),
        num_valid_frames=len(per_frame),
        success=bool(successful),
        warning_message=warning,
    )


def evaluate_ghost_image(
    image_bgr: np.ndarray,
    config: Optional[GhostEvaluationConfig] = None,
    *,
    camera_matrix: Optional[np.ndarray] = None,
    distortion: Optional[np.ndarray] = None,
    camera_model: Optional[CameraModelType] = None,
    pair_id: str = "",
) -> GhostEvaluationResult:
    """`config.mode`에 따라 세 evaluator 중 하나로 dispatch하는 단일
    진입점(STEP 8 stabilization 8번) - Evaluation Worker와 Suppression
    Worker가 각자 if/else를 중복해서 만들지 않도록 한다.

    Edge-target 모드는 여기서 이미지로부터 1D profile을 직접 추출한다
    (`extract_edge_profiles`) - 호출자가 profile을 직접 만들 필요가 없다
    (STEP 8 stabilization 2-B번).
    """
    cfg = config or GhostEvaluationConfig()
    mode = cfg.mode
    if mode == "point_source":
        return evaluate_ghost_point_source(
            image_bgr, cfg, camera_matrix=camera_matrix, distortion=distortion, camera_model=camera_model, pair_id=pair_id,
        )
    if mode == "edge_target":
        try:
            profiles = extract_edge_profiles(image_bgr, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            return GhostEvaluationResult(
                success=False, mode="edge_target", pair_id=pair_id,
                error_message=f"edge profile 추출 중 예외 발생: {exc}",
            )
        return evaluate_ghost_edge_target(profiles, cfg, pair_id=pair_id)
    if mode == "general_likelihood":
        return evaluate_ghost_general_likelihood(image_bgr, cfg, pair_id=pair_id)
    raise ValueError(f"Unknown ghost evaluation mode: {mode!r}")


def evaluate_ghost_dataset_from_paths(
    image_paths: list[str],
    config: Optional[GhostEvaluationConfig] = None,
    *,
    frame_ids: Optional[list[str]] = None,
    camera_matrix: Optional[np.ndarray] = None,
    distortion: Optional[np.ndarray] = None,
    camera_model: Optional[CameraModelType] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GhostDatasetResult:
    """이미지 경로 리스트를 읽어(`cv2.imread`) `evaluate_ghost_image()`로
    평가하고 `evaluate_ghost_dataset()`으로 집계하는 Qt-independent 순수
    함수 - `ui/ghost_evaluation_worker.py::GhostEvaluationWorker.run()`이
    이 함수 하나를 호출하기만 하므로, pytest에서 PySide6 없이도 dataset
    파이프라인(파일 로딩 + resolution gate + 평가 + 집계) 전체를 직접
    검증할 수 있다.

    STEP 8 semantic/safety fix 4번 - Multi-frame Resolution Consistency
    Gate: Ghost displacement는 pixel 좌표계로 측정되므로 서로 다른 해상도
    프레임을 하나의 dataset으로 섞으면 안 된다(같은 물리적 변위도 해상도에
    따라 다른 px 값으로 보인다). 첫 유효 프레임의 해상도를 기준으로 삼고,
    이후 프레임 중 하나라도 다르면 조용히 건너뛰거나 resize/crop하지 않고
    `ValueError`로 dataset 전체를 실패시킨다(어떤 파일이 문제인지 메시지에
    명시).
    """
    cfg = config or GhostEvaluationConfig()
    ids = frame_ids or [str(i) for i in range(len(image_paths))]
    n = len(image_paths)

    per_frame: list[GhostEvaluationResult] = []
    expected_shape: Optional[tuple[int, int]] = None
    expected_path: Optional[str] = None

    for i, (path, frame_id) in enumerate(zip(image_paths, ids)):
        if progress_callback is not None:
            progress_callback(f"Evaluating ghost {i + 1}/{n}...")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        shape = image.shape[:2]  # (height, width)
        if expected_shape is None:
            expected_shape = shape
            expected_path = path
        elif shape != expected_shape:
            eh, ew = expected_shape
            fh, fw = shape
            raise ValueError(
                "Ghost dataset contains mixed image resolutions.\n\n"
                f"Expected: {ew}x{eh} (from {os.path.basename(expected_path)})\n"
                f"Found:    {fw}x{fh}\n"
                f"File:     {os.path.basename(path)}\n\n"
                "Ghost displacement is measured in pixel coordinates, so "
                "mixed-resolution frames cannot be combined safely. No automatic "
                "resize/crop was applied - fix the dataset and retry."
            )

        res = evaluate_ghost_image(
            image, cfg, camera_matrix=camera_matrix, distortion=distortion, camera_model=camera_model, pair_id=frame_id,
        )
        per_frame.append(res)

    image_height, image_width = expected_shape if expected_shape is not None else (None, None)
    return evaluate_ghost_dataset(
        per_frame, mode=cfg.mode, image_width=image_width, image_height=image_height, num_input_frames=n,
    )
