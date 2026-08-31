"""Scene quality ranking and diversity-aware best-subset calibration.

Ranking은 per-view RMS만으로 정하지 않고 검출 완성도와 상대적 선명도를
함께 사용한다. Best-N 선택은 ranking 상위 N개가 아니라, 매 단계에서 아직
선택되지 않은 pose/coverage를 더 잘 채우는 scene에 보상을 준다.
"""

from __future__ import annotations

import copy
import numpy as np

from calibration.detector import maximum_pattern_corners
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.types import (
    CalibrationResult, CameraConfig, CameraModelType, Dataset, PatternConfig,
    SceneQualityAnalysis, SceneQualityEntry, SubsetCalibrationResult,
)

# 나중에 쉽게 tuning할 수 있도록 점수 가중치를 알고리즘에서 분리한다.
QUALITY_WEIGHTS = {
    "reprojection": 0.45,
    "detection": 0.35,
    "sharpness": 0.20,
}
SUBSET_WEIGHTS = {
    "quality": 0.55,
    "pose_diversity": 0.25,
    "coverage_gain": 0.20,
}


def _relative_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    """5~95 percentile clipping 후 0~1 정규화. 극단치 한 장의 지배를 줄인다."""
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    lo, hi = np.percentile(arr, [5, 95]) if len(arr) > 2 else (float(arr.min()), float(arr.max()))
    if float(hi - lo) < 1e-9:
        return {key: 0.5 for key in values}
    result = {}
    for key, value in values.items():
        normalized = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        result[key] = normalized if higher_is_better else 1.0 - normalized
    return result


def compute_scene_quality_analysis(
    dataset: Dataset,
    calibration_result: CalibrationResult,
    pattern_config: PatternConfig,
) -> SceneQualityAnalysis:
    """Initial model의 per-view RMS + 기존 detector/image metric으로 ranking을 계산."""
    eligible = [
        frame for frame in dataset.frames
        if frame.enabled and frame.detection and frame.detection.success
        and frame.image_info.image_id in calibration_result.per_frame_error
    ]
    rms_raw = {
        frame.image_info.image_id: float(calibration_result.per_frame_error[frame.image_info.image_id])
        for frame in eligible
    }
    sharp_raw = {
        frame.image_info.image_id: float(frame.image_info.sharpness)
        for frame in eligible if frame.image_info.sharpness is not None
    }
    rms_scores = _relative_scores(rms_raw, higher_is_better=False)
    sharp_scores = _relative_scores(sharp_raw, higher_is_better=True)
    expected = maximum_pattern_corners(pattern_config)

    scenes = []
    for frame in eligible:
        fid = frame.image_info.image_id
        detection_ratio = (
            float(np.clip(frame.detection.corner_confidence, 0.0, 1.0))
            if frame.detection.corner_confidence is not None
            else min(1.0, frame.detection.num_corners / max(1, expected))
        )
        reproj_score = rms_scores.get(fid, 0.5)
        sharp_score = sharp_scores.get(fid, 0.5)
        total = 100.0 * (
            QUALITY_WEIGHTS["reprojection"] * reproj_score
            + QUALITY_WEIGHTS["detection"] * detection_ratio
            + QUALITY_WEIGHTS["sharpness"] * sharp_score
        )
        scenes.append(SceneQualityEntry(
            frame_id=fid,
            quality_score=round(total, 1),
            reprojection_error=rms_raw.get(fid),
            detection_ratio=round(detection_ratio, 4),
            sharpness=frame.image_info.sharpness,
            reprojection_score=round(reproj_score * 100.0, 1),
            detection_score=round(detection_ratio * 100.0, 1),
            sharpness_score=round(sharp_score * 100.0, 1),
        ))
    scenes.sort(key=lambda item: (-item.quality_score, item.frame_id))
    for rank, scene in enumerate(scenes, 1):
        scene.rank = rank
    return SceneQualityAnalysis(model_name=calibration_result.model_name, scenes=scenes)


def _pose_vector(frame, camera_config: CameraConfig) -> np.ndarray:
    det = frame.detection
    center = det.board_center_px or (camera_config.width / 2, camera_config.height / 2)
    area = det.board_area_ratio if det.board_area_ratio is not None else 0.0
    tilt = det.board_tilt_deg if det.board_tilt_deg is not None else 0.0
    return np.array([
        center[0] / max(1, camera_config.width),
        center[1] / max(1, camera_config.height),
        min(1.0, area / 0.55),
        min(1.0, abs(tilt) / 60.0),
    ], dtype=float)


def _coverage_cells(frame, camera_config: CameraConfig, rows: int = 4, cols: int = 4) -> set[int]:
    if frame.detection is None or frame.detection.corners is None:
        return set()
    cells = set()
    for x, y in frame.detection.corners.reshape(-1, 2):
        col = min(cols - 1, max(0, int(x / max(1, camera_config.width) * cols)))
        row = min(rows - 1, max(0, int(y / max(1, camera_config.height) * rows)))
        cells.add(row * cols + col)
    return cells


def recommend_best_subset(
    dataset: Dataset,
    analysis: SceneQualityAnalysis,
    camera_config: CameraConfig,
    count: int,
) -> list[str]:
    """Quality + minimum pose distance + new coverage gain을 greedy로 최대화."""
    by_id = {f.image_info.image_id: f for f in dataset.frames}
    quality = {s.frame_id: s.quality_score / 100.0 for s in analysis.scenes if s.frame_id in by_id}
    candidates = list(quality)
    target = min(max(0, count), len(candidates))
    if target == 0:
        return []

    poses = {fid: _pose_vector(by_id[fid], camera_config) for fid in candidates}
    cells = {fid: _coverage_cells(by_id[fid], camera_config) for fid in candidates}
    rank_by_id = {scene.frame_id: scene.rank for scene in analysis.scenes}
    selected: list[str] = []
    covered: set[int] = set()
    while len(selected) < target:
        best_id = None
        best_key = None
        for fid in candidates:
            if fid in selected:
                continue
            if selected:
                distance = min(float(np.linalg.norm(poses[fid] - poses[chosen])) for chosen in selected)
                pose_score = min(1.0, distance)
            else:
                pose_score = 1.0
            new_cells = cells[fid] - covered
            coverage_gain = len(new_cells) / max(1, len(cells[fid]))
            value = (
                SUBSET_WEIGHTS["quality"] * quality[fid]
                + SUBSET_WEIGHTS["pose_diversity"] * pose_score
                + SUBSET_WEIGHTS["coverage_gain"] * coverage_gain
            )
            tie_key = (value, quality[fid], -rank_by_id[fid])
            if best_key is None or tie_key > best_key:
                best_id, best_key = fid, tie_key
        assert best_id is not None
        selected.append(best_id)
        covered.update(cells[best_id])
    return selected


def run_subset_calibration(
    dataset: Dataset,
    selected_frame_ids: list[str],
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    original_diversity=None,
    original_coverage_pct: float | None = None,
) -> SubsetCalibrationResult:
    """Deep-copied subset으로 동일 model을 재계산해 Original을 변경하지 않는다."""
    from calibration.compare import run_all_models
    from calibration.validation import split_train_test, validate_holdout

    if model is None:
        raise ValueError("Subset Calibration requires a Camera Model, but none was provided.")
    model = model if isinstance(model, CameraModelType) else CameraModelType(str(model))
    selected = set(selected_frame_ids)
    subset = Dataset(frames=copy.deepcopy([
        frame for frame in dataset.frames
        if frame.image_info.image_id in selected and frame.enabled
        and frame.detection and frame.detection.success
    ]))
    if len(subset.frames) < 3:
        raise ValueError(
            f"Only {len(subset.frames)} valid selected scenes remain; at least 3 are required."
        )
    analyze_dataset_quality(subset, camera_config)
    results = run_all_models(subset, camera_config, models=[model], model_jobs=1)
    result = results[0] if results else None
    train_ids, test_ids = split_train_test(subset, camera_config, 0.25, 42)
    validation = validate_holdout(subset, camera_config, pattern_config, model, train_ids, test_ids)
    subset_coverage = coverage_percentage(subset.coverage_grid)
    warnings = []
    if original_coverage_pct is not None and subset_coverage < original_coverage_pct - 20.0:
        warnings.append(
            f"Coverage dropped substantially: {original_coverage_pct:.1f}% → {subset_coverage:.1f}%."
        )
    if original_diversity is not None and subset.diversity is not None:
        if subset.diversity.overall < original_diversity.overall - 0.15:
            warnings.append(
                "Low-RMS subset is concentrated in similar target poses: "
                f"diversity {original_diversity.overall:.2f} → {subset.diversity.overall:.2f}."
            )
    return SubsetCalibrationResult(
        model_name=model,
        selected_frame_ids=[f.image_info.image_id for f in subset.frames],
        calibration_result=result,
        validation_result=validation,
        coverage_grid=subset.coverage_grid,
        diversity=subset.diversity,
        coverage_percentage=subset_coverage,
        original_coverage_percentage=original_coverage_pct or 0.0,
        original_diversity=copy.deepcopy(original_diversity),
        warnings=warnings,
    )


def add_original_comparison_warnings(
    subset: SubsetCalibrationResult,
    original_result: CalibrationResult | None,
    original_validation,
) -> None:
    """RMS 개선만으로 GOOD 판정하지 않도록 generalization/stability 저하를 경고."""
    subset_result = subset.calibration_result
    subset_validation = subset.validation_result
    if (
        original_validation and subset_validation
        and original_validation.success and subset_validation.success
        and original_validation.test_rms is not None and subset_validation.test_rms is not None
        and subset_validation.test_rms > original_validation.test_rms * 1.10
    ):
        subset.warnings.append(
            "Hold-out generalization became worse despite subset fitting: "
            f"{original_validation.test_rms:.3f}px → {subset_validation.test_rms:.3f}px."
        )

    def stability(result):
        if result is None:
            return None
        uncertainty = result.param_uncertainty_bootstrap or result.param_uncertainty
        return uncertainty.overall_stability if uncertainty else None

    original_stability = stability(original_result)
    subset_stability = stability(subset_result)
    if (
        original_stability is not None and subset_stability is not None
        and subset_stability < original_stability - 15.0
    ):
        subset.warnings.append(
            f"Parameter stability dropped: {original_stability:.0f}% → {subset_stability:.0f}%."
        )
