from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from calibration.windshield.reflection.alignment import align_reference_to_normal
from calibration.windshield.reflection.metrics import (
    apply_valid_mask,
    bottom_roi_metrics,
    contrast_retention,
    downsample_map,
    edge_retention,
    glare_coverage,
    glare_strength,
    gradient_magnitude,
    nan_safe_coverage,
    nan_safe_max,
    nan_safe_mean,
    nan_safe_median,
    nan_safe_percentile,
    normalize_reflection_map,
    region_metrics,
    saturation_coverage,
    severity_from_metrics,
    spatial_cells,
    to_luminance,
)
from calibration.windshield.reflection.types import (
    REFLECTION_METRIC_VERSION,
    ReflectionDatasetResult,
    ReflectionEvaluationConfig,
    ReflectionEvaluationResult,
    ReflectionImagePair,
)


def evaluate_reflection(
    normal_image: np.ndarray,
    reference_image: np.ndarray | None = None,
    config: ReflectionEvaluationConfig | None = None,
    *,
    pair_id: str = "",
) -> ReflectionEvaluationResult:
    cfg = config or ReflectionEvaluationConfig(mode="reference" if reference_image is not None else "no_reference")
    mode = cfg.mode.lower()
    if mode not in {"reference", "no_reference"}:
        raise ValueError(f"unsupported reflection evaluation mode: {cfg.mode}")
    if mode == "reference":
        if reference_image is None:
            raise ValueError("reference mode requires reference_image")
        return evaluate_reflection_reference(normal_image, reference_image, cfg, pair_id=pair_id)
    return evaluate_reflection_no_reference(normal_image, cfg, pair_id=pair_id)


def evaluate_reflection_reference(
    normal_image: np.ndarray,
    reference_image: np.ndarray,
    config: ReflectionEvaluationConfig | None = None,
    *,
    pair_id: str = "",
) -> ReflectionEvaluationResult:
    cfg = config or ReflectionEvaluationConfig(mode="reference")
    alignment_enabled = True if not cfg.allow_unsafe_reference_bypass else cfg.align
    normalization_enabled = True if not cfg.allow_unsafe_reference_bypass else cfg.photometric_normalize
    if normal_image.shape[:2] != reference_image.shape[:2]:
        return ReflectionEvaluationResult(
            mode="reference",
            pair_id=pair_id,
            no_reference_is_likelihood=False,
            alignment_status="invalid",
            success=False,
            error_message="normal/reference images must have the same resolution",
        )

    normal_luma = to_luminance(normal_image)
    reference_luma = to_luminance(reference_image)
    alignment = align_reference_to_normal(
        normal_luma,
        reference_luma,
        method=cfg.alignment_model,
        enabled=alignment_enabled,
    )
    if alignment.status == "invalid":
        return ReflectionEvaluationResult(
            mode="reference",
            pair_id=pair_id,
            no_reference_is_likelihood=False,
            alignment_score=alignment.score,
            alignment_error_px=alignment.error_px,
            alignment_status=alignment.status,
            alignment_method=alignment.method,
            success=False,
            error_message=alignment.warning_message or "alignment quality is invalid",
        )

    # Phase B-1 안정화 - warp로 생긴 인공적인 border 영역(alignment.valid_mask
    # 밖)을 이후의 모든 통계에서 제외한다. normal_luma도 같은 좌표계(warp
    # 안 된 "normal" 이미지 자체)이므로 같은 mask를 적용한다.
    masked_normal_luma = apply_valid_mask(normal_luma, alignment.valid_mask)
    masked_aligned_reference = apply_valid_mask(alignment.aligned_reference, alignment.valid_mask)

    reflection_map, positive_map, gain, bias, normalized_reference = normalize_reflection_map(
        masked_normal_luma,
        masked_aligned_reference,
        photometric_normalize=normalization_enabled,
    )
    return _result_from_map(
        reflection_map,
        positive_map,
        normal_image,
        # glare/saturation은 valid_mask를 별도 파라미터로 받아 명시적으로
        # 걸러내므로, 여기서는 NaN이 섞이지 않은 원본 normal_luma를 넘긴다
        # (NaN을 cv2.blur에 넣으면 local_contrast가 border 근처까지 blur로
        # 오염시켜 valid_mask의 정확한 경계와 어긋난다).
        normal_luma,
        normalized_reference,
        cfg,
        mode="reference",
        pair_id=pair_id,
        alignment_score=alignment.score,
        alignment_error_px=alignment.error_px,
        alignment_status=alignment.status,
        alignment_method=alignment.method,
        alignment_translation_x_px=alignment.translation_x_px,
        alignment_translation_y_px=alignment.translation_y_px,
        alignment_rotation_deg=alignment.rotation_deg,
        alignment_scale=alignment.scale,
        alignment_shear_deg=alignment.shear_deg,
        photometric_normalized=normalization_enabled,
        photometric_gain=gain,
        photometric_bias=bias,
        contrast_retention_value=contrast_retention(masked_normal_luma, normalized_reference),
        edge_retention_value=edge_retention(masked_normal_luma, normalized_reference),
        valid_mask=alignment.valid_mask,
        warning_message=alignment.warning_message,
    )


def evaluate_reflection_no_reference(
    normal_image: np.ndarray,
    config: ReflectionEvaluationConfig | None = None,
    *,
    pair_id: str = "",
) -> ReflectionEvaluationResult:
    cfg = config or ReflectionEvaluationConfig(mode="no_reference")
    luma = to_luminance(normal_image)
    blur = cv2.GaussianBlur(luma, (0, 0), sigmaX=9.0)
    contrast = cv2.GaussianBlur(gradient_magnitude(luma), (0, 0), sigmaX=3.0)
    bright = np.clip((blur - np.percentile(blur, 55.0)) / 80.0, 0.0, 1.0)
    low_detail = 1.0 - np.clip(contrast / (np.percentile(contrast, 90.0) + 1e-6), 0.0, 1.0)
    likelihood = (0.65 * bright + 0.35 * low_detail).astype(np.float32)
    return _result_from_map(
        likelihood,
        likelihood,
        normal_image,
        luma,
        None,
        cfg,
        mode="no_reference",
        pair_id=pair_id,
        alignment_status="not_run",
        alignment_method="none",
        no_reference_is_likelihood=True,
        warning_message="No-reference mode reports GT-free reflection likelihood only.",
        contrast_retention_value=None,
        edge_retention_value=None,
    )


def evaluate_reflection_pair(
    pair: ReflectionImagePair,
    config: ReflectionEvaluationConfig | None = None,
) -> ReflectionEvaluationResult:
    normal = _read_image(pair.normal_image_path)
    reference = _read_image(pair.reference_image_path) if pair.reference_image_path else None
    cfg = config or ReflectionEvaluationConfig(mode="reference" if reference is not None else "no_reference")
    pair_id = pair.pair_id or Path(pair.normal_image_path).stem
    return evaluate_reflection(normal, reference, cfg, pair_id=pair_id)


def evaluate_reflection_dataset(
    pairs: Iterable[ReflectionImagePair],
    config: ReflectionEvaluationConfig | None = None,
) -> ReflectionDatasetResult:
    pair_list = list(pairs)
    if not pair_list:
        return ReflectionDatasetResult(
            mode=(config.mode if config else "reference"),
            success=False,
            error_message="no reflection pairs were provided",
        )
    inferred_mode = "reference" if pair_list[0].reference_image_path else "no_reference"
    cfg = config or ReflectionEvaluationConfig(mode=inferred_mode)
    # Phase A-7 안정화 - 각 pair를 독립적으로 평가한다. 이전에는
    # `evaluate_reflection_pair()`가 던지는 예외(예: 이미지 파일을 읽을 수
    # 없음)가 list comprehension 전체를 중단시켜서, "Pair 03만 INVALID"인
    # 상황에서도 그 뒤에 있는 Pair 04/05가 아예 평가되지 않고 dataset 전체가
    # 예외로 죽었다. 한 pair의 실패가 다른 pair 평가를 막지 못하게 한다.
    results: list[ReflectionEvaluationResult] = []
    for pair in pair_list:
        try:
            results.append(evaluate_reflection_pair(pair, cfg))
        except Exception as exc:  # noqa: BLE001 - 한 pair의 실패를 dataset 전체 실패로 번지게 하지 않는다
            pair_id = pair.pair_id or Path(pair.normal_image_path).stem
            results.append(
                ReflectionEvaluationResult(
                    mode=cfg.mode, pair_id=pair_id, success=False, error_message=str(exc),
                )
            )
    successful = [r for r in results if r.success]
    if not successful:
        return ReflectionDatasetResult(
            mode=cfg.mode,
            pair_results=results,
            num_valid_pairs=0,
            num_invalid_pairs=len(results),
            success=False,
            error_message="no valid reflection pairs were evaluated",
        )

    means = np.array([r.mean_strength for r in successful], dtype=np.float32)
    coverages = np.array([r.coverage for r in successful], dtype=np.float32)
    worst = max(successful, key=lambda r: r.p95_strength)
    by_day_night: dict[str, list[ReflectionEvaluationResult]] = defaultdict(list)
    for pair, result in zip(pair_list, results):
        if result.success and pair.day_night:
            by_day_night[pair.day_night.lower()].append(result)

    grouped = {
        key: {
            "mean_strength": float(np.mean([r.mean_strength for r in vals])),
            "coverage": float(np.mean([r.coverage for r in vals])),
            "p95_strength": float(np.mean([r.p95_strength for r in vals])),
        }
        for key, vals in by_day_night.items()
    }
    is_no_reference = cfg.mode == "no_reference"
    return ReflectionDatasetResult(
        mode=cfg.mode,
        metric_version=REFLECTION_METRIC_VERSION,
        pair_results=results,
        reference_mean_strength=None if is_no_reference else float(np.mean(means)),
        reference_p95_strength=None if is_no_reference else float(np.mean([r.p95_strength for r in successful])),
        reference_coverage=None if is_no_reference else float(np.mean(coverages)),
        mean_reflection_likelihood=float(np.mean(means)) if is_no_reference else None,
        p95_reflection_likelihood=float(np.percentile(means, 95.0)) if is_no_reference else None,
        mean_strength=float(np.mean(means)),
        median_strength=float(np.median(means)),
        p95_strength=float(np.percentile(means, 95.0)),
        worst_pair_id=worst.pair_id,
        coverage=float(np.mean(coverages)),
        severity_score=None if is_no_reference else severity_from_metrics(float(np.mean(means)), float(np.mean([r.p95_strength for r in successful])), float(np.mean(coverages))),
        by_day_night=grouped,
        num_valid_pairs=len(successful),
        num_invalid_pairs=len(results) - len(successful),
        success=True,
        warning_message=(
            f"{len(results) - len(successful)} of {len(results)} pair(s) were invalid and excluded."
            if len(successful) != len(results) else None
        ),
    )


def _result_from_map(
    reflection_map: np.ndarray,
    positive_map: np.ndarray,
    normal_image: np.ndarray,
    normal_luma: np.ndarray,
    reference_luma: np.ndarray | None,
    config: ReflectionEvaluationConfig,
    *,
    mode: str,
    pair_id: str,
    alignment_score: float | None = None,
    alignment_error_px: float | None = None,
    alignment_status: str = "not_run",
    alignment_method: str = "none",
    alignment_translation_x_px: float | None = None,
    alignment_translation_y_px: float | None = None,
    alignment_rotation_deg: float | None = None,
    alignment_scale: float | None = None,
    alignment_shear_deg: float | None = None,
    no_reference_is_likelihood: bool = False,
    photometric_normalized: bool = False,
    photometric_gain: float | None = None,
    photometric_bias: float | None = None,
    warning_message: str | None = None,
    contrast_retention_value: float | None = None,
    edge_retention_value: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> ReflectionEvaluationResult:
    # Phase B-1 안정화 - `m`/`p`는 이미 NaN(invalid warp border)을 담고
    # 있을 수 있으므로(No-Reference 모드는 애초에 NaN이 없다) nan-aware
    # 집계 함수를 쓴다. `positive_mean_strength`/`positive_p95_strength`도
    # 동일하게 처리한다.
    m = np.asarray(reflection_map, dtype=np.float32)
    p = np.asarray(positive_map, dtype=np.float32)
    bottom_mean, bottom_coverage = bottom_roi_metrics(m, config.coverage_threshold, config.automotive_bottom_roi_fraction)
    mean_strength = nan_safe_mean(m)
    median_strength = nan_safe_median(m)
    p95_strength = nan_safe_percentile(m, 95.0)
    p99_strength = nan_safe_percentile(m, 99.0)
    max_strength = nan_safe_max(m)
    coverage = nan_safe_coverage(m, config.coverage_threshold)
    likelihood = mean_strength if mode == "no_reference" else None
    return ReflectionEvaluationResult(
        mode=mode,
        metric_version=REFLECTION_METRIC_VERSION,
        pair_id=pair_id,
        reflection_mean=mean_strength if mode == "reference" else None,
        reflection_median=median_strength if mode == "reference" else None,
        reflection_p95=p95_strength if mode == "reference" else None,
        reflection_p99=p99_strength if mode == "reference" else None,
        reflection_max=max_strength if mode == "reference" else None,
        reflection_coverage=coverage if mode == "reference" else None,
        reflection_likelihood=likelihood,
        no_reference_is_likelihood=no_reference_is_likelihood,
        mean_strength=mean_strength,
        median_strength=median_strength,
        p95_strength=p95_strength,
        p99_strength=p99_strength,
        max_strength=max_strength,
        positive_mean_strength=nan_safe_mean(p),
        positive_p95_strength=nan_safe_percentile(p, 95.0),
        coverage=coverage,
        coverage_threshold=config.coverage_threshold,
        saturation_threshold=config.saturation_threshold,
        glare_luminance_threshold=config.glare_luminance_threshold,
        glare_contrast_threshold=config.glare_contrast_threshold,
        severity_score=None if mode == "no_reference" else severity_from_metrics(mean_strength, p95_strength, coverage),
        saturation_coverage=saturation_coverage(normal_image, config.saturation_threshold, valid_mask=valid_mask),
        glare_coverage=glare_coverage(normal_luma, config, valid_mask=valid_mask),
        glare_strength=glare_strength(normal_luma, config, valid_mask=valid_mask),
        contrast_retention=contrast_retention_value,
        edge_retention=edge_retention_value,
        bottom_roi_mean_strength=bottom_mean,
        bottom_roi_coverage=bottom_coverage,
        regional_metrics=region_metrics(m, config.coverage_threshold),
        spatial_map=spatial_cells(m, config.spatial_rows, config.spatial_cols, config.coverage_threshold),
        alignment_score=alignment_score,
        alignment_error_px=alignment_error_px,
        alignment_status=alignment_status,
        alignment_method=alignment_method,
        alignment_translation_x_px=alignment_translation_x_px,
        alignment_translation_y_px=alignment_translation_y_px,
        alignment_rotation_deg=alignment_rotation_deg,
        alignment_scale=alignment_scale,
        alignment_shear_deg=alignment_shear_deg,
        photometric_normalized=photometric_normalized if mode == "reference" else False,
        photometric_gain=photometric_gain,
        photometric_bias=photometric_bias,
        heatmap_rows=24,
        heatmap_cols=32,
        downsampled_reflection_map=downsample_map(m, 24, 32),
        warning_message=warning_message,
        success=True,
    )


def _read_image(path: str | None) -> np.ndarray:
    if not path:
        raise ValueError("image path is required")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image
