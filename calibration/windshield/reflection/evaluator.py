from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from calibration.windshield.reflection.alignment import align_reference_to_normal
from calibration.windshield.reflection.metrics import (
    bottom_roi_metrics,
    contrast_retention,
    downsample_map,
    edge_retention,
    glare_coverage,
    glare_strength,
    gradient_magnitude,
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
        enabled=cfg.align,
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

    reflection_map, positive_map, gain, bias = normalize_reflection_map(
        normal_luma,
        alignment.aligned_reference,
        photometric_normalize=cfg.photometric_normalize,
    )
    return _result_from_map(
        reflection_map,
        positive_map,
        normal_image,
        normal_luma,
        alignment.aligned_reference,
        cfg,
        mode="reference",
        pair_id=pair_id,
        alignment_score=alignment.score,
        alignment_error_px=alignment.error_px,
        alignment_status=alignment.status,
        alignment_method=alignment.method,
        photometric_gain=gain,
        photometric_bias=bias,
        contrast_retention_value=contrast_retention(normal_luma, alignment.aligned_reference),
        edge_retention_value=edge_retention(normal_luma, alignment.aligned_reference),
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
    results = [evaluate_reflection_pair(pair, cfg) for pair in pair_list]
    successful = [r for r in results if r.success]
    if not successful:
        return ReflectionDatasetResult(
            mode=cfg.mode,
            pair_results=results,
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
    return ReflectionDatasetResult(
        mode=cfg.mode,
        metric_version=REFLECTION_METRIC_VERSION,
        pair_results=results,
        mean_strength=float(np.mean(means)),
        median_strength=float(np.median(means)),
        p95_strength=float(np.percentile(means, 95.0)),
        worst_pair_id=worst.pair_id,
        coverage=float(np.mean(coverages)),
        severity_score=severity_from_metrics(float(np.mean(means)), float(np.mean([r.p95_strength for r in successful])), float(np.mean(coverages))),
        by_day_night=grouped,
        success=True,
        warning_message="Some pairs were invalid and excluded." if len(successful) != len(results) else None,
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
    no_reference_is_likelihood: bool = False,
    photometric_gain: float | None = None,
    photometric_bias: float | None = None,
    warning_message: str | None = None,
    contrast_retention_value: float | None = None,
    edge_retention_value: float | None = None,
) -> ReflectionEvaluationResult:
    m = np.asarray(reflection_map, dtype=np.float32)
    p = np.asarray(positive_map, dtype=np.float32)
    bottom_mean, bottom_coverage = bottom_roi_metrics(m, config.coverage_threshold, config.automotive_bottom_roi_fraction)
    mean_strength = float(np.mean(m))
    median_strength = float(np.median(m))
    p95_strength = float(np.percentile(m, 95.0))
    p99_strength = float(np.percentile(m, 99.0))
    max_strength = float(np.max(m))
    coverage = float(np.mean(m > config.coverage_threshold))
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
        positive_mean_strength=float(np.mean(p)),
        positive_p95_strength=float(np.percentile(p, 95.0)),
        coverage=coverage,
        coverage_threshold=config.coverage_threshold,
        saturation_threshold=config.saturation_threshold,
        glare_luminance_threshold=config.glare_luminance_threshold,
        glare_contrast_threshold=config.glare_contrast_threshold,
        severity_score=severity_from_metrics(mean_strength, p95_strength, coverage),
        saturation_coverage=saturation_coverage(normal_image, config.saturation_threshold),
        glare_coverage=glare_coverage(normal_luma, config),
        glare_strength=glare_strength(normal_luma, config),
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
        photometric_normalized=config.photometric_normalize if mode == "reference" else False,
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
