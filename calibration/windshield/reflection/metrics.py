from __future__ import annotations

import cv2
import numpy as np

from calibration.windshield.reflection.types import (
    ReflectionEvaluationConfig,
    ReflectionRegionMetrics,
    ReflectionSpatialCell,
)


EPS = 1e-6


def to_luminance(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected grayscale/BGR/BGRA image, got shape {image.shape}")
    bgr = image[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32)


def robust_gain_bias(reference_luma: np.ndarray, normal_luma: np.ndarray) -> tuple[float, float]:
    ref = reference_luma.reshape(-1).astype(np.float32)
    norm = normal_luma.reshape(-1).astype(np.float32)
    diff = norm - ref
    lo, hi = np.percentile(diff, [10.0, 90.0])
    mask = (diff >= lo) & (diff <= hi)
    if int(np.count_nonzero(mask)) < 16:
        mask = np.ones_like(diff, dtype=bool)
    x = ref[mask].astype(np.float64)
    y = norm[mask].astype(np.float64)
    a = np.vstack([x, np.ones_like(x)]).T
    gain, bias = np.linalg.lstsq(a, y, rcond=None)[0]
    if not np.isfinite(gain) or not np.isfinite(bias) or gain <= 0:
        return 1.0, 0.0
    return float(gain), float(bias)


def normalize_reflection_map(
    normal_luma: np.ndarray,
    reference_luma: np.ndarray,
    *,
    photometric_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if normal_luma.shape != reference_luma.shape:
        raise ValueError("normal/reference images must have the same resolution")
    gain, bias = (1.0, 0.0)
    aligned_ref = reference_luma.astype(np.float32)
    if photometric_normalize:
        gain, bias = robust_gain_bias(aligned_ref, normal_luma)
        aligned_ref = np.clip(gain * aligned_ref + bias, 0.0, 255.0).astype(np.float32)
    absolute = np.abs(normal_luma.astype(np.float32) - aligned_ref) / np.maximum(aligned_ref, 20.0)
    positive = np.maximum(normal_luma.astype(np.float32) - aligned_ref, 0.0) / np.maximum(aligned_ref, 20.0)
    return absolute, positive, gain, bias


def local_contrast(luma: np.ndarray, ksize: int = 9) -> np.ndarray:
    luma = luma.astype(np.float32)
    mean = cv2.blur(luma, (ksize, ksize))
    mean_sq = cv2.blur(luma * luma, (ksize, ksize))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def gradient_magnitude(luma: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(luma.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def contrast_retention(normal_luma: np.ndarray, reference_luma: np.ndarray) -> float:
    ref = float(np.mean(gradient_magnitude(reference_luma)))
    if ref <= EPS:
        return 1.0
    return float(np.mean(gradient_magnitude(normal_luma)) / (ref + EPS))


def edge_retention(normal_luma: np.ndarray, reference_luma: np.ndarray) -> float:
    ref_grad = gradient_magnitude(reference_luma)
    norm_grad = gradient_magnitude(normal_luma)
    threshold = float(np.percentile(ref_grad, 85.0))
    mask = ref_grad >= max(threshold, EPS)
    if int(np.count_nonzero(mask)) == 0:
        return 1.0
    ratios = norm_grad[mask] / (ref_grad[mask] + EPS)
    return float(np.clip(np.mean(ratios), 0.0, 2.0))


def saturation_coverage(image: np.ndarray, threshold: float) -> float:
    if image.ndim == 2:
        mask = image >= threshold
    else:
        mask = np.max(image[:, :, :3], axis=2) >= threshold
    return float(np.mean(mask))


def glare_coverage(luma: np.ndarray, config: ReflectionEvaluationConfig) -> float:
    contrast = local_contrast(luma)
    mask = (luma >= config.glare_luminance_threshold) & (contrast <= config.glare_contrast_threshold)
    return float(np.mean(mask))


def glare_strength(luma: np.ndarray, config: ReflectionEvaluationConfig) -> float:
    contrast = local_contrast(luma)
    mask = (luma >= config.glare_luminance_threshold) & (contrast <= config.glare_contrast_threshold)
    if int(np.count_nonzero(mask)) == 0:
        return 0.0
    excess = (luma[mask].astype(np.float32) - config.glare_luminance_threshold) / max(255.0 - config.glare_luminance_threshold, EPS)
    return float(np.mean(np.clip(excess, 0.0, 1.0)))


def severity_from_metrics(mean_strength: float, p95_strength: float, coverage: float) -> float:
    raw = 100.0 * (0.35 * mean_strength / 0.20 + 0.40 * p95_strength / 0.50 + 0.25 * coverage / 0.30)
    return float(np.clip(raw, 0.0, 100.0))


def region_metrics(reflection_map: np.ndarray, threshold: float) -> dict[str, ReflectionRegionMetrics]:
    h, w = reflection_map.shape
    regions = {
        "center": reflection_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
        "top": reflection_map[: h // 4, :],
        "bottom": reflection_map[3 * h // 4 :, :],
        "left": reflection_map[:, : w // 4],
        "right": reflection_map[:, 3 * w // 4 :],
    }
    corner_masks = [
        reflection_map[: h // 4, : w // 4],
        reflection_map[: h // 4, 3 * w // 4 :],
        reflection_map[3 * h // 4 :, : w // 4],
        reflection_map[3 * h // 4 :, 3 * w // 4 :],
    ]
    regions["corners"] = np.concatenate([m.reshape(-1) for m in corner_masks])
    return {name: _metrics_for_values(np.asarray(values), threshold) for name, values in regions.items()}


def spatial_cells(reflection_map: np.ndarray, rows: int, cols: int, threshold: float) -> list[ReflectionSpatialCell]:
    h, w = reflection_map.shape
    cells: list[ReflectionSpatialCell] = []
    for r in range(rows):
        y0, y1 = int(round(r * h / rows)), int(round((r + 1) * h / rows))
        for c in range(cols):
            x0, x1 = int(round(c * w / cols)), int(round((c + 1) * w / cols))
            m = _metrics_for_values(reflection_map[y0:y1, x0:x1], threshold)
            cells.append(ReflectionSpatialCell(r, c, m.mean_strength, m.p95_strength, m.coverage))
    return cells


def downsample_map(reflection_map: np.ndarray, rows: int = 24, cols: int = 32) -> list[list[float]]:
    small = cv2.resize(reflection_map.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    return small.tolist()


def bottom_roi_metrics(reflection_map: np.ndarray, threshold: float, fraction: float) -> tuple[float, float]:
    h = reflection_map.shape[0]
    start = max(0, min(h - 1, int(round(h * (1.0 - fraction)))))
    roi = reflection_map[start:, :]
    return float(np.mean(roi)), float(np.mean(roi > threshold))


def _metrics_for_values(values: np.ndarray, threshold: float) -> ReflectionRegionMetrics:
    flat = values.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return ReflectionRegionMetrics()
    return ReflectionRegionMetrics(
        mean_strength=float(np.mean(flat)),
        p95_strength=float(np.percentile(flat, 95.0)),
        coverage=float(np.mean(flat > threshold)),
    )
