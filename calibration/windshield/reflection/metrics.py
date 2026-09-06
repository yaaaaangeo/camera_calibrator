from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from calibration.windshield.reflection.types import (
    ReflectionEvaluationConfig,
    ReflectionRegionMetrics,
    ReflectionSpatialCell,
)


EPS = 1e-6


# ---------------------------------------------------------------------------
# Phase B-1 안정화 - Valid Warp Mask.
#
# Reference 이미지를 alignment warp한 뒤 생기는 인공적인 border 영역
# (BORDER_REFLECT 등으로 채워진, 실제 reference 데이터가 아닌 픽셀)이
# Reflection metric에 섞이면 안 된다. 이 모듈의 모든 통계 함수는 invalid
# 영역이 NaN으로 채워진 배열을 받는다는 전제로 nan-aware 집계
# (np.nanmean/nanmedian/nanpercentile)를 쓴다 - 0으로 채우면 mask 경계에
# 가짜 edge가 생겨 gradient 기반 지표(contrast/edge retention)를 오염시키고,
# NaN은 Sobel 등 convolution 연산에서 자연스럽게 주변으로 "전파"되어 mask
# 경계 바로 안쪽까지 보수적으로 제외해준다는 장점도 있다.
# ---------------------------------------------------------------------------

def apply_valid_mask(values: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """`valid_mask`가 False인 위치를 NaN으로 채운다. `valid_mask`가
    None이면(정렬을 안 했거나 No-Reference 모드) 그대로 반환한다."""
    if valid_mask is None:
        return values.astype(np.float32)
    out = values.astype(np.float32).copy()
    out[~valid_mask] = np.nan
    return out


def nan_safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    flat = values.reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return default
    return float(np.mean(valid))


def nan_safe_median(values: np.ndarray, default: float = 0.0) -> float:
    flat = values.reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return default
    return float(np.median(valid))


def nan_safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    flat = values.reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return default
    return float(np.percentile(valid, q))


def nan_safe_max(values: np.ndarray, default: float = 0.0) -> float:
    flat = values.reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return default
    return float(np.max(valid))


def nan_safe_coverage(values: np.ndarray, threshold: float, default: float = 0.0) -> float:
    """valid(비-NaN) 픽셀 중 threshold를 넘는 비율 - invalid 픽셀은
    분자/분모 어디에도 포함하지 않는다."""
    flat = values.reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return default
    return float(np.mean(valid > threshold))


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
    """Phase B-1 안정화 - `reference_luma`/`normal_luma`에 이미 NaN(invalid
    warp border)이 섞여 있을 수 있으므로, percentile/least-squares 계산
    전에 NaN 쌍을 제외한다 - 안 그러면 gain/bias 자체가 NaN이 되어 버린다."""
    ref_full = reference_luma.reshape(-1).astype(np.float32)
    norm_full = normal_luma.reshape(-1).astype(np.float32)
    finite = np.isfinite(ref_full) & np.isfinite(norm_full)
    ref = ref_full[finite]
    norm = norm_full[finite]
    if ref.size < 16:
        return 1.0, 0.0
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
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """`normal_luma`/`reference_luma`에 이미 NaN이 섞여 있으면(Phase B-1,
    `apply_valid_mask`로 warp border를 표시한 경우) 그 NaN이 산술 연산을
    통해 자연스럽게 결과(`absolute`/`positive`)까지 전파된다 - 별도 마스킹
    로직이 필요 없다."""
    if normal_luma.shape != reference_luma.shape:
        raise ValueError("normal/reference images must have the same resolution")
    gain, bias = (1.0, 0.0)
    aligned_ref = reference_luma.astype(np.float32)
    if photometric_normalize:
        gain, bias = robust_gain_bias(aligned_ref, normal_luma)
        aligned_ref = np.clip(gain * aligned_ref + bias, 0.0, 255.0).astype(np.float32)
    absolute = np.abs(normal_luma.astype(np.float32) - aligned_ref) / np.maximum(aligned_ref, 20.0)
    positive = np.maximum(normal_luma.astype(np.float32) - aligned_ref, 0.0) / np.maximum(aligned_ref, 20.0)
    return absolute, positive, gain, bias, aligned_ref


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
    """`normal_luma`/`reference_luma`에 NaN(invalid warp border, Phase
    B-1)이 섞여 있으면 gradient_magnitude(Sobel)를 통해 그 NaN이 mask
    경계 주변으로 조금 더 넓게 전파된다 - 의도된 보수적 동작이다(경계
    바로 안쪽의 gradient 계산도 실제로는 일부 invalid 픽셀을 커널에
    포함하기 때문)."""
    ref = nan_safe_mean(gradient_magnitude(reference_luma), default=0.0)
    if ref <= EPS:
        return 1.0
    return float(nan_safe_mean(gradient_magnitude(normal_luma), default=0.0) / (ref + EPS))


def edge_retention(normal_luma: np.ndarray, reference_luma: np.ndarray) -> float:
    ref_grad = gradient_magnitude(reference_luma)
    norm_grad = gradient_magnitude(normal_luma)
    threshold = nan_safe_percentile(ref_grad, 85.0, default=0.0)
    valid = ~np.isnan(ref_grad) & ~np.isnan(norm_grad)
    mask = valid & (ref_grad >= max(threshold, EPS))
    if int(np.count_nonzero(mask)) == 0:
        return 1.0
    ratios = norm_grad[mask] / (ref_grad[mask] + EPS)
    return float(np.clip(np.mean(ratios), 0.0, 2.0))


def saturation_coverage(image: np.ndarray, threshold: float, valid_mask: Optional[np.ndarray] = None) -> float:
    if image.ndim == 2:
        mask = image >= threshold
    else:
        mask = np.max(image[:, :, :3], axis=2) >= threshold
    if valid_mask is None:
        return float(np.mean(mask))
    if not np.any(valid_mask):
        return 0.0
    return float(np.mean(mask[valid_mask]))


def glare_coverage(luma: np.ndarray, config: ReflectionEvaluationConfig, valid_mask: Optional[np.ndarray] = None) -> float:
    contrast = local_contrast(luma)
    mask = (luma >= config.glare_luminance_threshold) & (contrast <= config.glare_contrast_threshold)
    if valid_mask is None:
        return float(np.mean(mask))
    if not np.any(valid_mask):
        return 0.0
    return float(np.mean(mask[valid_mask]))


def glare_strength(luma: np.ndarray, config: ReflectionEvaluationConfig, valid_mask: Optional[np.ndarray] = None) -> float:
    contrast = local_contrast(luma)
    mask = (luma >= config.glare_luminance_threshold) & (contrast <= config.glare_contrast_threshold)
    if valid_mask is not None:
        mask = mask & valid_mask
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
    """NaN(invalid warp border)이 섞인 `reflection_map`을 그대로
    `cv2.resize`에 넘기면 NaN이 인접 cell로 blur되어 퍼진다 - heatmap
    표시용 downsample이므로, invalid 픽셀을 0으로 바꾼 뒤 리사이즈한다
    (통계 계산에는 이 함수를 쓰지 않으므로 여기서만 0-fill이 허용된다)."""
    safe = np.nan_to_num(reflection_map.astype(np.float32), nan=0.0)
    small = cv2.resize(safe, (cols, rows), interpolation=cv2.INTER_AREA)
    return small.tolist()


def bottom_roi_metrics(reflection_map: np.ndarray, threshold: float, fraction: float) -> tuple[float, float]:
    h = reflection_map.shape[0]
    start = max(0, min(h - 1, int(round(h * (1.0 - fraction)))))
    roi = reflection_map[start:, :]
    return nan_safe_mean(roi, default=0.0), nan_safe_coverage(roi, threshold, default=0.0)


def _metrics_for_values(values: np.ndarray, threshold: float) -> ReflectionRegionMetrics:
    flat = values.reshape(-1).astype(np.float32)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return ReflectionRegionMetrics()
    return ReflectionRegionMetrics(
        mean_strength=float(np.mean(valid)),
        p95_strength=float(np.percentile(valid, 95.0)),
        coverage=float(np.mean(valid > threshold)),
    )
