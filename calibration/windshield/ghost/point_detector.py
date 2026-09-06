"""
calibration.windshield.ghost.point_detector
==============================================================

STEP 8A - Point-Source Ghost Detection.

Image -> Luminance -> bright source detection -> blob(peak) 검출 -> Main
peak과 Ghost peak(secondary local maximum) 분리(사용자 스펙 6-10번).
2번째로 밝은 pixel을 그냥 고르는 방식은 절대 쓰지 않는다 - Gaussian
smoothing -> connected-component blob -> intensity-weighted subpixel
centroid + integrated blob energy를 사용한다.

STEP 8 stabilization 1번 - Multi-LED array에서는 단순 nearest-neighbor
pairing이 촘촘하게 배치된 Main LED끼리를 잘못 짝짓는 문제(main-main
mispair)를 일으킬 수 있다. 이를 막기 위해 `pair_main_and_ghost_blobs()`는
먼저 전체 candidate (main,ghost) 쌍들의 displacement vector에서 dominant
cluster(전체 array가 공유하는 진짜 ghost displacement)를 robust하게
추정하고, 그 vector와 일치하는 후보를 순수 거리 기반보다 우선한다.
Candidate가 너무 적어(예: LED 1개) consensus를 신뢰할 수 없으면 기존
local nearest-neighbor(+ghost energy < main energy 제약) 방식으로
fallback한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from calibration.windshield.ghost.config import (
    DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MAX_SEARCH_RADIUS_PX,
    DEFAULT_MIN_BLOB_AREA_PX,
    DEFAULT_MIN_CONSENSUS_CANDIDATES,
    DEFAULT_MIN_PEAK_DISTANCE_PX,
    DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
)
from calibration.windshield.ghost.types import GhostPointDetection

_ENERGY_EPS = 1e-6


@dataclass
class BrightBlob:
    x: float
    y: float
    energy: float
    area_px: int


def to_luminance(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8/float 이미지를 float32 luminance(0-255 scale)로 변환한다."""
    if image_bgr.ndim == 2:
        lum = image_bgr.astype(np.float32)
    else:
        lum = cv2.cvtColor(image_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if image_bgr.dtype == np.uint8:
        return lum
    # float 입력이 [0,1] 범위라고 가정되면 0-255로 스케일한다(threshold와 일관성 유지).
    if lum.max() <= 1.0 + 1e-6:
        return lum * 255.0
    return lum


def detect_bright_blobs(
    image_bgr: np.ndarray,
    *,
    threshold: float = DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_BLOB_AREA_PX,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    min_peak_distance_px: float = DEFAULT_MIN_PEAK_DISTANCE_PX,
) -> list[BrightBlob]:
    """밝은 광원 후보 blob들을 검출한다.

    Gaussian smoothing(노이즈로 인한 가짜 local maxima 억제) -> threshold
    mask -> `cv2.connectedComponentsWithStats`(blob 단위 분리) -> **각
    connected component 안에서 local maxima를 찾아, main과 ghost가 서로
    가까워 하나의 component로 뭉친 경우에도 peak 단위로 분리**한다(2번째로
    밝은 pixel을 그냥 고르는 대신, local maxima -> peak별 nearest-neighbor
    영역에 대한 intensity-weighted subpixel centroid + integrated energy를
    사용 - 사용자 스펙 7-8번).
    """
    lum = to_luminance(image_bgr)
    if gaussian_sigma > 0:
        ksize = max(3, int(round(gaussian_sigma * 6)) | 1)
        smoothed = cv2.GaussianBlur(lum, (ksize, ksize), gaussian_sigma)
    else:
        smoothed = lum

    mask_bool = smoothed >= threshold
    mask = mask_bool.astype(np.uint8)
    if not np.any(mask):
        return []

    footprint_size = max(3, int(round(min_peak_distance_px * 2 + 1)))
    local_max_mask = (maximum_filter(smoothed, size=footprint_size) == smoothed) & mask_bool

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[BrightBlob] = []
    for label in range(1, num_labels):  # 0 = background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        region_mask = labels == label
        ys, xs = np.nonzero(region_mask)
        weights = smoothed[region_mask].astype(np.float64)

        peak_mask_in_region = local_max_mask & region_mask
        peak_ys, peak_xs = np.nonzero(peak_mask_in_region)

        if peak_ys.size <= 1:
            total_weight = float(weights.sum())
            if total_weight <= _ENERGY_EPS:
                continue
            cx = float(np.sum(xs * weights) / total_weight)
            cy = float(np.sum(ys * weights) / total_weight)
            blobs.append(BrightBlob(x=cx, y=cy, energy=total_weight, area_px=area))
            continue

        # 하나의 component 안에 여러 local maximum(=main+ghost가 겹쳐 하나로
        # 뭉친 경우) -> 각 pixel을 가장 가까운 peak에 배정해(Voronoi 분할)
        # peak별로 독립된 subpixel centroid/energy를 계산한다.
        peaks = np.stack([peak_xs.astype(np.float64), peak_ys.astype(np.float64)], axis=1)
        pixel_xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        dists = np.linalg.norm(pixel_xy[:, None, :] - peaks[None, :, :], axis=2)
        assignment = np.argmin(dists, axis=1)

        for peak_idx in range(peaks.shape[0]):
            sel = assignment == peak_idx
            if not np.any(sel):
                continue
            sub_weights = weights[sel]
            total_weight = float(sub_weights.sum())
            if total_weight <= _ENERGY_EPS:
                continue
            sub_xs = xs[sel]
            sub_ys = ys[sel]
            cx = float(np.sum(sub_xs * sub_weights) / total_weight)
            cy = float(np.sum(sub_ys * sub_weights) / total_weight)
            blobs.append(BrightBlob(x=cx, y=cy, energy=total_weight, area_px=int(sel.sum())))

    return blobs


@dataclass
class _CandidatePair:
    main_idx: int
    ghost_idx: int
    dx: float
    dy: float
    distance: float


def _generate_candidate_pairs(blobs: list[BrightBlob], max_search_radius_px: float) -> list[_CandidatePair]:
    """모든 plausible (main,ghost) 후보를 만든다 - "ghost는 main보다 어둡다"
    (사용자 스펙 1-A번 "secondary energy < main energy")와
    `max_search_radius_px` 제약만 적용한, 방향성 있는(ordered) 쌍이다."""
    candidates: list[_CandidatePair] = []
    for i, main in enumerate(blobs):
        for j, ghost in enumerate(blobs):
            if i == j or ghost.energy >= main.energy:
                continue
            dx = ghost.x - main.x
            dy = ghost.y - main.y
            dist = math.hypot(dx, dy)
            if dist <= max_search_radius_px:
                candidates.append(_CandidatePair(main_idx=i, ghost_idx=j, dx=dx, dy=dy, distance=dist))
    return candidates


def estimate_dominant_ghost_vector(
    candidates: list[_CandidatePair],
    *,
    consensus_radius_px: float = DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
    min_consensus_candidates: int = DEFAULT_MIN_CONSENSUS_CANDIDATES,
) -> Optional[tuple[float, float, int]]:
    """전체 candidate displacement vector 중 dominant cluster를 찾는다
    (사용자 스펙 1-A번, "Option A - robust median + inlier"). scikit-learn
    같은 새 dependency 없이 직접 구현한다.

    절차: 각 candidate vector를 중심으로 `consensus_radius_px` 안에 몇 개의
    다른 candidate가 있는지 세어(=support) 가장 support가 큰 vector를
    초기 추정값으로 삼고, 그 반경 안의 inlier들로 median을 다시 계산해
    최종 dominant vector를 정한다.

    Candidate 수가 `min_consensus_candidates` 미만이거나 최종 inlier 수가
    그 기준에 못 미치면 `None`을 반환한다 - 호출자는 이 경우 global
    consensus 없이 local nearest-neighbor로 fallback해야 한다(사용자 스펙
    1-E번).
    """
    if len(candidates) < min_consensus_candidates:
        return None

    vectors = np.array([[c.dx, c.dy] for c in candidates], dtype=np.float64)
    best_support = -1
    best_center = vectors[0]
    for v in vectors:
        support = int(np.sum(np.linalg.norm(vectors - v, axis=1) <= consensus_radius_px))
        if support > best_support:
            best_support = support
            best_center = v

    dists = np.linalg.norm(vectors - best_center, axis=1)
    inliers = vectors[dists <= consensus_radius_px]
    if inliers.shape[0] < min_consensus_candidates:
        return None

    refined = np.median(inliers, axis=0)
    return float(refined[0]), float(refined[1]), int(inliers.shape[0])


def pair_main_and_ghost_blobs(
    blobs: list[BrightBlob],
    *,
    max_search_radius_px: float = DEFAULT_MAX_SEARCH_RADIUS_PX,
    consensus_radius_px: float = DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
    min_consensus_candidates: int = DEFAULT_MIN_CONSENSUS_CANDIDATES,
) -> list[GhostPointDetection]:
    """검출된 blob들을 밝기(energy) 내림차순으로 소비하며 Main/Ghost를
    짝짓는다.

    STEP 8 stabilization 1번 - 순수 거리 기반 nearest-neighbor 대신, 먼저
    전체 candidate에서 dominant ghost displacement vector를 추정하고
    (`estimate_dominant_ghost_vector`), 각 Main에 대해

        score = vector_error(candidate, dominant) + 0.1 * distance

    가 가장 작은 후보를 Ghost로 선택한다 - 촘촘한 LED array에서 이웃
    Main이 실제 Ghost보다 더 가깝더라도(순수 거리로는 이웃 Main이 이기는
    경우) displacement vector가 dominant와 맞지 않으면 걸러진다. Dominant
    vector를 신뢰할 수 없으면(candidate 부족) 기존 순수 거리 기반
    nearest-neighbor로 fallback한다(사용자 스펙 1-E번, 1-LED 케이스가
    여기 해당).

    반경 안에 후보가 없으면 그 Main은 ghost 없음(`detected=False`)으로
    남는다. 이미 다른 Main의 Ghost로 소비된 blob은 재사용하지 않는다
    (사용자 스펙 1-D번).
    """
    candidates = _generate_candidate_pairs(blobs, max_search_radius_px)
    dominant = estimate_dominant_ghost_vector(
        candidates, consensus_radius_px=consensus_radius_px, min_consensus_candidates=min_consensus_candidates,
    )

    # Main별 candidate lookup - O(n^2) 재계산을 피한다.
    candidates_by_main: dict[int, list[_CandidatePair]] = {}
    for c in candidates:
        candidates_by_main.setdefault(c.main_idx, []).append(c)

    order = sorted(range(len(blobs)), key=lambda i: blobs[i].energy, reverse=True)
    consumed = [False] * len(blobs)
    detections: list[GhostPointDetection] = []

    for i in order:
        if consumed[i]:
            continue
        consumed[i] = True
        main = blobs[i]

        best_j = -1
        best_score = None
        best_residual: Optional[float] = None
        for c in candidates_by_main.get(i, []):
            if consumed[c.ghost_idx]:
                continue
            if dominant is not None:
                ddx, ddy, _support = dominant
                vector_error = math.hypot(c.dx - ddx, c.dy - ddy)
                score = vector_error + 0.1 * c.distance
            else:
                vector_error = None
                score = c.distance
            if best_score is None or score < best_score:
                best_score = score
                best_j = c.ghost_idx
                best_residual = vector_error

        if best_j < 0:
            detections.append(GhostPointDetection(main_x=main.x, main_y=main.y, detected=False))
            continue

        consumed[best_j] = True
        ghost = blobs[best_j]
        offset_x = ghost.x - main.x
        offset_y = ghost.y - main.y
        distance = float(np.hypot(offset_x, offset_y))
        strength_ratio = float(ghost.energy / (main.energy + _ENERGY_EPS))
        detections.append(
            GhostPointDetection(
                main_x=main.x,
                main_y=main.y,
                ghost_x=ghost.x,
                ghost_y=ghost.y,
                offset_x_px=offset_x,
                offset_y_px=offset_y,
                distance_px=distance,
                strength_ratio=strength_ratio,
                detected=True,
                pair_residual_px=best_residual,
            )
        )

    return detections
