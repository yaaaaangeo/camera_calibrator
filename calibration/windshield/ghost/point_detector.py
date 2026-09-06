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
    DEFAULT_TWO_PASS_GRID_COLS,
    DEFAULT_TWO_PASS_GRID_ROWS,
    DEFAULT_TWO_PASS_MIN_CELL_SAMPLES,
    PAIR_SCORE_WEIGHT_DISTANCE,
    PAIR_SCORE_WEIGHT_ENERGY,
    PAIR_SCORE_WEIGHT_VECTOR,
)
from calibration.windshield.ghost.spatial_model import build_robust_spatial_map_from_detections
from calibration.windshield.ghost.types import GhostPointDetection, GhostSpatialCell

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
    energy_ratio: float


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
                energy_ratio = ghost.energy / max(main.energy, _ENERGY_EPS)
                candidates.append(
                    _CandidatePair(main_idx=i, ghost_idx=j, dx=dx, dy=dy, distance=dist, energy_ratio=energy_ratio)
                )
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


def estimate_dominant_energy_ratio(
    candidates: list[_CandidatePair],
    dominant_vector: Optional[tuple[float, float, int]],
    *,
    consensus_radius_px: float = DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
) -> Optional[float]:
    """Displacement consensus에 쓰인 것과 **같은 inlier 집합**에서 robust
    median energy ratio를 구한다(사용자 스펙 5-B번, "Dominant displacement
    cluster의 inlier pair들로 median energy ratio를 구한다") - 별도의
    독립적인 clustering을 하지 않는다. `dominant_vector`가 없으면(=
    displacement consensus 자체가 없으면) 에너지 비율도 신뢰할 수 없으므로
    `None`을 반환한다(사용자 스펙 5-E번, "consensus가 있을 때만 사용")."""
    if dominant_vector is None:
        return None
    ddx, ddy, _support = dominant_vector
    inliers = [c for c in candidates if math.hypot(c.dx - ddx, c.dy - ddy) <= consensus_radius_px]
    if not inliers:
        return None
    return float(np.median([c.energy_ratio for c in inliers]))


def _pick_best_ghost_for_main(
    candidates_for_main: list[_CandidatePair],
    consumed: list[bool],
    *,
    dominant_vector: Optional[tuple[float, float]],
    dominant_energy_ratio: Optional[float],
    consensus_radius_px: float,
    max_search_radius_px: float,
) -> tuple[int, Optional[float], Optional[float]]:
    """`pair_main_and_ghost_blobs()`/2-pass 버전이 공유하는 단일 main에
    대한 best-ghost 선택 로직(사용자 스펙 5-C/5-D/5-E번 3-항 score). PASS
    1(단일 global vector)과 PASS 2(main별 local vector)의 유일한 차이는
    호출자가 넘기는 `dominant_vector`/`dominant_energy_ratio`뿐이다 - 점수
    산식 자체는 완전히 동일하게 유지해야 두 pass의 결과가 일관적으로
    비교 가능하다."""
    best_j = -1
    best_score: Optional[float] = None
    best_residual: Optional[float] = None
    best_energy_residual: Optional[float] = None
    for c in candidates_for_main:
        if consumed[c.ghost_idx]:
            continue
        if dominant_vector is not None:
            ddx, ddy = dominant_vector
            vector_error = math.hypot(c.dx - ddx, c.dy - ddy)
            vector_term = vector_error / max(consensus_radius_px, _ENERGY_EPS)
            distance_term = c.distance / max(max_search_radius_px, _ENERGY_EPS)
            if dominant_energy_ratio is not None:
                energy_residual = abs(c.energy_ratio - dominant_energy_ratio)
                energy_term = energy_residual
            else:
                energy_residual = None
                energy_term = 0.0
            score = (
                PAIR_SCORE_WEIGHT_VECTOR * vector_term
                + PAIR_SCORE_WEIGHT_DISTANCE * distance_term
                + PAIR_SCORE_WEIGHT_ENERGY * energy_term
            )
        else:
            vector_error = None
            energy_residual = None
            score = c.distance
        if best_score is None or score < best_score:
            best_score = score
            best_j = c.ghost_idx
            best_residual = vector_error
            best_energy_residual = energy_residual
    return best_j, best_residual, best_energy_residual


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
    (`estimate_dominant_ghost_vector`), 각 Main에 대해 정규화된 3-항
    score(사용자 스펙 5-C번)가 가장 작은 후보를 Ghost로 선택한다:

        vector_term   = vector_error / consensus_radius_px
        distance_term = distance / max_search_radius_px
        energy_term   = |candidate.energy_ratio - dominant_energy_ratio|
        score = W_VECTOR*vector_term + W_DISTANCE*distance_term + W_ENERGY*energy_term

    세 항은 단위가 달라(px, px, ratio) 그대로 더치지 않고 각각
    `consensus_radius_px`/`max_search_radius_px` 기준으로 정규화한 뒤
    더한다. 우선순위는 displacement > energy > 단순 거리이며(사용자 스펙
    5-D번), 기본 weight(`PAIR_SCORE_WEIGHT_VECTOR=1.0` >
    `PAIR_SCORE_WEIGHT_ENERGY=0.5` > `PAIR_SCORE_WEIGHT_DISTANCE=0.15`)가
    이를 반영한다. 촘촘한 LED array에서 이웃 Main이 실제 Ghost보다
    거리상으로는 더 가깝거나 energy gate(ghost<main)를 우연히 만족하더라도,
    displacement/energy-ratio가 dominant와 맞지 않으면 걸러진다.

    Energy-ratio 항은 displacement consensus가 있을 때만 사용한다(사용자
    스펙 5-E번) - dominant vector를 신뢰할 수 없으면(candidate 부족)
    energy-ratio도 신뢰할 근거가 없으므로, score는 순수 거리로 fallback한다
    (사용자 스펙 1-E번, 1-LED 케이스가 여기 해당).

    반경 안에 후보가 없으면 그 Main은 ghost 없음(`detected=False`)으로
    남는다. 이미 다른 Main의 Ghost로 소비된 blob은 재사용하지 않는다
    (사용자 스펙 1-D번).
    """
    candidates = _generate_candidate_pairs(blobs, max_search_radius_px)
    dominant = estimate_dominant_ghost_vector(
        candidates, consensus_radius_px=consensus_radius_px, min_consensus_candidates=min_consensus_candidates,
    )
    dominant_energy_ratio = estimate_dominant_energy_ratio(
        candidates, dominant, consensus_radius_px=consensus_radius_px,
    )

    # Main별 candidate lookup - O(n^2) 재계산을 피한다.
    candidates_by_main: dict[int, list[_CandidatePair]] = {}
    for c in candidates:
        candidates_by_main.setdefault(c.main_idx, []).append(c)

    order = sorted(range(len(blobs)), key=lambda i: blobs[i].energy, reverse=True)
    consumed = [False] * len(blobs)
    detections: list[GhostPointDetection] = []

    dominant_vector_only = None if dominant is None else (dominant[0], dominant[1])

    for i in order:
        if consumed[i]:
            continue
        consumed[i] = True
        main = blobs[i]

        best_j, best_residual, best_energy_residual = _pick_best_ghost_for_main(
            candidates_by_main.get(i, []),
            consumed,
            dominant_vector=dominant_vector_only,
            dominant_energy_ratio=dominant_energy_ratio,
            consensus_radius_px=consensus_radius_px,
            max_search_radius_px=max_search_radius_px,
        )

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
                pair_energy_ratio=strength_ratio,
                pair_energy_residual=best_energy_residual,
            )
        )

    return detections


def _cell_indices_for_point(x: float, y: float, *, image_width: float, image_height: float, rows: int, cols: int) -> tuple[int, int]:
    """`spatial_model.py`의 grid bucketing과 동일한 규칙(사용자 spec과의
    일관성을 위해 두 곳에서 같은 공식을 씀)으로 (row,col)을 계산한다."""
    col = int(x / max(image_width, 1e-9) * cols)
    row = int(y / max(image_height, 1e-9) * rows)
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)
    return row, col


def pair_main_and_ghost_blobs_two_pass(
    blobs: list[BrightBlob],
    *,
    image_width: float,
    image_height: float,
    max_search_radius_px: float = DEFAULT_MAX_SEARCH_RADIUS_PX,
    consensus_radius_px: float = DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
    min_consensus_candidates: int = DEFAULT_MIN_CONSENSUS_CANDIDATES,
    coarse_grid_rows: int = DEFAULT_TWO_PASS_GRID_ROWS,
    coarse_grid_cols: int = DEFAULT_TWO_PASS_GRID_COLS,
    min_cell_samples_for_local: int = DEFAULT_TWO_PASS_MIN_CELL_SAMPLES,
) -> list[GhostPointDetection]:
    """Phase B-4 - 2-pass spatial pairing(사용자 스펙 B-4번).

    PASS 1: 기존 `pair_main_and_ghost_blobs()`(단일 global dominant
    vector)을 그대로 실행해 coarse 결과를 얻는다 - 이 결과 자체가 이미
    "detected=True인 pairing"으로 유효하다.

    PASS 2: PASS 1의 detection들을 `coarse_grid_rows x coarse_grid_cols`
    grid로 버켓팅(`build_robust_spatial_map_from_detections` 재사용 -
    median/MAD 기반, 새 통계 로직을 만들지 않음)해 "성긴 GhostField"를
    만들고, 각 Main 위치가 속한 cell의 median offset/energy-ratio를 그
    Main 전용 local dominant vector로 써서 **다시** pairing한다(점수 산식
    자체는 PASS 1과 동일한 `_pick_best_ghost_for_main` 공유 - 유일한 차이는
    "global 하나짜리 vector"냐 "main별 local vector"냐 뿐이다). 이렇게 하면
    windshield 곡률 등으로 위치마다 실제 ghost displacement가 달라지는
    경우에도(PASS 1의 단일 global vector로는 못 잡음) 지역적으로 더 정확한
    pairing이 가능하다.

    Local vector 신뢰 기준(사용자 스펙: "PASS 2는 PASS 1보다 세밀하게, 단
    항상 안전하게"): 어떤 cell의 PASS 1 detection 수가
    `min_cell_samples_for_local` 미만이면 그 cell의 local vector를 믿지
    않고 PASS 1의 global dominant vector로 fallback한다(빈 cell을 억지로
    extrapolate하지 않음 - `fill_empty_spatial_cells()`와 같은 보수적
    원칙).

    안전장치(사용자 스펙 "PASS 2 실패 시 PASS 1로 안전하게 fallback"):
    PASS 1 자체에 global consensus가 없거나(=candidate 부족, 이미
    거리 기반 fallback으로 pairing됨), coarse map을 만들 수 없거나, PASS 2
    계산 중 예외가 발생하면 PASS 1 결과를 그대로 반환한다 - 절대 예외를
    호출자에게 전파하지 않는다.
    """
    pass1_detections = pair_main_and_ghost_blobs(
        blobs,
        max_search_radius_px=max_search_radius_px,
        consensus_radius_px=consensus_radius_px,
        min_consensus_candidates=min_consensus_candidates,
    )
    detected_pass1 = [d for d in pass1_detections if d.detected]
    if len(detected_pass1) < min_consensus_candidates:
        # PASS 1 자체가 이미 global consensus 없이(순수 거리 기반) 짝지어진
        # 상태다 - 이런 소량의 detection으로 coarse spatial field를 만드는
        # 것은 근거가 없으므로 PASS 1을 그대로 최종 결과로 쓴다.
        return pass1_detections

    try:
        candidates = _generate_candidate_pairs(blobs, max_search_radius_px)
        global_dominant = estimate_dominant_ghost_vector(
            candidates, consensus_radius_px=consensus_radius_px, min_consensus_candidates=min_consensus_candidates,
        )
        if global_dominant is None:
            return pass1_detections
        global_dominant_vector = (global_dominant[0], global_dominant[1])
        global_dominant_energy_ratio = estimate_dominant_energy_ratio(
            candidates, global_dominant, consensus_radius_px=consensus_radius_px,
        )

        coarse_cells = build_robust_spatial_map_from_detections(
            detected_pass1,
            image_width=image_width,
            image_height=image_height,
            rows=coarse_grid_rows,
            cols=coarse_grid_cols,
        )
        cells_by_rc: dict[tuple[int, int], GhostSpatialCell] = {(c.row, c.col): c for c in coarse_cells}

        def _local_vector_and_ratio(main: BrightBlob) -> tuple[tuple[float, float], Optional[float]]:
            row, col = _cell_indices_for_point(
                main.x, main.y, image_width=image_width, image_height=image_height,
                rows=coarse_grid_rows, cols=coarse_grid_cols,
            )
            cell = cells_by_rc.get((row, col))
            if (
                cell is not None
                and cell.sample_count >= min_cell_samples_for_local
                and cell.mean_offset_x_px is not None
                and cell.mean_offset_y_px is not None
            ):
                ratio = cell.mean_strength_ratio if cell.mean_strength_ratio is not None else global_dominant_energy_ratio
                return (cell.mean_offset_x_px, cell.mean_offset_y_px), ratio
            return global_dominant_vector, global_dominant_energy_ratio

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
            local_vector, local_energy_ratio = _local_vector_and_ratio(main)

            best_j, best_residual, best_energy_residual = _pick_best_ghost_for_main(
                candidates_by_main.get(i, []),
                consumed,
                dominant_vector=local_vector,
                dominant_energy_ratio=local_energy_ratio,
                consensus_radius_px=consensus_radius_px,
                max_search_radius_px=max_search_radius_px,
            )

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
                    pair_energy_ratio=strength_ratio,
                    pair_energy_residual=best_energy_residual,
                )
            )

        return detections
    except Exception:
        # PASS 2는 어디까지나 PASS 1의 개선 시도일 뿐이다 - 어떤 이유로든
        # 실패하면 이미 유효한 PASS 1 결과로 안전하게 fallback한다(사용자
        # 스펙 B-4번, "must safely fall back to PASS 1 on failure").
        return pass1_detections
