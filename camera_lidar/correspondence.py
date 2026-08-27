"""
camera_calibrator.camera_lidar.correspondence
=================================================

Resolve LiDAR<->camera circle-center correspondence, including PARTIAL
(3-of-4) scenes.

Camera-side centers (camera_detector.detect_camera_target) are already in
a fixed semantic order (target_config.CORNER_ORDER: top_left, top_right,
bottom_right, bottom_left) established by the ArUco marker IDs, which is
itself a cyclic traversal of the rectangle's boundary. Only the subset in
CameraDetectionResult.detected_ids is trustworthy for correspondence --
see camera_detector.py's module-level note on why the OTHER (pose-inferred)
circle centers must not be used as if they were independently observed.

LiDAR-side centers (lidar_detector.detect_lidar_target) are only known up
to an arbitrary starting point and traversal direction around that same
cycle -- LiDAR geometry alone has no marker-ID equivalent to say "this one
is top-left".

Generalized correspondence (match_partial_centers): CORNER_ORDER is a
4-cycle, so removing any single corner always leaves the other 3 as one
contiguous arc of that cycle -- "which corner is missing" only ever has 4
possible hypotheses per side, not a combinatorial explosion. This searches
every length-L contiguous window of CORNER_ORDER that the camera actually
trusts (L = min(camera-trusted count, lidar-detected count)) against every
length-L contiguous window of the LiDAR's own cyclic (but unlabeled)
sequence -- for L=4 there is exactly one camera window (all 4 IDs) and 8
lidar candidates (4 rotations x 2 directions), which is exactly the
original full-scene search; for L=3 there are up to 4 camera windows (one
per possible missing corner) x 6 lidar candidates (3 rotations x 2
directions) each. Correspondence is verified by how well the resulting
rigid transform actually fits the data (lowest residual wins), not by a
geometric guess -- and never by detection order.

IMPORTANT, hard-mathematical-fact limitation, for BOTH the L=4 (FULL) and
L=3 (PARTIAL) case: a rectangle's symmetry group has 4 elements (identity,
180-degree in-plane rotation, and 2 mirror reflections -- a Klein four-
group), and because all 4 circle centers are coplanar, Kabsch/SVD's proper-
rotation correction can achieve exact zero residual for EVERY one of those
4 symmetry images, not just the true correspondence -- verified empirically
across many unrelated rotations and all 8 possible raw-detection orderings,
not a coincidence of one test. For L=3, additionally, removing any ONE
corner from a rectangle always leaves a triangle with the SAME three side
lengths (width, height, diagonal) regardless of which corner was removed,
so "which corner is missing" is *also* unresolvable by residual alone
whenever one side offers more than L points. Noise does not break any of
these ties, because they are *structural* congruences, not coincidences of
a particular measurement.

This is not fixable with a cleverer residual-based search -- it is an
information-theoretic fact about rectangular targets: point positions
alone cannot tell "this way up" from "rotated 180 degrees in-plane", nor
"this side left" from "mirrored". match_partial_centers breaks tied
candidates using, in priority order:
  1. `reference_transform` (typically the current best FULL-scene-only
     calibration estimate, once one exists) -- picks whichever candidate's
     recovered pose is closest to it.
  2. The "target held upright, not mirrored" assumption: the LiDAR's OWN
     points (before any transform) for canonical top_* corners should have
     a HIGHER Z than bottom_* corners (REP-103 up convention, already this
     project's convention -- see geometry/transform.py), AND right_* corners
     should have a higher Y than left_* corners. Both checks are needed --
     the rectangle's symmetry group has 2 independent axes of ambiguity
     (top/bottom flip and left/right flip), so a single-axis check leaves
     half the ambiguity unresolved (found the hard way: an earlier version
     of this fallback checked Z alone and still silently returned a wrong,
     mirrored correspondence about 50% of the time -- caught by testing all
     8 raw orderings, not just one). This depends only on how the physical
     board is held relative to the LiDAR's own axes -- NOT on the (unknown)
     camera-lidar extrinsic being solved for, unlike a naive "prefer the
     rotation closest to identity" guess (tried first and empirically found
     unreliable for realistic rig rotations, since it circularly depends on
     the very unknown it's trying to help solve).
Whenever tie-break #2 was needed (no reference available),
`CorrespondenceResult.ambiguous = True` is still set, since it rests on an
assumption (upright, non-mirrored target) rather than measured data --
callers (the Quality Gate, in particular) can choose to warn or refuse to
trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from camera_lidar.solver import compute_residuals, solve_rigid_transform
from camera_lidar.target_config import CORNER_ORDER
from geometry.transform import rotation_geodesic_distance

_MIN_CORRESPONDENCE_POINTS = 3
# How generously to treat two candidates' residuals as "tied" -- congruent
# rectangle-corner-triangle hypotheses tie almost exactly (see module
# docstring), so this needs to be generous enough to catch all of them,
# not just near-floating-point-identical residuals.
_TIE_RELATIVE_FACTOR = 2.0
_TIE_ABSOLUTE_FLOOR_M = 0.0005
# Two tied candidates are considered the "same" pose (not a real ambiguity)
# if they're within this close -- otherwise every exact-tie pair would
# register as 2 clusters purely from floating-point jitter.
_POSE_CLUSTER_ROTATION_DEG = 5.0
_POSE_CLUSTER_TRANSLATION_M = 0.01

_TOP_IDS = frozenset({"top_left", "top_right"})
_BOTTOM_IDS = frozenset({"bottom_left", "bottom_right"})
_LEFT_IDS = frozenset({"top_left", "bottom_left"})
_RIGHT_IDS = frozenset({"top_right", "bottom_right"})


@dataclass
class CorrespondenceResult:
    lidar_centers_matched: np.ndarray   # (L,3), reordered to align with camera_centers
    camera_centers: np.ndarray          # (L,3), same row order as lidar_centers_matched
    common_ids: frozenset               # canonical corner ids used (L of them)
    residual_rmse_m: float
    R_camera_from_lidar: Optional[np.ndarray] = None  # 3x3, the (L-point) fit -- useful as a future reference_transform
    t_camera_from_lidar: Optional[np.ndarray] = None  # (3,)
    ambiguous: bool = False             # True if multiple candidate corner-assignments tied and
                                         # no reference_transform was available to break the tie


def _cyclic_windows(sequence: list, length: int) -> list[list]:
    """Every length-`length` contiguous window of `sequence`, treated as
    cyclic (wraps around) -- one window per starting rotation."""
    n = len(sequence)
    return [[sequence[(start + i) % n] for i in range(length)] for start in range(n)]


def _orientation_score(window_ids: list[str], candidate_lidar: np.ndarray) -> Optional[float]:
    """Sum of two independent axis checks in the LiDAR's own frame (REP-103
    convention, already this project's convention -- see
    geometry/transform.py): (mean Z of top_* ids - mean Z of bottom_* ids)
    + (mean Y of right_* ids - mean Y of left_* ids). Positive/large means
    "upright and not mirrored", as expected for a target held/mounted
    normally and viewed (not viewed through a mirror or from behind).

    Both axes are required: a rectangle's symmetry group has 2 independent
    binary symmetries (180-degree in-plane rotation flips top<->bottom AND
    left<->right simultaneously; the 2 mirror reflections each flip only
    one axis) -- checking Z alone leaves the left/right mirror pair
    unresolved (found empirically: an early version of this function used
    Z only and still returned a mirrored-wrong correspondence about half
    the time). Returns None if `window_ids` doesn't include at least one
    id from each of the 4 corner groups (cannot happen for any length-3-
    or-4 contiguous window of CORNER_ORDER, but guarded anyway)."""
    top_z = [candidate_lidar[i][2] for i, cid in enumerate(window_ids) if cid in _TOP_IDS]
    bottom_z = [candidate_lidar[i][2] for i, cid in enumerate(window_ids) if cid in _BOTTOM_IDS]
    left_y = [candidate_lidar[i][1] for i, cid in enumerate(window_ids) if cid in _LEFT_IDS]
    right_y = [candidate_lidar[i][1] for i, cid in enumerate(window_ids) if cid in _RIGHT_IDS]
    if not top_z or not bottom_z or not left_y or not right_y:
        return None
    return float(np.mean(top_z) - np.mean(bottom_z)) + float(np.mean(right_y) - np.mean(left_y))


def _cluster_tied_poses(tied: list) -> int:
    """Number of meaningfully-distinct (rotation, translation) poses among
    tied candidates -- two candidates sharing the exact same common_ids SET
    (always true for the L=4 case, which only ever has one possible id set)
    can still represent totally different, both-exact-fitting poses (the
    180-degree in-plane "twin"), so pose similarity -- not id-set equality
    -- is what actually detects a real tie."""
    distinct: list[tuple[np.ndarray, np.ndarray]] = []
    for _, _, _, _, R, t in tied:
        if not any(
            rotation_geodesic_distance(R, R2, degrees=True) < _POSE_CLUSTER_ROTATION_DEG
            and float(np.linalg.norm(t - t2)) < _POSE_CLUSTER_TRANSLATION_M
            for R2, t2 in distinct
        ):
            distinct.append((R, t))
    return len(distinct)


def match_partial_centers(
    camera_ids_to_centers: dict,
    lidar_centers_cyclic: np.ndarray,
    reference_transform: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[CorrespondenceResult]:
    """camera_ids_to_centers: canonical id (from target_config.CORNER_ORDER)
    -> 3D center, restricted to IDs the camera actually, independently
    detected (see camera_detector.CameraDetectionResult.detected_ids).
    lidar_centers_cyclic: (N,3), N in {3,4}, cyclic order, unlabeled.
    reference_transform: optional (R_camera_from_lidar, t_camera_from_lidar)
    used ONLY to break residual ties among congruent PARTIAL-scene corner
    hypotheses (see module docstring) -- typically the current best FULL-
    scene-only estimate. Has no effect when there's no tie (in particular,
    never affects the FULL 4/4 case, which is never ambiguous).

    Returns None if there aren't enough points on either side (<3) to
    attempt any correspondence at all -- callers should treat that as an
    INVALID scene, not retry with a different strategy (see the
    "true 2-common-feature" scoping note in camera_lidar/types.py's
    SceneType docstring / the implementation plan)."""
    lidar_centers_cyclic = np.asarray(lidar_centers_cyclic, dtype=np.float64)
    n_lidar = lidar_centers_cyclic.shape[0]
    n_camera = len(camera_ids_to_centers)
    length = min(n_camera, n_lidar)
    if length < _MIN_CORRESPONDENCE_POINTS or n_lidar < _MIN_CORRESPONDENCE_POINTS:
        return None

    seen_id_sets: set[frozenset] = set()
    camera_windows: list[list[str]] = []
    for window_ids in _cyclic_windows(list(CORNER_ORDER), length):
        if not all(cid in camera_ids_to_centers for cid in window_ids):
            continue
        key = frozenset(window_ids)
        if key in seen_id_sets:
            continue
        seen_id_sets.add(key)
        camera_windows.append(window_ids)

    if not camera_windows:
        return None

    lidar_index_windows = _cyclic_windows(list(range(n_lidar)), length)

    candidates = []  # (rmse, window_ids, candidate_lidar, camera_points, R, t)
    for window_ids in camera_windows:
        camera_points = np.array([camera_ids_to_centers[cid] for cid in window_ids])
        for idx_window in lidar_index_windows:
            for candidate_idx in (idx_window, idx_window[::-1]):
                candidate_lidar = lidar_centers_cyclic[candidate_idx]
                R, t = solve_rigid_transform(candidate_lidar, camera_points)
                residual = compute_residuals(candidate_lidar, camera_points, R, t)
                candidates.append((residual.rmse, window_ids, candidate_lidar.copy(), camera_points.copy(), R, t))

    if not candidates:
        return None

    best_rmse = min(c[0] for c in candidates)
    tie_band = max(best_rmse * _TIE_RELATIVE_FACTOR, _TIE_ABSOLUTE_FLOOR_M)
    tied = [c for c in candidates if c[0] <= best_rmse + tie_band]
    genuinely_tied = _cluster_tied_poses(tied) > 1

    if genuinely_tied and reference_transform is not None:
        ref_R, ref_t = reference_transform

        def _pose_distance(candidate):
            _, _, _, _, R, t = candidate
            return rotation_geodesic_distance(ref_R, R, degrees=True) + float(np.linalg.norm(ref_t - t)) * 1000.0

        tied.sort(key=_pose_distance)
        chosen = tied[0]
        ambiguous = False
    elif genuinely_tied:
        # No reference available -- fall back to the "target held upright,
        # not mirrored" assumption (see _orientation_score / module docstring).
        scored = [(c, _orientation_score(c[1], c[2])) for c in tied]
        scored = [(c, s) for c, s in scored if s is not None]
        if scored and any(abs(s) > 1e-9 for _, s in scored):
            scored.sort(key=lambda item: -item[1])
            chosen = scored[0][0]
        else:
            tied.sort(key=lambda c: c[0])
            chosen = tied[0]
        ambiguous = True
    else:
        tied.sort(key=lambda c: c[0])
        chosen = tied[0]
        ambiguous = False

    rmse, window_ids, candidate_lidar, camera_points, R, t = chosen
    return CorrespondenceResult(
        lidar_centers_matched=candidate_lidar,
        camera_centers=camera_points,
        common_ids=frozenset(window_ids),
        residual_rmse_m=rmse,
        R_camera_from_lidar=R,
        t_camera_from_lidar=t,
        ambiguous=ambiguous,
    )


def match_centers(camera_centers: np.ndarray, lidar_centers_cyclic: np.ndarray) -> CorrespondenceResult:
    """Original full-scene (4-of-4) entry point: camera_centers must
    already be the 4 canonical centers in CORNER_ORDER order. A thin
    wrapper around match_partial_centers for callers that don't need to
    deal with partial detected_ids bookkeeping."""
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    lidar_centers_cyclic = np.asarray(lidar_centers_cyclic, dtype=np.float64)
    if camera_centers.shape != (4, 3) or lidar_centers_cyclic.shape != (4, 3):
        raise ValueError(
            f"match_centers expects (4,3) arrays, got camera={camera_centers.shape}, "
            f"lidar={lidar_centers_cyclic.shape}"
        )
    camera_ids_to_centers = dict(zip(CORNER_ORDER, camera_centers))
    result = match_partial_centers(camera_ids_to_centers, lidar_centers_cyclic)
    if result is None:  # pragma: no cover -- unreachable: 4 camera IDs + 4 lidar points always yields a candidate
        raise ValueError("match_centers: no valid correspondence found (unexpected for 4/4 input)")
    return result
