"""Calibration room: several boards in view at once, one capture.

The single-board tool gets its pose variety by moving one board between scenes.
A calibration room gets it by standing several boards around the vehicle, so
one capture holds what used to take a dozen -- and every board in it is a
"scene" to `solve`. Three things change, and this file is those three:

1. Camera: markers have to be grouped into boards. The boards in this room all
   carry the same four ids (1, 2, 4, 3), so the id cannot say which board a
   marker is on. Each marker, though, fixes the plane it lies on, and from its
   own four corners predicts where the other three markers of its board must
   be. Markers that predict each other are one board; PnP over the group then
   confirms it. Nothing here needs the ids to differ -- when the boards get
   distinct ids, list them in `layouts` and the same code keeps them apart
   more easily.

2. LiDAR: nobody draws a box per board. Board-sized planar patches are found by
   region growing on normals, and each patch is handed to the existing hole
   detector in a box fitted to it. The detector, not the patch search, decides
   what is a board: a car door is board-sized and planar, but it has no four
   holes on the right rectangle.

3. Correspondence: the LiDAR boards come out unlabelled, and with identical
   boards nothing on a board says which camera board it is. One pair is
   enough to solve an extrinsic (that is what `solve` does with one scene), so
   every pairing is tried as a hypothesis and kept by how many *other* boards
   it lines up. A wrong pair puts the rest of the room in the wrong place, so
   the count separates them sharply.

Boards the patch search misses but the camera saw are then placed from the
extrinsic, the way `roi_auto.sweep` places scenes from a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from gui.core import camera_ops
from gui.core import detect_lidar as dl
from gui.core.detect_camera import MARKERS, CameraDetection, detect_markers, pose_from_markers
from gui.core.project import FilterBox, Target

# Quadrant order used throughout: top-left, top-right, bottom-right, bottom-left,
# matching detect_camera.MARKERS. A layout is the four ids in that order.
STOCK_LAYOUT = tuple(m[2] for m in MARKERS)  # (1, 2, 4, 3)


# =========================================================== camera: grouping


@dataclass
class CameraBoard:
    """One board as the camera sees it."""

    label: str
    layout: int
    det: CameraDetection
    marker_idx: list[int] = field(default_factory=list)  # indices into the frame's markers
    centre_px: np.ndarray | None = None  # board centre in the image
    frames: int = 1  # how many frames were averaged into det.centers
    spread_mm: float = 0.0  # RMS spread of the centres across those frames


def _marker_centres_2d(target: Target) -> dict[int, np.ndarray]:
    """Board-plane (x, y) of each quadrant's marker centre, keyed by quadrant."""
    return {q: np.array([sx * target.delta_width_qr_center, sy * target.delta_height_qr_center])
            for q, (sx, sy, _) in enumerate(MARKERS)}


def _marker_corners_2d(target: Target, q: int) -> np.ndarray:
    c = _marker_centres_2d(target)[q]
    h = target.marker_size / 2
    # Same corner order detectMarkers returns, board +y up.
    return np.array([c + (dx * h, dy * h) for dx, dy in ((-1, 1), (1, 1), (1, -1), (-1, -1))])


def group_markers(ids: list[int], corners: list[np.ndarray], camera, target: Target,
                  layouts=(STOCK_LAYOUT,), tol: float = 1.2, min_markers: int = 3,
                  max_rms_px: float = 4.0) -> list[tuple[int, list[int], CameraDetection]]:
    """Split one frame's markers into boards.

    Each marker's four corners give a homography from its board's plane to the
    image (in undistorted coordinates, where a plane maps exactly). Through it
    the marker predicts its three siblings; whichever marker of the right id
    lies within `tol` marker widths of a prediction joins the candidate. Every
    marker proposes a candidate, PnP scores them, and the best are taken
    greedily without sharing a marker.

    Returns (layout index, marker indices, detection) per board.
    """
    if not ids or not camera.is_set:
        return []
    norm = [camera_ops.undistort_points(camera, c) for c in corners]
    centres = np.array([n.mean(axis=0) for n in norm])
    widths = np.array([np.mean([np.linalg.norm(n[k] - n[(k + 1) % 4]) for k in range(4)]) for n in norm])
    marker_xy = _marker_centres_2d(target)

    candidates: dict[tuple[int, frozenset], None] = {}
    for i, mid in enumerate(ids):
        for li, layout in enumerate(layouts):
            if mid not in layout:
                continue
            q = layout.index(mid)
            H, _ = cv2.findHomography(_marker_corners_2d(target, q), norm[i], 0)
            if H is None:
                continue
            group = {q: i}
            for q2 in range(4):
                if q2 == q:
                    continue
                p = cv2.perspectiveTransform(marker_xy[q2].reshape(1, 1, 2), H).reshape(2)
                same = [j for j, m in enumerate(ids) if m == layout[q2] and j != i]
                if not same:
                    continue
                d = [np.linalg.norm(centres[j] - p) for j in same]
                k = int(np.argmin(d))
                if d[k] < tol * widths[i]:
                    group[q2] = same[k]
            if len(group) >= min_markers and len(set(group.values())) == len(group):
                candidates[(li, frozenset(group.items()))] = None

    scored = []
    for li, items in candidates:
        group = dict(items)
        known = [(STOCK_LAYOUT[q], np.asarray(corners[j], np.float64)) for q, j in sorted(group.items())]
        det = pose_from_markers(known, camera, target)
        if not det.ok or det.reproj_rms > max_rms_px:
            continue
        # Behind the camera is a mirror solution, not a board.
        if det.tvec is None or float(np.ravel(det.tvec)[2]) <= 0:
            continue
        scored.append((-len(group), det.reproj_rms, li, sorted(group.values()), det))

    scored.sort(key=lambda s: (s[0], s[1]))
    used: set[int] = set()
    out = []
    for _, _, li, members, det in scored:
        if used.intersection(members):
            continue
        used.update(members)
        out.append((li, members, det))
    return out


def detect_boards(image: np.ndarray, camera, target: Target, layouts=(STOCK_LAYOUT,),
                  markers=None, **kw) -> list[CameraBoard]:
    """Every board in one image, labelled C0, C1, ... left to right.

    Markers are found with the small-marker second pass on: room boards stand
    10 m and more away, where the stock pass found as few as 10 of 32 markers.
    `markers` takes a precomputed (ids, corners) for the same image, so ranking
    many presets does not detect them over and over.
    """
    ids, corners = markers if markers is not None else detect_markers(image, small=True)
    found = group_markers(ids, corners, camera, target, layouts, **kw)
    boards = []
    for li, members, det in found:
        det.image_corners = np.array([corners[j] for j in members])
        px = camera_ops.project_points(camera, det.centers.mean(axis=0, keepdims=True))[0]
        boards.append(CameraBoard("", li, det, members, px))
    boards.sort(key=lambda b: (b.centre_px[0], b.centre_px[1]))
    for k, b in enumerate(boards):
        b.label = f"C{k}"
    return boards


def detect_boards_multi(images: list[np.ndarray], camera, target: Target,
                        layouts=(STOCK_LAYOUT,), **kw) -> list[CameraBoard]:
    """Boards found in a still scene over several frames, centres averaged.

    The room and the vehicle stand still, so every frame sees the same boards;
    averaging is free precision on the camera side, which is the noisier one at
    range. Boards are linked across frames by where they sit in the image.

    Every frame gets a vote on *which* markers a board has, not only the first.
    Grouping used to happen on frame 0 alone: a small marker that flickered out
    of that one frame left its board at three markers for good -- the view then
    showed the fourth marker detected right next to a board that did not use it
    -- and a board missing from frame 0 was missing altogether. Now each board
    is represented by the frame that saw the most of its markers (then the
    lowest residual), boards seen in any frame are kept, and the centres are
    averaged over the frames that saw as many markers as that best one.
    """
    from gui.core.multiview import match_order, robust_mean

    if not images:
        return []
    groups: list[list[CameraBoard]] = []
    for img in images:
        for b in detect_boards(img, camera, target, layouts, **kw):
            best, best_d = None, np.inf
            for g in groups:
                ref = g[0]
                d = float(np.linalg.norm(ref.centre_px - b.centre_px))
                if ref.layout == b.layout and d < max(0.5 * ref.det.marker_px, 8.0) and d < best_d:
                    best, best_d = g, d
            if best is None:
                groups.append([b])
            else:
                best.append(b)

    boards = []
    for g in groups:
        rep = max(g, key=lambda b: (b.det.n_markers, -b.det.reproj_rms))
        full = [b for b in g if b.det.n_markers == rep.det.n_markers]
        mean, kept, spread = robust_mean([match_order(rep.det.centers, b.det.centers) for b in full])
        if mean is not None:
            rep.det.centers = mean
            rep.frames, rep.spread_mm = kept, spread
        rep.seen = len(g)
        boards.append(rep)
    boards.sort(key=lambda b: (b.centre_px[0], b.centre_px[1]))
    for k, b in enumerate(boards):
        b.label = f"C{k}"
    return boards


# ============================================================ lidar: search


@dataclass
class LidarBoard:
    label: str
    centers: np.ndarray  # (4, 3)
    box: FilterBox
    det: object = None
    extent: tuple[float, float] = (0.0, 0.0)  # patch width, height in metres
    source: str = "자동"  # 자동 / 수동 / 배치


def panel_size(target: Target, margin: float = 0.03) -> tuple[float, float]:
    """Full width and height of a panel holding the printed target.

    Not `synth.board_extent`: that one halves the marker offsets, while
    `detect_camera.board_model` -- which the real boards match -- places marker
    centres at the full +-delta_*_qr_center. On the stock CAD it says 0.81 x
    0.70 m for a panel that is 1.36 x 0.96.
    """
    hw = max(target.delta_width_circles / 2 + target.circle_radius,
             target.delta_width_qr_center + target.marker_size / 2) + margin
    hh = max(target.delta_height_circles / 2 + target.circle_radius,
             target.delta_height_qr_center + target.marker_size / 2) + margin
    return 2 * hw, 2 * hh


def _normals(pts: np.ndarray, k: int, radius: float):
    """Per-point normal and surface variation from the k nearest neighbours."""
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k, distance_upper_bound=radius)
    valid = np.isfinite(dist)
    idx = np.where(valid, idx, 0)
    nb = pts[idx]  # (N, k, 3)
    w = valid[..., None].astype(np.float64)
    cnt = np.maximum(w.sum(axis=1), 1.0)
    mean = (nb * w).sum(axis=1) / cnt
    d = (nb - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", d, d) / cnt[:, None]
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, :, 0]
    variation = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-12)
    return normal, variation, idx, valid


def planar_patches(cloud: np.ndarray, voxel: float = 0.03, k: int = 24,
                   max_angle_deg: float = 12.0, max_variation: float = 0.05,
                   min_points: int = 60) -> list[np.ndarray]:
    """Planar segments of a cloud, by region growing on normals.

    Two neighbours join when both are locally flat, their normals agree, and
    each lies on the other's plane. A board joins its stand only through the
    pole, where the normals turn, so the panel comes out on its own.
    """
    # k = 24 rather than 12: a sensor scanning in lines puts a point's dozen
    # nearest neighbours all on its own line, and a line has no normal. Twice as
    # many reach the next line. Measured: rear AT128 boards found 4 -> 5, and on
    # the front recordings the same boards with a lower image error.
    pts = dl.voxel_down(np.asarray(cloud, np.float64), voxel)
    if len(pts) < min_points:
        return []
    normal, variation, idx, valid = _normals(pts, k, 4.0 * voxel)
    flat = variation < max_variation

    rows = np.repeat(np.arange(len(pts)), k)
    cols = idx.ravel()
    ok = valid.ravel() & flat[rows] & flat[cols] & (rows != cols)
    rows, cols = rows[ok], cols[ok]
    cos = np.abs(np.einsum("ij,ij->i", normal[rows], normal[cols]))
    off = np.abs(np.einsum("ij,ij->i", normal[rows], pts[cols] - pts[rows]))
    ok = (cos > np.cos(np.radians(max_angle_deg))) & (off < 1.5 * voxel)
    rows, cols = rows[ok], cols[ok]

    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pts), len(pts)))
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n)
    return [pts[labels == lab] for lab in np.flatnonzero(sizes >= min_points)]


def _patch_frame(patch: np.ndarray):
    """(centre, rotation with columns right/up/normal-to-sensor, width, height)."""
    centre = np.median(patch, axis=0)
    _, _, vt = np.linalg.svd(patch - patch.mean(axis=0), full_matrices=False)
    n = vt[2]
    if n @ centre > 0:
        n = -n
    up = np.array([0.0, 0.0, 1.0])
    ex = np.cross(up, n)
    if np.linalg.norm(ex) < 1e-3:
        ex = vt[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    rot = np.column_stack([ex, ey, n])
    local = (patch - centre) @ rot
    lo, hi = np.percentile(local[:, :2], 1, axis=0), np.percentile(local[:, :2], 99, axis=0)
    centre = centre + rot[:, :2] @ ((lo + hi) / 2)
    w, h = hi - lo
    return centre, rot, float(w), float(h)


def size_matches(w: float, h: float, target: Target, lo: float = 0.35, hi: float = 1.5) -> bool:
    """Could this patch be (part of) a board?

    The lower bound is loose on purpose. A board is often seen cut: the merged
    topic on this rig drops everything above about 2 m, which takes the top off
    the lower boards and the elevated ones entirely, and a partly hidden board
    is still worth a look -- the four holes may all be there. The hole detector
    is what decides, so a false patch costs one detection, a missed board costs
    the board.
    """
    pw, ph = panel_size(target)
    a, b = sorted((w, h)), sorted((pw, ph))
    return lo * b[1] <= a[1] <= hi * b[1] and 0.25 * b[0] <= a[0] <= hi * b[0]


def box_for_patch(centre, rot, target: Target, pad: float = 0.25,
                  thickness: float = 0.12) -> FilterBox:
    """A slab the size of a whole panel and then some, on the patch's plane.

    Sized from the target rather than the patch, because the patch may be only
    the part of the board region growing held together. Thin across the plane,
    so the wall behind stays out of it.
    """
    pw, ph = panel_size(target)
    half = max(pw, ph) / 2 + pad
    return dl.box_from_frame(centre, rot, (half, half, thickness / 2))


# Three holes are completed to four only when they already sit on the board's
# rectangle this closely -- every side and the diagonal, as a fraction of length.
COMPLETE_TOLERANCE = 0.04


def complete_rectangle(centres: np.ndarray, target: Target,
                       tolerance: float = COMPLETE_TOLERANCE) -> np.ndarray | None:
    """The fourth hole, from three that form a corner of the hole rectangle.

    A board cut by the sensor's field of view -- the merged topic here stops at
    about 2 m -- can lose a hole the occupancy raster needs to see enclosed. The
    other three still fix the board: they are a right angle with known sides,
    and the missing corner is the parallelogram's fourth vertex. Its error is
    the other three's, added, so it is only taken when they fit tightly.
    """
    from gui.core.detect_auto import shape_error

    pts = np.asarray(centres, np.float64).reshape(-1, 3)
    if len(pts) != 3 or shape_error(pts, target) > tolerance:
        return None
    # The corner is the point opposite the longest side (the diagonal).
    d = [np.linalg.norm(pts[(i + 1) % 3] - pts[(i + 2) % 3]) for i in range(3)]
    k = int(np.argmax(d))
    a, b = pts[(k + 1) % 3], pts[(k + 2) % 3]
    return np.vstack([pts, a + b - pts[k]])


# ----------------------------------------------------------- template fit
#
# Sizes for the model fit, as fractions of the hole radius. Inside `EMPTY` a
# hole must hold no points; between `RING_IN` and `RING_OUT` the panel must.
TEMPLATE_EMPTY = 0.75
TEMPLATE_RING_IN = 1.25
TEMPLATE_RING_OUT = 1.75
# Points inside a hole, per square metre, as a share of the panel's own density
# in the ring around it. A merged cloud puts the odd return in mid-air, and a
# dense board puts proportionally more there -- so a count would be either too
# strict up close or too loose far away. Real holes measured 0-6 %; a board
# placed on solid panel reads near 100 %.
TEMPLATE_STRAY_DENSITY = 0.15
# How far the fine fit may move from the coarse one. The coarse grid is 10 mm
# and 1 degree, so this is several cells of slack -- and it is what stops the
# fit from "emptying" the holes by sliding the pattern off the board.
TEMPLATE_FINE_SHIFT = 0.05
TEMPLATE_FINE_TURN = np.radians(2.0)
# Of the ring around each hole, at least this share of the angular sectors must
# hold panel points, or the "hole" could be open space beside the board.
TEMPLATE_RING_COVER = 0.7
# A ring sector counts as panel when this share of it is solid on the grown
# raster. Half was too strict for a board sampled at 40 mm: it read 58-67 %
# surrounded and lost both elevated front boards that had been found before.
TEMPLATE_SECTOR_FILL = 0.3
# Penalty for a hole that is completely filled, against 1 per ring.
TEMPLATE_HOLE_WEIGHT = 2.0
# Weight of "the whole panel is there" in the coarse kernel, against 1 per ring.
TEMPLATE_PANEL_WEIGHT = 2.0
# |normal . z| above this is floor or ceiling, never a board.
TEMPLATE_MAX_NORMAL_Z = 0.85
# Scan lines further apart than this are not bridged when growing the raster.
# 16 cm covers a VLP-32C's widest spacing a few degrees below the horizon at
# 6-8 m; much wider and a hole (240 mm across) would be grown shut.
TEMPLATE_MAX_GAP = 0.16
# A spot between scan lines counts as panel when there is panel on both sides
# of it, across the lines, within this distance (see the checks in
# `fit_board_template`). Wider than any channel gap met on a board in practice.
TEMPLATE_BRACKET = 0.30
# A board fitted as a whole pattern, rather than found hole by hole, is only
# believed this close. Further out the scan lines are 10-16 cm apart, a hole is
# crossed by one or two of them, and walls at the far end of the hall passed
# the checks: three such "boards" at 23-24 m on the recordings here, while every
# real board stood within 16 m.
TEMPLATE_MAX_RANGE = 20.0


def _plane_points(cloud, box: FilterBox, voxel: float = 0.005, threshold: float = 0.02):
    """Board-plane points in the box, and the 2D frame they are expressed in."""
    xyz = np.asarray(cloud, np.float64)[box.mask(cloud)]
    xyz = dl.voxel_down(xyz, voxel)
    if len(xyz) < 50:
        return None
    normal, d, inliers = dl.fit_plane(xyz, threshold)
    if normal is None or inliers.sum() < 50:
        return None
    plane = xyz[inliers]
    centre, rot, _, _ = _patch_frame(plane)
    uv = (plane - centre) @ rot[:, :2]
    rms = float(np.sqrt(((plane @ normal + d) ** 2).mean()))
    return plane, uv, centre, rot, rms


def line_gap(uv: np.ndarray, sample: int = 1500, seed: int = 0) -> float:
    """Spacing *between* scan lines on a plane, in metres.

    Many sensors sample a surface in lines: a spinning unit's rings, the AT128's
    scan rows. Along a line points sit 1-2 cm apart, between lines 6-8 cm, and
    stacking sweeps only makes the lines denser -- so any nearest-neighbour
    distance, however many neighbours deep, measures along the line.

    So for each sampled point: find its line's direction from the closest
    neighbours, take a narrow strip across the line, sort the points in it by
    their perpendicular offset, and keep the widest empty interval. That is the
    space between lines. Asking for the nearest point off the line instead does
    not work on the AT128: each of its rows is two sub-rows 1 cm apart, and the
    nearest point "off the line" is the other sub-row. On evenly sampled data
    the widest interval is simply about the point spacing.
    """
    _, gaps, _ = _line_gaps(uv, sample, seed)
    return float(np.median(gaps)) if len(gaps) else 0.0


def local_line_gap(uv: np.ndarray, sample: int = 1500, seed: int = 0) -> np.ndarray:
    """`line_gap` per point: the spacing between the scan lines around each one.

    One number per board is not enough for a spinning sensor whose channels are
    unevenly spaced. A VLP-32C's are 0.33 degrees apart near the horizon and
    1-1.6 degrees a few degrees below it, so one board low in its view has lines
    3 cm apart at the top and 10-15 cm apart at the bottom. Grown by the median,
    the bottom stays striped and its holes look half off the panel.

    Each point takes the median of the gaps measured at the sampled points
    within 10 cm of it (or the nearest sample's, if none are that close).
    """
    idx, gaps, _ = _line_gaps(uv, sample, seed)
    if not len(gaps):
        return np.zeros(len(uv))
    tree = cKDTree(uv[idx])
    out = np.empty(len(uv))
    near = tree.query_ball_point(uv, 0.10)
    _, nearest = tree.query(uv, k=1)
    for i, nb in enumerate(near):
        out[i] = np.median(gaps[nb]) if nb else gaps[nearest[i]]
    return out


def scan_direction(uv: np.ndarray, sample: int = 600, seed: int = 0) -> np.ndarray | None:
    """Unit vector along the scan lines on a plane (the dominant one), or None."""
    _, _, dirs = _line_gaps(uv, sample, seed)
    if not len(dirs):
        return None
    # Directions have no sign: average the doubled angle.
    a = 2 * np.arctan2(dirs[:, 1], dirs[:, 0])
    m = np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 2
    return np.array([np.cos(m), np.sin(m)])


def _line_gaps(uv: np.ndarray, sample: int, seed: int):
    """(sampled point indices, the gap measured at each, line direction there) -- see `line_gap`."""
    if len(uv) < 50:
        return np.zeros(0, int), np.zeros(0), np.zeros((0, 2))
    tree = cKDTree(uv)
    rng = np.random.default_rng(seed)
    idx, gaps, dirs = [], [], []
    for i in rng.choice(len(uv), min(len(uv), sample), replace=False):
        off = uv[tree.query_ball_point(uv[i], 0.12)] - uv[i]
        near = off[np.linalg.norm(off, axis=1) < 0.03]
        if len(near) < 4:
            continue
        _, _, vt = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
        along, perp = off @ vt[0], off @ np.array([-vt[0, 1], vt[0, 0]])
        strip = np.sort(perp[np.abs(along) < 0.04])
        if len(strip) < 3:
            continue
        idx.append(i)
        gaps.append(float(np.diff(strip).max()))
        dirs.append(vt[0])
    return np.asarray(idx, int), np.asarray(gaps), np.asarray(dirs).reshape(-1, 2)


def _hole_offsets(target: Target) -> np.ndarray:
    w, h = target.delta_width_circles / 2, target.delta_height_circles / 2
    return np.array([[-w, h], [w, h], [w, -h], [-w, -h]])


def fit_board_template(cloud, box: FilterBox, target: Target, spacing: float = 0.0):
    """Fit the whole hole pattern at once, for boards too sparse to find holes one by one.

    The occupancy raster finds each hole as an empty region after closing the
    gaps between samples. At 40 mm spacing -- the elevated boards here, seen by
    few rings at a steep angle -- six samples cross a 240 mm hole, the closing
    eats into it, and a single stray return splits it. Yet the four holes are
    plain to the eye, because the eye uses what the raster does not: they come
    as a rigid rectangle of known size.

    So the model is fitted instead. Coarse: for each in-plane angle, correlate
    the point raster with a kernel that rewards points in a ring around each
    hole and heavily punishes points inside it -- an FFT finds the best shift
    for every angle at once. Fine: move the rigid four-hole pattern to make the
    holes as empty as possible, measured by each hole's nearest few points, so
    every rim point on all four holes pulls on one answer.

    Returns a LidarDetection (method "template") or one with ok False and a reason.
    """
    from scipy.optimize import minimize
    from scipy.signal import fftconvolve

    got = _plane_points(cloud, box)
    if got is None:
        return dl.LidarDetection(False, method="template", reason="평면을 찾지 못했습니다")
    plane, uv, centre, rot, rms = got
    # Boards stand up; even the elevated ones here lean back no more than about
    # 45 degrees. A near-horizontal plane is floor or ceiling, and a floor's tiles
    # and drains can pass for four holes -- that is how a "board" was once found
    # 20 m behind the vehicle, and another lying on the floor.
    if abs(rot[2, 2]) > TEMPLATE_MAX_NORMAL_Z:
        return dl.LidarDetection(False, method="template",
                                 reason=f"수평면입니다 (법선 z {rot[2, 2]:+.2f}) — 바닥/천장")
    r = target.circle_radius
    holes = _hole_offsets(target)

    # ---- coarse: angle grid x FFT over shifts
    cell = 0.01
    lo = uv.min(axis=0) - 0.3
    size = np.ceil((uv.max(axis=0) + 0.3 - lo) / cell).astype(int)
    # Presence, not count, and each point grown by its own sampling gap. A board
    # is rarely sampled evenly: here two units split it into a dense upper half
    # and a sparse lower one, and a spinning sensor's rings bunch up near the
    # horizon. Scored on raw presence, the pattern slides towards wherever the
    # points are thickest -- on one board a full row upwards. Growing each point
    # by its local gap makes every part of the panel read as solid.
    #
    # Growth is at least the spacing between scan lines (`line_gap`). The AT128
    # here samples 1-2 cm along a line and 6-8 cm between lines; grown by the
    # along-line gap alone the panel stays striped and the pattern locks onto
    # whatever else is solid.
    #
    # And the spacing between lines is taken where each point is, not once per
    # board (`local_line_gap`): a Velodyne's channels bunch up near the horizon,
    # so one board can have lines 3 cm apart at its top and 15 cm at its bottom.
    ij = np.floor((uv - lo) / cell).astype(int)
    nn, _ = cKDTree(uv).query(uv, k=4)
    across = local_line_gap(uv)
    gap = np.clip(np.maximum(np.median(nn[:, 1:], axis=1), across), cell, TEMPLATE_MAX_GAP)
    grow = np.clip(np.round(0.6 * gap / cell).astype(int), 0, int(round(0.6 * TEMPLATE_MAX_GAP / cell)))
    img = np.zeros((size[1], size[0]), bool)
    from scipy import ndimage
    for g_ in np.unique(grow):
        layer = np.zeros_like(img)
        sel = grow == g_
        layer[ij[sel, 1], ij[sel, 0]] = True
        if g_ > 0:
            yy, xx = np.mgrid[-g_:g_ + 1, -g_:g_ + 1]
            layer = ndimage.binary_dilation(layer, xx * xx + yy * yy <= g_ * g_)
        img |= layer
    img = img.astype(float)
    # Growing every point eats into the holes by the same amount, so the disc
    # that must stay empty shrinks with it. Left at a fixed 0.6 r, a board grown
    # by 43 mm had its holes narrowed to about 60 mm and the disc overlapped
    # their rims at the right place.
    grown_m = float(np.median(grow)) * cell
    r_empty = max(0.3 * r, r - grown_m - 0.015)

    # The kernel also rewards the panel itself -- the whole rectangle minus the
    # holes. Rings alone let the pattern hang off the board's lower edge, where
    # the space below is as empty as any hole and the rim above still counts;
    # that is where it went on a sparse AT128 board.
    pw, ph = panel_size(target)
    half = int(np.ceil(max(np.hypot(pw, ph) / 2, np.abs(holes).max() + TEMPLATE_RING_OUT * r) / cell)) + 1
    g = (np.arange(-half, half + 1)) * cell
    gx, gy = np.meshgrid(g, g)
    best = None
    for ang in np.radians(np.arange(-15.0, 15.01, 1.0)):
        c, s_ = np.cos(ang), np.sin(ang)
        k = np.zeros_like(gx)
        lx, ly = c * gx + s_ * gy, -s_ * gx + c * gy  # kernel grid in the board's own axes
        panel = (np.abs(lx) < pw / 2 - 0.03) & (np.abs(ly) < ph / 2 - 0.03)
        for hx, hy in holes:
            panel &= np.hypot(lx - hx, ly - hy) > TEMPLATE_RING_IN * r
        k[panel] += TEMPLATE_PANEL_WEIGHT / max(panel.sum(), 1)
        for hx, hy in holes:
            px, py = c * hx - s_ * hy, s_ * hx + c * hy
            d = np.hypot(gx - px, gy - py)
            ring = (d > TEMPLATE_RING_IN * r) & (d < TEMPLATE_RING_OUT * r)
            k[ring] += 1.0 / ring.sum()
            # Normalised like the rewards. As a flat -0.5 per cell it came to
            # about -100 for four slightly filled holes against at most +6 of
            # reward, and the best score was wherever there was nothing at all
            # -- below the board.
            disc = d < r_empty
            k[disc] -= TEMPLATE_HOLE_WEIGHT / max(disc.sum(), 1)
        score = fftconvolve(img, k[::-1, ::-1], mode="same")
        j = np.unravel_index(np.argmax(score), score.shape)
        if best is None or score[j] > best[0]:
            best = (score[j], ang, np.array([j[1], j[0]]) * cell + lo)
    _, ang0, t0 = best

    # ---- fine: rigid pattern, holes as empty as possible
    tree = cKDTree(uv)
    kn = 6

    def placed(x):
        c, s_ = np.cos(x[0]), np.sin(x[0])
        return holes @ np.array([[c, s_], [-s_, c]]) + x[1:]

    x0 = np.array([ang0, *t0])

    def cost(x):
        over = max(abs(x[0] - x0[0]) / TEMPLATE_FINE_TURN, np.linalg.norm(x[1:] - x0[1:]) / TEMPLATE_FINE_SHIFT)
        d, _ = tree.query(placed(x), k=kn)
        return -float(d.mean()) + (10.0 * (over - 1.0) if over > 1.0 else 0.0)

    res_ = minimize(cost, x0, method="Nelder-Mead",
                    options=dict(initial_simplex=np.vstack([x0, x0 + np.diag([0.02, 0.02, 0.02])]),
                                 xatol=1e-4, fatol=1e-6, maxiter=800))
    x = res_.x if np.isfinite(res_.fun) else x0
    centres2d = placed(x)

    # ---- checks: each hole empty and surrounded by panel
    #
    # Whether the panel surrounds each hole is read off the grown raster, not
    # the raw points: a 60 mm ring falls between scan lines 60-80 mm apart, and
    # counted on raw points a solid panel read as 33 % surrounded.
    #
    # A spot on the ring with no panel drawn there may still be panel: it can
    # fall between two scan lines too far apart to bridge. A VLP-32C has 23 cm
    # between two of its channels a few degrees below the horizon, and a board
    # there has its lower holes' rims in that gap. What the check is for -- the
    # pattern hanging off the panel's edge -- leaves panel on one side only, so a
    # spot with panel on both sides of it across the scan lines, within
    # TEMPLATE_BRACKET, counts as panel.
    H, W = img.shape
    along_dir = scan_direction(uv)
    across_dir = None if along_dir is None else np.array([-along_dir[1], along_dir[0]])

    def bracketed(pts: np.ndarray) -> np.ndarray:
        if across_dir is None:
            return np.zeros(len(pts), bool)
        out = np.zeros(len(pts), bool)
        for i, nb in enumerate(tree.query_ball_point(pts, TEMPLATE_BRACKET)):
            if len(nb) < 2:
                continue
            off = uv[nb] - pts[i]
            a, c_ = off @ along_dir, off @ across_dir
            near_line = np.abs(a) < 0.03
            out[i] = bool(np.any(near_line & (c_ > 0)) and np.any(near_line & (c_ < 0)))
        return out

    radii, reasons, covers = [], [], []
    for k_, c2 in enumerate(centres2d):
        d = np.linalg.norm(uv - c2, axis=1)
        inside = int((d < TEMPLATE_EMPTY * r).sum())
        ring = uv[(d > TEMPLATE_RING_IN * r) & (d < TEMPLATE_RING_OUT * r)] - c2
        ring_density = len(ring) / (np.pi * r * r * (TEMPLATE_RING_OUT ** 2 - TEMPLATE_RING_IN ** 2))
        inside_density = inside / (np.pi * (TEMPLATE_EMPTY * r) ** 2)
        covered = drawn = 0
        for sec in range(12):
            a0 = sec / 12 * 2 * np.pi
            angs = a0 + np.linspace(0, 2 * np.pi / 12, 6)
            rads = np.linspace(TEMPLATE_RING_IN * r, TEMPLATE_RING_OUT * r, 4)
            pts = c2 + np.stack([np.outer(rads, np.cos(angs)).ravel(), np.outer(rads, np.sin(angs)).ravel()], axis=1)
            q = np.floor((pts - lo) / cell).astype(int)
            okq = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            if not okq.any():
                continue
            filled = img[q[okq, 1], q[okq, 0]] > 0
            drawn += filled.mean() >= TEMPLATE_SECTOR_FILL
            if filled.mean() < TEMPLATE_SECTOR_FILL and (~filled).any():
                gapfill = filled.copy()
                gapfill[~filled] = bracketed(pts[okq][~filled])
                filled = gapfill
            if filled.mean() >= TEMPLATE_SECTOR_FILL:
                covered += 1
        cover = covered / 12
        covers.append((drawn / 12, cover))
        radii.append(float(np.sort(d)[:kn].mean()))
        if inside_density > TEMPLATE_STRAY_DENSITY * max(ring_density, 1e-9):
            reasons.append(f"구멍{k_ + 1} 안에 점 {inside}개 (둘레 밀도의 {inside_density / max(ring_density, 1e-9):.0%})")
        if cover < TEMPLATE_RING_COVER:
            reasons.append(f"구멍{k_ + 1} 둘레 {cover:.0%}만 패널")
    centres = np.array([centre + rot[:, :2] @ c2 for c2 in centres2d])
    reach = float(np.linalg.norm(centres.mean(axis=0)))
    if reach > TEMPLATE_MAX_RANGE:
        reasons.append(f"{reach:.1f} m — 모델 정합은 {TEMPLATE_MAX_RANGE:.0f} m 안에서만 믿습니다")
    det = dl.LidarDetection(
        ok=not reasons, centers=centres, method="template", plane=plane.astype(np.float32),
        plane_rms=rms, spacing=spacing, radii=radii, n_clusters=4,
        reason="; ".join(reasons) if reasons else "보드 모델 정합",
    )
    det.template_angle_deg = float(np.degrees(x[0]))
    det.template_debug = {"raster": img, "origin": lo, "cell": cell, "coarse": (float(ang0), t0), "gap": float(np.median(gap)),
                          "cover": covers}
    return det


def detect_in_box(cloud, single, box: FilterBox, target: Target, method: str = "occupancy",
                  ring=None, complete: bool = True, template: bool = True):
    """The single-board detector, run on one box, with spacing from one sweep.

    With `complete`, a detection of exactly three holes on the right rectangle
    is completed to four; the result then carries `completed = True`. With
    `template`, a board the hole detector cannot finish is tried once more by
    fitting the whole pattern (`fit_board_template`); that result carries
    `templated = True`. Method "template" goes straight to the model fit.
    """
    sweep = single[box.mask(single)] if single is not None and len(single) else np.empty((0, 3))
    spacing = dl.point_spacing(sweep) if len(sweep) > 20 else 0.0
    if method == "template":
        det = fit_board_template(cloud, box, target, spacing)
        det.completed, det.templated = False, det.ok
        return det
    params = dl.derive_params(spacing)
    params.method = method
    det = dl.detect(cloud, box, target, params, ring=ring)
    det.completed = False
    det.templated = False
    if complete and not det.ok and det.centers is not None and len(det.centers) == 3:
        four = complete_rectangle(det.centers, target)
        if four is not None:
            det.centers, det.ok, det.completed = four, True, True
            det.reason = "구멍 3개 + 직사각형 보완"
    if det.ok and len(det.centers) == 4:
        err = _shape_error(det.centers, target)
        if err > SHAPE_TOLERANCE:
            # The occupancy raster can take four gaps that are not the board's
            # holes -- low-return squares on a checkerboard, a gap between two
            # panels -- and does not itself check that they make its rectangle.
            det.ok = False
            det.reason = f"구멍 4개가 보드 직사각형이 아님 (형상 오차 {err:.0%})"
    if template and not det.ok:
        tdet = fit_board_template(cloud, box, target, spacing)
        if tdet.ok:
            tdet.completed, tdet.templated = False, True
            return tdet
    return det


# Four holes are accepted only when every side and diagonal is within this
# fraction of the board's own (`detect_auto.shape_error`). Measured on every
# recording here: real boards 0.00-0.084, the false four-gap sets the occupancy
# raster finds on checkerboards, walls and ceilings 0.25 and up.
SHAPE_TOLERANCE = 0.15


def _shape_error(centres, target: Target) -> float:
    from gui.core.detect_auto import shape_error

    return shape_error(np.asarray(centres, np.float64), target)


# A plane bigger than one panel may be several boards standing edge to edge --
# a column of three on one frame, say. Region growing joins them, since their
# planes differ only by the slight hinge where they meet, and the slab around
# the whole column then cuts through each board at a slant. So a plane up to
# STACK_MAX_PANELS panels long is searched again in panel-sized windows, each
# with its own plane fitted to the points inside it. A window takes the model fit
# (which has no shape to check, and finds board-like spots on bare walls) only
# once another window of the same plane has found a board by its holes: then
# the plane is known to be a stack of boards, and one cut by the floor or
# sparsely hit can still be fitted.
STACK_MAX_PANELS = 3.5


def board_windows(patch: np.ndarray, centre, rot, w: float, h: float, target: Target):
    """(centre, rotation) of panel-sized windows over an oversized patch.

    Both panel orientations are tried (landscape and portrait), windows half a
    panel apart, so each board is covered by at least one window that holds
    all of it and little of its neighbours.
    """
    pw, ph = panel_size(target)
    big = max(pw, ph)
    if max(w, h) > STACK_MAX_PANELS * big or min(w, h) > 1.6 * big:
        return []
    loc = (patch - centre) @ rot
    step = 0.5 * min(pw, ph)

    def offsets(extent, size):
        k = max(0.0, (extent - size) / 2)
        return np.linspace(-k, k, int(np.ceil(2 * k / step)) + 1) if k > 0.05 else np.zeros(1)

    out = []
    for wd, hd in ((pw, ph), (ph, pw)):
        for sx in offsets(w, wd):
            for sy in offsets(h, hd):
                m = (np.abs(loc[:, 0] - sx) < wd / 2) & (np.abs(loc[:, 1] - sy) < hd / 2)
                if m.sum() < 30:
                    continue
                c, r, _, _ = _patch_frame(patch[m])
                out.append((c, r))
    return out


# Detection boxes are fitted to a patch and reach past it by up to this much, so
# the cloud handed to the detector keeps this margin beyond the ROI -- a board
# whose edge the ROI clips still gets its whole panel and the wall behind it.
ROI_MARGIN = 0.4


def roi_mask(xyz: np.ndarray, rois, pad: float = 0.0) -> np.ndarray:
    """Points inside any of the ROI boxes, each grown by `pad` metres."""
    xyz = np.asarray(xyz)
    if not rois:
        return np.ones(len(xyz), bool)
    m = np.zeros(len(xyz), bool)
    for b in rois:
        if pad:
            b = FilterBox(b.x_min - pad, b.x_max + pad, b.y_min - pad, b.y_max + pad,
                          b.z_min - pad, b.z_max + pad, b.yaw, b.pitch, b.roll)
        # The cheap axis-aligned bound first: most of a 1M-point cloud is far
        # from every ROI, and the rotated test is several times dearer per point.
        c = np.asarray(b.corners())
        lo, hi = c.min(axis=0), c.max(axis=0)
        cand = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
        if len(cand):
            m[cand[b.mask(xyz[cand])]] = True
    return m


def _on_board(boards: list, point, target: Target) -> bool:
    """Is this window centre already on a found board?

    Boards on one frame stand at least a panel apart, so a window less than
    half a panel from a found board's centre is mostly that board again.
    """
    reach = 0.55 * min(panel_size(target))
    return any(np.linalg.norm(b.centers.mean(axis=0) - point) < reach for b in boards)


def _add_board(boards: list, det, box, size) -> None:
    c = np.asarray(det.centers, np.float64)
    if any(np.linalg.norm(b.centers.mean(axis=0) - c.mean(axis=0)) < 0.2 for b in boards):
        return
    boards.append(LidarBoard("", c, box, det, size))


def find_lidar_boards(cloud: np.ndarray, single: np.ndarray | None, target: Target,
                      method: str = "occupancy", max_range: float = 25.0,
                      min_range: float = 0.8, voxels=(0.03, 0.05), complete: bool = True,
                      template: bool = True, progress=None, roi=None) -> tuple[list[LidarBoard], dict]:
    """Every board the LiDAR can see, without being told where to look.

    With `roi` (a list of `FilterBox`), patches are only sought inside those
    boxes, and the detector only sees the cloud near them. The room's walls,
    floor and the vehicle itself are most of the cloud and most of the planar
    patches, so this is where the time goes.

    Returns the boards and a few counts for the report.
    """
    xyz = np.asarray(cloud, np.float32)
    stats_in = len(xyz)
    if roi:
        cloud = xyz[roi_mask(xyz, roi, ROI_MARGIN)]
        if single is not None and len(single):
            single = np.asarray(single)[roi_mask(single, roi, ROI_MARGIN)]
        xyz = cloud[roi_mask(cloud, roi)]
    r = np.linalg.norm(xyz, axis=1)
    near = xyz[(r > min_range) & (r < max_range)]
    # Two scales: the finer keeps neighbouring boards apart, the coarser holds a
    # noisy board together. A merged cloud from several units carries a few
    # millimetres of disagreement between them, which breaks fine normals.
    patches = [p for v in voxels for p in planar_patches(near, voxel=v)]
    stats = {"patches": len(patches), "sized": 0, "stacked": 0, "detected": 0, "points": stats_in, "searched": len(near)}

    boards: list[LidarBoard] = []
    tried: list[np.ndarray] = []
    for i, patch in enumerate(patches):
        centre, rot, w, h = _patch_frame(patch)
        if abs(rot[2, 2]) > TEMPLATE_MAX_NORMAL_Z:
            continue
        if size_matches(w, h, target):
            candidates = [(centre, rot)]
        else:
            candidates = board_windows(patch, centre, rot, w, h, target)
            stats["stacked"] += bool(candidates)
        if progress is not None:
            progress(i, len(patches))
        windowed = len(candidates) > 1 or not size_matches(w, h, target)
        later = []  # windows left for the model fit, once the plane proves a stack
        stack = False
        for centre, rot in candidates:
            if abs(rot[2, 2]) > TEMPLATE_MAX_NORMAL_Z:
                continue
            if any(np.linalg.norm(centre - t) < 0.15 for t in tried):
                continue
            if windowed and _on_board(boards, centre, target):
                continue
            tried.append(centre)
            stats["sized"] += 1
            box = box_for_patch(centre, rot, target)
            det = detect_in_box(cloud, single, box, target, method, complete=complete,
                                template=template and not windowed)
            if not det.ok:
                later.append(box)
                continue
            stack = True
            _add_board(boards, det, box, (w, h))
        if windowed and template and stack:
            for box in later:
                if _on_board(boards, np.asarray(box.center()), target):
                    continue
                det = detect_in_box(cloud, single, box, target, method, complete=complete, template=True)
                if det.ok:
                    _add_board(boards, det, box, (w, h))
    stats["detected"] = len(boards)
    # Label by bearing, left to right as seen from the sensor (+y is left).
    boards.sort(key=lambda b: -np.arctan2(b.centers[:, 1].mean(), b.centers[:, 0].mean()))
    for k, b in enumerate(boards):
        b.label = f"L{k}"
    return boards, stats


def board_at(cloud, single, point, target: Target, method: str = "occupancy",
             complete: bool = True, label: str = "M", template: bool = True):
    """A board around a point someone clicked on it, or (None, reason).

    The plane is fitted in a ball the size of a panel around the click, then
    the same slab and detector the search uses take over.
    """
    p = np.asarray(point, np.float64)
    pw, ph = panel_size(target)
    reach = 0.6 * max(pw, ph)
    xyz = np.asarray(cloud, np.float64)
    near = xyz[np.linalg.norm(xyz - p, axis=1) < reach]
    if len(near) < 100:
        return None, f"클릭 주변 점이 {len(near)}개뿐입니다"
    normal, d, inliers = dl.fit_plane(dl.voxel_down(near, 0.01), 0.02)
    if normal is None:
        return None, "평면을 찾지 못했습니다"
    patch = dl.voxel_down(near, 0.01)[inliers]
    centre, rot, w, h = _patch_frame(patch)
    box = box_for_patch(centre, rot, target)
    det = detect_in_box(cloud, single, box, target, method, complete=complete, template=template)
    if not det.ok:
        return None, det.reason or "검출 실패"
    return LidarBoard(label, np.asarray(det.centers, np.float64), box, det, (w, h), "수동"), ""


# ===================================================== correspondence + solve


def _gate(cam_centre: np.ndarray, base: float, frac: float) -> float:
    """How far a predicted board may sit from the camera's and still match.

    The camera's own range error grows with distance, so the gate does too.
    """
    return base + frac * float(np.linalg.norm(cam_centre))


def _assign(sol, lidar: list[LidarBoard], cams: list[CameraBoard], base: float, frac: float):
    """Mutually nearest (lidar, camera) pairs under `sol`, within the gate."""
    if not lidar or not cams:
        return []
    L = np.array([sol.transform(b.centers).mean(axis=0) for b in lidar])
    C = np.array([c.det.centers.mean(axis=0) for c in cams])
    D = np.linalg.norm(L[:, None, :] - C[None, :, :], axis=2)
    pairs = []
    for i in range(len(lidar)):
        j = int(np.argmin(D[i]))
        if int(np.argmin(D[:, j])) == i and D[i, j] < _gate(C[j], base, frac):
            pairs.append((i, j, float(D[i, j])))
    return pairs


def _scenes(pairs, lidar, cams):
    return [(f"{cams[j].label}={lidar[i].label}", lidar[i].centers, cams[j].det.centers)
            for i, j, _ in pairs]


# Two pairings count as rival explanations when they line up as many boards and
# put the camera this far apart. Below it they are the same answer with a board
# more or less; above it one of them is wrong.
AMBIGUOUS_SHIFT_M = 0.2
AMBIGUOUS_TURN_DEG = 1.0
# ...and fit within this factor of the best one's per-board distance, or under
# this many metres per board, whichever is looser.
AMBIGUOUS_RESIDUAL_RATIO = 2.0
AMBIGUOUS_FLOOR_M = 0.08
# The beam test (Plausible) is there to tell apart explanations the boards
# cannot -- one board fits every LiDAR board. It is not a veto over the boards
# themselves: a rear camera here sat where 42 beams pass close by (more than
# BLOCKED_RAYS), its true pairing lined up all 10 boards at 4 cm and was thrown
# out, and the 7-board wrong pairings left over tied as "ambiguous". So a
# pairing dropped only for beams is taken back when it lines up at least this
# many boards more than anything that passed, at most AMBIGUOUS_FLOOR_M per
# board, and still faces the way the camera does.
BEAM_OVERRIDE_BOARDS = 2


def _refine_pairs(pairs, lidar, cams, gate, gate_frac):
    """Refit on a pairing and reassign until it stops changing."""
    from gui.core.solve import solve

    sol = solve(_scenes(pairs, lidar, cams))
    for _ in range(5):
        new = _assign(sol, lidar, cams, gate, gate_frac)
        if len(new) < 1 or sorted(p[:2] for p in new) == sorted(p[:2] for p in pairs):
            break
        pairs = new
        sol = solve(_scenes(pairs, lidar, cams))
    return sol, pairs


def _apart(a, b) -> tuple[float, float]:
    """(camera position difference m, rotation difference deg) of two extrinsics."""
    pa, pb = -a.R.T @ a.t, -b.R.T @ b.t
    turn = np.degrees(np.arccos(np.clip((np.trace(a.R.T @ b.R) - 1) / 2, -1.0, 1.0)))
    return float(np.linalg.norm(pa - pb)), float(turn)


def board_evidence(sol, cams, matched: set, cloud, radius: float = 0.45, slab: float = 0.06) -> float:
    """How much of the cloud sits where `sol` says the unmatched camera boards are.

    A camera board the LiDAR search missed is usually still *there* in the
    cloud -- just not detected. The right extrinsic puts it on those points; a
    wrong one puts it in empty space. Counted as points on the predicted panel
    plane near its centre, summed over the unmatched boards.
    """
    if cloud is None or not len(cloud):
        return 0.0
    from gui.core.roi_auto import board_frame_in_lidar

    xyz = np.asarray(cloud, np.float32)
    total = 0.0
    for j, cb in enumerate(cams):
        if j in matched or getattr(cb.det, "rvec", None) is None:
            continue
        c, n = board_frame_in_lidar(sol, cb.det)
        near = np.linalg.norm(xyz - c.astype(np.float32), axis=1) < radius
        if near.any():
            total += float((np.abs((xyz[near] - c) @ n) < slab).sum())
    return total


def _rescue_beam_dropped(dropped, cands, check):
    """Pairings dropped only by the beam test that the boards clearly favour, best first.

    `dropped` and `cands` as in `match_boards`: (boards, summed distance,
    solution, pairs[, reason]). See BEAM_OVERRIDE_BOARDS.
    """
    top = max((c[0] for c in cands), default=0)
    facing = getattr(check, "facing", None)
    out = []
    for d in dropped:
        n, total, sol, _, why = d
        if not why.startswith("빔통과"):
            continue
        if n < max(WEAK_BOARDS, top + BEAM_OVERRIDE_BOARDS) or total / n > AMBIGUOUS_FLOOR_M:
            continue
        if facing is not None and facing(sol):
            continue
        out.append(d)
    out.sort(key=lambda d: (-d[0], d[1]))
    return out


def match_boards(lidar: list[LidarBoard], cams: list[CameraBoard],
                 gate: float = 0.35, gate_frac: float = 0.05, cloud=None, prior=None,
                 info: dict | None = None, check=None):
    """Which LiDAR board is which camera board, and the extrinsic that says so.

    Every (LiDAR, camera) pair seeds a one-board extrinsic; the seed that lines
    up the most other boards wins, ties broken by how closely. Then refit on
    all of them and reassign until the pairing stops changing.

    Identical boards can make that ambiguous. Measured with known truth: a row
    of boards all facing the same way at equal spacing, with the LiDAR missing
    the board at one end, is explained just as well by every board shifted one
    place -- and that pairing won about half the time, giving an extrinsic off
    by the board spacing with a perfectly ordinary residual. Boards that differ
    in height or angle, as in the room here, never did (0 of 2,600 pairings).

    So the runner-up is kept too. When it lines up as many boards but puts the
    camera somewhere else, the tie is broken by evidence -- the extrinsic
    loaded as `prior` if there is one, else which pairing puts the unmatched
    camera boards on actual points in `cloud` -- and `info["ambiguous"]` says
    what happened, so it can be shown rather than silently decided.

    `check(solution) -> reason` (see `Plausible`) drops explanations that put
    the camera where it cannot be -- inside space the LiDAR sees through, or
    facing a way it does not face. A camera that sees one board fits every
    LiDAR board equally well, and this is what tells them apart. If it would
    drop them all, it is ignored and `info["note"]` says so.

    Returns (solution, pairs as (lidar index, camera index, distance)).
    """
    from gui.core.solve import solve

    seeds = []
    for i in range(len(lidar)):
        for j in range(len(cams)):
            sol = solve([("seed", lidar[i].centers, cams[j].det.centers)])
            if not sol.ok:
                continue
            pairs = _assign(sol, lidar, cams, gate, gate_frac)
            if not any(p[0] == i and p[1] == j for p in pairs):
                continue
            seeds.append(((-len(pairs), sum(p[2] for p in pairs)), pairs))
    if not seeds:
        return None, []
    seeds.sort(key=lambda s: s[0])

    # Distinct explanations, best first, each refined.
    cands, seen, dropped = [], set(), []
    for key, pairs in seeds:
        sig = frozenset(p[:2] for p in pairs)
        if sig in seen:
            continue
        seen.add(sig)
        sol, ref = _refine_pairs(pairs, lidar, cams, gate, gate_frac)
        if sol is None or not sol.ok:
            continue
        why = check(sol) if check is not None else ""
        if why:
            dropped.append((len(ref), sum(p[2] for p in ref), sol, ref, why))
            continue
        cands.append((len(ref), sum(p[2] for p in ref), sol, ref))
        if len(cands) >= 6:
            break
    check_note = ""
    rescued = _rescue_beam_dropped(dropped, cands, check)
    if rescued:
        r = rescued[0]
        dropped = [d for d in dropped if d is not r]
        cands.append(r[:4])
        top = max((c[0] for c in cands[:-1]), default=0)
        check_note = (f"빔통과 검사에 걸린 대응을 되살렸습니다: 보드 {r[0]}개를 보드당 {100 * r[1] / r[0]:.1f} cm 로 "
                      f"맞춥니다 (통과한 후보는 최대 {top}개) — {r[4]}")
    if dropped and not cands:
        cands = [d[:4] for d in dropped[:6]]
        check_note = (f"⚠ 대응 후보 {len(dropped)}개가 모두 위치·방향 검사에 걸렸습니다 ({dropped[0][4]}). "
                      f"검사 없이 골랐으니 결과를 확인하세요")
    elif dropped:
        why = sorted({d[4].split(":")[0] for d in dropped})
        check_note = " · ".join(n for n in (check_note, f"대응 후보 {len(dropped)}개를 제외했습니다 ({', '.join(why)})") if n)
    if not cands:
        return None, []
    cands.sort(key=lambda c: (-c[0], c[1]))
    best = cands[0]
    # A rival must also fit plausibly well -- judged per board and against an
    # absolute floor, not only relative to the best. With boards at varied
    # angles a pairing can line up as many boards at 12-40 cm per board, which
    # is a worse answer and raised a false "ambiguous" 2 times in 30. But a
    # ratio alone is not safe either: in a symmetric row the *wrong* pairing
    # fitted three times better than the true one (1.6 vs 4.8 cm per board,
    # camera range noise decides it), and a 2x ratio dropped the truth from the
    # rivals and let the wrong answer through without a warning.
    per = lambda c: c[1] / max(c[0], 1)
    rivals = [c for c in cands[1:] if c[0] == best[0]
              and per(c) <= max(AMBIGUOUS_RESIDUAL_RATIO * per(best), AMBIGUOUS_FLOOR_M)
              and (lambda d: d[0] > AMBIGUOUS_SHIFT_M or d[1] > AMBIGUOUS_TURN_DEG)(_apart(c[2], best[2]))]

    note = ""
    if rivals:
        contenders = [best] + rivals
        if prior is not None:
            dist = [_apart(c[2], prior)[0] for c in contenders]
            k = int(np.argmin(dist))
            note = (f"대응 후보 {len(contenders)}개가 똑같이 맞습니다 — 기존 extrinsic 에 가장 가까운 것을 골랐습니다 "
                    f"(차이 {dist[k]:.2f} m, 다음 후보 {sorted(dist)[1]:.2f} m)")
            best = contenders[k]
        else:
            ev = [board_evidence(c[2], cams, {p[1] for p in c[3]}, cloud) for c in contenders]
            order = np.argsort(ev)[::-1]
            k = int(order[0])
            decisive = ev[order[0]] > 0 and ev[order[0]] >= 3.0 * max(ev[order[1]], 1.0)
            if decisive:
                note = (f"대응 후보 {len(contenders)}개가 똑같이 맞았지만, LiDAR 가 놓친 보드 자리에 실제 점이 있는 쪽을 "
                        f"골랐습니다 (점 {ev[order[0]]:.0f} 대 {ev[order[1]]:.0f})")
                best = contenders[k]
            else:
                shift = _apart(rivals[0][2], best[2])
                note = (f"⚠ 대응이 모호합니다: 보드 {best[0]}개를 똑같이 잘 맞추는 다른 대응이 있고, 둘은 카메라 위치가 "
                        f"{shift[0]:.2f} m 다릅니다. 같은 보드가 같은 간격·같은 방향으로 늘어서 있으면 생깁니다. "
                        f"보드 높이나 각도를 다르게 두거나, 기존 extrinsic 을 불러와 비교하세요.")
    if info is not None:
        info["ambiguous"] = bool(rivals)
        info["note"] = " · ".join(n for n in (check_note, note) if n)
        info["dropped"] = len(dropped)
        info["candidates"] = [(c[0], c[1]) for c in cands]
    return best[2], best[3]


@dataclass
class RoomResult:
    ok: bool
    solution: object = None
    pairs: list = field(default_factory=list)  # (lidar idx, cam idx, dist)
    lidar: list = field(default_factory=list)
    cams: list = field(default_factory=list)
    scenes: list = field(default_factory=list)
    loo: dict = field(default_factory=dict)
    image_rmse: float = 0.0
    per_board_px: dict = field(default_factory=dict)
    placed: list = field(default_factory=list)  # camera labels found by placement, not search
    search: dict = field(default_factory=dict)
    weighted: bool = False
    holdout: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    ambiguous: bool = False  # another pairing fitted as well (see match_boards)


def place_missing(res: RoomResult, cloud, single, target: Target, method: str, rounds: int = 3):
    """Camera boards the search missed, looked for where the extrinsic puts them."""
    from gui.core.roi_auto import place
    from gui.core.solve import solve

    for _ in range(rounds):
        matched = {j for _, j, _ in res.pairs}
        fresh = 0
        for j, cb in enumerate(res.cams):
            if j in matched:
                continue
            p = place(cloud, res.solution, cb.det, target, cb.label)
            if not p.ok:
                continue
            det = detect_in_box(cloud, single, p.box, target, method)
            if not det.ok:
                continue
            lb = LidarBoard(f"P{len(res.lidar)}", np.asarray(det.centers, np.float64), p.box, det,
                            source="배치")
            pred = res.solution.transform(lb.centers).mean(axis=0)
            cam_c = cb.det.centers.mean(axis=0)
            if np.linalg.norm(pred - cam_c) > _gate(cam_c, 0.35, 0.05):
                continue
            res.lidar.append(lb)
            res.pairs.append((len(res.lidar) - 1, j, float(np.linalg.norm(pred - cam_c))))
            res.placed.append(cb.label)
            fresh += 1
        if not fresh:
            break
        res.solution = solve(_scenes(res.pairs, res.lidar, res.cams))


def _noise(scenes, cams_by_label, camera, target):
    """Per-board 12x12 residual covariance, for the weighted refinement."""
    from gui.core.scene_pick import lidar_sigma, pose_covariance
    from gui.core.solve import scene_covariance

    out = {}
    for sid, _, cam_pts in scenes:
        cb = cams_by_label[sid.split("=")[0]]
        cov = pose_covariance(cb.det, camera, target)
        dist = float(np.linalg.norm(np.asarray(cam_pts).mean(axis=0)))
        out[sid] = scene_covariance(cam_pts, cov, lidar_sigma(dist))
    return out


def finalize(res: RoomResult, camera, target: Target, solver: str = "auto",
             exclude: set[str] | None = None) -> RoomResult:
    """Final fit over the matched pairs, minus any the user switched off.

    `solver` is "auto" (Kabsch or noise-weighted, whichever predicts held-out
    boards better -- in-sample error favours whichever leans on the camera,
    which is not the same as being right), "kabsch" or "weighted".
    """
    from gui.core import solve as S

    exclude = exclude or set()
    all_scenes = _scenes(res.pairs, res.lidar, res.cams)
    res.scenes = [s for s in all_scenes if s[0] not in exclude]
    res.loo, res.per_board_px, res.image_rmse, res.holdout = {}, {}, 0.0, {}
    if not res.scenes:
        res.ok = False
        return res
    by_label = {c.label: c for c in res.cams}
    noise = _noise(res.scenes, by_label, camera, target)
    plain = S.solve(res.scenes)
    weighted = S.solve(res.scenes, noise)
    res.holdout = {
        "kabsch": S.holdout_image_error(res.scenes, camera),
        "weighted": S.holdout_image_error(res.scenes, camera, noise),
    }
    if solver == "weighted":
        use_w = weighted.ok and weighted.weighted
    elif solver == "kabsch":
        use_w = False
    else:
        use_w = (weighted.ok and weighted.weighted and np.isfinite(res.holdout["weighted"])
                 and res.holdout["weighted"] < res.holdout["kabsch"])
    res.solution, res.weighted = (weighted, True) if use_w else (plain, False)
    res.loo = S.leave_one_out(res.scenes, noise if use_w else None)
    res.image_rmse, res.per_board_px = S.image_rmse(res.solution, res.scenes, camera)
    res.ok = bool(res.solution.ok)
    res.notes = [n for n in res.notes if not n.startswith("보드 ")]
    if len(res.scenes) < 3:
        res.notes.append(f"보드 {len(res.scenes)}개만 대응 — 3개 이상이어야 믿을 만합니다")
    return res


RELIABLE_BOARDS = 3  # finalize() warns below this many pairs


def reliable(res: RoomResult | None) -> bool:
    return bool(res is not None and res.ok and res.solution is not None and len(res.scenes) >= RELIABLE_BOARDS)


def match(cams: list[CameraBoard], lidar: list[LidarBoard], cloud=None, prior=None, check=None) -> RoomResult:
    """Pair the two sensors' boards; no final fit yet.

    `cloud` and `prior` (a Solution) only matter when the pairing is ambiguous;
    see `match_boards`.
    """
    res = RoomResult(False, cams=list(cams), lidar=list(lidar))
    if not cams:
        res.notes.append("카메라에서 보드를 찾지 못했습니다")
        return res
    if not lidar:
        res.notes.append("LiDAR 에서 보드를 찾지 못했습니다")
        return res
    info: dict = {}
    res.solution, res.pairs = match_boards(res.lidar, res.cams, cloud=cloud, prior=prior, info=info, check=check)
    res.ambiguous = info.get("ambiguous", False)
    if info.get("note"):
        res.notes.append(info["note"])
    if res.solution is None or not res.solution.ok:
        res.notes.append("카메라·LiDAR 보드 대응을 찾지 못했습니다")
        res.solution = None
        return res
    res.ok = True
    return res


# ============================================ where a camera can be, and faces
#
# Every board is the same, so a camera that sees one board fits any LiDAR
# board perfectly: the board alone does not say which it is. Two facts about
# the camera do. It sits on the vehicle, which the LiDAR's beams do not pass
# through -- a wrong board puts it out in the room, in air the beams cross. And
# it faces a known way (left, rear ...), relative to the vehicle's forward
# direction, which the cameras that see several boards establish.

DIRECTIONS = {"정면": 0.0, "좌전": 45.0, "좌": 90.0, "좌후": 135.0,
              "후": 180.0, "우후": -135.0, "우": -90.0, "우전": -45.0}
# Wide enough for a camera turned between two of the eight names; adjacent
# boards in a room seen from one camera position differ by far more.
DIRECTION_TOL_DEG = 50.0
# A camera position is impossible when more than BLOCKED_RAYS beams of one
# sweep pass within BLOCKED_RADIUS of it and end at least BLOCKED_MARGIN beyond.
# Measured on the debug room: true positions 0 beams, wrong ones 46-728 (the
# few wrong ones that land on the vehicle too are left to the direction).
BLOCKED_RADIUS = 0.15
BLOCKED_MARGIN = 0.25
BLOCKED_RAYS = 20
# Cameras with fewer camera boards than this are solved after the others, and
# with their direction.
WEAK_BOARDS = 3


def guess_direction(name: str) -> str:
    """A direction from a camera's name or topic: 'front_left' -> '좌전'. '' if it says none."""
    n = name.lower()
    front = "front" in n or "fwd" in n
    rear = "rear" in n or "back" in n
    left, right = "left" in n, "right" in n
    if front and left:
        return "좌전"
    if front and right:
        return "우전"
    if rear and left:
        return "좌후"
    if rear and right:
        return "우후"
    return "좌" if left else "우" if right else "정면" if front else "후" if rear else ""


def camera_centre(sol) -> np.ndarray:
    return -sol.R.T @ np.asarray(sol.t, np.float64).reshape(3)


def optical_heading(sol) -> float:
    """Azimuth of the camera's optical axis in the LiDAR's xy plane, degrees."""
    a = sol.R.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(a[1], a[0])))


def _wrap(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


class Rays:
    """The beams of one sweep, to ask whether a point is in space they cross."""

    def __init__(self, sweep: np.ndarray):
        pts = np.asarray(sweep, np.float64).reshape(-1, 3)
        r = np.linalg.norm(pts, axis=1)
        keep = r > 0.3
        self.range = r[keep]
        self.dirs = pts[keep] / self.range[:, None]

    def blocked(self, point) -> int:
        """Beams passing within BLOCKED_RADIUS of `point` and ending well past it."""
        c = np.asarray(point, np.float64)
        t = self.dirs @ c
        m = (t > 0) & (t < self.range - BLOCKED_MARGIN)
        d = np.linalg.norm(c[None] - t[m, None] * self.dirs[m], axis=1)
        return int((d < BLOCKED_RADIUS).sum())


@dataclass
class Plausible:
    """`check` for `match_boards`: why an extrinsic cannot be right, or ''."""

    rays: Rays | None = None
    forward: float | None = None  # vehicle forward, azimuth in the LiDAR frame
    direction: str = ""  # a key of DIRECTIONS

    def __call__(self, sol) -> str:
        if self.rays is not None:
            n = self.rays.blocked(camera_centre(sol))
            if n > BLOCKED_RAYS:
                return f"빔통과: 카메라 자리를 LiDAR 빔 {n}개가 지나감"
        return self.facing(sol)

    def facing(self, sol) -> str:
        """The direction half of the check alone: why `sol` faces the wrong way, or ''."""
        if self.forward is not None and self.direction in DIRECTIONS:
            rel = _wrap(optical_heading(sol) - self.forward)
            want = DIRECTIONS[self.direction]
            if abs(_wrap(rel - want)) > DIRECTION_TOL_DEG:
                return f"방향: 정면 기준 {rel:+.0f}° (지정 {self.direction} {want:+.0f}°)"
        return ""


def vehicle_forward(solved) -> float | None:
    """Vehicle forward azimuth in the LiDAR frame, from [(RoomResult, direction)].

    Each solved camera with a direction says where forward is; weighted by its
    boards, averaged on the circle.
    """
    v = np.zeros(2)
    for res, direction in solved:
        if res is None or not res.ok or direction not in DIRECTIONS:
            continue
        a = np.radians(optical_heading(res.solution) - DIRECTIONS[direction])
        v += len(res.pairs) * np.array([np.cos(a), np.sin(a)])
    if not v.any():
        return None
    return float(np.degrees(np.arctan2(v[1], v[0])))


@dataclass
class CameraJob:
    boards: list  # CameraBoard, already filtered
    camera: object  # intrinsics
    prior: object = None  # Solution
    direction: str = ""


def solve_cameras(jobs: list[CameraJob], lidar: list[LidarBoard], cloud, single, target: Target,
                  method: str = "occupancy", place: bool = True, solver: str = "auto",
                  forward: float | None = None, known_forward: float | None = None):
    """Pair and solve several cameras against one set of LiDAR boards.

    Cameras that see WEAK_BOARDS or more go first; their directions fix the
    vehicle's forward (unless `forward` is given; `known_forward`, from the
    last session, is used when none of them has a direction), and the cameras
    that see fewer are then held to theirs. Boards one camera places from its extrinsic
    are offered to the next. Returns ([(RoomResult, placed boards)] in job
    order, forward or None).
    """
    rays = Rays(single if single is not None and len(single) else cloud)
    pool = list(lidar)
    out: list = [None] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda k: -len(jobs[k].boards))
    strong = [k for k in order if len(jobs[k].boards) >= WEAK_BOARDS]
    weak = [k for k in order if len(jobs[k].boards) < WEAK_BOARDS]

    def run(k, fwd):
        j = jobs[k]
        res = match(j.boards, pool, cloud, j.prior, check=Plausible(rays, fwd, j.direction))
        if res.ok and place:
            place_missing(res, cloud, single, target, method)
        if res.ok:
            finalize(res, j.camera, target, solver, set())
        known = {id(b) for b in pool}
        placed = [b for b in res.lidar if id(b) not in known]
        pool.extend(placed)
        out[k] = (res, placed)

    for k in strong:
        run(k, forward)
    if forward is None:
        forward = vehicle_forward([(out[k][0], jobs[k].direction) for k in strong])
    if forward is None:
        forward = known_forward
    for k in weak:
        run(k, forward)
        res = out[k][0]
        if res.ok and len(res.pairs) < WEAK_BOARDS and res.ambiguous:
            res.notes.append(
                "카메라 방향을 지정하면 어느 보드인지 가려집니다." if not jobs[k].direction else
                "차량 정면을 알 수 없어 방향을 쓰지 못했습니다 — 보드 3개 이상 보이는 카메라 하나에도 방향을 지정하세요.")
    return out, forward


def calibrate(cams: list[CameraBoard], cloud: np.ndarray, single: np.ndarray | None,
              camera, target: Target, method: str = "occupancy",
              lidar: list[LidarBoard] | None = None) -> RoomResult:
    """The whole room, end to end, from camera boards and a still cloud."""
    search = {}
    if lidar is None:
        lidar, search = find_lidar_boards(cloud, single, target, method)
    res = match(cams, lidar, cloud)
    res.search = search
    if not res.ok:
        return res
    place_missing(res, cloud, single, target, method)
    return finalize(res, camera, target)


# ==================================================================== inputs


def _dedupe(xyz: np.ndarray):
    """Indices of the first occurrence of each exactly repeated point."""
    if len(xyz) == 0:
        return np.arange(0)
    _, idx = np.unique(np.ascontiguousarray(xyz, np.float32).view(np.dtype((np.void, 12))), return_index=True)
    return np.sort(idx)


def dedupe_topics(cloud, single, intensity=None):
    """Drop points that appear in more than one stacked topic.

    Not hypothetical: on one recording here the ground filter put the same
    board returns into both `ground_points` and `no_ground_points`. Each point
    then had an exact twin, so every nearest-neighbour distance came out zero --
    the point spacing read 0 mm, and everything sized from it (the occupancy
    grid, the template's per-point growth) collapsed.
    """
    keep = _dedupe(cloud)
    cloud = cloud[keep]
    intensity = None if intensity is None else intensity[keep]
    single = single[_dedupe(single)]
    return cloud, single, intensity


def load_capture(bag: str, lidar_topic, camera_topic: str, at: float = 0.5,
                 sweeps: int = 20, frames: int = 5, with_intensity: bool = False):
    """(stacked cloud, middle sweep, ring, images, sweep count) around one moment.

    One camera; `load_capture_multi` reads several at once.
    """
    cap = load_capture_multi(bag, lidar_topic, [camera_topic] if camera_topic else [], at, sweeps, frames)
    images = cap.images.get(camera_topic, [])
    if with_intensity:
        return cap.cloud, cap.single, cap.ring, images, cap.n_sweeps, cap.intensity
    return cap.cloud, cap.single, cap.ring, images, cap.n_sweeps


@dataclass
class Capture:
    """One moment of the room: the LiDAR once, every camera once."""

    cloud: np.ndarray
    single: np.ndarray
    ring: np.ndarray | None
    images: dict  # camera topic -> [BGR]
    n_sweeps: int
    intensity: np.ndarray | None
    stamp_ns: int = 0  # recording time of that moment


def load_capture_multi(bag: str, lidar_topic, camera_topics, at: float = 0.5,
                       sweeps: int = 1, frames: int = 1) -> Capture:
    """The LiDAR and every camera in `camera_topics`, around one moment of a recording.

    The room stands still, so no scene picking is needed: the middle of the
    recording is as good as any other moment, and one sweep sees the boards as
    well as twenty -- stacking does not densify a spinning LiDAR that repeats
    its own beam positions, it only averages range noise.

    `lidar_topic` may be several topics (a list, or comma-separated), stacked
    together. On this rig the merged cloud is split by a ground filter that
    sends far more than the ground to `ground_points` -- everything above about
    2 m at board range, the two elevated boards included -- so the whole scene
    is only there when both halves are read.
    """
    from gui.core.bag_reader import BagSource, accumulate_cloud, sweep_window
    from gui.core.decode import image_to_bgr

    topics = [t.strip() for t in lidar_topic.split(",")] if isinstance(lidar_topic, str) else list(lidar_topic)
    with BagSource(bag) as src:
        t = src.at_fraction(at)
        parts = [accumulate_cloud(src, tp, t, sweeps) for tp in topics if tp]
        cloud = np.vstack([p[0] for p in parts])
        n = min(p[1] for p in parts)
        single = np.vstack([p[2] for p in parts])
        # Ring numbers from different topics cannot be combined.
        ring = parts[0][4] if len(parts) == 1 else None
        intens = [p[3] for p in parts]
        intensity = np.concatenate(intens) if all(i is not None for i in intens) else None
        if len(parts) > 1:
            cloud, single, intensity = dedupe_topics(cloud, single, intensity)
        images = {}
        for cam in camera_topics:
            stamps = sorted(src.timestamps(cam))
            imgs = []
            if stamps:
                # Shifted rather than cut at either end, as `sweep_window` does
                # for the clouds: near the end of a short recording the old
                # window returned three frames when five were asked for.
                lo, hi = sweep_window(stamps, t, frames)
                for ts in stamps[lo:hi]:
                    _, msg = src.first_after(cam, ts)
                    if msg is not None:
                        imgs.append(image_to_bgr(msg))
            images[cam] = imgs
    return Capture(cloud, single, ring, images, n, intensity, int(t))


# A principal point further than this from the image centre, as a fraction of
# that dimension, means the preset was calibrated on a different camera or
# resolution. Real lenses sit within a few percent.
PRINCIPAL_POINT_TOL = 0.10


def principal_point_offset(cam, width: int, height: int) -> float:
    """Largest offset of (cx, cy) from the image centre, as a fraction of the size."""
    return max(abs(cam.cx - width / 2) / width, abs(cam.cy - height / 2) / height)


def rank_cameras(image: np.ndarray, cameras: dict, target: Target, layouts=(STOCK_LAYOUT,),
                 width: int | None = None, height: int | None = None):
    """Presets ordered by how well they explain the boards in this image.

    Wrong intrinsics show up as boards that fail the PnP residual check, or pass
    it with a larger error. More boards first, then the smaller median error.

    That is a weak test, and it has been fooled: on a 1920x1080 camera a preset
    made for a 1920x1536 one (principal point 745 px down a 1080 px image)
    ranked first at 1.1 px, because PnP bends each board's pose until its
    markers fit. The poses come out wrong in depth, the extrinsic's translation
    absorbs it, and objects at other distances than the boards land 20-25 px off.
    So presets whose principal point is far from this image's centre are left
    out before ranking; when none is left, the list is empty and the caller
    must say that this camera has no intrinsics here.

    Returns [(name, boards found, median reprojection px)].
    """
    out = []
    markers = detect_markers(image, small=True)
    for name, cam in cameras.items():
        if not cam.is_set:
            continue
        if width and height and not (0 < cam.cx < width and 0 < cam.cy < height):
            continue
        if width and height and principal_point_offset(cam, width, height) > PRINCIPAL_POINT_TOL:
            continue
        boards = detect_boards(image, cam, target, layouts, markers=markers)
        rms = float(np.median([b.det.reproj_rms for b in boards])) if boards else float("inf")
        out.append((name, len(boards), rms))
    out.sort(key=lambda r: (-r[1], r[2]))
    return out


# ================================================================= reporting


def report(res: RoomResult) -> str:
    lines = []
    if res.search:
        s = res.search
        lines.append(f"LiDAR 탐색: 평면 {s['patches']}개 → 보드 크기 {s['sized']}개 → 구멍 4개 {s['detected']}개")
    lines.append(f"카메라 보드 {len(res.cams)}개, LiDAR 보드 {len(res.lidar)}개, 대응 {len(res.pairs)}개"
                 + (f" (배치로 추가: {', '.join(res.placed)})" if res.placed else ""))
    for n in res.notes:
        lines.append("! " + n)
    if not res.ok:
        return "\n".join(lines)
    sol = res.solution
    rvec = cv2.Rodrigues(sol.R)[0].ravel()
    lines.append(f"풀이: {'가중' if res.weighted else 'Kabsch'}  "
                 f"(holdout px: Kabsch {res.holdout.get('kabsch', float('nan')):.2f}, "
                 f"가중 {res.holdout.get('weighted', float('nan')):.2f})")
    lines.append(f"RMSE {sol.rmse * 1000:.1f} mm   이미지 RMSE {res.image_rmse:.2f} px")
    lines.append(f"rvec [{rvec[0]:+.5f} {rvec[1]:+.5f} {rvec[2]:+.5f}]  t [{sol.t[0]:+.4f} {sol.t[1]:+.4f} {sol.t[2]:+.4f}]")
    lines.append("")
    lines.append(f"  {'보드':<10}{'거리':>7}{'기울기':>8}{'재투영':>8}{'RMSE':>9}{'px':>7}{'빼면 이동':>12}{'회전':>8}  비고")
    by_label = {c.label: c for c in res.cams}
    for sid, _, cam_pts in res.scenes:
        cb = by_label[sid.split("=")[0]]
        loo = res.loo.get(sid, {})
        flag = ""
        lb = next((res.lidar[i] for i, j, _ in res.pairs if res.cams[j].label == cb.label), None)
        if lb is not None and getattr(lb.det, "completed", False):
            flag += "3+1보완 "
        if lb is not None and (getattr(lb.det, "templated", False) or lb.det.method == "template"):
            flag += "모델정합 "
        if loo and (loo["shift_mm"] > 30 or loo["rotation_deg"] > 0.5):
            flag += "← 확인"
        lines.append(
            f"  {sid:<10}{cb.det.distance:6.2f}m{cb.det.tilt_deg:7.1f}°{cb.det.reproj_rms:7.2f}px"
            f"{sol.per_scene_rmse.get(sid, 0) * 1000:7.1f}mm{res.per_board_px.get(sid, 0):7.2f}"
            f"{loo.get('shift_mm', float('nan')):10.1f}mm{loo.get('rotation_deg', float('nan')):7.2f}°  {flag}")
    unmatched = [c.label for j, c in enumerate(res.cams) if j not in {p[1] for p in res.pairs}]
    if unmatched:
        lines.append(f"  LiDAR 와 대응 못 한 카메라 보드: {', '.join(unmatched)}")
    return "\n".join(lines)


def report_html(res: RoomResult, meta: dict, pose: str, overlay: str | None = None) -> str:
    """카메라 하나의 결과를 HTML 로 (Camera Intrinsic 리포트와 같은 틀 - shared.report_html)."""
    from shared.report_html import esc, kv_table, page, table

    title = f"{meta.get('vehicle') or ''} {meta.get('camera_name') or ''} — 카메라·LiDAR 캘리브레이션".strip()
    body = [f"<h1>{esc(title)}</h1>",
            f"<p class='meta'>{esc(meta.get('date', ''))} · {esc(meta.get('camera_topic', ''))}</p>",
            "<h2>입력</h2>", kv_table([(k, v) for k, v in meta.items() if isinstance(v, (str, int, float))])]
    if res.notes:
        body.append("<h2>주의</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in res.notes) + "</ul>")
    if res.ok and res.solution is not None:
        sol = res.solution
        body += ["<h2>결과</h2>", kv_table([
            ("보드 쌍", len(res.scenes)),
            ("RMSE (구멍 중심, 3D)", f"{sol.rmse * 1000:.1f} mm"),
            ("이미지 재투영 RMSE", f"{res.image_rmse:.2f} px"),
            ("풀이", "가중" if res.weighted else "Kabsch"),
            ("holdout px (Kabsch / 가중)", f"{res.holdout.get('kabsch', float('nan')):.2f} / "
                                          f"{res.holdout.get('weighted', float('nan')):.2f}"),
        ]), f"<pre>{esc(pose)}</pre>"]
        by_label = {c.label: c for c in res.cams}
        rows = []
        for sid, _, _ in res.scenes:
            cb = by_label[sid.split("=")[0]]
            loo = res.loo.get(sid, {})
            rows.append([sid, f"{cb.det.distance:.2f} m", f"{cb.det.tilt_deg:.1f}°", f"{cb.det.reproj_rms:.2f} px",
                         f"{sol.per_scene_rmse.get(sid, 0) * 1000:.1f} mm", f"{res.per_board_px.get(sid, 0):.2f} px",
                         f"{loo.get('shift_mm', float('nan')):.1f} mm", f"{loo.get('rotation_deg', float('nan')):.2f}°"])
        body += ["<h2>보드별</h2>",
                 table(["보드", "거리", "기울기", "재투영", "RMSE", "이미지 px", "빼면 이동", "빼면 회전"], rows)]
    if overlay:
        body.append(f"<h2>오버레이</h2><img src='{esc(overlay)}' style='width:100%;border:1px solid #ddd'>")
    body.append(f"<h2>상세</h2><pre>{esc(report(res))}</pre>")
    return page(title, "\n".join(body))


def draw_overlay(image: np.ndarray, res: RoomResult, camera, cloud: np.ndarray | None = None,
                 **style) -> np.ndarray:
    """Cloud projected by the result, plus each board's holes from both sensors."""
    from gui.core.room_overlay import OverlayStyle, render

    return render(image, camera, res, cloud, None, OverlayStyle(**style))[0]
