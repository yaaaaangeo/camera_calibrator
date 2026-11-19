"""Choosing which moments in a bag to calibrate from.

Scanning a recording is already cheap; deciding which frames to keep is the part
that was left to the eye, and it is the part repeated seven to eighteen times per
project. What follows is that decision written down.

Three stages, and they are kept apart on purpose.

A frame has to be *usable*: four markers, inside the range the distortion model
was fitted over, and a pose the markers actually support. That is a floor, not a
standard -- everything above it is a matter of degree.

It has to be *steady enough to pair*. The two sensors are read at different
instants, so a board still travelling has already been measured in two places
before the fit sees it, and no later step can undo that. What decides is how far
it moves in the gap, in millimetres, which makes a slow recording strict and a
fast one permissive without anything to configure.

And the set as a whole has to leave the fit as little error as possible. Spread
is half of that -- ten views from one spot pin down far less than three from
different ones -- and noise is the other half: the camera's hole centres come
out of one PnP pose, whose error along the line of sight grows with the square
of the distance. Both are measured the way the fit will feel them, as the
covariance of the least-squares answer, carried through to pixels on screen.

Nothing about how comfortable a frame looks enters the third stage. A sharper
copy of a pose already held adds nothing to the answer.

Kept free of Qt so it can be run against a bag from a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Only what makes a frame unusable is rejected. The scrubber's warning levels --
# 40 px markers, 12 degrees of tilt -- were tried here first and threw away 17 of
# the 18 scenes a person had chosen on this rig: at 9 m the markers are 18 px and
# the board is often held near head-on, and both still detect cleanly with under
# 1 px of reprojection error. Those numbers tell the eye "look closer", not
# "discard", and using them as a pass mark also collapsed the distance spread,
# because everything far away went out at once.
MIN_MARKERS = 4

# Kept only to score with, never to reject on.
GOOD_MARKER_PX = 40.0
GOOD_TILT_DEG = 12.0

# Reprojection error is reported in pixels, and a pixel is not a fixed amount of
# anything: the same one is 2.9 mm on the board at 4 m and 8.9 mm at 9 m. Scored
# in pixels it therefore *undoes* the marker-size term rather than adding to it
# -- on an eighteen-scene recording spanning 3.9 to 9.7 m it contributed -10% of
# the score's variance, rewarding distant frames for an error that is physically
# three times worse. Converted to millimetres on the board the sign comes right,
# but it then tracks marker size at +0.86 and says little the size term has not
# already said. So: converted, kept as a floor, and given a small weight.
#
#   mm on the board = reprojection px * distance / fx
GOOD_REPROJ_MM = 5.0
MAX_REPROJ_MM = 12.0

# Two samples belong to the same hold if the board barely moved between them.
# Generous, because a camera topic can be as slow as 1.7 Hz and the board drifts
# noticeably between frames even while someone is holding it still.
STILL_SHIFT_M = 0.06
STILL_TURN_DEG = 3.0

# How far the board may travel between the image and the sweep it gets paired
# with. The scene's instant comes off the camera clock, so the image is exact and
# the nearest sweep is up to half a sweep interval away; multiply that by how
# fast the board is moving and the answer is how far apart the two sensors think
# the same board is. Nothing downstream can recover it -- it is baked into the
# correspondence before the fit ever runs.
#
# 20 mm costs almost nothing and catches a great deal. On a 19 Hz recording the
# median frame is 6.4 mm out and the threshold keeps 211 of 264 stretches, with
# the conditioning of the chosen set moving 38.1 to 37.5. On one that came out
# at 0.7 Hz the median is 141 mm and the worst 1231 mm -- twice the spacing of
# the holes being matched -- and cutting at 20 mm *improves* the chosen set,
# 72.4 to 73.6, because what it removes is unusable.
#
# The threshold is in millimetres rather than seconds on purpose: a slow
# recording tightens itself and a fast one relaxes, with no setting to get wrong.
MAX_PAIR_MM = 20.0

# Deepest stack worth reporting. A board genuinely at rest gives an unbounded
# answer, and "41 sweeps" is not a useful thing to tell anyone -- past a handful
# the gain is gone and the risk of the board having drifted is not.
MAX_STACK = 15

# Requiring instead that the board hold still long enough to stack five sweeps
# was tried and is the wrong lever. Still moments cluster where the person
# paused, so demanding depth collapses the spread: on the 19 Hz recording the
# pool fell from 264 stretches to 33 and conditioning from 38.1 to 10.6, more
# than three times worse, in exchange for a depth most scenes have no need of.
# Stacking is per scene and optional -- `Candidate.safe_frames` says how deep
# each one may go, and a scene that can only take one sweep takes one and has no
# stacking error at all.

# Image is cut this many ways along each axis when reporting and rewarding
# coverage. Three is coarse on purpose -- it asks "did the board ever go left,
# or high", not "is the spread even".
COVER_CELLS = 3

# How much opening a new image cell, or a new band of tilt, is worth against the
# conditioning it costs. Conditioning cannot see either: two poses at opposite
# corners of the image with the same board geometry look identical to it.
COVER_BONUS = 0.25
TILT_BAND_DEG = 8.0

# The noise model `select_informed` chooses by. Corner noise is read off each
# frame's own PnP residual and only floored here. The LiDAR figure is per axis
# for one hole centre: the synthetic benchmark puts detected holes at 1-3 mm,
# and real sweeps are worse than synthetic ones, so this is set well above it.
CORNER_PX_FLOOR = 0.2
LIDAR_BASE_M = 0.010
LIDAR_PER_M = 0.0025

# Ranges the expected error is averaged over -- where projected LiDAR is used.
EVAL_RANGES_M = (5.0, 10.0, 20.0, 40.0)

# Two holds closer than this are the same pose held twice.
REPEAT_SHIFT_M = 0.25
REPEAT_TURN_DEG = 8.0


@dataclass
class Candidate:
    """One moment worth calibrating from."""

    t_ns: int
    distance: float
    tilt_deg: float
    reproj_rms: float  # pixels, as the detector reports it
    reproj_mm: float  # the same error as a length on the board
    marker_px: float
    centre: np.ndarray  # board centre in the camera frame
    normal: np.ndarray  # board normal in the camera frame, unit
    centers: np.ndarray  # the four hole centres, camera frame -- what gets fitted
    off_axis: float  # tan of the angle from the optical axis
    score: float = 0.0  # 0..1 comfort, for ranking within a pose
    hold: int = 0  # which still stretch it came from
    # How many sweeps could be stacked here before the board has moved too far.
    # Filled in by `measure_depths`, which reads it off the track.
    safe_frames: int = 1
    speed: float = 0.0  # m/s, from the frames either side
    pair_mm: float = 0.0  # how far the board moves in half a sweep, at that speed
    note: str = ""
    # Filled in by `attach_noise`: what this scene adds to the least-squares
    # normal matrix, and to its noise, both 6x6 in the camera frame.
    gram: np.ndarray | None = field(default=None, repr=False)
    noise: np.ndarray | None = field(default=None, repr=False)
    pose_cov: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Coverage:
    """What the chosen set actually spans -- the part a better picker cannot fix.

    Both recordings measured here put every board within 18% of the image height,
    because nobody held it high or low. No choice among those frames recovers the
    other 82%; only recording again does. Saying so is the useful thing the tool
    can do about it.
    """

    x_frac: float = 0.0  # of the image width
    y_frac: float = 0.0
    cells: int = 0  # of COVER_CELLS^2
    distance: tuple[float, float] = (0.0, 0.0)
    tilt_range: float = 0.0
    condition: float = 0.0  # sigma2 * sigma3 of the fitted points, see `spread`
    expected_px: float = 0.0  # by the noise model, see `expected_px`; 0 if unknown
    pair_mm: float = 0.0  # the worst of the chosen set
    safe_frames: int = 1  # the shallowest of the chosen set


@dataclass
class PickResult:
    chosen: list[Candidate] = field(default_factory=list)
    holds: list[Candidate] = field(default_factory=list)  # middle frame of every hold
    coverage: Coverage = field(default_factory=Coverage)
    n_scanned: int = 0
    n_usable: int = 0
    rejected: dict[str, int] = field(default_factory=dict)


def _board_normal(rvec) -> np.ndarray:
    """Board normal in the camera frame, pointing back towards the camera."""
    import cv2

    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    n = R[:, 2]
    return -n if n[2] > 0 else n


def reproj_mm(det, fx: float) -> float:
    """Reprojection error as a length on the board rather than in the image.

    A pixel subtends more of the board the further away the board is, so the
    pixel figure flatters distant frames exactly where they are worst. Dividing
    by the focal length and multiplying by the range undoes that.
    """
    if not fx:
        return 0.0
    return float(det.reproj_rms * det.distance / fx * 1000.0)


def usable(det, radial_limit: float | None = None, fx: float = 0.0) -> str:
    """Empty string when the frame can be used, otherwise why not.

    Deliberately permissive: a frame is only thrown out when the board could not
    be measured at all, or when it sits where the distortion model is guessing.
    Everything else is a matter of degree and belongs in `quality`.
    """
    if det is None or not getattr(det, "ok", False):
        return "검출 실패"
    if getattr(det, "n_markers", 0) < MIN_MARKERS:
        return "마커 부족"
    if radial_limit is not None and det.tvec is not None:
        t = np.asarray(det.tvec, float).reshape(3)
        if t[2] > 1e-6 and np.hypot(t[0], t[1]) / t[2] > radial_limit:
            # Past here the distortion model is extrapolating, so the board's
            # measured centre is not to be trusted -- see verify.radial_limit.
            return "왜곡 모델 범위 밖"
    # A floor, not a standard. Across two recordings the usable frames all sat
    # between 2.9 and 11.1 mm, so tightening this cuts nothing but the far
    # frames -- and those are what give the set its range of depths, which is
    # what conditions the fit. Tightening here would make the answer worse.
    if fx and reproj_mm(det, fx) > MAX_REPROJ_MM:
        return "재투영 오차 과다"
    return ""


def quality(det, fx: float = 0.0) -> float:
    """0 to 1, how comfortable a frame is. Reported; it decides nothing.

    Each term saturates at the level the scrubber calls comfortable, so a frame
    that clears all three scores 1 and further improvement buys nothing.

    It used to break ties -- which frame of a hold to keep, where to start the
    diversity search. Both are now settled on their own terms, because a frame
    being pleasant to look at is not a reason to calibrate from it. What decides
    is stillness and the spread of the set.
    """
    size = min(det.marker_px / GOOD_MARKER_PX, 1.0)
    tilt = min(det.tilt_deg / GOOD_TILT_DEG, 1.0)
    mm = reproj_mm(det, fx) if fx else 0.0
    reproj = min(GOOD_REPROJ_MM / mm, 1.0) if mm > 0 else 1.0
    return float(size * 0.5 + tilt * 0.4 + reproj * 0.1)


def to_candidate(t_ns: int, det, fx: float = 0.0) -> Candidate:
    t = np.asarray(det.tvec, float).reshape(3)
    return Candidate(
        t_ns=int(t_ns),
        distance=float(det.distance),
        tilt_deg=float(det.tilt_deg),
        reproj_rms=float(det.reproj_rms),
        reproj_mm=reproj_mm(det, fx) if fx else 0.0,
        marker_px=float(det.marker_px),
        centre=t,
        normal=_board_normal(det.rvec),
        centers=np.asarray(det.centers, float).reshape(-1, 3),
        off_axis=float(np.hypot(t[0], t[1]) / max(t[2], 1e-9)),
        score=quality(det, fx),
    )


def measure_motion(cands: list[Candidate], sweep_s: float) -> None:
    """Fill in how fast the board was moving, and what that costs at pairing time.

    Speed comes from the frames either side rather than from a fitted track: the
    board is carried by hand, so there is no motion to model, only the distance
    it happened to cover. In place, because every caller wants it on the
    candidates it already has.
    """
    order = sorted(cands, key=lambda c: c.t_ns)
    for i, c in enumerate(order):
        a, b = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
        dt = (b.t_ns - a.t_ns) / 1e9
        c.speed = float(np.linalg.norm(b.centre - a.centre) / dt) if dt > 0 else 0.0
        c.pair_mm = c.speed * sweep_s / 2 * 1000.0


def measure_depths(cands: list[Candidate], sweep_s: float,
                   limit_mm: float = MAX_PAIR_MM, cap: int = MAX_STACK) -> None:
    """Fill in how many sweeps each candidate could stack before the board moves.

    A stack of n is centred on the instant, so it spans (n-1)/2 sweep intervals
    either side and the board has travelled that much further by the outermost
    one. The same millimetres that decide whether a moment can be paired at all
    decide how deep it can go: one limit, applied twice.

    It has to be the speed, extrapolated, rather than the track read directly
    over the window -- which was tried, and cannot work. The window is a sweep or
    two wide, 53 ms on a 19 Hz LiDAR, while the scan that produced these
    candidates samples the camera a few hundred milliseconds apart to keep the
    button responsive. There is no frame inside the window to read. Measured that
    way every candidate in a 30 Hz recording came back at one sweep, not because
    the board was moving but because nothing had been sampled to say otherwise.

    Counting frames of the still stretch instead reads wrong the other way: the
    detector drops out on two frames in three, chopping a motionless board into
    stretches two samples long, and 222 of 264 stretches came out at one sweep
    while their boards were moving at 2 to 26 cm/s.

    So the speed over the sampling interval stands in for the speed over the
    window. It is an average, and it would miss a board set down and picked up
    again inside a single interval; nothing available at this sampling rate would
    catch that.
    """
    if sweep_s <= 0:
        for c in cands:
            c.safe_frames = 1
        return
    limit = limit_mm / 1000.0
    for c in cands:
        if c.speed <= 1e-6:
            c.safe_frames = cap
            continue
        half = limit / (c.speed * sweep_s)
        c.safe_frames = int(np.clip(int(half) * 2 + 1, 1, cap))


def group_holds(cands: list[Candidate]) -> list[Candidate]:
    """One frame per stretch where the board stayed put -- the middle of it.

    Without this the picker keeps returning neighbours of whatever moment scored
    best, since a board held still for two seconds gives twenty near-identical
    frames. It also drops the blurred ones taken while the board was being moved,
    because those sit alone in their own short stretch.

    The middle, not the sharpest. Step 5 stacks sweeps *centred* on whatever
    instant is chosen here, so a frame at the edge of a stretch has the stack
    reaching straight out into the motion that ended it. Measured on a 19 Hz
    recording, over the 42 stretches at least three frames long: taking the
    sharpest left a median of 3 sweeps that could safely be stacked and put 13
    stretches down to a single one, while taking the middle left a median of 11
    and none at 1. Same recording, same stretches -- only where in them the
    instant was taken.

    """
    if not cands:
        return []
    order = sorted(cands, key=lambda c: c.t_ns)
    holds: list[list[Candidate]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        shift = float(np.linalg.norm(cur.centre - prev.centre))
        turn = float(np.degrees(np.arccos(np.clip(prev.normal @ cur.normal, -1.0, 1.0))))
        if shift <= STILL_SHIFT_M and turn <= STILL_TURN_DEG:
            holds[-1].append(cur)
        else:
            holds.append([cur])

    best = []
    for i, group in enumerate(holds):
        span = (group[0].t_ns + group[-1].t_ns) / 2
        pick = min(group, key=lambda c: abs(c.t_ns - span))
        pick.hold = i
        held = (group[-1].t_ns - group[0].t_ns) / 1e9
        pick.note = f"{len(group)}프레임 {held:.1f}초 정지"
        best.append(pick)
    return best


def spread(cands: list[Candidate]) -> np.ndarray:
    """Singular values of the correspondences these candidates would contribute.

    This is not a proxy for how well conditioned the extrinsic will be; it is
    the thing itself. The fit is a rigid 3D-3D one, so the rotation is pinned
    down by how the matched points lie about their own centroid, and how they
    lie about it is what these three numbers say. The camera side alone is
    enough to compute it -- no LiDAR, no bag reading, so it can be evaluated
    once per candidate at selection time.

    The rotation that is worst determined is the one about the axis the points
    are most strung out along, and what pins *that* down is the two smaller
    values. Hence `sigma2 * sigma3` as the quantity to maximise, rather than the
    largest value or the condition number.
    """
    if not cands:
        return np.zeros(3)
    pts = np.vstack([c.centers for c in cands])
    if len(pts) < 3:
        return np.zeros(3)
    return np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)


def condition(cands: list[Candidate]) -> float:
    v = spread(cands)
    return float(v[1] * v[2]) if len(v) >= 3 else 0.0


def _bearing(c: Candidate) -> np.ndarray:
    """Where the board sits on the normalised image plane, before distortion.

    Enough to say which part of the frame it was in, and it needs no intrinsics
    and no image size -- both of which the picker would otherwise have to be
    handed just to bucket something into ninths.
    """
    z = max(c.centre[2], 1e-9)
    return np.array([c.centre[0] / z, c.centre[1] / z])


def _cells(cands: list[Candidate], n: int = COVER_CELLS) -> dict[int, tuple[int, int]]:
    """Which patch of the frame each candidate falls in, over the pool's own span.

    Relative to what the recording contains rather than to the sensor's field.
    The question the bonus needs answered is "does this open up somewhere the set
    has not been yet", and that is about the frames on offer. What the set covers
    of the *image* is a separate question, and `coverage` answers it in the
    honest units -- against the lens, where it can say 18% and mean it.
    """
    b = np.array([_bearing(c) for c in cands])
    lo, hi = b.min(axis=0), b.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    idx = np.clip(((b - lo) / span * n).astype(int), 0, n - 1)
    return {id(c): (int(i), int(j)) for c, (i, j) in zip(cands, idx)}


def select(cands: list[Candidate], n: int) -> list[Candidate]:
    """The n whose correspondences condition the fit best.

    Greedy: start from the furthest board, then repeatedly add whichever
    candidate lifts `condition` the most. Greedy rather than exhaustive because
    the choice is 15 from a few hundred, and each step only needs one small SVD.

    Two things conditioning cannot see, so they are nudged for. Where the board
    sat in the image: two poses at opposite corners with the same geometry look
    identical to it, and the corners are where the distortion model is doing its
    guessing, so a set that never goes there never finds out the model is wrong.
    And how far the board was tilted: the fit does not care, but a set at one
    tilt lets any systematic error in hole-finding be absorbed whole into the
    extrinsic instead of showing up as residual.

    Superseded by `select_informed` wherever the camera is known, and kept for
    when it is not. On simulated recordings with real PnP noise the informed
    picker left 11-14% less projection error on average, across a 900 px and an
    1840 px lens and LiDAR noise from 5 to 15 mm -- most where the recording kept
    the board near the middle, and where the lens is long.

    Measured against the nearest-neighbour spread this replaces, picking 15:
    conditioning doubled on one recording (19.4 -> 38.1) and rose 23% on the
    other (63.0 -> 77.3), with the tilt range going up on both and the image
    coverage no worse. Conditioning alone did better still on the number it
    optimises and paid for it elsewhere -- 37 degrees of tilt down to 20 on one
    recording, image coverage from six patches to four on the other.
    """
    if n >= len(cands):
        return list(cands)
    if n <= 0 or not cands:
        return []

    cell = _cells(cands)
    # Furthest first: it is the one decision no later step can undo, since range
    # is the axis a hand-held board covers least willingly.
    taken = [max(cands, key=lambda c: c.distance)]
    rest = [c for c in cands if c is not taken[0]]
    while len(taken) < n and rest:
        seen_cell = {cell[id(c)] for c in taken}
        seen_tilt = {int(c.tilt_deg // TILT_BAND_DEG) for c in taken}
        best, best_score = None, -1.0
        for c in rest:
            score = condition(taken + [c])
            if cell[id(c)] not in seen_cell:
                score *= 1 + COVER_BONUS
            if int(c.tilt_deg // TILT_BAND_DEG) not in seen_tilt:
                score *= 1 + COVER_BONUS
            if score > best_score:
                best, best_score = c, score
        taken.append(best)
        rest.remove(best)
    return sorted(taken, key=lambda c: c.t_ns)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rigid_jacobian(pts: np.ndarray) -> np.ndarray:
    """How points in the camera frame move under a small rotation and shift.

    (3N, 6), columns (rotation, translation). It is the same matrix for an error
    in the board pose and for an error in the extrinsic -- which is why a scene
    whose PnP is off drags the extrinsic by exactly that much.
    """
    return np.vstack([np.hstack([-_skew(p), np.eye(3)]) for p in pts])


def _marker_corners(det, target) -> np.ndarray | None:
    """The marker corners the PnP fitted, placed in the camera frame."""
    import cv2

    from gui.core.detect_camera import board_model

    if target is None or det.rvec is None or not det.ids:
        return None
    model, _ = board_model(target)
    obj = [model[i] for i in det.ids if i in model]
    if not obj:
        return None
    R = cv2.Rodrigues(np.asarray(det.rvec, float).reshape(3))[0]
    return (R @ np.vstack(obj).T).T + np.asarray(det.tvec, float).reshape(3)


def pose_covariance(det, camera, target=None) -> np.ndarray | None:
    """6x6 covariance of the board pose, rotation then translation, camera frame.

    The camera's four hole centres are not measured, they are read off one PnP
    pose, so the whole scene shares that pose's error. And that error is far from
    uniform: range is judged from how large the pattern looks, so it grows with
    the *square* of the distance while sideways position grows only linearly.
    On the 0.594 board with a 900 px lens at 0.3 px of corner noise that is
    about 2 cm along the line of sight at 5 m and 6 cm at 9 m -- more than the
    LiDAR side, and enough on its own to explain RMSEs of 30 mm.

    Linearised at the solved pose, with the projection differentiated
    numerically so lens distortion is included: a board in the corner of a wide
    lens is imaged smaller and is measured worse, and the pinhole formula would
    not say so.
    """
    pts = _marker_corners(det, target)
    if pts is None:
        return None
    from gui.core.camera_ops import project_points

    # The residual of the fit is the best estimate of the corner noise on hand,
    # after putting back the six degrees of freedom the fit absorbed. Floored,
    # because a perfect fit to sixteen corners is luck rather than precision.
    n = 2 * len(pts)
    sigma = max(float(det.reproj_rms) * np.sqrt(n / max(n - 6, 1)), CORNER_PX_FLOOR)

    h = 1e-4 * max(float(np.linalg.norm(pts.mean(axis=0))), 1.0)
    steps = np.vstack([pts + s * h * e for e in np.eye(3) for s in (1.0, -1.0)])
    uv = project_points(camera, steps).reshape(6, len(pts), 2)
    D = np.stack([(uv[2 * k] - uv[2 * k + 1]) / (2 * h) for k in range(3)], axis=-1)  # (N,2,3)
    J = np.einsum("nij,njk->nik", D, _rigid_jacobian(pts).reshape(len(pts), 3, 6)).reshape(-1, 6)
    H = J.T @ J
    if np.linalg.cond(H) > 1e12:
        return None
    return sigma**2 * np.linalg.inv(H)


def lidar_sigma(distance: float) -> float:
    """Per-axis error of one LiDAR hole centre, metres."""
    return LIDAR_BASE_M + LIDAR_PER_M * distance


def attach_noise(c: Candidate, det, camera, target=None) -> None:
    """Work out, once, what this scene would contribute to the fit.

    Stored as the two 6x6 matrices the least-squares covariance is assembled
    from, so choosing a set is sums and one small inverse per trial:

        cov = A^-1 B A^-1,   A = sum G^T G,   B = sum G^T Sigma G

    That is the covariance of the solver actually in use -- plain Kabsch, every
    point weighted alike -- rather than of an ideal weighted one, so a far,
    noisy scene is charged for the harm it really does.
    """
    cov = pose_covariance(det, camera, target)
    G = _rigid_jacobian(c.centers)
    M = G.T @ G
    s2 = lidar_sigma(c.distance) ** 2
    c.gram = M
    if cov is None:
        # No model of the board to hand: fall back to the pattern-size rule the
        # covariance reduces to, range error ~ d^2 and sideways ~ d.
        fx = float(getattr(camera, "fx", 0.0) or 1000.0)
        z = c.distance
        cov = np.zeros((6, 6))
        cov[3:, 3:] = np.diag([z / fx, z / fx, z * z / (fx * 0.5)]) ** 2 * CORNER_PX_FLOOR**2
    c.pose_cov = cov
    c.noise = M @ cov @ M + s2 * M


def _eval_weight(camera, size=None) -> np.ndarray:
    """6x6 turning an extrinsic covariance into mean squared pixels on screen.

    Averaged over a grid across the image at 5, 10, 20 and 40 m, because that is
    where the answer gets used: projected LiDAR, most of it well beyond the
    board. At those ranges rotation is nearly everything, and a centimetre of
    translation is worth a fraction of a pixel.
    """
    fx = float(getattr(camera, "fx", 0.0) or 1000.0)
    fy = float(getattr(camera, "fy", 0.0) or fx)
    cx = float(getattr(camera, "cx", 0.0) or 960.0)
    cy = float(getattr(camera, "cy", 0.0) or 540.0)
    w, h = size if size else (2 * cx, 2 * cy)
    W = np.zeros((6, 6))
    k = 0
    for u in np.linspace(0.1 * w, 0.9 * w, 5):
        for v in np.linspace(0.1 * h, 0.9 * h, 5):
            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            ray /= np.linalg.norm(ray)
            perp = np.eye(3) - np.outer(ray, ray)
            for d in EVAL_RANGES_M:
                A = fx / d * perp @ np.hstack([-_skew(ray * d), np.eye(3)])
                W += A.T @ A
                k += 1
    return W / k


def expected_px(cands: list[Candidate], W: np.ndarray) -> float:
    """RMS pixel error the set is expected to leave, by the noise model."""
    have = [c for c in cands if c.gram is not None]
    if not have:
        return float("inf")
    A = sum(c.gram for c in have)
    B = sum(c.noise for c in have)
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(max(np.trace(W @ Ai @ B @ Ai), 0.0)))


def _same_pose(a: Candidate, b: Candidate) -> bool:
    turn = float(np.degrees(np.arccos(np.clip(a.normal @ b.normal, -1.0, 1.0))))
    return float(np.linalg.norm(a.centre - b.centre)) < REPEAT_SHIFT_M and turn < REPEAT_TURN_DEG


def select_informed(cands: list[Candidate], n: int, W: np.ndarray) -> list[Candidate]:
    """The n that leave the smallest expected error on screen.

    Greedy on `expected_px`. Unlike `select` it knows which scenes are noisy:
    a board at 9 m spreads the points beautifully and is also measured three
    times worse along the line of sight than one at 5 m, and conditioning alone
    counted only the first half of that.

    One thing the model cannot see is kept out by hand. It treats every scene's
    error as independent, so two holds of the same pose look like twice the
    evidence; in practice they share whatever the pose got wrong. A pose already
    taken is only offered again once nothing new is left.
    """
    usable_ = [c for c in cands if c.gram is not None]
    if n <= 0 or not usable_:
        return []
    if n >= len(usable_):
        return sorted(usable_, key=lambda c: c.t_ns)
    taken: list[Candidate] = []
    A = np.zeros((6, 6))
    B = np.zeros((6, 6))
    # Tiny prior so the very first step, a single scene, can be scored at all.
    eps = 1e-9 * np.eye(6)
    rest = list(usable_)
    while len(taken) < n and rest:
        fresh = [c for c in rest if not any(_same_pose(c, t) for t in taken)] or rest
        best, best_err = None, float("inf")
        for c in fresh:
            Ai = np.linalg.inv(A + c.gram + eps)
            err = float(np.trace(W @ Ai @ (B + c.noise) @ Ai))
            if err < best_err:
                best, best_err = c, err
        taken.append(best)
        rest.remove(best)
        A += best.gram
        B += best.noise
    return sorted(taken, key=lambda c: c.t_ns)


def coverage(cands: list[Candidate], camera=None, size=None) -> Coverage:
    """What the chosen set spans, for the report next to it.

    The image fractions need the lens and the frame size, since the whole point
    of them is to be absolute -- "the board never left the middle fifth" is only
    worth saying against the actual frame. Without those they are left at zero
    and the rest still reports.
    """
    if not cands:
        return Coverage()
    dist = [c.distance for c in cands]
    tilt = [c.tilt_deg for c in cands]
    cell = _cells(cands)
    cov = Coverage(
        cells=len(set(cell.values())),
        distance=(float(min(dist)), float(max(dist))),
        tilt_range=float(np.ptp(tilt)),
        condition=condition(cands),
        pair_mm=float(max(c.pair_mm for c in cands)),
        safe_frames=int(min(c.safe_frames for c in cands)),
    )
    if camera is not None and size:
        from gui.core.camera_ops import project_points

        px = project_points(camera, np.array([c.centre for c in cands], float))
        cov.x_frac = float(np.ptp(px[:, 0]) / size[0])
        cov.y_frac = float(np.ptp(px[:, 1]) / size[1])
    return cov


def pick(samples, radial_limit: float | None = None, want: int = 12,
         camera=None, sweep_s: float = 0.0, size=None,
         max_pair_mm: float = MAX_PAIR_MM, target=None) -> PickResult:
    """Whole decision, from scanned frames to the moments worth keeping.

    In three separate stages, and deliberately so. What is unusable is thrown
    out; what remains is reduced to one instant per still stretch; and the final
    set is chosen by the error it is expected to leave. Nothing about how
    comfortable a frame looks reaches the third stage -- a sharper duplicate of a
    pose already held is worth nothing.

    `samples` is (t_ns, CameraDetection) in any order. `target` lets each
    frame's PnP covariance be worked out exactly; without it a range-squared
    rule stands in. Without `camera` there is no noise model at all, and the set
    is chosen on spread by `select`.
    """
    fx = float(getattr(camera, "fx", 0.0) or 0.0)
    res = PickResult(n_scanned=len(samples))
    good = []
    for t_ns, det in samples:
        why = usable(det, radial_limit, fx)
        if why:
            res.rejected[why] = res.rejected.get(why, 0) + 1
            continue
        cand = to_candidate(t_ns, det, fx)
        if camera is not None:
            attach_noise(cand, det, camera, target)
        good.append(cand)
    res.n_usable = len(good)
    measure_motion(good, sweep_s)
    measure_depths(good, sweep_s)
    res.holds = group_holds(good)

    # Drop the stretches where the board was still travelling far enough between
    # the image and its sweep to matter. Relaxed if it would leave too little to
    # choose from -- a recording bad enough for that has a problem no picker can
    # fix, and the report will say so.
    steady = [c for c in res.holds if c.pair_mm <= max_pair_mm]
    if sweep_s > 0 and len(steady) >= want:
        res.rejected["짝 어긋남 과다"] = len(res.holds) - len(steady)
        pool = steady
    else:
        pool = res.holds
    if camera is not None:
        W = _eval_weight(camera, size)
        res.chosen = select_informed(pool, want, W)
        res.coverage = coverage(res.chosen, camera, size)
        res.coverage.expected_px = expected_px(res.chosen, W)
    else:
        res.chosen = select(pool, want)
        res.coverage = coverage(res.chosen, camera, size)
    return res
