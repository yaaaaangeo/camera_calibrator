"""Measuring a scene several times over and keeping the average.

One image and one sweep per scene leaves each scene's hole centres as noisy as a
single measurement can be, and that noise -- not the hand holding the board --
is most of what is left in the extrinsic. Measured on six recordings, a board
held "still" drifts at a median 23 mm/s, which over the ~70 ms the pairing is
off by is under 2 mm. The camera's PnP range error at 5 m is ten times that.

So each scene is measured `views` times, over consecutive sweeps and the images
taken during them, and the centres are averaged. On simulated scenes that halved
the projection error; real errors are partly shared between neighbouring
frames, so expect less.

Why detect on every sweep and average the centres, instead of stacking the
sweeps and detecting once: stacking is what `detect_auto` tried, and it blurs
exactly what the ring detectors read -- and a board that moved between sweeps
still passes the shape check, in the wrong place. Separate detections fail or
succeed one by one, and a view that landed somewhere else is visible as such and
dropped.

Drift does not need a window short enough to freeze the board. Both sensors are
averaged over the same stretch, centred on the same instant, so steady drift
moves both means to the middle and cancels. What it cannot cancel is a board
that stopped and started again inside the window, and those views are what the
outlier test is for.

Kept free of Qt so it can be driven from a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

# How many views to average per scene when nothing else is said. Five sweeps is
# half a second on a 10 Hz LiDAR: long enough to average, short enough that a
# person holding a board has mostly not moved on.
DEFAULT_VIEWS = 5

# A view whose board centre sits further than this from the median of its scene
# is not the same measurement. Scaled from the spread actually seen, with a
# floor, because the camera's range error alone reaches several centimetres at
# 9 m and a fixed cut would throw out honest far views.
OUTLIER_MAD = 3.0
OUTLIER_FLOOR_M = 0.03

# A LiDAR scene counts only when most of its views found the board, and found it
# in the same place. More views means more chances for one sweep to line up by
# luck: on synthetic scenes, those found by one or two of five sweeps were off by
# 85 and 105 mm, while those found by three or more were within 5-11 mm -- the
# same trap `detect_auto` fell into, more scenes and a worse answer. The single
# view path has no second opinion to ask and is left as it was.
MAX_LIDAR_SPREAD_M = 0.03

# Noise handed to the weighted solve. The LiDAR figure is read off the views'
# own spread where there are several, floored because consecutive sweeps share
# part of their error; a single view falls back to the picker's model.
LIDAR_SIGMA_FLOOR_M = 0.003

# How far a camera range may be off for reasons averaging cannot touch, as a
# fraction of the range. A focal length 1% wrong puts every board 1% too near
# or far, in every frame alike. In simulation charging for it cut the error left
# by miscalibrated intrinsics from -9% to -34% against Kabsch on six scenes --
# and with the real detectors and correct intrinsics made it worse, 0.59 to
# 1.38 px. Which one a recording is cannot be known in advance, so it is one of
# the candidates step 6 compares on held-out scenes, not a default.
RANGE_BIAS_FRAC = 0.01


@dataclass
class SceneMeasure:
    lidar: np.ndarray | None = None  # (4, 3) averaged, LiDAR frame
    camera: np.ndarray | None = None  # (4, 3) averaged, camera frame
    views: int = 1  # asked for
    n_lidar: int = 0  # views that detected and were kept
    n_camera: int = 0
    lidar_spread_mm: float = 0.0  # RMS of kept views about their mean
    camera_spread_mm: float = 0.0
    drift_mm: float = 0.0  # how far the camera saw the board move across the window
    time_gap_ms: float = 0.0  # LiDAR scan instant minus the images' own
    speed_mm_s: float = 0.0  # how fast the board was travelling, from the camera
    motion_mm: float = 0.0  # how far it moved across that gap -- what was corrected
    reason: str = ""
    # What the weighted solve needs, kept apart so step 6 can try more than one
    # weighting without measuring again. See `covariance`.
    pose_cov: np.ndarray | None = None  # 6x6 camera pose, already divided by the views
    pose_range: float = 0.0  # metres to the board, for the range-bias term
    lidar_sigma: float = 0.0  # per axis, one averaged hole centre

    def covariance(self, range_frac: float = 0.0) -> np.ndarray | None:
        """12x12 covariance of the residual, camera centres in the order of `camera`.

        `range_frac` charges for a camera range that is off by a fixed fraction
        in every frame -- what a wrong focal length does. It helps a great deal
        when the intrinsics are wrong and costs when they are right, which is
        why it is a parameter and step 6 decides by held-out error.
        """
        from gui.core.solve import scene_covariance

        if self.pose_cov is None or self.camera is None:
            return None
        pose = self.pose_cov.copy()
        c = self.camera.mean(axis=0)
        ray = c / np.linalg.norm(c)
        pose[3:, 3:] += (range_frac * self.pose_range) ** 2 * np.outer(ray, ray)
        return scene_covariance(self.camera, pose, self.lidar_sigma)

    @property
    def ok(self) -> bool:
        return self.lidar is not None and self.camera is not None

    def note(self) -> str:
        if not self.ok or self.views <= 1:
            return self.reason
        text = (f"평균 L {self.n_lidar}/{self.views} C {self.n_camera} · "
                f"흩어짐 L {self.lidar_spread_mm:.0f} C {self.camera_spread_mm:.0f} mm")
        if self.motion_mm >= 3:
            text += f" · 시간차 {self.time_gap_ms:+.0f} ms 보정 {self.motion_mm:.0f} mm"
        if self.drift_mm >= 30:
            text += f" · 보드 이동 {self.drift_mm:.0f} mm"
        return text


def match_order(reference: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """`pts` reordered to line up with `reference`, hole for hole.

    The LiDAR detector returns holes in whatever order it found them. Four
    points give 24 orderings, so all are tried. Safe because the holes are
    24-30 cm apart and a board does not move that far in half a second.
    """
    best, best_err = pts, np.inf
    for perm in permutations(range(len(pts))):
        cand = pts[list(perm)]
        err = float(((cand - reference) ** 2).sum())
        if err < best_err:
            best, best_err = cand, err
    return best


def robust_mean(sets: list[np.ndarray]) -> tuple[np.ndarray | None, int, float]:
    """(mean, how many kept, RMS spread in mm) of already-ordered (4, 3) sets."""
    if not sets:
        return None, 0, 0.0
    arr = np.stack(sets)
    if len(arr) >= 3:
        cen = arr.mean(axis=1)
        med = np.median(cen, axis=0)
        dev = np.linalg.norm(cen - med, axis=1)
        mad = float(np.median(dev)) * 1.4826
        arr = arr[dev <= max(OUTLIER_MAD * mad, OUTLIER_FLOOR_M)]
    mean = arr.mean(axis=0)
    spread = float(np.sqrt(((arr - mean) ** 2).sum(axis=2).mean())) * 1000 if len(arr) > 1 else 0.0
    return mean, len(arr), spread


def measure_scene(src, scene, lidar_topic: str, camera_topic: str, camera, target,
                  method: str, views: int = DEFAULT_VIEWS) -> SceneMeasure:
    """Hole centres for one scene from both sensors, averaged over `views`.

    `views` of 1 is exactly the single measurement this replaces: the first
    image at or after the scene's instant, and `scene.frames` sweeps stacked
    around it.
    """
    from gui.core.bag_reader import stack_sweeps, sweep_window, sweeps_around
    from gui.core.decode import image_to_bgr
    from gui.core.detect_camera import detect as detect_camera
    from gui.core.detect_lidar import (
        apply_box, derive_params, detect as detect_lidar, point_spacing,
    )

    views = max(int(views), 1)
    per = max(int(scene.frames), 1)
    out = SceneMeasure(views=views)

    # Line the two sensors up on when they were *measured*, not when they were
    # written to the bag. The image at bag time T was taken at T - cam_lag; the
    # sweep in a message at bag time S covers S - lidar_lag onwards, so the board
    # in it was seen around S - lidar_lag + sweep/2. Setting those equal puts the
    # LiDAR window about 70 ms later than the scene's own instant on these
    # recordings, and the overlay error on a moving board says the same: it
    # tracks board speed with a slope of 49 ms.
    shift = 0
    try:
        stamps_l = sorted(src.timestamps(lidar_topic))
        sweep = int(np.median(np.diff(stamps_l))) if len(stamps_l) > 2 else 0
        shift = src.log_lag_ns(lidar_topic) - src.log_lag_ns(camera_topic) - sweep // 2
    except Exception:  # noqa: BLE001 - a bag without headers just pairs as before
        shift = 0
    t_lidar = int(scene.t_ns) + shift

    # --- camera: every image taken while the sweeps below were --------------
    # First, as the single-view path always did: it is the cheaper of the two
    # and the more likely to say a scene is unusable.
    images = []
    if views > 1:
        stamps = sorted(src.timestamps(lidar_topic))
        if stamps:
            lo, hi = sweep_window(stamps, t_lidar, views * per)
            gap = int(np.median(np.diff(stamps))) if len(stamps) > 1 else 100_000_000
            images = list(src.range(
                camera_topic, stamps[lo] - gap // 2, stamps[hi - 1] + gap // 2))
    if not images:
        ts_cam, msg = src.first_after(camera_topic, scene.t_ns)
        images = [(ts_cam, msg)] if msg is not None else []

    # Not every source can say how far behind the moment a message was written
    # -- a synthetic one has no headers to read it from -- and pairing then just
    # works as it did before.
    cam_lag = 0
    try:
        cam_lag = src.log_lag_ns(camera_topic)
    except Exception:  # noqa: BLE001
        cam_lag = 0

    dets, cam_dets, cam_times, reason = [], [], [], ""
    for ts_img, msg in images:
        det = detect_camera(image_to_bgr(msg), camera, target)
        if det.ok:
            dets.append(np.asarray(det.centers, np.float64))  # fixed order: board model
            cam_dets.append(det)
            cam_times.append(ts_img - cam_lag)
        elif not reason:
            reason = det.reason
    out.camera, out.n_camera, out.camera_spread_mm = robust_mean(dets)
    if out.camera is None:
        out.reason = f"카메라: {reason or '이미지 없음'}"
        return out
    if len(dets) > 1:
        # Sideways travel only: range is where PnP is noisy, and reporting its
        # jitter as movement would say the board moved when it did not.
        a, b = dets[0].mean(axis=0), dets[-1].mean(axis=0)
        ray = (a + b) / np.linalg.norm(a + b)
        d = b - a
        out.drift_mm = float(np.linalg.norm(d - (d @ ray) * ray)) * 1000

    # --- LiDAR: one detection per view, each view `scene.frames` sweeps -------
    chunks, intens, rings = sweeps_around(src, lidar_topic, t_lidar, views * per)
    if not chunks:
        out.reason = "LiDAR: 스윕 없음"
        return out
    inside = apply_box(chunks[len(chunks) // 2], scene.filter)
    params = derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
    params.method = method

    n_views = max(len(chunks) // per, 1)
    found, reason = [], ""
    # Middle view first, so the others are ordered against the one nearest the
    # scene's own instant.
    for k in sorted(range(n_views), key=lambda k: abs(k - (n_views - 1) / 2)):
        sl = slice(k * per, (k + 1) * per)
        cloud, _, _, _, ring = stack_sweeps(chunks[sl], intens[sl], rings[sl])
        det = detect_lidar(cloud, scene.filter, target, params, ring=ring)
        if det.ok and det.centers is not None and len(det.centers) == 4:
            c = np.asarray(det.centers, np.float64)
            found.append(c if not found else match_order(found[0], c))
        elif not reason:
            reason = det.reason
    if not found:
        out.reason = f"LiDAR: {reason or '구멍 4개를 찾지 못함'}"
        return out

    # `found` runs outwards from the middle, so its first k entries are the k
    # views closest to the scene's instant. A board being carried can travel far
    # enough across half a second to disagree with itself -- on a real recording
    # seven of fifteen scenes came back 31 to 74 mm apart -- and the window is
    # then simply too wide for that moment. Narrow it rather than lose the scene;
    # the views are detected already, so this costs nothing.
    for size in range(len(found), 0, -1):
        mean, kept, spread = robust_mean(found[:size])
        if (size == 1 or kept >= size // 2 + 1) and spread <= MAX_LIDAR_SPREAD_M * 1000:
            out.lidar, out.n_lidar, out.lidar_spread_mm = mean, kept, spread
            out.views = size
            break
    if out.lidar is None:
        out.lidar_spread_mm = robust_mean(found)[2]
        out.reason = (f"LiDAR: 측정끼리 {out.lidar_spread_mm:.0f} mm 어긋남 — "
                      "보드가 움직였거나 오검출")
        return out
    if n_views > 1 and len(found) < n_views // 2 + 1:
        # Not a window that was too wide: the board was mostly not found at all,
        # and the few hits are as likely to be luck as measurement.
        out.lidar = None
        out.reason = (f"LiDAR: {n_views}회 중 {len(found)}회만 검출 — 우연히 맞았을 수 있어 제외"
                      + (f" ({reason})" if reason else ""))
        return out

    _compensate_motion(out, src, scene, lidar_topic, camera_topic, t_lidar,
                       np.array(cam_times, np.float64), dets)
    if out.ok:
        _attach_noise(out, cam_dets[len(cam_dets) // 2], camera, target)
    return out


def board_scan_time(src, scene, lidar_topic: str, t_ns: int) -> int | None:
    """When the LiDAR actually swept the board, on the sensor's own clock.

    Compared against the camera's header stamp, which is the same clock where
    the two are synchronised -- and they are here: the difference of the two
    write delays (43 ms) matches the offset the overlay error shows against
    board speed (49 ms), which it could not if the clocks disagreed.

    A sweep takes a tenth of a second and the board occupies a few degrees of
    it, so "the sweep nearest the image" is only accurate to a twentieth of a
    second -- 12 mm at the speed these boards are carried. The points carry
    their own acquisition time, and the ones inside the scene's box are the
    board, so the moment can be read rather than assumed.
    """
    from gui.core.decode import cloud_extra, cloud_to_struct
    from gui.core.detect_lidar import region_mask

    stamp, msg = src.nearest(lidar_topic, t_ns)
    if msg is None:
        return None
    times = cloud_extra(msg, "timestamp")
    if times is None:
        return None
    rec = cloud_to_struct(msg)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    keep = region_mask(xyz, scene.filter) & np.isfinite(np.asarray(times, np.float64))
    if keep.sum() < 20:
        return None
    # Seconds on the sensor's own clock; the header says what that is in bag time.
    header = getattr(msg, "header", None)
    if header is None:
        return None
    head_ns = header.stamp.sec * 10**9 + header.stamp.nanosec
    mid = float(np.median(np.asarray(times, np.float64)[keep]))
    if abs(mid * 1e9 - head_ns) > 5 * 10**9:
        # Not the same clock (some drivers count from the sweep start)
        mid_ns = head_ns + int(mid * 1e9) if mid < 1.0 else None
        if mid_ns is None:
            return None
    else:
        mid_ns = int(mid * 1e9)
    return mid_ns


def _compensate_motion(m: SceneMeasure, src, scene, lidar_topic: str, camera_topic: str,
                       t_lidar: int, cam_times: np.ndarray, dets: list[np.ndarray]) -> None:
    """Move the camera's board to the instant the LiDAR actually saw it.

    Both sensors are right about where the board was when each looked; what is
    left after `t_lidar` is chosen is the gap between those two instants, and on
    a board being carried at a quarter of a metre per second that gap is what
    the fit sees as disagreement. The camera gives the velocity -- several
    frames of it, sideways, where it is accurate to a millimetre or two -- so
    the correction is just velocity times the gap.
    """
    if m.camera is None or len(cam_times) == 0:
        return
    try:
        t_scan = board_scan_time(src, scene, lidar_topic, t_lidar)
    except Exception:  # noqa: BLE001 - a source without per-point times
        t_scan = None
    if t_scan is None:
        return
    gap = (t_scan - float(cam_times.mean())) / 1e9
    m.time_gap_ms = gap * 1000
    if len(dets) < 2 or abs(gap) < 1e-4:
        return
    t = (cam_times - cam_times.mean()) / 1e9
    c = np.array([d.mean(axis=0) for d in dets])
    if float(np.ptp(t)) < 1e-3:
        return
    vel = np.polyfit(t, c, 1)[0]          # metres per second, camera frame
    m.speed_mm_s = float(np.linalg.norm(vel)) * 1000
    m.motion_mm = float(np.linalg.norm(vel * gap)) * 1000
    m.camera = m.camera + vel * gap


def _attach_noise(m: SceneMeasure, det, camera, target) -> None:
    from gui.core.scene_pick import lidar_sigma, pose_covariance

    pose = pose_covariance(det, camera, target)
    if pose is None:
        return
    m.pose_cov = pose / max(m.n_camera, 1)
    m.pose_range = float(np.linalg.norm(np.asarray(det.tvec, np.float64)))
    if m.n_lidar >= 2:
        # spread is the RMS 3D distance of one view from the mean
        sigma = m.lidar_spread_mm / 1000 / np.sqrt(3) / np.sqrt(m.n_lidar)
        m.lidar_sigma = max(sigma, LIDAR_SIGMA_FLOOR_M)
    else:
        m.lidar_sigma = lidar_sigma(m.pose_range)
