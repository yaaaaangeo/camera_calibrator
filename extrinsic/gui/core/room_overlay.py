"""Drawing a room result onto the photo, several ways.

One picture cannot answer every question about an extrinsic. Points coloured
by range show whether boards land on boards; edges show a pixel or two of
misalignment along silhouettes that range colour hides; a crop at 1:1 shows
whether the holes sit in the holes. So the renderer takes a style and the
verification page offers them all.

Points are splatted with numpy rather than `cv2.circle` per point: half a
million circles take seconds, a few array scatters take tens of milliseconds,
and the page redraws on every slider move.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from gui.core import camera_ops

COLOR_BY = ("range", "height", "intensity", "flat")
COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "viridis": cv2.COLORMAP_VIRIDIS,
             "hot": cv2.COLORMAP_HOT, "rainbow": cv2.COLORMAP_RAINBOW}

# One colour per board, for groups and labels. BGR.
BOARD_COLORS = [(255, 0, 255), (0, 200, 255), (255, 160, 0), (0, 255, 0), (80, 80, 255),
                (255, 255, 0), (0, 128, 255), (200, 120, 255)]


@dataclass
class OverlayStyle:
    show_points: bool = True
    color_by: str = "range"
    colormap: str = "turbo"
    point_px: int = 2
    dim: float = 0.55  # photo brightness under the points
    alpha: float = 1.0  # point opacity
    max_range: float = 15.0
    max_points: int = 600_000
    only_boards: bool = False  # keep only points inside the matched boards' boxes
    show_holes: bool = True
    show_markers: bool = True
    show_labels: bool = True
    hide_occluded: bool = True  # drop points the camera cannot see (behind boards, through holes)
    edges: bool = False  # compare image edges against LiDAR depth edges
    edge_jump: float = 0.3  # metres of range step that counts as a LiDAR edge
    # Draw on the straightened image instead: the image goes through the lens
    # model and the points do not, rather than the other way round. The two are
    # the same comparison -- any misalignment shows equally in both -- but
    # straight walls and pillars stay straight, which is easier to judge by eye.
    undistort: bool = False
    undistort_view: str = "crop"  # "crop" (no blank corners) or "full" (whole field)
    side_by_side: bool = False  # original and undistorted next to each other


def nudged(sol, delta):
    """A copy of `sol` turned and shifted by (rx, ry, rz deg, tx, ty, tz mm), camera frame.

    For seeing how far the picture moves per degree or centimetre -- a quick
    feel for how well the result is pinned down, without refitting anything.
    """
    if sol is None or delta is None or not np.any(delta):
        return sol
    from copy import copy

    d = np.asarray(delta, float)
    dR = cv2.Rodrigues(np.radians(d[:3]))[0]
    out = copy(sol)
    out.R = dR @ sol.R
    out.t = dR @ sol.t + d[3:] / 1000.0
    return out


def splat(img: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int, alpha: float = 1.0):
    """Discs of `radius` px at each uv, later rows drawn over earlier ones."""
    h, w = img.shape[:2]
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius + radius]
    u, v = uv[:, 0].astype(np.int64), uv[:, 1].astype(np.int64)
    if alpha >= 0.999:
        for dx, dy in offs:
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = colors[ok]
        return img
    layer = img.copy()
    mask = np.zeros((h, w), bool)
    for dx, dy in offs:
        uu, vv = u + dx, v + dy
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        layer[vv[ok], uu[ok]] = colors[ok]
        mask[vv[ok], uu[ok]] = True
    img[mask] = (img[mask] * (1 - alpha) + layer[mask] * alpha).astype(np.uint8)
    return img


def _values(xyz_l, pc, intensity, by):
    if by == "height":
        return xyz_l[:, 2]
    if by == "intensity" and intensity is not None:
        return intensity.astype(np.float64)
    return np.linalg.norm(pc, axis=1)


def _depth_edges(uv, rng, shape, jump, cell=4):
    """Which projected points sit on a range discontinuity in front of what is behind.

    A coarse min-range raster, then each point compared with the farthest
    neighbour cell: a foreground silhouette has empty-or-far cells next to it.
    """
    from scipy import ndimage

    h, w = shape[:2]
    H, W = h // cell + 1, w // cell + 1
    cu, cv_ = (uv[:, 0] // cell).astype(int), (uv[:, 1] // cell).astype(int)
    near = np.full((H, W), np.inf)
    np.minimum.at(near, (cv_, cu), rng)
    filled = np.where(np.isfinite(near), near, 0.0)
    far = ndimage.maximum_filter(filled, size=3)
    occupied = np.isfinite(near)
    # An empty neighbour cell next to an occupied one is a silhouette too, but
    # only where points are dense enough that empty means "nothing there".
    return (far[cv_, cu] - rng > jump) & occupied[cv_, cu]


def visible_mask(uv: np.ndarray, rng: np.ndarray, shape, cell: int = 4, slack: float = 0.25) -> np.ndarray:
    """Points nearest the camera in their neighbourhood -- a coarse z-buffer.

    The LiDAR sees from somewhere else. Through a board's holes, and past its
    edges, it measures the wall behind; from the camera, two metres further
    forward, those wall points sit behind the board and would be hidden. Drawn
    anyway they land on the panel as blotches beside the holes, which reads as
    misalignment when it is only parallax.
    """
    from scipy import ndimage

    h, w = shape[:2]
    H, W = h // cell + 1, w // cell + 1
    cu, cv_ = (uv[:, 0] // cell).astype(int), (uv[:, 1] // cell).astype(int)
    near = np.full((H, W), np.inf)
    np.minimum.at(near, (cv_, cu), rng)
    near = ndimage.minimum_filter(near, size=3)
    return rng <= near[cv_, cu] * (1.0 + 0.02) + slack


def edge_alignment(image: np.ndarray, camera, sol, cloud: np.ndarray, max_range: float = 12.0,
                   search: bool = True, max_points: int = 400_000):
    """How well LiDAR depth edges sit on image edges, and whether a nudge helps.

    Returns (score px, best delta, best score px). The score is the mean
    distance from each projected LiDAR silhouette point to the nearest image
    edge, clipped at 15 px. If the best nudge found is a few hundredths of a
    degree and millimetres, the extrinsic already sits where the picture says
    it should; a large one with a clearly lower score points at a real error.
    """
    h, w = image.shape[:2]
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 140)
    dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5)
    xyz = np.asarray(cloud, np.float64)
    xyz = xyz[np.linalg.norm(xyz, axis=1) < max_range]
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    pc = sol.transform(xyz)
    f = pc[:, 2] > 0.3
    xyz, pc = xyz[f], pc[f]
    uv = camera_ops.project_points(camera, pc)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    xyz, pc, uv = xyz[ok], pc[ok], uv[ok]
    E = xyz[_depth_edges(uv, np.linalg.norm(pc, axis=1), image.shape, 0.3)]
    if len(E) < 50:
        return float("nan"), np.zeros(6), float("nan")

    def score(delta):
        q = nudged(sol, delta).transform(E)
        u = camera_ops.project_points(camera, q)
        ok = np.isfinite(u).all(1) & (u[:, 0] >= 0) & (u[:, 0] < w - 1) & (u[:, 1] >= 0) & (u[:, 1] < h - 1) & (q[:, 2] > 0.3)
        return float(np.mean(np.minimum(dt[u[ok, 1].astype(int), u[ok, 0].astype(int)], 15.0)))

    base = score(np.zeros(6))
    if not search:
        return base, np.zeros(6), base
    from scipy.optimize import minimize

    r = minimize(score, np.zeros(6), method="Nelder-Mead",
                 options=dict(initial_simplex=np.vstack([np.zeros(6), np.diag([0.2, 0.2, 0.2, 30, 30, 30])]),
                              maxiter=600, xatol=0.01, fatol=0.002))
    return base, np.asarray(r.x), float(r.fun)


def render(image: np.ndarray, camera, res, cloud: np.ndarray | None, intensity: np.ndarray | None,
           style: OverlayStyle, delta=None, focus: str | None = None):
    """(picture, info dict). `focus` is a camera board label to crop around.

    With `side_by_side`, the original and the undistorted overlay come back as
    one picture, original on the left.
    """
    if style.side_by_side:
        a, info = _render_one(image, camera, res, cloud, intensity, replace(style, undistort=False), delta, focus)
        b, _ = _render_one(image, camera, res, cloud, intensity, replace(style, undistort=True), delta, focus)
        h = max(a.shape[0], b.shape[0])
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h)) if a.shape[0] != h else a
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h)) if b.shape[0] != h else b
        for img, txt in ((a, "original (lens model on the points)"), (b, f"undistorted ({style.undistort_view})")):
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        sep = np.full((h, 8, 3), 255, np.uint8)
        return np.hstack([a, sep, b]), info
    return _render_one(image, camera, res, cloud, intensity, style, delta, focus)


def _render_one(image, camera, res, cloud, intensity, style: OverlayStyle, delta=None, focus=None):
    sol = nudged(getattr(res, "solution", None), delta)
    cam = camera
    base = image
    K_und = None
    if style.undistort:
        base, K_und = camera_ops.undistorted(camera, image, style.undistort_view)
        cam = camera_ops.pinhole_like(K_und)
    info = {"n_points": 0, "range": None}

    if style.edges:
        g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 140)
        out = (cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) * 0.25).astype(np.uint8)
        out[e > 0] = (255, 255, 255)
    else:
        out = (base.astype(np.float32) * style.dim).astype(np.uint8)

    if style.show_points and cloud is not None and sol is not None and getattr(sol, "ok", False) and len(cloud):
        xyz = np.asarray(cloud, np.float64)
        inten = None if intensity is None or len(intensity) != len(xyz) else np.asarray(intensity)
        keep = np.linalg.norm(xyz, axis=1) < style.max_range
        if style.only_boards and res is not None and res.pairs:
            inside = np.zeros(len(xyz), bool)
            for i, _, _ in res.pairs:
                inside |= res.lidar[i].box.mask(xyz)
            keep &= inside
        xyz = xyz[keep]
        inten = None if inten is None else inten[keep]
        if len(xyz) > style.max_points:
            pick = np.random.default_rng(0).choice(len(xyz), style.max_points, replace=False)
            xyz = xyz[pick]
            inten = None if inten is None else inten[pick]
        pc = sol.transform(xyz)
        front = pc[:, 2] > 0.3
        # Past the angle where the distortion curve turns, directions the camera
        # cannot see fold back inside the frame (radial streaks through the
        # centre). Drop them before projecting, as verify.py does. On the
        # undistorted view `cam` is flat and the limit is far out of the way.
        theta = np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2])
        front &= theta < camera_ops.incidence_limit(cam)
        xyz, pc = xyz[front], pc[front]
        inten = None if inten is None else inten[front]
        uv = camera_ops.project_points(cam, pc)
        h, w = out.shape[:2]
        ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv, pc, xyz = uv[ok], pc[ok], xyz[ok]
        inten = None if inten is None else inten[ok]
        rng = np.linalg.norm(pc, axis=1)
        if style.hide_occluded and len(uv):
            m = visible_mask(uv, rng, out.shape)
            uv, rng, xyz, pc = uv[m], rng[m], xyz[m], pc[m]
            inten = None if inten is None else inten[m]
        if style.edges:
            m = _depth_edges(uv, rng, out.shape, style.edge_jump)
            uv, rng, xyz, pc = uv[m], rng[m], xyz[m], pc[m]
            inten = None if inten is None else inten[m]
        if len(uv):
            if style.color_by == "flat":
                colors = np.tile(np.array([0, 255, 255], np.uint8), (len(uv), 1))
            else:
                val = _values(xyz, pc, inten, style.color_by)
                lo, hi = np.percentile(val, 2), np.percentile(val, 98)
                info["range"] = (float(lo), float(hi))
                t = np.clip((val - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                colors = cv2.applyColorMap(t.reshape(-1, 1), COLORMAPS.get(style.colormap, cv2.COLORMAP_TURBO)).reshape(-1, 3)
            order = np.argsort(-rng)
            splat(out, uv[order], colors[order], max(int(style.point_px), 0), style.alpha)
        info["n_points"] = int(len(uv))

    if res is not None:
        for k, cb in enumerate(res.cams):
            col = BOARD_COLORS[k % len(BOARD_COLORS)]
            if style.show_markers and cb.det.image_corners is not None:
                for pts in cb.det.image_corners:
                    p = pts if K_und is None else _undistort_px(camera, pts, K_und)
                    cv2.polylines(out, [p.astype(np.int32)], True, col, 2)
            if style.show_holes:
                for (u, v) in camera_ops.project_points(cam, cb.det.centers):
                    cv2.circle(out, (int(u), int(v)), 10, (255, 0, 255), 2)
            if style.show_labels:
                u, v = camera_ops.project_points(cam, cb.det.centers.mean(axis=0, keepdims=True))[0].astype(int)
                cv2.putText(out, cb.label, (u - 25, v - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
                cv2.putText(out, cb.label, (u - 25, v - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
        if sol is not None and getattr(sol, "ok", False) and style.show_holes:
            for i, j, _ in res.pairs:
                for (u, v) in camera_ops.project_points(cam, sol.transform(res.lidar[i].centers)):
                    cv2.drawMarker(out, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
                if style.show_labels:
                    u, v = camera_ops.project_points(cam, sol.transform(res.lidar[i].centers).mean(axis=0, keepdims=True))[0].astype(int)
                    cv2.putText(out, res.lidar[i].label, (u - 20, v + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
                    cv2.putText(out, res.lidar[i].label, (u - 20, v + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if info["range"] is not None and style.color_by != "flat":
        lo, hi = info["range"]
        unit = {"range": "m", "height": "m (z)", "intensity": ""}.get(style.color_by, "")
        _legend(out, lo, hi, unit, style.colormap)

    if focus is not None and res is not None:
        cb = next((c for c in res.cams if c.label == focus), None)
        if cb is not None:
            out = _crop_board(out, cam, cb)
    return out, info


def _undistort_px(camera, pts, K):
    """Original pixels -> pixels of the undistorted image whose matrix is K."""
    n = camera_ops.undistort_points(camera, pts)
    return n @ K[:2, :2].T + K[:2, 2]


def _legend(img, lo, hi, unit, cmap):
    h, w = img.shape[:2]
    bar_w, bar_h = min(400, w // 3), 18
    x0, y0 = 20, h - 50
    ramp = cv2.applyColorMap(np.linspace(0, 255, bar_w).astype(np.uint8).reshape(1, -1),
                             COLORMAPS.get(cmap, cv2.COLORMAP_TURBO))
    img[y0:y0 + bar_h, x0:x0 + bar_w] = np.repeat(ramp, bar_h, axis=0)
    for txt, x in ((f"{lo:.1f}", x0), (f"{hi:.1f} {unit}", x0 + bar_w - 10)):
        cv2.putText(img, txt, (x, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(img, txt, (x, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _crop_board(img, cam, cb, pad: float = 1.0):
    """The board and its surroundings, at full resolution."""
    from gui.core.detect_camera import board_model

    R = cv2.Rodrigues(np.asarray(cb.det.rvec, float).reshape(3))[0]
    t = np.asarray(cb.det.tvec, float).reshape(3)
    corners = np.array([[sx * 0.8, sy * 0.6, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    uv = camera_ops.project_points(cam, (R @ corners.T).T + t)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    size = (hi - lo) * (1 + pad)
    mid = (lo + hi) / 2
    h, w = img.shape[:2]
    x0, y0 = np.clip((mid - size / 2).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip((mid + size / 2).astype(int), 1, [w, h])
    return img[y0:y1, x0:x1].copy()


__all__ = ["OverlayStyle", "render", "nudged", "splat", "COLOR_BY", "COLORMAPS", "replace"]
