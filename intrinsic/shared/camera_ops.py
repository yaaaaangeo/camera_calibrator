"""Model-aware wrappers for the OpenCV calls that take camera intrinsics (shared).

Camera Intrinsic 과 CAMERA–LiDAR 가 같이 쓴다. `camera` 는 matrix(), dist(), is_fisheye,
model, fx/fy/cx/cy 를 가진 객체면 된다 - CAMERA–LiDAR 의 project.Camera, 또는 인트린직
결과를 감싼 shared.camera.KD.

Two projection models are in play, and OpenCV keeps them in separate
namespaces. `cv2.projectPoints` and friends implement the pinhole model --
plumb_bob or rational_polynomial, picked by how many coefficients they are
handed -- while `cv2.fisheye.*` implements the equidistant one.

They are not interchangeable, and the difference is not a refinement. A pinhole
lens maps a ray arriving at angle θ to radius f·tan θ, a fisheye to f·θ. Running
fisheye coefficients through `cv2.projectPoints` does not approximate the right
answer; it computes a different curve, and past 60 degrees off-axis the two
disagree by hundreds of pixels.

So every call site goes through here instead of reaching for cv2 directly: a
camera carries its own model with it, and a third model later is one more branch
in this file rather than a hunt through the package.
"""

from __future__ import annotations

import numpy as np

# Beyond this the equidistant model is not describing a lens any more. Real
# fisheyes reach 100 degrees off-axis (a 200-degree field); the fitted
# polynomial can stay well behaved far past that, and without a ceiling a point
# well behind the camera would be handed a plausible-looking pixel.
MAX_FISHEYE_THETA = np.radians(110.0)

# x, y, z in BGR, the same colours cv2.drawFrameAxes uses.
_AXIS_BGR = ((0, 0, 255), (0, 255, 0), (255, 0, 0))


def _pose(rvec, tvec):
    """rvec/tvec as the 3x1 float64 columns the fisheye calls insist on."""
    r = np.zeros((3, 1)) if rvec is None else np.asarray(rvec, np.float64).reshape(3, 1)
    t = np.zeros((3, 1)) if tvec is None else np.asarray(tvec, np.float64).reshape(3, 1)
    return r, t


def _fisheye_dist(camera) -> np.ndarray:
    """The four coefficients as a column; cv2.fisheye refuses a bare (4,)."""
    return np.asarray(camera.dist(), np.float64).reshape(4, 1)


# ----------------------------------------------------------------- projection


def project_points(camera, obj_pts, rvec=None, tvec=None) -> np.ndarray:
    """3D points -> (N, 2) pixels. No pose given means the points are already
    in camera coordinates."""
    import cv2

    pts = np.asarray(obj_pts, np.float64).reshape(-1, 1, 3)
    if not len(pts):
        return np.empty((0, 2), np.float64)
    r, t = _pose(rvec, tvec)
    if camera.is_fisheye:
        uv, _ = cv2.fisheye.projectPoints(pts, r, t, camera.matrix(), _fisheye_dist(camera))
    else:
        uv, _ = cv2.projectPoints(pts, r, t, camera.matrix(), camera.dist())
    return uv.reshape(-1, 2)


def solve_pnp(camera, obj_pts, img_pts, flags=None):
    """Pose from 2D-3D correspondences, as (ok, rvec, tvec).

    There is no `cv2.fisheye.solvePnP`. The equivalent is to undistort the image
    points into normalised camera coordinates first and then solve against an
    identity intrinsic -- which is what solvePnP does internally anyway once the
    distortion is out of the way.
    """
    import cv2

    flags = cv2.SOLVEPNP_ITERATIVE if flags is None else flags
    obj = np.asarray(obj_pts, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_pts, np.float64).reshape(-1, 1, 2)
    if camera.is_fisheye:
        undist = cv2.fisheye.undistortPoints(img, camera.matrix(), _fisheye_dist(camera))
        return cv2.solvePnP(obj, undist, np.eye(3), np.zeros(5), flags=flags)
    return cv2.solvePnP(obj, img, camera.matrix(), camera.dist(), flags=flags)


def undistort_points(camera, img_pts) -> np.ndarray:
    """Pixels -> (N, 2) normalised camera coordinates, distortion removed.

    In these coordinates a plane is imaged by a homography exactly, whatever
    the lens -- which is what grouping markers into boards leans on.
    """
    import cv2

    img = np.asarray(img_pts, np.float64).reshape(-1, 1, 2)
    if not len(img):
        return np.empty((0, 2), np.float64)
    if camera.is_fisheye:
        out = cv2.fisheye.undistortPoints(img, camera.matrix(), _fisheye_dist(camera))
    else:
        out = cv2.undistortPoints(img, camera.matrix(), camera.dist())
    return _refine_undistorted(camera, out.reshape(-1, 2), img.reshape(-1, 2))


def _refine_undistorted(camera, n: np.ndarray, px: np.ndarray, iters: int = 20, tol_px: float = 1e-3):
    """Newton steps on `project(n) == px`, starting from OpenCV's answer.

    cv2.undistortPoints inverts the lens model by a fixed handful of
    fixed-point iterations. Near the middle that converges; towards the edge of
    a strongly distorted wide lens it stops short. On the 128-degree camera here
    the corner of a marker 1700 px across the frame came back 33-50 px from
    where the undistorted image actually shows it -- the overlay then drew the
    marker beside itself. Solving the forward projection directly does not
    depend on how many iterations OpenCV chose, and works for either model.
    """
    n = n.copy()
    h = 1e-6
    for _ in range(iters):
        pts = np.column_stack([n, np.ones(len(n))])
        f0 = project_points(camera, pts)
        r = px - f0
        if not np.isfinite(r).all():
            break
        if np.abs(r).max() < tol_px:
            break
        fx = project_points(camera, np.column_stack([n[:, 0] + h, n[:, 1], np.ones(len(n))]))
        fy = project_points(camera, np.column_stack([n[:, 0], n[:, 1] + h, np.ones(len(n))]))
        J = np.stack([(fx - f0) / h, (fy - f0) / h], axis=-1)  # (N, 2, 2)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        ok = np.abs(det) > 1e-12
        step = np.zeros_like(n)
        step[ok, 0] = (J[ok, 1, 1] * r[ok, 0] - J[ok, 0, 1] * r[ok, 1]) / det[ok]
        step[ok, 1] = (-J[ok, 1, 0] * r[ok, 0] + J[ok, 0, 0] * r[ok, 1]) / det[ok]
        # Damped: a lens polynomial far out can be steep, and a full step there
        # can overshoot into the region where the model folds back.
        n += np.clip(step, -0.2, 0.2)
    return n


def undistort_image(camera, image, view: str = "native", balance: float | None = None):
    """Straighten an image with this camera's model.

    A straightened image has to be framed, and there is no neutral choice --
    correcting moves every pixel, so something is either cut off or left blank.
    `view` says which:

      "native"  the least reframing the model allows. For a pinhole that is the
                original intrinsics, black wedges and all. A fisheye has no such
                option: f·tan θ against f·θ means keeping its own focal length
                would blow the middle of the image up past the frame, so it
                falls through to "crop".
      "crop"    trim to the largest region that is all image, no blank corners.
      "full"    keep every corrected pixel, blank corners included. The edges are
                where a wrong coefficient shows, which is what this is for.
    """
    import cv2

    if view not in ("native", "crop", "full"):
        raise ValueError(f"view must be native, crop or full -- got {view!r}")

    h, w = image.shape[:2]
    K = camera.matrix()
    if camera.is_fisheye:
        D = _fisheye_dist(camera)
        if balance is None:
            balance = 1.0 if view == "full" else 0.0
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_k, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    D = camera.dist()
    if view == "native":
        return cv2.undistort(image, K, D)
    new_k, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0 if view == "full" else 0.0, (w, h))
    return cv2.undistort(image, K, D, None, new_k)


_UNDIST_CACHE: dict = {}


def undistorted(camera, image, view: str = "crop"):
    """(straightened image, its 3x3 pinhole matrix), with the remap cached.

    Unlike `undistort_image`, this also says what camera the straightened image
    now is -- a plain pinhole with no distortion -- so points can be projected
    onto it. The maps are cached per camera, size and view, because a live
    overlay redraws several times a second and building them dominates.

    `view` as in `undistort_image`: "crop" keeps only pixels that exist, "full"
    keeps every corrected pixel and shows the blank corners. A pinhole camera
    also accepts "native" (keep its own matrix).
    """
    import cv2

    h, w = image.shape[:2]
    key = (id(camera), camera.model, tuple(np.round(camera.dist(), 12)), camera.fx, camera.fy,
           camera.cx, camera.cy, w, h, view)
    hit = _UNDIST_CACHE.get(key)
    if hit is None:
        K = camera.matrix()
        if camera.is_fisheye:
            D = _fisheye_dist(camera)
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=1.0 if view == "full" else 0.0)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
        else:
            D = camera.dist()
            if view == "native":
                new_k = K.copy()
            else:
                new_k, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0 if view == "full" else 0.0, (w, h))
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, new_k, (w, h), cv2.CV_16SC2)
        if len(_UNDIST_CACHE) > 16:
            _UNDIST_CACHE.clear()
        hit = _UNDIST_CACHE[key] = (m1, m2, np.asarray(new_k, np.float64))
    m1, m2, new_k = hit
    return cv2.remap(image, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), new_k


def pinhole_like(K):
    """A distortion-free pinhole Camera with matrix K -- what an undistorted image is."""
    from gui.core.project import Camera

    return Camera(model="pinhole", fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))


def draw_axes(image, camera, rvec, tvec, length: float = 0.2, thickness: int = 2):
    """Draw the board's axes in place. `cv2.drawFrameAxes` is pinhole-only."""
    import cv2

    if not camera.is_fisheye:
        cv2.drawFrameAxes(image, camera.matrix(), camera.dist(), rvec, tvec, length, thickness)
        return image

    pts = np.array(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float64
    )
    uv = project_points(camera, pts, rvec, tvec)
    origin = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    for end, colour in zip(uv[1:], _AXIS_BGR):
        cv2.line(image, origin, (int(round(end[0])), int(round(end[1]))), colour, thickness)
    return image


# ------------------------------------------------------- model validity range


_LIMIT_CACHE: dict[tuple, float] = {}


def _fisheye_theta_d(camera, theta):
    """Distorted angle: θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)."""
    t2 = theta * theta
    return theta * (
        1.0 + camera.k1 * t2 + camera.k2 * t2**2 + camera.k3 * t2**3 + camera.k4 * t2**4
    )


def incidence_limit(camera, steps: int = 20001) -> float:
    """Largest incidence angle, in radians, the model still maps outward.

    The distortion is fitted from board photos, so it only describes the angles
    the camera actually saw. Past that it keeps returning numbers, and at some
    point those numbers start getting *smaller* -- a direction further off-axis
    is reported as landing nearer the image centre. Neither `cv2.projectPoints`
    nor `cv2.fisheye.projectPoints` checks, so directions the camera could never
    see get drawn inside the frame.

    Measured on 31CAM_HFOV54, a 55-degree pinhole lens: the curve turns at 57.8
    degrees incidence and 63-65 degrees lands back inside the image. Dropping
    everything past the turn is what stops it. Points between the real field of
    view and the turn are past the image edge anyway, and the bounds test
    removes those.

    Radial terms only. The tangential p1, p2 shift a point sideways rather than
    change how far out it lands, and are ~0.001 on these cameras; the fisheye
    model has no tangential terms at all.
    """
    key = (camera.model, np.asarray(camera.dist(), float).tobytes(), steps)
    cached = _LIMIT_CACHE.get(key)
    if cached is not None:
        return cached

    if camera.is_fisheye:
        theta = np.linspace(0.0, MAX_FISHEYE_THETA, steps)
        curve = _fisheye_theta_d(camera, theta)
        ceiling = MAX_FISHEYE_THETA
    else:
        # Pinhole works in tan θ, which is what the projection itself works in.
        # 6.0 is 80.5 degrees -- far past any lens this covers.
        r = np.linspace(0.0, 6.0, steps)
        theta = np.arctan(r)
        r2 = r * r
        d = np.asarray(camera.dist(), float)
        k1, k2, k3 = d[0], d[1], d[4]
        k4, k5, k6 = (d[5], d[6], d[7]) if len(d) >= 8 else (0.0, 0.0, 0.0)
        num = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        den = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        curve = r * num / den
        ceiling = float(theta[-1])

    falling = np.diff(curve) <= 0
    if not falling.any():
        limit = float(ceiling)
    else:
        # Fail closed. An index of 0 would mean the curve never rises at all,
        # which no real lens does -- but if it happens, clamp hard rather than
        # let everything through.
        limit = float(theta[int(np.argmax(falling))])
    _LIMIT_CACHE[key] = limit
    return limit


def radial_tan_limit(camera) -> float:
    """`incidence_limit` as tan θ, for callers comparing hypot(x, y) / z.

    Infinite for a fisheye that stays valid to 89 degrees or past it: tan runs
    away there, and such a camera has no meaningful off-axis ratio to test
    against. The angle itself is the thing to compare in that case.
    """
    theta = incidence_limit(camera)
    return float(np.tan(theta)) if theta < np.radians(89.0) else float("inf")


def field_of_view(camera, width: int, height: int) -> tuple[float, float]:
    """Horizontal and vertical field of view in degrees, for this image size.

    The pinhole figure ignores distortion, as it always has -- it is the number
    to compare against a lens datasheet. For a fisheye the distortion *is* the
    projection, so it has to be inverted: find the incidence angle whose
    distorted radius reaches the image edge.

    A fisheye whose edge falls past `incidence_limit` comes back clamped to twice
    that limit, which reads as "at least this wide". It usually means fx is wrong
    rather than that the lens really sees 220 degrees.
    """
    if not camera.is_fisheye:
        return (
            float(2 * np.degrees(np.arctan(width / 2 / camera.fx))),
            float(2 * np.degrees(np.arctan(height / 2 / camera.fy))),
        )

    limit = incidence_limit(camera)
    theta = np.linspace(0.0, limit, 4001)
    curve = _fisheye_theta_d(camera, theta)  # rising over [0, limit] by construction
    return (
        float(2 * np.degrees(np.interp(width / 2 / camera.fx, curve, theta))),
        float(2 * np.degrees(np.interp(height / 2 / camera.fy, curve, theta))),
    )
