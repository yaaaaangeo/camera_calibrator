"""Solving for the extrinsic, and judging how much to trust it.

Both sensors give four 3D points per scene, so this is a rigid 3D-3D fit rather
than PnP -- which is why no initial guess is needed.

The two point sets carry no labels, though. Nothing marks which LiDAR hole is
the top-left one. Correspondence comes from sorting both sets the same way, and
a sort that disagrees between the two sides produces a confident, completely
wrong extrinsic. `sort_centers` is therefore the most delicate part of this
file, not the least.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sort_centers(points: np.ndarray, frame: str = "") -> np.ndarray:
    """Order four coplanar centres so both sensors agree on which is which.

    Bearings are taken in the board's own plane, about its centre, with the
    in-plane axes oriented against the sensor's line of sight. Both sensors see
    the same face of the board, so that single rule makes the two orderings run
    the same way round -- no per-frame axis swap, and no assumption about where
    the target sits relative to the sensor.

    The original does swap axes by a fixed rule, which only yields a head-on
    view when the board lies on the LiDAR's +x axis. With the board off to one
    side -- the common case, and what happens as soon as you move it around to
    get pose variety -- the four bearings collapse together and the ordering
    comes out of numerical noise.

    `frame` is accepted for readability at call sites; the rule is the same
    either way.
    """
    points = np.asarray(points, np.float64)
    if len(points) != 4:
        raise ValueError(f"expected 4 centres, got {len(points)}")

    centroid = points.mean(axis=0)
    if np.linalg.norm(centroid) < 1e-6:
        raise ValueError("target centroid sits on the sensor origin")

    rel = points - centroid
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    u, v = vt[0], vt[1]
    if np.cross(u, v) @ centroid < 0:
        v = -v
    ring = points[np.argsort(np.arctan2(rel @ v, rel @ u))]

    # Start the cycle on a long edge. The holes sit on a rectangle -- 297 x 238
    # mm on the stock board -- so "long edge first" is decidable from the points
    # alone and lands on the same edge from either sensor. What it cannot decide
    # is *which* long edge: the rectangle looks identical after a half turn.
    # solve() resolves that ambiguity, where more than one scene is available.
    if np.linalg.norm(ring[1] - ring[0]) < np.linalg.norm(ring[2] - ring[1]):
        ring = np.roll(ring, -1, axis=0)
    return ring


def is_upright(R: np.ndarray) -> bool:
    """Does this extrinsic keep the world's up direction pointing up in the image?

    The one thing a single scene cannot decide is the half turn: swapping every
    hole with the one diagonally opposite fits exactly as well and describes an
    extrinsic turned 180 degrees about the board's normal. Since the board faces
    the sensors, that normal runs roughly along the line of sight, so the turn
    barely moves the LiDAR origin -- neither the residual nor the size of the
    translation can tell the two apart.

    What the turn does do is put the picture upside down. It is a half turn about
    an axis pointing at the camera, so up becomes down. Taking the LiDAR's +z as
    roughly upward and the camera's +y as downward -- the first is how a LiDAR is
    mounted, the second is the OpenCV convention the intrinsics already follow --
    the real solution has up landing on negative y.

    Measured on eighteen scenes the two came out at -0.96 and +0.94; the sign is
    not close to the fence. It would only mislead a LiDAR mounted on its side.
    """
    return float((np.asarray(R, np.float64) @ np.array([0.0, 0.0, 1.0]))[1]) < 0.0


@dataclass
class Solution:
    ok: bool
    R: np.ndarray | None = None  # 3x3, LiDAR -> camera
    t: np.ndarray | None = None  # 3,
    rmse: float = 0.0
    per_point: np.ndarray | None = None  # residual per correspondence, metres
    scene_ids: list[str] = field(default_factory=list)
    per_scene_rmse: dict[str, float] = field(default_factory=dict)
    n_pairs: int = 0
    reason: str = ""
    weighted: bool = False  # refined by each scene's noise, see `refine_weighted`

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        if self.R is not None:
            T[:3, :3] = self.R
            T[:3, 3] = self.t
        return T

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(xyz, np.float64).T).T + self.t


def solve_rigid(lidar: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: the rotation and translation taking `lidar` points onto `camera`."""
    mu_l, mu_c = lidar.mean(axis=0), camera.mean(axis=0)
    H = (lidar - mu_l).T @ (camera - mu_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_c - R @ mu_l


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def refine_weighted(L: np.ndarray, C: np.ndarray, covs: list[np.ndarray],
                    R: np.ndarray, t: np.ndarray, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on the same pairs Kabsch fitted, each scene weighted by its noise.

    Kabsch trusts every point alike, and the points are nothing alike. A
    camera's hole centres come out of one PnP pose, whose error along the line
    of sight grows with the square of the range and is shared by all four
    holes; sideways the same centres are good to a millimetre or two. The LiDAR's
    are the reverse -- a few millimetres every way. Weighting by each scene's
    12x12 covariance lets the camera say where the board is across the image and
    the LiDAR say how far away, which is what each is actually good at.

    Measured on simulated scenes against plain Kabsch, the projection error fell
    in every regime tried -- most where the camera is the noisier sensor, which
    is where poorly fitted intrinsics put it.

    `L` and `C` are the (4N, 3) paired points; `covs` one (12, 12) per scene in
    that same order. Returns the refined (R, t).
    """
    n = len(covs)
    Si = [np.linalg.inv(S) for S in covs]
    for _ in range(iters):
        H = np.zeros((6, 6))
        g = np.zeros(6)
        q = (R @ L.T).T + t
        for i in range(n):
            qi = q[4 * i:4 * i + 4]
            J = np.vstack([np.hstack([-_skew(p), np.eye(3)]) for p in qi])
            r = (C[4 * i:4 * i + 4] - qi).ravel()
            H += J.T @ Si[i] @ J
            g += J.T @ Si[i] @ r
        try:
            xi = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        dR = cv2_rodrigues(xi[:3])
        R, t = dR @ R, dR @ t + xi[3:]
        if np.linalg.norm(xi) < 1e-10:
            break
    return R, t


def cv2_rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector, without importing OpenCV here."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    k = _skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


# A scene's face is flipped only when that at least halves the joint residual
# and takes off this much -- a wrong face costs hundreds of millimetres.
MIRROR_GAIN = 0.5
MIRROR_MIN_M = 0.02
MIRROR_MAX_SCENES = 8  # every combination is tried
# ...and only when the fit is already poor; a right one sits at 2-8 cm.
MIRROR_SUSPECT_M = 0.10


def solve(scenes: list[tuple[str, np.ndarray, np.ndarray]],
          noise: dict[str, np.ndarray] | None = None) -> Solution:
    """Fit one extrinsic to every scene at once.

    `scenes` is (id, lidar centres, camera centres); each pair gets sorted here
    so the caller does not have to think about correspondence.

    Sorting alone cannot say which hole the cycle starts at -- the choice drifts
    with board orientation, and a rule that disagrees between the two sensors
    yields a confident, badly wrong extrinsic. So all eight relabellings of the
    cycle (four rotations, each with or without a flip) are tried.

    Residual alone cannot pick the winner. The holes sit on a rectangle, which
    is symmetric under a half turn, so the correspondence that swaps every hole
    with the one diagonally opposite fits *exactly as well* while describing an
    extrinsic rotated by 180 degrees. Physics breaks the tie: the board is in
    front of the camera, so the solution that puts it there is the real one.

    `noise`, when given, maps a scene id to the 12x12 covariance of that scene's
    residual, with the camera centres in the order they were passed in. The
    Kabsch answer is then refined by `refine_weighted`. Scenes missing from it
    borrow the typical covariance of the rest.
    """
    if not scenes:
        return Solution(False, reason="scene이 없습니다")

    ids, L_sets, C_sets, C_orders = [], [], [], []
    for sid, lidar_pts, cam_pts in scenes:
        try:
            cam = np.asarray(cam_pts, np.float64)
            L_sets.append(sort_centers(np.asarray(lidar_pts, np.float64), "lidar"))
            C_sets.append(sort_centers(cam, "camera"))
            # Where each sorted camera centre came from, so a covariance given in
            # the caller's order can follow it.
            C_orders.append([int(np.argmin(np.linalg.norm(cam - p, axis=1))) for p in C_sets[-1]])
            ids.append(sid)
        except ValueError:
            continue

    if len(ids) < 1:
        return Solution(False, reason="사용할 수 있는 scene이 없습니다")

    def fit(assignment):
        L_ = np.vstack([np.roll(s_, sh, axis=0) for s_, sh in zip(L_sets, assignment)])
        C_ = np.vstack(C_sets)
        R_, t_ = solve_rigid(L_, C_)
        return L_, C_, R_, t_

    def rmse_of(assignment) -> float:
        L_, C_, R_, t_ = fit(assignment)
        return float(np.sqrt((np.linalg.norm((R_ @ L_.T).T + t_ - C_, axis=1) ** 2).mean()))

    # Each scene independently starts on one of the two long edges, and its own
    # residual cannot say which: the rectangle is unchanged by the half turn, so
    # the two fit to the same digit. Which way up the answer comes out does say
    # -- see is_upright -- and that reads off a single scene, so every scene can
    # settle its own turn before any of them are combined.
    #
    # Leaving it to a joint search instead is what this used to do, and it seeds
    # the search with an answer it has no reason to believe: one scene's turn
    # chosen, every other scene left at zero. The extrinsic fitted to that is
    # meaningless, and the scenes then agree with the meaningless one and stay
    # there. On three scenes here that converged on an assignment fitting at
    # 382 mm while the upright assignment fits the same points at 25 mm.
    assignment, undecided = [], []
    for i, (s_, c_) in enumerate(zip(L_sets, C_sets)):
        up = [sh for sh in (0, 2)
              if is_upright(solve_rigid(np.roll(s_, sh, axis=0), c_)[0])]
        if len(up) == 1:
            assignment.append(up[0])
        else:
            assignment.append(0)
            undecided.append(i)

    # The test abstains on a board lying flat, or a LiDAR mounted on its side --
    # then up says nothing about the turn. Those scenes fall back to agreeing
    # with the rest, and only those: a scene the test did settle must not be
    # talked out of it, least of all by the residual, which is the same to the
    # last digit either way and so flips on nothing but rounding.
    best_rmse = rmse_of(assignment)
    for _ in range(3):
        changed = False
        for i in undecided:
            trial = list(assignment)
            trial[i] = 2 - trial[i]
            err = rmse_of(trial)
            if err < best_rmse:
                assignment, best_rmse, changed = trial, err, True
        if not changed:
            break

    # Which face each LiDAR board was seen from comes out of sort_centers, which
    # judges it from the origin. A merged cloud in the vehicle frame has its
    # origin on the ground, not at the sensor, and a steeply tilted board can
    # then read as seen from behind: its holes come out mirrored against the
    # camera's, and no turn lines them up (385 mm where the true fit is 33 mm,
    # boards on a stack beside a bus). One board cannot say which face -- a
    # rectangle is its own mirror image -- but two or more can, so each scene
    # may flip if the joint fit then gets clearly better.
    if 2 <= len(L_sets) <= MIRROR_MAX_SCENES and best_rmse > MIRROR_SUSPECT_M:
        base_sets, base_assign = list(L_sets), list(assignment)
        flipped = []
        for s_, c_ in zip(base_sets, C_sets):
            m_ = s_[::-1].copy()  # reversed cycle, still long edge first
            up = [sh for sh in (0, 2) if is_upright(solve_rigid(np.roll(m_, sh, axis=0), c_)[0])]
            flipped.append((m_, up[0] if len(up) == 1 else 0))
        best = (best_rmse, base_sets, base_assign)
        for mask in range(1, 1 << len(base_sets)):
            L_sets[:] = [flipped[i][0] if mask >> i & 1 else base_sets[i] for i in range(len(base_sets))]
            trial = [flipped[i][1] if mask >> i & 1 else base_assign[i] for i in range(len(base_sets))]
            err = rmse_of(trial)
            if err < best[0]:
                best = (err, list(L_sets), trial)
        if best[0] < MIRROR_GAIN * best_rmse and best_rmse - best[0] > MIRROR_MIN_M:
            best_rmse, L_sets[:], assignment = best[0], best[1], best[2]
        else:
            L_sets[:] = base_sets

    L, C, R, t = fit(assignment)
    weighted = False
    if noise:
        covs = []
        for sid, order in zip(ids, C_orders):
            S = noise.get(sid)
            if S is None:
                covs.append(None)
                continue
            idx = np.array([[3 * k, 3 * k + 1, 3 * k + 2] for k in order]).ravel()
            covs.append(np.asarray(S, np.float64)[np.ix_(idx, idx)])
        known = [S for S in covs if S is not None]
        if known:
            typical = np.median(np.stack(known), axis=0)
            covs = [typical if S is None else S for S in covs]
            R_w, t_w = refine_weighted(L, C, covs, R, t)
            # A refinement that turns the answer upside down has left the basin
            # the correspondence was decided in; keep the Kabsch answer then.
            if is_upright(R_w) == is_upright(R) and np.all(np.isfinite(t_w)):
                R, t, weighted = R_w, t_w, True
    residual = np.linalg.norm((R @ L.T).T + t - C, axis=1)

    per_scene = {
        sid: float(np.sqrt((residual[i * 4 : i * 4 + 4] ** 2).mean())) for i, sid in enumerate(ids)
    }
    return Solution(
        ok=True,
        R=R,
        t=t,
        rmse=float(np.sqrt((residual**2).mean())),
        per_point=residual,
        scene_ids=ids,
        per_scene_rmse=per_scene,
        n_pairs=len(L),
        weighted=weighted,
    )


def leave_one_out(scenes: list[tuple[str, np.ndarray, np.ndarray]],
                  noise: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """Refit without each scene in turn, and see how far the answer moves.

    A scene that is quietly wrong often keeps the overall RMSE low while pulling
    the extrinsic somewhere else. Dropping it and watching the result jump finds
    that, where the residual alone does not.
    """
    if len(scenes) < 3:
        return {}

    full = solve(scenes, noise)
    if not full.ok:
        return {}

    out = {}
    for i, (sid, _, _) in enumerate(scenes):
        rest = scenes[:i] + scenes[i + 1 :]
        sub = solve(rest, noise)
        if not sub.ok:
            continue
        d_rot = np.degrees(
            np.arccos(np.clip((np.trace(full.R.T @ sub.R) - 1) / 2, -1.0, 1.0))
        )
        out[sid] = {
            "rmse_without": sub.rmse,
            "rmse_delta": sub.rmse - full.rmse,
            "shift_mm": float(np.linalg.norm(sub.t - full.t) * 1000),
            "rotation_deg": float(d_rot),
        }
    return out


def coverage(scenes: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """How much the board actually moved between scenes.

    Extrinsic rotation is only well determined when the correspondences are
    spread out; repeats of the same pose add points without adding information.
    """
    if not scenes:
        return {}
    centroids = np.array([np.asarray(c, np.float64).mean(axis=0) for _, _, c in scenes])
    normals = []
    for _, _, cam in scenes:
        pts = np.asarray(cam, np.float64)
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        n = vt[2]
        view = pts.mean(axis=0)
        normals.append(n if n @ view < 0 else -n)
    normals = np.array(normals)

    tilts = np.degrees(
        np.arccos(np.clip(np.abs(np.sum(normals * (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)), axis=1)), 0, 1))
    )
    return {
        "x_range": float(np.ptp(centroids[:, 0])),
        "y_range": float(np.ptp(centroids[:, 1])),
        "z_range": float(np.ptp(centroids[:, 2])),
        "tilt_min": float(tilts.min()),
        "tilt_max": float(tilts.max()),
        "tilt_range": float(np.ptp(tilts)),
    }


def to_fast_livo2(sol: Solution, camera, width: int = 0, height: int = 0) -> str:
    """The text format the original tool writes, so results stay drop-in."""
    lines = ["# FAST-LIVO2 calibration format"]
    if width and height:
        lines += [
            f"cam_model: {'EquiDistant' if camera.is_fisheye else 'Pinhole'}",
            f"cam_width: {width}",
            f"cam_height: {height}",
            "scale: 1.0",
        ]
    lines += [
        f"cam_fx: {camera.fx:.6f}",
        f"cam_fy: {camera.fy:.6f}",
        f"cam_cx: {camera.cx:.6f}",
        f"cam_cy: {camera.cy:.6f}",
    ]
    if camera.is_fisheye:
        # The equidistant camera in this format takes four coefficients and no
        # cam_d4, which is exactly the fisheye model -- nothing is lost here.
        lines += [
            f"cam_d0: {camera.k1:.8f}",
            f"cam_d1: {camera.k2:.8f}",
            f"cam_d2: {camera.k3:.8f}",
            f"cam_d3: {camera.k4:.8f}",
            "# 어안(등거리) 모델입니다. 읽는 쪽이 EquiDistant 카메라를 지원하는지 확인하세요 —",
            "#       Pinhole 로 읽히면 화면 주변부가 크게 틀어집니다.",
        ]
    else:
        lines += [
            f"cam_d0: {camera.k1:.8f}",
            f"cam_d1: {camera.k2:.8f}",
            f"cam_d2: {camera.p1:.8f}",
            f"cam_d3: {camera.p2:.8f}",
            f"cam_d4: {camera.k3:.8f}",
        ]
        # The pinhole form has five slots and no more. A rational calibration
        # cannot be written here without changing what it means, so say so in the
        # file rather than let the missing denominator pass for a complete camera.
        if camera.rational:
            lines += [
                f"# 주의: 이 카메라는 rational_polynomial(8계수)입니다 —"
                f" k4={camera.k4:.8f}, k5={camera.k5:.8f}, k6={camera.k6:.8f} 는",
                "#       이 형식에 담을 자리가 없어 빠졌습니다. 그대로 쓰면 화면 주변부가 틀어집니다.",
            ]
    lines += [""]
    R, t = sol.R, sol.t
    lines.append(
        "Rcl: [ {:9.6f}, {:9.6f}, {:9.6f},\n"
        "       {:9.6f}, {:9.6f}, {:9.6f},\n"
        "       {:9.6f}, {:9.6f}, {:9.6f}]".format(*R.ravel())
    )
    lines.append("Pcl: [ {:9.6f}, {:9.6f}, {:9.6f}]".format(*t))
    return "\n".join(lines) + "\n"


def scene_covariance(cam_centres: np.ndarray, pose_cov: np.ndarray | None,
                     lidar_sigma: float) -> np.ndarray:
    """12x12 covariance of one scene's residual, camera centres in the given order.

    The camera part is the PnP pose covariance carried to the four centres --
    they share that one pose, so their errors are almost perfectly correlated.
    The LiDAR part is independent per centre and per axis.
    """
    c = np.asarray(cam_centres, np.float64)
    S = (lidar_sigma ** 2) * np.eye(3 * len(c))
    if pose_cov is not None:
        G = np.vstack([np.hstack([-_skew(p), np.eye(3)]) for p in c])
        S = S + G @ np.asarray(pose_cov, np.float64) @ G.T
    return S


def image_rmse(sol: Solution, scenes, camera) -> tuple[float, dict[str, float]]:
    """RMS pixel distance between each LiDAR hole, projected, and the camera's.

    The number that matches what an overlay looks like. RMSE in millimetres is
    mostly the camera's range noise -- the board measured a few centimetres too
    far is a few centimetres of residual and no pixels at all -- so a fit can
    have a worse RMSE and a better picture. Holes are paired by proximity.
    """
    from gui.core.camera_ops import project_points

    if sol is None or not sol.ok:
        return 0.0, {}
    per, all_sq = {}, []
    for sid, lidar_pts, cam_pts in scenes:
        cam = np.asarray(cam_pts, np.float64)
        pred = sol.transform(np.asarray(lidar_pts, np.float64))
        idx = [int(np.argmin(np.linalg.norm(pred - c, axis=1))) for c in cam]
        d = np.linalg.norm(project_points(camera, pred[idx]) - project_points(camera, cam), axis=1)
        per[sid] = float(np.sqrt((d ** 2).mean()))
        all_sq.extend(d ** 2)
    return (float(np.sqrt(np.mean(all_sq))) if all_sq else 0.0), per


def holdout_image_error(scenes, camera, noise=None) -> float:
    """RMS pixel error of each scene's holes under an extrinsic fitted without it.

    What decides between ways of solving on a recording with no ground truth.
    In-sample error cannot: a weighting that leans on the camera sideways
    shrinks the in-sample pixel error by construction, and on synthetic scenes
    did so while moving the answer further from the truth. Held out, a scene
    has no say in its own fit.

    Checked against ground truth on simulated sets of 9-18 scenes, choosing by
    this never did worse than Kabsch at the median, and where the camera was
    the noisier sensor it recovered most of the best choice (4.6 -> 0.9 px).
    """
    if len(scenes) < 4:
        return float("nan")
    sq = []
    for i in range(len(scenes)):
        sol = solve(scenes[:i] + scenes[i + 1:], noise)
        if not sol.ok:
            continue
        _, per = image_rmse(sol, [scenes[i]], camera)
        sq.extend([v ** 2 for v in per.values()])
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def range_disagreement(scenes, sol) -> float:
    """How far the two sensors disagree along the camera's line of sight.

    Fitted as a fraction of the range, because that is how the camera's error
    grows: the board's distance is read from how large the pattern looks, so a
    focal length slightly wrong, or a pattern measured slightly small, moves
    every board by the same proportion. Measured on real scenes it came to
    about half a percent -- 28 mm at 5.5 m -- which is an order more than the
    LiDAR's few millimetres, and is why weighting the fit matters here.

    The alternative is to guess the fraction, and a guess that is wrong in the
    generous direction throws away the camera's good sideways information.
    """
    if sol is None or not getattr(sol, "ok", False) or len(scenes) < 3:
        return 0.0
    fr = []
    for _, lidar_pts, cam_pts in scenes:
        C = np.asarray(cam_pts, np.float64)
        L = sol.transform(np.asarray(lidar_pts, np.float64))
        idx = [int(np.argmin(np.linalg.norm(L - c, axis=1))) for c in C]
        d = (L[idx] - C).mean(axis=0)
        centre = C.mean(axis=0)
        rng = float(np.linalg.norm(centre))
        if rng > 1e-6:
            fr.append(abs(float(d @ (centre / rng))) / rng)
    return float(np.median(fr)) if fr else 0.0
