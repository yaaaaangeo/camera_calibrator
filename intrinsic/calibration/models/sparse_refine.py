"""
camera_calibrator.calibration.models.sparse_refine
=====================================================

Perspective(Brown-Conrady / Rational) 수렴 완료(P7)용 희소 Levenberg-Marquardt.

왜 따로 두는가: ``cv2.calibrateCameraExtended``는 반복마다 내부 파라미터와
모든 프레임의 외부 파라미터(프레임당 6개)를 한 덩어리 행렬로 푼다. 행렬
크기가 ``내부 + 6 x 프레임 수``라서 비용이 프레임 수의 세제곱으로 커진다.
실측(960x540 광각, 단일 스레드): 146프레임에서 반복 1회 약 2.3초. P7의
수렴 확인(5000회 x 2후보)이 Brown/Rational 적합 1회에 15~25분, 전체
파이프라인은 1시간 이상 걸렸다.

여기서는 같은 목적함수(OpenCV와 같은 재투영 오차 제곱합, 같은 파라미터화:
fx, fy, cx, cy, 왜곡 계수, 프레임별 rvec/tvec)를 Schur complement로 푼다.
프레임별 6x6 블록만 역행렬을 구하므로 반복 1회 비용이 프레임 수에 선형이다.

OpenCV의 해를 대체하는 솔버가 아니다. native 결과에서 "더 내려갈 수 있는지"만
확인하는 후보 생성기이고, 채택 여부는 ``common.complete_perspective_convergence``가
같은 관측의 RMS로만 판단한다.
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV distortion 배열에서 k1,k2,p1,p2,k3,k4,k5,k6의 위치와 고정 플래그.
_DIST_FIX_FLAGS = (
    (0, cv2.CALIB_FIX_K1),
    (1, cv2.CALIB_FIX_K2),
    (4, cv2.CALIB_FIX_K3),
    (5, cv2.CALIB_FIX_K4),
    (6, cv2.CALIB_FIX_K5),
    (7, cv2.CALIB_FIX_K6),
)

# 이 모듈이 재현하지 않는 파라미터화. 이런 플래그가 오면 None을 돌려주고,
# 호출부가 기존 OpenCV 경로로 대신 확인한다.
_UNSUPPORTED_FLAGS = (
    cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_THIN_PRISM_MODEL
    | cv2.CALIB_TILTED_MODEL
    | cv2.CALIB_FIX_S1_S2_S3_S4
    | cv2.CALIB_FIX_TAUX_TAUY
    | getattr(cv2, "CALIB_FIX_TANGENT_DIST", 0)
)

# OpenCV stdDeviationsIntrinsics 순서: fx, fy, cx, cy, k1, k2, p1, p2, k3,
# k4, k5, k6, s1..s4, tau_x, tau_y (18개).
_STD_INTRINSICS_LEN = 18

MAX_ITERATIONS = 2000
# 수용된 스텝의 상대 목적함수 감소가 이 값보다 작은 일이 연속 3번이면 수렴.
RELATIVE_COST_TOLERANCE = 1e-13
_SMALL_STEPS_TO_STOP = 3
_MAX_DAMPING = 1e12


def free_intrinsic_mask(flags: int, n_dist: int) -> np.ndarray | None:
    """[fx, fy, cx, cy, d0..d(n-1)] 중 추정 대상인 항목. 지원하지 않는 플래그면 None."""
    if flags & _UNSUPPORTED_FLAGS:
        return None
    allowed = 8 if flags & cv2.CALIB_RATIONAL_MODEL else 5
    if n_dist < allowed:
        return None
    mask = np.zeros(4 + n_dist, dtype=bool)
    mask[:4] = True
    if flags & cv2.CALIB_FIX_FOCAL_LENGTH:
        mask[0:2] = False
    if flags & cv2.CALIB_FIX_PRINCIPAL_POINT:
        mask[2:4] = False
    mask[4:4 + allowed] = True
    if flags & cv2.CALIB_ZERO_TANGENT_DIST:
        mask[4 + 2] = mask[4 + 3] = False
    for index, flag in _DIST_FIX_FLAGS:
        if index < allowed and flags & flag:
            mask[4 + index] = False
    return mask


def _camera(q: np.ndarray, n_dist: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([[q[0], 0.0, q[2]], [0.0, q[1], q[3]], [0.0, 0.0, 1.0]])
    return k, q[4:4 + n_dist].copy()


def _residual(obj, img, k, d, rvec, tvec):
    return (cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2) - img).reshape(-1)


def _cost(obj, img, q, n_dist, rvecs, tvecs) -> float:
    k, d = _camera(q, n_dist)
    total = 0.0
    for o, p, r, t in zip(obj, img, rvecs, tvecs):
        res = _residual(o, p, k, d, r, t)
        total += float(res @ res)
    return total


def _linearize(obj, img, q, n_dist, rvecs, tvecs, mask):
    k, d = _camera(q, n_dist)
    total = 0.0
    blocks = []
    for o, p, r, t in zip(obj, img, rvecs, tvecs):
        proj, jac = cv2.projectPoints(o, r, t, k, d)
        res = (proj.reshape(-1, 2) - p).reshape(-1)
        total += float(res @ res)
        # projectPoints jacobian 열: rvec(3), tvec(3), f(2), c(2), distortion(n).
        blocks.append((jac[:, :6], jac[:, 6:10 + n_dist][:, mask], res))
    return total, blocks


def _schur_step(blocks, damping: float):
    n_int = blocks[0][1].shape[1]
    reduced = np.zeros((n_int, n_int))
    rhs = np.zeros(n_int)
    per_frame = []
    for j_ext, j_int, res in blocks:
        u = j_int.T @ j_int
        v = j_ext.T @ j_ext
        w = j_int.T @ j_ext
        g_int = j_int.T @ res
        g_ext = j_ext.T @ res
        v_inv = np.linalg.inv(v + damping * np.diag(np.diag(v)))
        reduced += u + damping * np.diag(np.diag(u)) - w @ v_inv @ w.T
        rhs += -g_int + w @ v_inv @ g_ext
        per_frame.append((v_inv, w, g_ext))
    delta_int = _solve_scaled(reduced, rhs)
    delta_ext = [v_inv @ (-g_ext - w.T @ delta_int) for v_inv, w, g_ext in per_frame]
    return delta_int, delta_ext, reduced, per_frame


def _solve_scaled(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """대각 스케일링(Jacobi) 후 푼다. Rational 계수(k1..k6)는 상관이 |r|~1이라
    스케일링 없이 lstsq(rcond 기본값)로 풀면 약한 방향이 잘려 최소점 전에 멈춘다
    (실측: 합성 self-check seed 3에서 0.26253 vs OpenCV 긴 조건 0.26217px)."""
    scale = np.sqrt(np.abs(np.diag(matrix)))
    scale[scale == 0] = 1.0
    scaled = matrix / np.outer(scale, scale)
    try:
        delta = np.linalg.solve(scaled, rhs / scale)
        if not np.all(np.isfinite(delta)):
            raise np.linalg.LinAlgError("non-finite step")
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(scaled, rhs / scale, rcond=1e-15)[0]
    return delta / scale


def refine_perspective(object_points, image_points, flags, camera_matrix, distortion,
                       rvecs, tvecs, *, max_iterations: int = MAX_ITERATIONS):
    """주어진 상태(K, D, 프레임별 rvec/tvec)에서 재투영 오차 제곱합을 끝까지 줄인다.

    반환: ``cv2.calibrateCameraExtended``와 같은 모양의 튜플
    ``(rms, K, D, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view_errors)``.
    표준편차는 OpenCV와 같은 정규 근사(sigma^2 = SSE / (2N - 파라미터 수))로
    계산한다. 지원하지 않는 플래그면 None.
    """
    dist = np.asarray(distortion, dtype=np.float64).reshape(-1)
    n_dist = dist.size
    mask = free_intrinsic_mask(int(flags), n_dist)
    if mask is None:
        return None
    obj = [np.asarray(o, dtype=np.float64).reshape(-1, 3) for o in object_points]
    img = [np.asarray(i, dtype=np.float64).reshape(-1, 2) for i in image_points]
    k0 = np.asarray(camera_matrix, dtype=np.float64)
    q = np.r_[k0[0, 0], k0[1, 1], k0[0, 2], k0[1, 2], dist]
    rv = [np.asarray(r, dtype=np.float64).reshape(3).copy() for r in rvecs]
    tv = [np.asarray(t, dtype=np.float64).reshape(3).copy() for t in tvecs]

    cost, blocks = _linearize(obj, img, q, n_dist, rv, tv, mask)
    damping = 1e-3
    small_steps = 0
    for _ in range(max_iterations):
        delta_int, delta_ext, _, _ = _schur_step(blocks, damping)
        q_new = q.copy()
        q_new[mask] += delta_int
        rv_new = [r + d[:3] for r, d in zip(rv, delta_ext)]
        tv_new = [t + d[3:] for t, d in zip(tv, delta_ext)]
        cost_new = _cost(obj, img, q_new, n_dist, rv_new, tv_new)
        if np.isfinite(cost_new) and cost_new < cost:
            relative = (cost - cost_new) / cost
            q, rv, tv = q_new, rv_new, tv_new
            cost, blocks = _linearize(obj, img, q, n_dist, rv, tv, mask)
            damping = max(damping / 10.0, 1e-15)
            small_steps = small_steps + 1 if relative < RELATIVE_COST_TOLERANCE else 0
            if small_steps >= _SMALL_STEPS_TO_STOP:
                break
        else:
            damping *= 10.0
            if damping > _MAX_DAMPING:
                break

    n_points = sum(len(o) for o in obj)
    _, _, reduced, per_frame = _schur_step(blocks, 0.0)
    n_params = int(mask.sum()) + 6 * len(obj)
    dof = 2 * n_points - n_params
    sigma2 = cost / dof if dof > 0 else np.nan
    reduced_inv = np.linalg.pinv(reduced)
    var_int = np.zeros(4 + n_dist)
    var_int[mask] = np.diag(reduced_inv) * sigma2
    std_intrinsics = np.zeros((_STD_INTRINSICS_LEN, 1))
    count = min(4 + n_dist, _STD_INTRINSICS_LEN)
    std_intrinsics[:count, 0] = np.sqrt(np.maximum(var_int[:count], 0.0))
    std_extrinsics = []
    for v_inv, w, _ in per_frame:
        cov = v_inv + v_inv @ w.T @ reduced_inv @ w @ v_inv
        std_extrinsics.append(np.sqrt(np.maximum(np.diag(cov) * sigma2, 0.0)))
    std_extrinsics = np.concatenate(std_extrinsics).reshape(-1, 1)

    k_out, d_out = _camera(q, n_dist)
    per_view = []
    for o, p, r, t in zip(obj, img, rv, tv):
        res = _residual(o, p, k_out, d_out, r, t)
        per_view.append(np.sqrt(float(res @ res) / len(o)))
    return (
        float(np.sqrt(cost / n_points)),
        k_out,
        d_out.reshape(1, -1),
        tuple(r.reshape(3, 1) for r in rv),
        tuple(t.reshape(3, 1) for t in tv),
        std_intrinsics,
        std_extrinsics,
        np.asarray(per_view, dtype=np.float64).reshape(-1, 1),
    )


def reinitialize_extrinsics(object_points, image_points, camera_matrix, distortion):
    """K/D를 고정하고 프레임별 자세를 새로 구한다(OpenCV의 USE_INTRINSIC_GUESS 재시작과 같은 역할)."""
    k = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    rvecs, tvecs = [], []
    for o, p in zip(object_points, image_points):
        o = np.asarray(o, dtype=np.float64).reshape(-1, 3)
        p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        ok, r, t = cv2.solvePnP(o, p, k, d)
        if not ok:
            raise cv2.error("solvePnP failed while re-initializing extrinsics")
        r, t = cv2.solvePnPRefineLM(o, p, k, d, r, t)
        rvecs.append(r)
        tvecs.append(t)
    return rvecs, tvecs
