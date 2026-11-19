"""Regressions found on 2026-11-06 rosbag data (960x540, ~150 deg wide angle).

1. Fisheye full fit failed from a bad Ideal-Pinhole seed (fx 881 vs ~434) ->
   Fisheye dropped from ranking -> Rational auto-selected.
2. One ill-posed test frame (4 corners / corners on one board row) made the
   hold-out Test RMS 27px (pose solver accepted a 181px local minimum).
3. Iterative median+3*MAD pruning peeled normal frames (threshold 0.518 ->
   0.362 -> 0.300px, 18/50 removed, fx moved 14px).
4. Motion-blurred bag frames: board-edge sharpness filter for auto selection.
5. Native OpenCV perspective fits capped at 30 iterations above 40 views.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import calibration.outlier as outlier
from calibration.image_quality import compute_board_edge_sharpness
from calibration.models import common
from calibration.models import fisheye as fe
from calibration.models.common import (
    DEFAULT_TERM_CRITERIA,
    DENSE_CONVERGENCE_MAX_VIEWS,
    LARGE_NATIVE_MAX_ITERATIONS,
    corner_spread_ratio,
    is_pose_well_posed,
    native_perspective_criteria,
    solve_pnp_for_model_robust,
)
from calibration.types import (
    CalibrationResult, CameraConfig, CameraModelType, Dataset, DetectionResult, Frame, ImageInfo,
)
from calibration.validation import split_train_test

BOARD = np.array([[x * .08, y * .08, 0] for y in range(1, 7) for x in range(1, 10)], np.float64)
K = np.array([[440., 0, 480], [0, 440, 270], [0, 0, 1]])


def _frame(fid, obj, img=None, n=None):
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    img = np.zeros((len(obj), 1, 2), np.float32) if img is None else np.asarray(img, np.float32).reshape(-1, 1, 2)
    det = DetectionResult(fid, True, corners=img, object_points=obj, num_corners=n or len(obj),
                          board_center_px=(480.0, 270.0), board_area_ratio=0.2)
    return Frame(ImageInfo(fid, f'/x/{fid}.png', 960, 540), det)


# --- well-posedness --------------------------------------------------------------

def test_well_posedness_rejects_few_and_collinear_corners():
    assert is_pose_well_posed(BOARD)
    assert not is_pose_well_posed(BOARD[:4])                        # 4 corners
    one_row = np.vstack([BOARD[:9], BOARD[9:10]])                     # a row + 1 point
    assert corner_spread_ratio(one_row) < 0.15 and not is_pose_well_posed(one_row)
    assert is_pose_well_posed(BOARD[:18])                             # two full rows


def test_split_keeps_ill_posed_frames_out_of_test():
    frames = [_frame(f'g{i}', BOARD) for i in range(12)] + [_frame('few', BOARD[:4]), _frame('row', BOARD[:10])]
    train, test = split_train_test(Dataset(frames=frames), CameraConfig(960, 540), test_ratio=0.5, seed=1)
    assert 'few' in train and 'row' in train
    assert 'few' not in test and 'row' not in test and test


# --- pose candidate selection ------------------------------------------------------

def test_wrong_first_pass_pose_is_replaced_by_lower_objective(monkeypatch):
    rv, tv = np.array([[0.1], [-0.2], [0.05]]), np.array([[-0.3], [-0.2], [1.2]])
    img, _ = cv2.projectPoints(BOARD, rv, tv, K, np.zeros(5))
    bad_r, bad_t = np.array([[2.5], [0.3], [0.1]]), np.array([[0.5], [0.4], [3.0]])
    monkeypatch.setattr(common, '_first_pass_pose', lambda *a: (True, bad_r, bad_t, 'solvePnP'))
    ok, r, t, reason = solve_pnp_for_model_robust(BOARD, img, K, np.zeros(5), CameraModelType.BROWN_CONRADY)
    assert ok and 'IPPE+LM' in reason
    np.testing.assert_allclose(r, rv, atol=1e-6)
    np.testing.assert_allclose(t, tv, atol=1e-6)


def test_good_first_pass_pose_is_kept():
    rv, tv = np.array([[0.1], [-0.2], [0.05]]), np.array([[-0.3], [-0.2], [1.2]])
    img, _ = cv2.projectPoints(BOARD, rv, tv, K, np.zeros(5))
    img = img + np.random.default_rng(0).normal(0, .2, img.shape)
    ok, _, _, reason = solve_pnp_for_model_robust(BOARD, img, K, np.zeros(5), CameraModelType.BROWN_CONRADY)
    assert ok and reason == 'solvePnP'


# --- outlier pruning -----------------------------------------------------------------

def test_outlier_threshold_is_not_lowered_after_first_pass(monkeypatch):
    rng = np.random.default_rng(3)
    errors = {f'f{i}': float(v) for i, v in enumerate(np.r_[rng.normal(.28, .05, 44), [1.56, 1.35, 1.32, 1.27, .8, .74]])}
    dataset = Dataset(frames=[_frame(fid, BOARD) for fid in errors])

    def fake_fit(ds, cam, model, fisheye_initial_guess=None):
        alive = {f.image_info.image_id for f in ds.enabled_frames}
        pfe = {k: (v * .8 if len(alive) < 50 else v) for k, v in errors.items() if k in alive}   # tighter after refit
        return CalibrationResult(model_name=model, success=True, camera_matrix=K.copy(), distortion=np.zeros((4, 1)),
                                 rms_error=float(np.sqrt(np.mean(np.square(list(pfe.values()))))), per_frame_error=pfe)

    monkeypatch.setattr(outlier, '_calibrate_by_model', fake_fit)
    first_pass, first_threshold = outlier.recommend_outliers(errors)
    _, res = outlier.recalibrate_with_outlier_pruning(dataset, CameraConfig(960, 540), CameraModelType.FISHEYE)
    # Only the first-pass outliers go; the tighter refit distribution must not lower the bar.
    assert sorted(res.removed_frame_ids) == sorted(first_pass)
    assert res.threshold_used == pytest.approx(first_threshold)
    assert len(res.removed_frame_ids) <= int(50 * outlier.MAX_OUTLIER_REMOVAL_FRACTION)


# --- fisheye seed fallback -----------------------------------------------------------

def test_fisheye_retries_without_bad_pinhole_seed(monkeypatch):
    calls = []
    real = fe._robust_fisheye_calibrate

    def flaky(obj, img, size, k_init, d_init, flags, crit, min_frames):
        calls.append(k_init is not None)
        if k_init is not None:
            raise cv2.error('InitExtrinsics')
        return real(obj, img, size, k_init, d_init, flags, crit, min_frames)

    monkeypatch.setattr(fe, '_robust_fisheye_calibrate', flaky)
    rng = np.random.default_rng(5)
    D = np.array([[-0.02], [0.003], [-0.01], [0.004]])
    frames = []
    while len(frames) < 12:
        rv = rng.normal(0, .35, 3); tv = np.array([rng.uniform(-.4, 0), rng.uniform(-.3, 0), rng.uniform(.6, 1.2)])
        uv, _ = cv2.fisheye.projectPoints(BOARD.reshape(-1, 1, 3), rv, tv, K, D)
        uv = uv.reshape(-1, 2)
        if (uv < 0).any() or (uv[:, 0] > 960).any() or (uv[:, 1] > 540).any():
            continue
        frames.append(_frame(f'f{len(frames)}', BOARD, uv))
    seed = CalibrationResult(model_name=CameraModelType.PINHOLE, success=True,
                             camera_matrix=np.array([[881., 0, 503], [0, 909, 307], [0, 0, 1]]),
                             distortion=np.zeros((5, 1)), rms_error=3.6)
    r = fe.calibrate_fisheye(Dataset(frames=frames), CameraConfig(960, 540), initial_guess=seed)
    assert calls == [True, False] and r.success
    assert r.camera_matrix[0, 0] == pytest.approx(440, rel=0.01)


# --- motion blur metric ----------------------------------------------------------------

def test_board_edge_sharpness_drops_with_motion_blur():
    img = np.full((540, 960), 255, np.uint8)
    for y in range(7):
        for x in range(10):
            if (x + y) % 2 == 0:
                img[100 + y * 40:140 + y * 40, 200 + x * 40:240 + x * 40] = 30
    corners = np.array([[240 + x * 40, 140 + y * 40] for y in range(6) for x in range(9)], np.float32)
    kernel = np.zeros((1, 15), np.float32); kernel[0, :] = 1 / 15
    sharp = compute_board_edge_sharpness(img, corners)
    blurred = compute_board_edge_sharpness(cv2.filter2D(img, -1, kernel), corners)
    assert sharp is not None and blurred is not None and blurred < 0.6 * sharp


# --- native criteria --------------------------------------------------------------------

def test_native_iterations_capped_only_for_large_view_counts():
    assert native_perspective_criteria(DENSE_CONVERGENCE_MAX_VIEWS, 0) == DEFAULT_TERM_CRITERIA
    big = native_perspective_criteria(DENSE_CONVERGENCE_MAX_VIEWS + 1, cv2.CALIB_RATIONAL_MODEL)
    assert big[1] == LARGE_NATIVE_MAX_ITERATIONS and big[2] == DEFAULT_TERM_CRITERIA[2]
    assert native_perspective_criteria(200, cv2.CALIB_FIX_ASPECT_RATIO) == DEFAULT_TERM_CRITERIA
