"""P7: perspective solver convergence completion.

Real-data reproductions (not stored here, user data): Train28 Rational stopped at
3.0008px with (100, 1e-6) while the same start converges to 1.6422px; a seeded
start stopped at 2.3641px and a restart from it reached 1.4336px.

Above DENSE_CONVERGENCE_MAX_VIEWS views the completion candidates are a sparse
(Schur-complement) LM; at or below it (and for flags the sparse solver does not
parameterize) the original OpenCV long-criteria candidates are used. Reason: the dense OpenCV solve is cubic in frame count
(measured 2.3s per iteration at 146 frames, 15-25 min per fit).
"""
import copy
import time

import cv2
import numpy as np
import pytest

from calibration.models import common
from calibration.models.common import (
    CONVERGENCE_CHECK_CRITERIA,
    DEFAULT_TERM_CRITERIA,
    complete_perspective_convergence,
)
from calibration.models.sparse_refine import free_intrinsic_mask, refine_perspective
from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, ImageInfo

K1 = np.array([[1978.7, 0, 1835.2], [0, 2004.0, 1125.7], [0, 0, 1.]])
K2 = np.array([[1920.6, 0, 1933.2], [0, 1927.9, 1098.8], [0, 0, 1.]])


def native(rms, k):
    return (rms, k.copy(), np.zeros((8, 1)), (np.zeros(3),), (np.ones(3),), np.zeros((18, 1)), np.zeros((6, 1)), np.zeros((1, 1)))


def fake_candidates(monkeypatch, first_rms, second_rms, k=K2, first='continue_sparse', second='restart_sparse'):
    """Replace candidate generation; None rms -> the candidate raises cv2.error."""
    def make(rms):
        def solve():
            if rms is None:
                raise cv2.error('injected')
            return native(rms, k)
        return solve

    monkeypatch.setattr(common, '_convergence_candidates',
                        lambda *a, **kw: ('sparse_lm', [(first, make(first_rms)), (second, make(second_rms))]))


# --- adoption rule (unchanged from P7) ---------------------------------------

def test_unconverged_native_replaced_by_lower_objective(monkeypatch):
    fake_candidates(monkeypatch, first_rms=1.6422, second_rms=1.6391)
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), cv2.CALIB_RATIONAL_MODEL, None, None, native(3.0008, K1))
    assert audit['adopted'] == 'restart_sparse' and chosen[0] == pytest.approx(1.6391)
    np.testing.assert_array_equal(chosen[1], K2)


def test_converged_native_is_returned_unchanged(monkeypatch):
    fake_candidates(monkeypatch, first_rms=1.4336465638, second_rms=1.4336465637)
    start = native(1.4336465638, K1)
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), 0, None, None, start)
    assert chosen is start and audit['adopted'] == 'native'


def test_diverging_restart_is_rejected(monkeypatch):
    # Real Train28 Brown: restart from the native result diverged 28 -> 47px.
    fake_candidates(monkeypatch, first_rms=2.7139, second_rms=47.5588)
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), 0, None, None, native(3.6524, K1))
    assert audit['adopted'] == 'continue_sparse' and chosen[0] == pytest.approx(2.7139)


def test_both_candidates_worse_or_failing_keep_native(monkeypatch):
    fake_candidates(monkeypatch, first_rms=None, second_rms=9.0)
    start = native(3.0, K1)
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), 0, None, None, start)
    assert chosen is start and audit['adopted'] == 'native'
    assert 'error' in audit['candidates'][0]


def test_seeded_premature_stop_completed_by_restart(monkeypatch):
    fake_candidates(monkeypatch, first_rms=2.3641, second_rms=1.4336)
    seed_k = np.array([[1990., 0, 1970], [0, 1800, 1030], [0, 0, 1]])
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), flags, seed_k, np.zeros((8, 1)), native(2.3641, K1))
    assert audit['adopted'] == 'restart_sparse' and chosen[0] == pytest.approx(1.4336)


def test_nonfinite_candidate_never_adopted(monkeypatch):
    def solve():
        out = list(native(0.1, K2)); out[1] = np.full((3, 3), np.nan); return tuple(out)
    monkeypatch.setattr(common, '_convergence_candidates',
                        lambda *a, **kw: ('sparse_lm', [('continue_sparse', solve), ('restart_sparse', solve)]))
    start = native(3.0, K1)
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), 0, None, None, start)
    assert chosen is start and all(not c['finite'] for c in audit['candidates'])


# --- candidate selection -------------------------------------------------------

def test_unsupported_flags_fall_back_to_opencv_long_candidates(monkeypatch):
    calls = []

    def fake(obj, img, size, k_in, d_in, flags, criteria):
        calls.append((flags, criteria, None if k_in is None else k_in.copy()))
        assert criteria == CONVERGENCE_CHECK_CRITERIA
        return native(1.0, K2)

    monkeypatch.setattr(common.cv2, 'calibrateCameraExtended', fake)
    flags = cv2.CALIB_FIX_ASPECT_RATIO
    chosen, audit = complete_perspective_convergence([], [], (3840, 2160), flags, None, None, native(3.0, K1))
    assert audit['method'] == 'opencv_long' and audit['check_criteria'] == list(CONVERGENCE_CHECK_CRITERIA)
    assert [c['name'] for c in audit['candidates']] == ['same_start_long', 'restart_long']
    assert len(calls) == 2 and calls[0][2] is None
    np.testing.assert_array_equal(calls[1][2], K1)
    assert calls[1][0] & cv2.CALIB_USE_INTRINSIC_GUESS


def test_free_intrinsic_mask_follows_flags():
    assert free_intrinsic_mask(0, 5).tolist() == [True] * 9
    m = free_intrinsic_mask(cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_ZERO_TANGENT_DIST, 14)
    assert m[:4].all() and m[4:6].all() and not m[6:8].any() and m[8:12].all() and not m[12:].any()
    assert free_intrinsic_mask(cv2.CALIB_FIX_K3, 5)[8] == np.False_
    assert free_intrinsic_mask(cv2.CALIB_FIX_ASPECT_RATIO, 5) is None
    assert free_intrinsic_mask(cv2.CALIB_RATIONAL_MODEL, 5) is None


# --- real solves on synthetic data ---------------------------------------------

def _synthetic_views(n_views, seed=3, d=(-0.30, 0.10, 0.0005, 0.0002, -0.015)):
    rng = np.random.default_rng(seed)
    k = np.array([[1900., 0, 1916], [0, 1905, 1083], [0, 0, 1]])
    d = np.array(d)
    obj = np.array([[x * .08, y * .08, 0] for y in range(1, 7) for x in range(1, 10)], np.float32)
    objs, imgs = [], []
    while len(objs) < n_views:
        rv = rng.normal(0, .45, 3)
        t = np.array([rng.uniform(-.5, .5), rng.uniform(-.3, .3), rng.uniform(.5, 1.6)]) - cv2.Rodrigues(rv)[0] @ np.array([.4, .28, 0])
        uv = cv2.projectPoints(obj, rv, t, k, d)[0].reshape(-1, 2)
        if ((obj @ cv2.Rodrigues(rv)[0].T + t)[:, 2] <= 0).any() or (uv < 0).any() or (uv[:, 0] > 3840).any() or (uv[:, 1] > 2160).any():
            continue
        objs.append(obj.reshape(-1, 1, 3))
        imgs.append((uv + rng.normal(0, .3, uv.shape)).astype(np.float32).reshape(-1, 1, 2))
    return objs, imgs


def _synthetic_dataset():
    objs, imgs = _synthetic_views(16)
    frames = []
    for i, (o, p) in enumerate(zip(objs, imgs)):
        fid = f'f{i}'
        det = DetectionResult(fid, True, corners=p, object_points=o, num_corners=len(o))
        frames.append(Frame(ImageInfo(fid, 'synthetic', 3840, 2160), det))
    return Dataset(frames=frames)


def test_small_view_counts_keep_p7_opencv_candidates(monkeypatch):
    """<= DENSE_CONVERGENCE_MAX_VIEWS: unchanged P7 path (validated solutions)."""
    seen = []
    monkeypatch.setattr(common.cv2, 'calibrateCameraExtended',
                        lambda *a, **kw: seen.append(kw['criteria']) or native(1.0, K2))
    views = [np.zeros((4, 1, 3))] * common.DENSE_CONVERGENCE_MAX_VIEWS
    _, audit = complete_perspective_convergence(views, views, (3840, 2160), 0, None, None, native(3.0, K1))
    assert audit['method'] == 'opencv_long' and seen == [CONVERGENCE_CHECK_CRITERIA] * 2


@pytest.mark.parametrize('flags', [0, cv2.CALIB_RATIONAL_MODEL])
def test_sparse_completion_reaches_opencv_long_minimum_from_premature_stop(flags, monkeypatch):
    monkeypatch.setattr(common, 'DENSE_CONVERGENCE_MAX_VIEWS', 0)
    cv2.setNumThreads(1)
    objs, imgs = _synthetic_views(20)
    size = (3840, 2160)
    premature = cv2.calibrateCameraExtended(objs, imgs, size, None, None, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_MAX_ITER, 4, 0))
    reference = cv2.calibrateCameraExtended(objs, imgs, size, None, None, flags=flags,
                                            criteria=CONVERGENCE_CHECK_CRITERIA)
    chosen, audit = complete_perspective_convergence(objs, imgs, size, flags, None, None, premature)
    assert audit['method'] == 'sparse_lm' and audit['adopted'] != 'native'
    assert chosen[0] == pytest.approx(reference[0], rel=1e-6)
    np.testing.assert_allclose(chosen[1], reference[1], rtol=1e-4)


def test_sparse_outputs_match_opencv_layout_and_uncertainty():
    cv2.setNumThreads(1)
    objs, imgs = _synthetic_views(20)
    ref = cv2.calibrateCameraExtended(objs, imgs, (3840, 2160), None, None, flags=0,
                                      criteria=CONVERGENCE_CHECK_CRITERIA)
    out = refine_perspective(objs, imgs, 0, ref[1], ref[2], ref[3], ref[4])
    assert out[0] == pytest.approx(ref[0], rel=1e-9)
    for got, want in zip(out[1:], ref[1:]):
        assert np.shape(got) == np.shape(want) or (isinstance(got, tuple) and len(got) == len(want))
    np.testing.assert_allclose(out[5][:9], ref[5][:9], rtol=1e-2, atol=1e-9)  # std intrinsics
    np.testing.assert_allclose(out[6], ref[6], rtol=1e-2, atol=1e-9)          # std extrinsics
    np.testing.assert_allclose(out[7], ref[7], rtol=1e-6)                     # per-view RMS


def test_completion_cost_is_linear_in_frames():
    """Regression for the 1h+ pipeline: 150 views must complete in seconds."""
    cv2.setNumThreads(1)
    objs, imgs = _synthetic_views(150, seed=11)
    size = (3840, 2160)
    nat = cv2.calibrateCameraExtended(objs, imgs, size, None, None, flags=cv2.CALIB_RATIONAL_MODEL,
                                      criteria=(cv2.TERM_CRITERIA_MAX_ITER, 2, 0))
    start = time.perf_counter()
    chosen, audit = complete_perspective_convergence(objs, imgs, size, cv2.CALIB_RATIONAL_MODEL, None, None, nat)
    assert time.perf_counter() - start < 60
    assert audit['method'] == 'sparse_lm' and chosen[0] <= nat[0]


def test_converged_wrapper_output_bitwise_equals_single_native_solve():
    from calibration.models.brown_conrady import calibrate_brown_conrady
    from calibration.models.common import collect_calibration_inputs
    cv2.setNumThreads(1)
    data = _synthetic_dataset()
    r = calibrate_brown_conrady(copy.deepcopy(data), CameraConfig(3840, 2160))
    _, obj, img = collect_calibration_inputs(copy.deepcopy(data))
    rms, k, d, *_ = cv2.calibrateCameraExtended(obj, img, (3840, 2160), None, None, flags=0, criteria=DEFAULT_TERM_CRITERIA)
    assert r.success and r.optimization_audit['convergence']['adopted'] == 'native'
    assert r.rms_error == float(rms)
    np.testing.assert_array_equal(r.camera_matrix, k)
    np.testing.assert_array_equal(np.asarray(r.distortion).reshape(-1), np.asarray(d).reshape(-1)[:5])
