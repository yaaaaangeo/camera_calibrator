"""Opt-in, nested-validation initialization selection. Outer Test is never read.

Selection measures fixed-intrinsic pose errors on a TRAIN-internal validation
split. It never compares native RMS values over differently trimmed view sets.
Native model solvers, their objectives, flags and stopping criteria are retained.
"""
from __future__ import annotations
import copy
import time
from collections import OrderedDict
import threading
import cv2

_DEVELOPMENT_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
import numpy as np
from calibration.types import CameraModelType as M, CalibrationResult
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye

STRATEGIES = ('legacy', 'cold', 'pinhole_k', 'brown_compatible', 'same_model_subset')


def fit_strategy(data, camera, model, strategy, seed_data):
    """All seed pixels must belong to caller-supplied development data."""
    data, seed_data = copy.deepcopy(data), copy.deepcopy(seed_data)
    fn = {M.PINHOLE: calibrate_pinhole, M.BROWN_CONRADY: calibrate_brown_conrady,
          M.EXTENDED_PINHOLE: calibrate_extended_pinhole, M.FISHEYE: calibrate_fisheye}[model]
    guess = None
    if (strategy == 'legacy' and model == M.FISHEYE) or strategy == 'pinhole_k':
        guess = calibrate_pinhole(seed_data if strategy != 'legacy' else data, camera)
    elif strategy == 'brown_compatible':
        guess = calibrate_brown_conrady(seed_data, camera)
    elif strategy == 'same_model_subset':
        from calibration.validation import _subset_dataset
        ids = [f.image_info.image_id for f in seed_data.enabled_frames]
        subset = _subset_dataset(seed_data, ids[::2])
        guess = fn(subset, camera)
    init = {'strategy': strategy, 'seed_ids': [f.image_info.image_id for f in seed_data.enabled_frames],
            'seed_success': guess.success if guess is not None else None,
            'seed_error': guess.error_message if guess is not None else None,
            'K': None if guess is None or guess.camera_matrix is None else guess.camera_matrix.tolist(),
            'source_D': None if guess is None or guess.distortion is None else guess.distortion.tolist(),
            'distortion_transfer': 'K only; zero D' if model == M.FISHEYE else 'model-aware compatible_seed',
            'native_termination': 'EPS+MAX_ITER,100,1e-6; native iteration count not exposed'}
    if strategy == 'same_model_subset':
        init['seed_ids'] = [f.image_info.image_id for f in subset.enabled_frames]
    if model == M.FISHEYE:
        from calibration.models.fisheye import _fisheye_base_flags, _fisheye_flag
        flags = _fisheye_base_flags()
        if guess is not None and guess.success:
            flags |= _fisheye_flag('CALIB_USE_INTRINSIC_GUESS')
        init['native_flags'] = int(flags)
        init['transferred_D'] = [0.0]*4 if guess is not None and guess.success else None
    else:
        from calibration.models.pinhole import _PINHOLE_FLAGS
        flags = _PINHOLE_FLAGS if model == M.PINHOLE else (cv2.CALIB_RATIONAL_MODEL if model == M.EXTENDED_PINHOLE else 0)
        if guess is not None:
            from calibration.initialization import compatible_seed
            _, initial_d = compatible_seed(guess, model)
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            init['transferred_D'] = initial_d.reshape(-1).tolist()
        init['native_flags'] = int(flags)
    if model == M.FISHEYE:
        # Existing fisheye wrapper deliberately copies K only and starts D=0.
        result = fn(data, camera, initial_guess=guess)
    else:
        result = fn(data, camera, initial_guess=guess)
    result.optimization_audit = {'initialization': init}
    return result


def admissible(result, camera=None, data=None):
    if not result.success or result.camera_matrix is None or result.distortion is None:
        return False
    k, d = result.camera_matrix, result.distortion
    if k.shape != (3, 3) or np.asarray(d).size not in (4,5,8):
        return False
    if not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
        return False
    if any(not np.isfinite(t).all() or np.asarray(t).reshape(-1)[2] <= 0 for t in result.tvecs):
        return False
    if data is not None:
        from calibration.models.common import collect_calibration_inputs
        frames, objects, _ = collect_calibration_inputs(data)
        kept = [(f, obj) for f, obj in zip(frames, objects) if f.image_info.image_id in result.per_frame_error]
        if len(kept) != len(result.rvecs) or len(kept) != len(result.tvecs):
            return False
        for (_, obj), rv, tv in zip(kept, result.rvecs, result.tvecs):
            points = np.asarray(obj).reshape(-1,3) @ cv2.Rodrigues(rv)[0].T + np.asarray(tv).reshape(1,3)
            if not np.isfinite(points).all() or np.any(points[:,2] <= 0):
                return False
    if result.model_name == M.FISHEYE:
        dv = np.asarray(d).reshape(-1)
        roots = np.roots([9*dv[3], 7*dv[2], 5*dv[1], 3*dv[0], 1.0])
        if any(abs(x.imag) < 1e-9 and 0 <= x.real <= (np.pi/2)**2 for x in roots):
            return False  # Angular radial mapping folds in the front hemisphere.
    if result.model_name == M.EXTENDED_PINHOLE:
        # Reject a rational denominator pole anywhere between axis and fitted points.
        dv = np.asarray(d).reshape(-1)
        if len(dv) >= 8:
            roots = np.roots([dv[7], dv[6], dv[5], 1.0])
            # Conservative full image domain is checked by the caller below.
            poles = [float(x.real) for x in roots if abs(x.imag) < 1e-9 and x.real >= 0]
            if camera is not None:
                radius2 = max(((x-k[0,2])/k[0,0])**2+((y-k[1,2])/k[1,1])**2
                              for x in (0,camera.width) for y in (0,camera.height))
                if any(p <= radius2 for p in poles): return False
    return True


def _fit_development(development_data, camera, model, seed):
    data = development_data
    from calibration.validation import split_train_test, _subset_dataset, _test_reprojection_errors
    train, val = split_train_test(development_data, camera, .25, seed)
    inner = _subset_dataset(development_data, train)
    validation = _subset_dataset(development_data, val)
    audit = {'selection_train_ids': train, 'selection_validation_ids': val,
             'fit_ids': [f.image_info.image_id for f in data.enabled_frames],
             'criterion': 'complete inner-validation frame-equal RMS; no outer Test selection',
             'candidates': [], 'native_termination': 'EPS+MAX_ITER,100,1e-6; native iteration count not exposed'}
    eligible = []
    if len(train) >= 3 and val:
        for strategy in STRATEGIES:
            if strategy == "pinhole_k" and model == M.FISHEYE:
                continue  # Same as legacy on this identical inner Train.
            if strategy == "cold" and model != M.FISHEYE:
                continue  # Identical to legacy; do not repeat the same solve.
            start = time.monotonic()
            try:
                r = fit_strategy(inner, camera, model, strategy, inner)
                valid = admissible(r, camera, inner)
                per, failed = {}, []
                if valid:
                    per, failed, *_ = _test_reprojection_errors(validation.enabled_frames, r.camera_matrix, r.distortion, model)
                score = float(np.sqrt(np.mean(np.square(list(per.values()))))) if per and not failed and len(per) == len(val) else None
                if score is not None and not np.isfinite(score):
                    score = None
                entry = {'strategy': strategy, 'success': r.success, 'admissible': valid,
                         'fit_rms': r.rms_error, 'validation_rms': score, 'error': r.error_message,
                         'warning': r.warning_message, 'fit_per_frame_rms': r.per_frame_error,
                         'residual_stats': __import__('dataclasses').asdict(r.residual_stats) if r.residual_stats else None,
                         'initialization': r.optimization_audit.get('initialization', {}), 'retained_fit_ids': list(r.per_frame_error),
                         'K': None if r.camera_matrix is None else r.camera_matrix.tolist(),
                         'D': None if r.distortion is None else r.distortion.tolist()}
                if valid and score is not None: eligible.append((score, STRATEGIES.index(strategy), strategy))
            except Exception as e:
                entry = {'strategy': strategy, 'success': False, 'error': f'{type(e).__name__}: {e}'}
            entry['seconds'] = time.monotonic()-start
            audit['candidates'].append(entry)
    strategy = min(eligible)[2] if eligible else 'legacy'
    # Evidence currently supports warm full refit and fisheye retries only.
    # Keep perspective defaults: real-data refits did not demonstrate stable improvement.
    if model in (M.PINHOLE, M.BROWN_CONRADY, M.EXTENDED_PINHOLE):
        strategy = 'legacy'
        audit['policy'] = 'Default retained: no demonstrated stable improvement'
    start = time.monotonic()
    try:
        result = fit_strategy(development_data, camera, model, strategy, development_data)
    except Exception as e:
        result = CalibrationResult(model_name=model, success=False, error_message=f'{type(e).__name__}: {e}')
    if not admissible(result, camera, development_data):
        audit['fallback_reason'] = result.error_message or 'Selected development refit is physically inadmissible'
        if strategy != 'legacy':
            strategy = 'legacy'
            result = fit_strategy(development_data, camera, model, strategy, development_data)
        audit['fallback_requires_review'] = True
    audit.update(selected_strategy=strategy, development_seconds=time.monotonic()-start,
                 fallback=not bool(eligible) or 'fallback_reason' in audit)
    return result, audit


def robust_fit(data, camera, model, development_data, seed=42, *, estimate_uncertainty=False, bootstrap_jobs=1):
    from calibration.cache import dataset_fingerprint, camera_fingerprint
    key = (dataset_fingerprint(development_data), camera_fingerprint(camera), model.value,
           seed, cv2.__version__, cv2.getNumThreads())
    with _CACHE_LOCK:
        cached = copy.deepcopy(_DEVELOPMENT_CACHE.get(key))
    if cached is None:
        result, audit = _fit_development(development_data, camera, model, seed)
        with _CACHE_LOCK:
            _DEVELOPMENT_CACHE[key] = copy.deepcopy((result, audit))
            while len(_DEVELOPMENT_CACHE) > 16:
                _DEVELOPMENT_CACHE.popitem(last=False)
        audit['development_cache_hit'] = False
    else:
        result, audit = cached
        audit['development_cache_hit'] = True
    strategy = audit['selected_strategy']
    audit['fit_ids'] = [f.image_info.image_id for f in data.enabled_frames]
    start = time.monotonic()
    development_ids = [f.image_info.image_id for f in development_data.enabled_frames]
    target_ids = [f.image_info.image_id for f in data.enabled_frames]
    if target_ids != development_ids:
        if model == M.FISHEYE and result.success:
            fn = calibrate_fisheye
            try:
                warm = fn(copy.deepcopy(data), camera, initial_guess=result,
                          estimate_uncertainty=estimate_uncertainty, bootstrap_jobs=bootstrap_jobs)
            except Exception as e:
                warm = CalibrationResult(model_name=model, success=False, error_message=f'{type(e).__name__}: {e}')
            audit['full_refit_seed'] = {'scope': 'outer_train_only', 'ids': development_ids,
                                      'K': result.camera_matrix.tolist(), 'D': result.distortion.tolist(),
                                      'distortion_transfer': 'K only; D reset to zero'}
            if admissible(warm, camera, data):
                result = warm
            else:
                audit['fallback_reason'] = warm.error_message or 'Full refit physically inadmissible'
                result = fit_strategy(data, camera, model, 'legacy', data)
        else:
            result = fit_strategy(data, camera, model, 'legacy', data)
    audit['final_physical_check'] = 'PASS' if admissible(result, camera, data) else 'FAIL'
    if result.success and audit['final_physical_check'] == 'FAIL':
        result.success = False
        result.error_message = 'Robust optimization: no physically admissible final result, including legacy fallback'
    audit['refit_initialization'] = result.optimization_audit.get('initialization', {})
    audit.update(selected_strategy=strategy, refit_seconds=time.monotonic()-start,
                 fallback=audit['fallback'] or 'fallback_reason' in audit,
                 seed_source_ids=development_ids,
                 actual_fit_ids=list(result.per_frame_error))
    result.optimization_audit = audit
    return result
