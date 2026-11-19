"""Train-only optional experiment, shared external-comparison evaluation."""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import html
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from calibration.external_compare import (
    _evaluate_side, build_parameter_diff_rows, build_fov_diff_rows,
    build_spatial_comparison_grid, build_bootstrap_comparison, build_parameter_diagnostics,
)
from calibration.json_utils import json_safe
from calibration.models.common import collect_calibration_inputs
from calibration.project_io import load_project
from calibration.types import CameraModelType
from calibration.validation import _subset_dataset, refit_on_train_split
from automation.factory.provider import FactoryError, FileProvider, validate_coordinates, validate_parameters


def seeded_fit(dataset, camera, train_ids, reference):
    """Experimental optimizer; both supplied K and D are used, TRAIN only.

    Never patches the default model solvers or their initialization behavior.
    """
    frames, obj, img = collect_calibration_inputs(_subset_dataset(dataset, train_ids))
    if len(frames) < 3:
        raise FactoryError('Seeded fit requires at least three usable train frames')
    K, D = reference.K.copy(), reference.D.copy()
    size = (camera.width, camera.height)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    if reference.model == CameraModelType.FISHEYE:
        from calibration.models.fisheye import _fisheye_flag, _to_fisheye_points
        flags = (_fisheye_flag('CALIB_USE_INTRINSIC_GUESS')
            | _fisheye_flag('CALIB_RECOMPUTE_EXTRINSIC') | _fisheye_flag('CALIB_FIX_SKEW'))
        obj64, img64 = _to_fisheye_points(obj, img)
        _, K, D, _, _ = cv2.fisheye.calibrate(obj64, img64, size,
            K, D.reshape(4, 1), flags=flags, criteria=criteria)
    else:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if reference.model == CameraModelType.EXTENDED_PINHOLE:
            flags |= cv2.CALIB_RATIONAL_MODEL
        _, K, D, _, _ = cv2.calibrateCamera(obj, img, size, K, D, flags=flags, criteria=criteria)
        D = D.ravel()[:len(reference.D)]
    validate_parameters(reference.model, K, D, *size)
    return K, D


def _split(project, model):
    validation = project.validation_results.get(model)
    if validation is None:
        raise FactoryError('Recorded Local hold-out split missing')
    train, test = list(validation.train_frame_ids), list(validation.test_frame_ids)
    if len(train) < 3 or not test or len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise FactoryError('Missing, duplicate or overlapping train/test IDs')
    frames = project.dataset.frames
    by_id = {f.image_info.image_id: f for f in frames}
    if len(by_id) != len(frames) or not (set(train) | set(test)) <= set(by_id):
        raise FactoryError('Ambiguous or unknown split IDs')
    # Prevent aliases of the same image/corner observation crossing the split.
    def fingerprints(ids):
        output = set()
        for fid in ids:
            f = by_id[fid]
            if not f.enabled or f.detection is None or not f.detection.success:
                raise FactoryError('Split contains disabled or undetected frame')
            obj, img = f.detection.object_points, f.detection.corners
            if obj is None or img is None or len(obj) < 4:
                raise FactoryError('Split has insufficient detected points')
            output.add(hashlib.sha256(np.asarray(obj).tobytes() + np.asarray(img).tobytes()).hexdigest())
            p = Path(f.image_info.path)
            if p.is_file():
                output.add(hashlib.sha256(p.read_bytes()).hexdigest())
        return output
    if fingerprints(train) & fingerprints(test):
        raise FactoryError('Train/test share image contents or detected observations')
    return train, test


def compare_project(project, reference, local_model, *, seeded=False):
    # Existing fit/evaluation helpers annotate Frame.reprojection_error.
    # Keep those annotations out of the caller's project as well as disk.
    project = copy.deepcopy(project)
    camera, pattern, dataset = project.camera_config, project.pattern_config, project.dataset
    if (camera.width, camera.height) != (reference.metadata['width'], reference.metadata['height']):
        raise FactoryError('Session/reference resolution mismatch')
    train, test = _split(project, local_model)
    observations = []
    for frame in dataset.frames:
        if frame.image_info.image_id in test:
            observations.append({'frame_id':frame.image_info.image_id,
                'num_corners':len(frame.detection.corners),
                'corners_sha256':hashlib.sha256(frame.detection.corners.tobytes()).hexdigest(),
                'object_points_sha256':hashlib.sha256(frame.detection.object_points.tobytes()).hexdigest()})
    sides, failures = {}, {}
    candidates = {'Factory-fixed': lambda: (reference.K.copy(), reference.D.copy(), reference.model)}
    def local():
        result = refit_on_train_split(dataset, camera, local_model, train)
        if not result.success:
            raise FactoryError(result.error_message or 'Local train fit failed')
        return result.camera_matrix, result.distortion, local_model
    candidates['Local'] = local
    if seeded:
        candidates['Factory-seeded'] = lambda: (*seeded_fit(dataset, camera, train, reference), reference.model)
    for name, fit in candidates.items():
        try:
            K, D, model = fit()
            if not np.isfinite(K).all() or not np.isfinite(D).all():
                raise FactoryError('Nonfinite candidate')
            sides[name] = _evaluate_side(dataset, camera, pattern, test, K, D, model, name)
        except (FactoryError, cv2.error, ValueError, RuntimeError) as exc:
            failures[name] = str(exc)
    report = {'schema_version': 1, 'status': 'COMPLETED', 'conclusion': 'UNCLEAR',
        'approval_status': 'UNREVIEWED', 'deployment_status': 'NOT_DEPLOYED',
        'factory_source': reference.metadata['source'], 'factory_authenticated': False,
        'factory_date': reference.metadata.get('calibration_date'),
        'factory_rms_not_ranked': reference.metadata.get('factory_rms'),
        'train_frame_ids': train, 'test_frame_ids': test,
        'test_observations': observations,
        'rms_definition': 'test_rms is equal-frame RMS; residual_stats.rmse is pooled-corner RMS',
        'evaluation': 'Existing external_compare._evaluate_side: identical corners, model-aware PnP, no Test intrinsic optimization',
        'independence': 'Recorded P1 hold-out; independence of manufacturer calibration data is not authenticated',
        'candidates': {}, 'failures': failures, 'pairwise': {},
        'limitations': ['Comparison does not change P1/P2 acceptance or model recommendation.',
            'No manufacturer authentication; source and coordinate metadata are declarations.',
            'Seeded optimization is experimental and need not improve.',
            'FOV is the existing nominal model estimate, not measured distorted boundary FOV.',
            'Parameter diagnostics use Test fixed poses and approximate local sensitivity/covariance; not training observability or repeat-capture stability.']}
    for name, side in sides.items():
        value = dataclasses.asdict(side)
        try:
            value['parameter_diagnostics'] = dataclasses.asdict(build_parameter_diagnostics(dataset, test, side))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as exc:
            value['parameter_diagnostics'] = {'status':'UNAVAILABLE', 'error':str(exc)}
        value['parameter_count'] = 4 if side.model_name == CameraModelType.PINHOLE else 4 + len(np.asarray(side.distortion).ravel())
        report['candidates'][name] = value
        expected = set(test)
        if (not side.success or side.failed_frame_ids or set(side.per_frame_error) != expected
                or side.test_rms is None or not np.isfinite(side.test_rms)
                or not all(np.isfinite(v) for v in side.per_frame_error.values())
                or len(side.point_error_details) != sum(x['num_corners'] for x in observations)):
            failures[name] = side.error_message or 'Incomplete Test evaluation (no survivor-only ranking)'
    factory = sides.get('Factory-fixed')
    if factory is not None:
        for name, side in sides.items():
            if name == 'Factory-fixed':
                continue
            report['pairwise'][name] = {
                'parameters': [dataclasses.asdict(x) for x in build_parameter_diff_rows(factory, side)
                    if factory.model_name == side.model_name or x.name in ('fx','fy','cx','cy')],
                'fov_nominal': [dataclasses.asdict(x) for x in build_fov_diff_rows(factory, side, (camera.width, camera.height))],
                'spatial': dataclasses.asdict(build_spatial_comparison_grid(factory, side, (camera.width, camera.height), rows=4, cols=4)),
                'bootstrap': dataclasses.asdict(build_bootstrap_comparison(factory, side)),
            }
            def ray_record(candidate):
                return {'camera':{}, 'resolution':[camera.width,camera.height],
                    'chosen_model':candidate.model_name.value, 'coordinate_system':{'kind':'full_frame'},
                    'K':candidate.camera_matrix.tolist(), 'D':candidate.distortion.ravel().tolist()}
            report['pairwise'][name]['shared_ray_projection'] = compare_runs(ray_record(factory), ray_record(side))
    # Conservative frame-level paired evidence for every candidate against all
    # alternatives; no extrapolation from single RMS or unequal failure subsets.
    if failures:
        report['conclusion'] = 'INCOMPARABLE'
    else:
        winners = []
        for name, side in sides.items():
            superior = True
            for other, ref in sides.items():
                if other == name:
                    continue
                b = build_bootstrap_comparison(ref, side)
                metrics_ok = all(getattr(side, metric) is not None and getattr(ref, metric) is not None
                    and getattr(side, metric) <= getattr(ref, metric) + 1e-6
                    for metric in ('edge_rms', 'straightness_residual'))
                if len(test) < 5 or b.improvement_ci_low is None or b.improvement_ci_low <= 0 or not metrics_ok or ref.test_rms - side.test_rms <= 1e-3:
                    superior = False
            if superior:
                winners.append(name)
        if len(winners) == 1:
            report['conclusion'] = {'Factory-fixed': 'FACTORY_SUPERIOR', 'Local': 'LOCAL_SUPERIOR', 'Factory-seeded': 'SEEDED_SUPERIOR'}[winners[0]]
    return json_safe(report, ndarray_wrapper=False)


def compare_session(session, factory, *, coordinates, seeded=False, output_root=None):
    root = resolve_session(session)
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf8'))
    if manifest.get('status') != 'completed':
        raise FactoryError('Local Session is not completed')
    identity = manifest['camera']
    reference = FileProvider(factory).lookup(identity['camera_model'], identity['serial_number'])
    project_path = root / 'project/calibration.ccproj'
    project, warnings = load_project(str(project_path))
    validate_coordinates(coordinates, project.camera_config.width, project.camera_config.height)
    if coordinates != reference.metadata['coordinate_system']:
        raise FactoryError('Reference/session coordinate mismatch')
    chosen = project.final_result.chosen_model if project.final_result else None
    if chosen is None:
        raise FactoryError('Session chosen Local model missing')
    report = compare_project(project, reference, chosen, seeded=seeded)
    report.update(camera=identity, coordinate_system=coordinates,
        session_manifest_sha256=hashlib.sha256((root / 'run_manifest.json').read_bytes()).hexdigest(),
        project_sha256=hashlib.sha256(project_path.read_bytes()).hexdigest(),
        factory_file_sha256=hashlib.sha256(Path(factory).read_bytes()).hexdigest(),
        load_warnings=warnings)
    parent = Path(output_root) if output_root else root.parent / 'factory_comparisons'
    if parent.resolve().is_relative_to(root.resolve()):
        raise FactoryError('Comparison output must be outside original Session')
    out = parent / uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (out / 'factory_comparison.json').write_text(text, encoding='utf8')
    (out / 'factory_comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Factory comparison</title><h1>Factory comparison — review required</h1><pre>' + html.escape(text) + '</pre>', encoding='utf8')
    return out, report


# ------------------------------------------------- session helpers
# (예전 automation.publish 에 있던 것 - 게시 기능을 빼면서 Factory 비교가 쓰는 것만 옮겼다)


def _confined(root, relative):
    """Reject traversal, symlinks and Windows junction/reparse redirects."""
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(x in {'.', '..'} for x in rel.parts) or ':' in str(rel) or '\\' in str(relative):
        raise FactoryError('UNSAFE_PATH')
    path = root
    for component in rel.parts:
        path = path / component
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise FactoryError('UNSAFE_LINK')
    if not path.resolve().is_relative_to(root):
        raise FactoryError('UNSAFE_PATH')
    if not path.is_file():
        raise FactoryError('REQUIRED_FILE_MISSING')
    return path


def resolve_session(session):
    """A P1 session folder, or the last completed run inside a P2 (live) session folder."""
    root = Path(session).expanduser().resolve()
    if (root / 'p2_manifest.json').is_file():
        try:
            data = json.loads(_confined(root, 'p2_manifest.json').read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            raise FactoryError('INVALID_RESULT_JSON') from None
        attempts = data.get('attempts') or []
        if not attempts or data.get('state') != 'COMPLETED':
            raise FactoryError('P2_NOT_COMPLETED')
        child = Path(attempts[-1]['outcome']['session_dir']).resolve()
        if not child.is_relative_to(root / 'runs'):
            raise FactoryError('UNSAFE_P2_SESSION')
        _confined(root, child.relative_to(root).as_posix() + '/run_manifest.json')
        return child
    return root


def compare_runs(left,right):
    reasons=[]
    if left.get('camera')!=right.get('camera'):reasons.append('CAMERA_IDENTITY_MISMATCH')
    if left.get('resolution')!=right.get('resolution'):reasons.append('RESOLUTION_MISMATCH')
    if left.get('chosen_model')!=right.get('chosen_model'):reasons.append('DISTORTION_MODEL_MISMATCH')
    a,b=left.get('coordinate_system') or {},right.get('coordinate_system') or {}
    if a.get('kind') not in {'full_frame','roi'} or b.get('kind') not in {'full_frame','roi'}:reasons.append('ROI_UNCONFIRMED')
    elif a!=b:reasons.append('ROI_MISMATCH')
    result={'comparable':not reasons,'limitations':reasons,
            'left_verdict':left.get('acceptance_verdict'),'right_verdict':right.get('acceptance_verdict'),
            'interpretation':'Parameter or projection differences are not a quality verdict.'}
    if reasons:return result
    import cv2
    import numpy as np
    try:
        ka,kb=np.asarray(left['K'],float),np.asarray(right['K'],float)
        da,db=np.asarray(left['D'],float).ravel(),np.asarray(right['D'],float).ravel()
        if ka.shape!=(3,3) or kb.shape!=(3,3) or da.shape!=db.shape:raise ValueError()
        if not all(np.isfinite(x).all() for x in [ka,kb,da,db]):raise ValueError()
        if min(ka[0,0],ka[1,1],kb[0,0],kb[1,1])<=0:raise ValueError()
        result['intrinsic_delta']={name:float(y-x) for name,x,y in zip(['fx','fy','cx','cy'],[ka[0,0],ka[1,1],ka[0,2],ka[1,2]],[kb[0,0],kb[1,1],kb[0,2],kb[1,2]])}
        result['distortion_delta']=(db-da).tolist()
        width,height=left['resolution']
        pixels=np.array([(x,y) for y in np.linspace(.1*height,.9*height,7) for x in np.linspace(.1*width,.9*width,9)],float).reshape(-1,1,2)
        fish=left['chosen_model']=='fisheye'
        rays=(cv2.fisheye.undistortPoints(pixels,ka,da) if fish else cv2.undistortPoints(pixels,ka,da)).reshape(-1,2)
        xyz=np.column_stack([rays,np.ones(len(rays))]).reshape(-1,1,3)
        def project(k,d):
            fn=cv2.fisheye.projectPoints if fish else cv2.projectPoints
            return fn(xyz,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
        delta=np.linalg.norm(project(ka,da)-project(kb,db),axis=1)
        if not np.isfinite(delta).all():raise ValueError()
        result['shared_ray_projection_px']={'samples':len(delta),'rms':float(np.sqrt(np.mean(delta**2))),
            'p95':float(np.percentile(delta,95)),'max':float(delta.max()),'reference':'left 10%-90% pixel grid'}
    except (ValueError,TypeError,KeyError,cv2.error):
        return {**result,'comparable':False,'limitations':['INVALID_OR_UNSUPPORTED_CALIBRATION']}
    result['metrics']={'left':left.get('models'),'right':right.get('models')}
    return result
