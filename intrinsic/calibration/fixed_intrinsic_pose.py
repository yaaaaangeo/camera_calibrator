"""Audited pose-only evaluation. Intrinsics, pixels and training are never changed.

Cheirality is necessary, not an accuracy certificate. Geometry rejection is
independent of reprojection RMS: rank-deficient or numerically near-collinear
views cannot provide a well-supported pose. The relative conditioning floor is
sqrt(float32 epsilon), matching the detector's coordinate precision; it is not
an operational pixel-error threshold. Remaining pose ambiguity is recorded.
"""
from __future__ import annotations
import cv2
import numpy as np
from calibration.types import CameraModelType
from calibration.models.common import project_points_for_model

GEOMETRY_RELATIVE_FLOOR = float(np.sqrt(np.finfo(np.float32).eps))


def geometry_diagnostics(object_points, image_points):
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    evidence = {'num_points': len(obj), 'relative_conditioning_floor': GEOMETRY_RELATIVE_FLOOR,
                'floor_basis': 'sqrt(float32 epsilon); numerical geometry guard, not RMS selection'}
    if len(obj) != len(img) or not np.isfinite(obj).all() or not np.isfinite(img).all():
        return evidence, 'invalid_correspondences'
    for name, points in [('object', obj), ('image', img)]:
        centered = points - points.mean(axis=0) if len(points) else points
        singular = np.linalg.svd(centered, compute_uv=False) if len(points) else np.array([])
        rank = int(np.linalg.matrix_rank(centered)) if len(points) else 0
        ratio = float(singular[1] / singular[0]) if len(singular) > 1 and singular[0] > 0 else 0.
        evidence[name+'_rank'] = rank
        evidence[name+'_singular_values'] = singular.tolist()
        evidence[name+'_minor_major_ratio'] = ratio
    if len(obj) < 4:
        return evidence, 'insufficient_points'
    for name in ('object', 'image'):
        if evidence[name+'_rank'] < 2:
            return evidence, name+'_points_collinear'
        if evidence[name+'_minor_major_ratio'] <= GEOMETRY_RELATIVE_FLOOR:
            return evidence, name+'_points_near_collinear'
    return evidence, None


def evaluate_fixed_pose(object_points, image_points, camera_matrix, distortion, model):
    """Return audit plus projected pixels; preserve valid native poses exactly.

    Planar IPPE solutions are inspected in addition to the native initializer.
    A retry is allowed only if native fails validity, never because its RMS is
    high. Indistinguishable retry solutions are not promoted to a unique pose.
    """
    audit = {'status': 'unevaluable', 'reason': None, 'attempts': [],
             'selected_pose': None, 'intrinsics_refitted': False,
             'reprojection_quality_threshold_applied': False}
    if object_points is None or image_points is None:
        audit['reason'] = 'missing_correspondences'
        return audit, None
    try:
        geometry, reason = geometry_diagnostics(object_points, image_points)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        audit.update(reason='invalid_correspondences', error=str(exc))
        return audit, None
    audit['geometry'] = geometry
    if reason:
        audit['reason'] = reason
        return audit, None
    fish = model == CameraModelType.FISHEYE
    dtype = np.float64 if fish else np.asarray(object_points).dtype
    obj = np.ascontiguousarray(object_points, dtype=dtype).reshape(-1, 1, 3)
    img = np.ascontiguousarray(image_points, dtype=dtype).reshape(-1, 1, 2)
    K = np.asarray(camera_matrix, dtype=np.float64).copy()
    D = np.asarray(distortion, dtype=np.float64).copy()
    candidates = []

    def inspect(method, rv, tv):
        entry = {'method': method, 'valid': False}
        audit['attempts'].append(entry)
        if rv is None or tv is None or not np.isfinite(rv).all() or not np.isfinite(tv).all():
            entry['rejection'] = 'nonfinite_pose'
            return
        rv, tv = np.asarray(rv).reshape(3, 1), np.asarray(tv).reshape(3, 1)
        entry.update(rvec=rv.reshape(-1).tolist(), tvec=tv.reshape(-1).tolist(), origin_depth=float(tv[2, 0]))
        points = obj.reshape(-1, 3) @ cv2.Rodrigues(rv)[0].T + tv.T
        entry['corner_depths'] = points[:, 2].tolist()
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            entry['rejection'] = 'nonpositive_observed_depth'
            return
        projected = project_points_for_model(obj, rv, tv, K, D, model)
        if not np.isfinite(projected).all():
            entry['rejection'] = 'nonfinite_projection'
            return
        residual = projected - img.reshape(-1, 2)
        errors = np.linalg.norm(residual, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        if not np.isfinite(rms):
            entry['rejection'] = 'nonfinite_reprojection_error'
            return
        entry.update(valid=True, rms_px=rms, corner_errors_px=errors.tolist(),
                     residual_xy=residual.tolist(), normal_camera=cv2.Rodrigues(rv)[0][:, 2].tolist())
        candidates.append((entry, projected))

    try:
        fn = cv2.fisheye.solvePnP if fish else cv2.solvePnP
        ok, rv, tv = fn(obj, img, K, D)
        if ok:
            inspect('native', rv, tv)
        else:
            audit['attempts'].append({'method': 'native', 'valid': False, 'rejection': 'solver_returned_false'})
    except cv2.error as exc:
        audit['attempts'].append({'method': 'native', 'valid': False, 'rejection': 'solver_error', 'error': str(exc)})
    native = candidates[0] if candidates else None
    # This is diagnostic unless native is invalid. No good native pose is
    # replaced by a lower-RMS retry, keeping existing valid results reproducible.
    if geometry['object_rank'] == 2:
        try:
            pnp_img = cv2.fisheye.undistortPoints(img, K, D, P=K) if fish else img
            n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, pnp_img, K, None if fish else D, flags=cv2.SOLVEPNP_IPPE)
            for index, (rv, tv) in enumerate(zip(rvs, tvs)):
                inspect('IPPE_'+str(index), rv, tv)
        except cv2.error as exc:
            audit['attempts'].append({'method': 'IPPE', 'valid': False, 'rejection': 'solver_error', 'error': str(exc)})
    ranked = sorted(candidates, key=lambda item: item[0]['rms_px'])
    if len(ranked) >= 2:
        audit['candidate_rms_gap_px'] = ranked[1][0]['rms_px'] - ranked[0][0]['rms_px']
    chosen = native
    if chosen is None and ranked:
        if len(ranked) > 1 and np.isclose(ranked[0][0]['rms_px'], ranked[1][0]['rms_px'],
                                        rtol=GEOMETRY_RELATIVE_FLOOR, atol=np.finfo(float).eps):
            first, second = ranked[0][0], ranked[1][0]
            r0 = cv2.Rodrigues(np.asarray(first['rvec']))[0]
            r1 = cv2.Rodrigues(np.asarray(second['rvec']))[0]
            t0, t1 = np.asarray(first['tvec']), np.asarray(second['tvec'])
            translation_scale = max(np.linalg.norm(t0), np.linalg.norm(t1), np.finfo(float).tiny)
            same_pose = (np.allclose(r0, r1, rtol=0, atol=GEOMETRY_RELATIVE_FLOOR)
                         and np.linalg.norm(t0-t1) <= GEOMETRY_RELATIVE_FLOOR * translation_scale)
            if not same_pose:
                audit['reason'] = 'ambiguous_retry_solutions'
                return audit, None
        chosen = ranked[0]
    if chosen is None:
        audit.update(status='failed', reason='no_valid_pose')
        return audit, None
    audit.update(status='valid', reason=None, selected_pose=chosen[0],
                 retry_used=chosen[0]['method'] != 'native',
                 ambiguity_note='Candidate gap is diagnostic; planar/noise ambiguity is not an accuracy certificate.')
    return audit, chosen[1]
