"""P6: conservative continuous-domain distortion certificate (no fitting).

Coordinates: raw pixels -> K^-1 -> distorted normalized coordinates; forward
mapping input is the ideal camera ray (X/Z,Y/Z), NOT K^-1 raw pixels.
A disk encloses the requested raw/output rectangles and every retained fit ray.
Occlusion masks never shrink this geometric domain. PASS is necessary, not an
accuracy, inverse uniqueness outside the disk, or vehicle acceptance guarantee.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from numpy.polynomial import Polynomial as P
from calibration.types import CameraModelType as M

VERSION = 'p6-continuous-domain-v2'


def _positive_roots(poly):
    # Near-real roots are conservatively included; repeated roots additionally
    # cannot escape the Bernstein positivity certificate below.
    return sorted(float(z.real) for z in poly.roots()
                  if abs(z.imag) <= 1e-7 * max(1., abs(z.real)) and z.real >= 0.)


def _positive_on_disk(poly, radius, *, relative_margin=None):
    """Bernstein convex-hull certificate on [0,radius], including endpoints.

    No sampling gap can certify a polynomial crossing/touching zero. Numerical
    uncertainty/depth exhaustion fails closed. Padding covers conversion and
    subdivision roundoff; this is not a user quality threshold.
    """
    c = np.asarray(poly.coef, float) * radius ** np.arange(len(poly.coef))
    if not np.isfinite(c).all():
        return False
    n = len(c)-1
    b = np.array([sum(c[j]*math.comb(i,j)/math.comb(n,j) for j in range(i+1))
                  for i in range(n+1)])
    tol = (128*np.finfo(float).eps if relative_margin is None else relative_margin)*max(1., float(np.sum(np.abs(c))))
    stack = [(b, 0)]
    visits = 0
    while stack:
        v, depth = stack.pop(); visits += 1
        if np.min(v) > tol:
            continue
        if np.max(v) <= tol or depth >= 40 or visits > 8192:
            return False
        levels = [v]
        for _ in range(n): levels.append((levels[-1][:-1]+levels[-1][1:])/2)
        stack.append((np.array([level[0] for level in levels]), depth+1))
        stack.append((np.array([level[-1] for level in levels[::-1]]), depth+1))
    return True


def _rectangle_radius(k, roi):
    k = np.asarray(k,float)
    if k.shape != (3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0:
        raise ValueError('Invalid ROI camera matrix')
    x,y,w,h = map(float, roi)
    if not np.isfinite([x,y,w,h]).all(): raise ValueError('Nonfinite ROI')
    if w <= 0 or h <= 0 or min(x,y) < 0:
        raise ValueError('ROI must be a nonempty pixel-edge rectangle')
    uv = np.array([[x,y,1], [x+w,y,1], [x,y+h,1], [x+w,y+h,1]], float)
    rays = uv @ np.linalg.inv(k).T
    if np.any(rays[:,2] <= 0): raise ValueError('Invalid ROI rays')
    return float(np.max(np.linalg.norm(rays[:,:2]/rays[:,2,None], axis=1)))


def fitted_rays(result, data):
    """Retained OpenCV board->camera extrinsics, identical order to solver."""
    if data is None: return np.empty((0,3))
    from calibration.models.common import collect_calibration_inputs
    frames, objects, _ = collect_calibration_inputs(data)
    kept = [obj for f,obj in zip(frames,objects) if f.image_info.image_id in result.per_frame_error]
    if len(kept) != len(result.rvecs) or len(kept) != len(result.tvecs):
        raise ValueError('Retained frame/pose correspondence mismatch')
    chunks = [np.asarray(obj).reshape(-1,3) @ cv2.Rodrigues(rv)[0].T + np.asarray(tv).reshape(1,3)
              for obj,rv,tv in zip(kept,result.rvecs,result.tvecs)]
    return np.concatenate(chunks) if chunks else np.empty((0,3))


def rational_jacobian(xy, distortion):
    """Exact normalized 2-D forward Jacobian including p1,p2."""
    x,y = np.asarray(xy,float).reshape(2)
    d = np.asarray(distortion,float).reshape(-1)
    t = x*x+y*y
    p = P([1,d[0],d[1],d[4]]); q = P([1,d[5],d[6],d[7]])
    scale = p(t)/q(t)
    slope = (p.deriv()(t)*q(t)-p(t)*q.deriv()(t))/q(t)**2
    return scale*np.eye(2)+2*slope*np.outer([x,y],[x,y])+np.array([
        [2*d[2]*y+6*d[3]*x,2*d[2]*x+2*d[3]*y],
        [2*d[2]*x+2*d[3]*y,6*d[2]*y+2*d[3]*x]])


def _certified_boundary(reach, candidate):
    """Step a near-real boundary root outward until reach(r) > roundoff.

    A near-real/ill-conditioned polynomial root is a candidate, not proof of
    boundary coverage. Step outward only by a residual/roundoff-derived amount
    and verify separation. Returns (radius, residual, error) or None.
    """
    for _ in range(8):
        coefficients = np.abs(reach.coef)*candidate**np.arange(len(reach.coef))
        error = 256*np.finfo(float).eps*max(1.,float(coefficients.sum()))
        residual = float(reach(candidate))
        if np.isfinite(residual) and residual > error:
            return candidate, residual, error
        slope = float(reach.deriv()(candidate))
        if not np.isfinite(slope) or slope <= 0:
            return None
        candidate += max(256*abs(float(np.spacing(candidate))),2*(error-residual)/slope)
    return None


def _polynomial_minimum(poly, radius):
    positions=[0.,radius]+[r for r in _positive_roots(poly.deriv()) if r<=radius]
    return float(min(poly(r) for r in positions))


def _fisheye_usage_domain(report, fail, d, derivative, observed, rd, rout):
    """Base-calibration Fisheye domain: retained rays, output ROI, raw ROI.

    theta_d(theta) must rise strictly from the axis to the largest used angle,
    and the raw ROI must be reached on that monotone branch before pi/2. A
    fold beyond every used angle is reported, never silently approved as a
    hemisphere-valid model (Robust Optimization keeps the hemisphere gate).
    """
    report['domain_policy'] = ('usage domain: max(retained ray angle, output ROI angle, '
                               'raw ROI inverse angle) within the front hemisphere')
    theta = max(float(np.arctan(observed)), float(np.arctan(rout)))
    report.update(output_theta_max_rad=float(np.arctan(rout)))
    if rd:
        reach = P([-rd,1,0,d[0],0,d[1],0,d[2],0,d[3]])
        roots = [r for r in _positive_roots(reach) if 0 < r < np.pi/2]
        if not roots: return fail('raw_roi_not_reached_before_fold_or_front_hemisphere')
        boundary = _certified_boundary(reach, min(roots))
        if boundary is None: return fail('raw_roi_inverse_boundary_not_certified')
        report.update(raw_inverse_theta_rad=boundary[0],
                      inverse_boundary_polynomial_residual=boundary[1],
                      inverse_boundary_roundoff_bound=boundary[2])
        theta = max(theta, boundary[0])
    report['checked_theta_max_rad'] = theta
    if not theta < np.pi/2: return fail('usage_domain_outside_front_hemisphere')
    if not _positive_on_disk(derivative, theta):
        return fail('fisheye_angular_fold_in_usage_domain_or_uncertified')
    report['fold_outside_usage_domain_only'] = not report['hemisphere_derivative_certified']
    report['status'] = 'PASS'
    return report


def distortion_diagnostic(result, camera=None, data=None, *, raw_roi=None,
                          output_matrix=None, output_roi=None, observed_xyz=None,
                          fisheye_domain='hemisphere'):
    """JSON-safe evidence without changing result, K/D, poses or RMS.

    ROI is in pixel-edge coordinates (x,y,width,height) of its own matrix.
    Defaults: full sensor rectangle; current perspective remap output uses K.
    A supplied cropped output needs its matching K (and caller-managed origin).
    Mask is intentionally not an argument: visibility != geometric validity.
    fisheye_domain: 'hemisphere' (legacy Robust Optimization gate, [0,pi/2])
    or 'usage' (base calibration gate, actual rays/output/raw ROI only).
    PASS is a necessary geometric condition, not quality or vehicle approval.
    """
    report = {'version': VERSION, 'status': 'FAIL', 'reasons': [],
              'interpretation': 'necessary geometric condition only; not calibration quality PASS or vehicle approval',
              'mask_policy': 'occlusion does not crop geometric ROI',
              'domain_policy': 'axis-connected enclosing disk; raw inverse envelope + output ROI + retained rays',
              'continuous_certificate': 'Bernstein polynomial positivity; uncertain means FAIL'}
    def fail(reason):
        report['reasons'].append(reason)
        return report
    try:
        k = np.asarray(result.camera_matrix, float)
        d = np.asarray(result.distortion, float).reshape(-1)
        if k.shape != (3,3) or not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0,0],k[1,1]) <= 0:
            return fail('invalid_intrinsics')
        xyz = fitted_rays(result,data) if observed_xyz is None else np.asarray(observed_xyz,float).reshape(-1,3)
        if len(xyz) and (not np.isfinite(xyz).all() or np.any(xyz[:,2] <= 0)):
            return fail('nonpositive_or_nonfinite_observed_depth')
        observed = float(np.max(np.linalg.norm(xyz[:,:2]/xyz[:,2,None],axis=1))) if len(xyz) else 0.
        report.update(observed_ray_radius_max=observed, observed_points=len(xyz),
                      observed_domain_included=bool(len(xyz)),
                      observed_depth_min=float(xyz[:,2].min()) if len(xyz) else None)
        rd = rout = 0.
        if camera is not None:
            if data is not None and any((f.image_info.width,f.image_info.height) != (camera.width,camera.height)
                                        for f in data.enabled_frames):
                return fail('camera_frame_resolution_mismatch')
            full = (0,0,camera.width,camera.height)
            raw_roi = tuple(raw_roi or full)
            if raw_roi[0]+raw_roi[2]>camera.width or raw_roi[1]+raw_roi[3]>camera.height:
                return fail('raw_roi_outside_sensor')
            rd = _rectangle_radius(k,raw_roi)
            # Current perspective display uses K and same-size output. Explicit
            # rectification callers must supply their output K/ROI together.
            if (output_matrix is None) != (output_roi is None):
                return fail('output_matrix_and_roi_required_together')
            out_k = k if output_matrix is None else np.asarray(output_matrix,float)
            if output_matrix is None and result.model_name == M.FISHEYE:
                # Same identity rectification and balance=0 as the current
                # undistortion-quality default; hemisphere gate is independent.
                out_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    k,d.reshape(-1,1),(camera.width,camera.height),np.eye(3),balance=0.)
            rout = _rectangle_radius(out_k,full if output_roi is None else output_roi)
            report['output_matrix'] = np.asarray(out_k).tolist()
            report.update(raw_roi=list(raw_roi), output_roi=list(full if output_roi is None else output_roi),
                          raw_normalized_radius_max=rd, output_ideal_radius_max=rout)
        elif output_matrix is not None or raw_roi is not None or output_roi is not None:
            return fail('camera_resolution_required_for_roi')
        radius = max(observed,rd,rout)
        report['checked_radius'] = radius
        if result.model_name == M.FISHEYE:
            if len(d)!=4: return fail('invalid_fisheye_coefficient_count')
            derivative = P([1,0,3*d[0],0,5*d[1],0,7*d[2],0,9*d[3]])
            report.update(fisheye_domain=fisheye_domain,
                          observed_theta_max_rad=float(np.arctan(observed)),
                          angular_derivative_roots_rad=_positive_roots(derivative),
                          hemisphere_derivative_certified=bool(_positive_on_disk(derivative,np.pi/2)))
            if fisheye_domain == 'usage':
                return _fisheye_usage_domain(report, fail, d, derivative, observed, rd, rout)
            if fisheye_domain != 'hemisphere': return fail('unknown_fisheye_domain')
            report.update(checked_theta_max_rad=float(np.pi/2),
                          domain_policy='legacy entire front hemisphere retained (not inferred hardware FOV)')
            if not report['hemisphere_derivative_certified']: return fail('fisheye_angular_fold_or_uncertified')
            theta_max = np.pi/2
            image_radius_max = theta_max*(1+sum(d[i]*theta_max**(2*i+2) for i in range(4)))
            report['front_hemisphere_distorted_radius_max'] = float(image_radius_max)
            if rd > image_radius_max:
                return fail('raw_roi_outside_front_hemisphere_inverse_domain')
        elif result.model_name == M.EXTENDED_PINHOLE:
            if len(d)!=8: return fail('invalid_rational_coefficient_count')
            if radius<=0: return fail('missing_physical_domain')
            p = P([1,0,d[0],0,d[1],0,d[4]])
            q = P([1,0,d[5],0,d[6],0,d[7]])
            pr, qr = _positive_roots(p), _positive_roots(q)
            report.update(numerator_roots_radius=pr, denominator_roots_radius=qr,
                          numerator_at_denominator_roots=[float(p(r)) for r in qr],
                          nearest_root_separation=min((abs(a-b) for a in pr for b in qr),default=None))
            radial_num = (p+P([0,1])*p.deriv())*q-P([0,1])*p*q.deriv()
            report.update(radial_derivative_roots_radius=_positive_roots(radial_num),
                          denominator_min_on_initial_domain=_polynomial_minimum(q,radius),
                          denominator_relative_margin=float(np.sqrt(np.finfo(float).eps)),
                          near_cancellation_policy='never remove common/near-common factors from stored OpenCV D')
            # Never cancel even exactly common factors: OpenCV evaluates P/Q.
            if not _positive_on_disk(q,radius,relative_margin=np.sqrt(np.finfo(float).eps)): return fail('rational_denominator_zero_near_zero_or_uncertified')
            tangent = float(np.hypot(d[2],d[3]))
            if rd:
                # |tangential(r,phi)| <= 3*|p|*r^2. The first root of
                # r*P/Q - 3*|p|*r^2 = Rd encloses the distorted ROI by
                # boundary separation. Final SPD certificate establishes a
                # one-to-one, axis-connected map inside the resulting disk.
                reach = P([0,1])*p - (P([rd,0,3*tangent]))*q
                roots = [r for r in _positive_roots(reach) if r>0 and q(r)>0]
                if not roots: return fail('raw_roi_inverse_envelope_not_certified')
                boundary = _certified_boundary(reach, min(roots))
                if boundary is None: return fail('raw_roi_inverse_boundary_not_certified')
                inverse, residual, error = boundary
                radius = max(radius,inverse)
                report.update(raw_inverse_envelope_radius=inverse,checked_radius=radius,
                              inverse_boundary_polynomial_residual=residual,
                              inverse_boundary_roundoff_bound=error)
            if not _positive_on_disk(q,radius,relative_margin=np.sqrt(np.finfo(float).eps)): return fail('rational_denominator_zero_near_zero_or_uncertified')
            # d(r*P/Q)/dr numerator, and symmetric Jacobian tangential
            # perturbation bound ||J_t|| <= 6*|p|*r (all azimuths).
            radial_num = (p+P([0,1])*p.deriv())*q-P([0,1])*p*q.deriv()
            report['radial_derivative_roots_radius'] = _positive_roots(radial_num)
            if not _positive_on_disk(p,radius) or not _positive_on_disk(radial_num,radius):
                return fail('radial_fold_or_uncertified')
            lower_scale = p-P([0,6*tangent])*q
            lower_radial = radial_num-P([0,6*tangent])*q*q
            if not (_positive_on_disk(lower_scale,radius) and _positive_on_disk(lower_radial,radius)):
                return fail('tangential_2d_jacobian_not_certified_positive_definite')
            report['jacobian_certificate'] = 'SPD everywhere in disk: min(s, d(r*s)/dr) > 6*norm(p1,p2)*r'
        report['status'] = 'PASS'
        return report
    except (ValueError,TypeError,IndexError,np.linalg.LinAlgError,cv2.error,OverflowError) as exc:
        return fail('invalid_or_uncertified_domain: '+str(exc))


def enforce_distortion_validity(result, camera, data):
    """Gate newly computed results only; preserve fitted numbers as evidence."""
    if result.success and result.model_name in (M.EXTENDED_PINHOLE,M.FISHEYE):
        diagnostic = distortion_diagnostic(result,camera,data,fisheye_domain='usage')
        result.optimization_audit['physical_validity'] = diagnostic
        if diagnostic['status'] != 'PASS':
            result.success = False
            result.error_message = 'Physical distortion validity: '+', '.join(diagnostic['reasons'])
    return result
