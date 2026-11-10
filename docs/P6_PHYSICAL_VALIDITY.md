# P6 distortion validity

Diagnostic version: `p6-continuous-domain-v2` (v1 applied the Fisheye front
hemisphere to every path; v2 uses the usage domain in base calibration only).

`calibration.physical_validity.distortion_diagnostic` is a read-only, JSON-safe
numeric diagnostic. It never fits K/D, solves poses, changes frame selection or
reads outer-Test metrics. `enforce_distortion_validity` gates newly calculated
Rational/Fisheye results while retaining K/D, poses and RMS as failure evidence.
The native wrappers, robust optimizer and model-result cache-hit path apply it.
Failure sets `success=False`; existing recommendation/acceptance/export rules
then handle failure without new score weights or quality thresholds.
`optimization_audit.physical_validity` persists the version/domain/reasons.
Loading a historical project does not rewrite or retroactively invalidate it;
run the read-only diagnostic separately to audit its frozen numeric output.

## Coordinates and domain

OpenCV extrinsics map board to camera: `Xc=R*Xb+t`. Ideal forward-map input is
`(Xc/Zc,Yc/Zc)`. Applying `K^-1` to **raw distorted** pixels instead gives
**distorted normalized** coordinates. These radii are not interchangeable.
All retained fitted corners must have finite positive camera depth; the board
coordinate origin need not be visible or have positive depth.

The perspective certificate uses an axis-connected disk enclosing:

1. All retained fitted ideal camera rays (after native frame/corner exclusions).
2. The full-size corrected output pixel-edge rectangle, using its own K.
   Current perspective remap uses the input K and resolution. A custom output
   requires both `output_matrix` and `output_roi=(x,y,width,height)`.
3. The raw ROI's inverse envelope, as well as the old raw/K bound (no relaxation).
   Default raw ROI is the complete sensor rectangle `[0,width] x [0,height]`.

For Rational radial scale `s=P(r²)/Q(r²)` and `p=hypot(p1,p2)`, tangential
magnitude is bounded by `3*p*r²`. Solve the polynomial boundary equation
`r*P - (Rd+3*p*r²)*Q = 0`, where Rd encloses the raw ROI after K inverse.
The first positive boundary root is only a candidate. An outward correction
derived from its residual and coefficient roundoff bound must establish strict
boundary separation; otherwise fail closed. This boundary plus the continuous positive-definite Jacobian
certificate below encloses an axis-connected inverse branch covering that ROI.
Do not use iterative `undistortPoints` as a guaranteed global inverse.
Resolution mismatches, missing domain, inability to establish an envelope or
uncertified positivity fail closed. Explicit raw crops must be in original
sensor coordinates; custom cropped output needs a correspondingly shifted K.
This API does not infer crop/resize/rectification transforms.

A hood Occlusion Mask describes scene visibility for coverage/readiness. It
neither crops the sensor/output geometry nor removes calibration corners. It
cannot reduce this validity domain. Only an explicit geometric ROI can do so;
retained rays and the actual output domain still constrain the disk. The disk
can include rays outside a noncircular usable region, so rejection can be
conservative. It is not a maximal visible-region proof.

## Continuous checks

Report positive real numerator/denominator roots and their separation, with
near-real roots conservatively included. Never cancel common/near-common
factors: the stored OpenCV model still evaluates P/Q. Check Q strictly positive
on the whole domain, including endpoints. The relative denominator safety
margin is `sqrt(float64 epsilon)` against the polynomial coefficient scale,
a numerical conditioning guard, not a tuned calibration-quality threshold.

Check positive scale and positive radial derivative. For the exact symmetric
2-D Jacobian, the radial eigenvalues are `s` and `d(r*s)/dr`; the tangential
Jacobian perturbation has spectral norm at most `6*p*r`. Certify both
`s > 6*p*r` and `d(r*s)/dr > 6*p*r` across the full disk, after multiplying by
positive Q or Q² to obtain polynomials in r. This sufficient SPD condition
proves local orientation preservation and injectivity on the convex disk.
It is conservative: an uncertified Jacobian is not necessarily a witnessed fold.
`rational_jacobian` provides the exact analytic Jacobian for point diagnostics.

Polynomial positivity uses a Bernstein convex-hull certificate with adaptive
de Casteljau subdivision, coefficient roundoff padding and fixed work/depth
limits. No uniformly spaced radial/angular samples decide PASS. Unresolved
near-zero intervals, repeated roots and work-limit exhaustion fail closed.
Roots are diagnostic/envelope candidates; they alone never certify positivity.
This is a float64 numerical certificate, not formal exact-arithmetic verification.

## Fisheye domain by path

`theta_d = theta*(1+k1*theta^2+k2*theta^4+k3*theta^6+k4*theta^8)`.
`fisheye_domain` selects the angular interval on which `d(theta_d)/d(theta)>0`
is certified (same Bernstein certificate, never sampled):

| Path | `fisheye_domain` | Certified interval |
|---|---|---|
| Base calibration gate (`enforce_distortion_validity`: native wrappers, cache hit) | `usage` | `[0, theta_use]` |
| Robust Optimization `admissible` (unchanged legacy rule) | `hemisphere` (default) | `[0, pi/2]` |

`theta_use = max(atan(max retained fitted ray radius), atan(output ROI ideal
radius), theta_raw)`, where `theta_raw` is the first root of
`theta_d(theta) = Rd` (Rd: raw ROI radius after K inverse), moved outward
by the same residual/roundoff boundary certificate as Rational. Base FAIL when:

* the raw ROI is not reached on the monotone branch before `pi/2`
  (`raw_roi_not_reached_before_fold_or_front_hemisphere`: sensor pixels with no
  inverse ray, e.g. theta_d peaks below the corner radius);
* `theta_use >= pi/2` (`usage_domain_outside_front_hemisphere`);
* a fold or uncertified derivative inside `[0, theta_use]`, including exactly
  at its boundary (`fisheye_angular_fold_in_usage_domain_or_uncertified`).

A fold beyond `theta_use` alone does not reject base calibration; it is
recorded (`angular_derivative_roots_rad`, `hemisphere_derivative_certified`,
`fold_outside_usage_domain_only=true`). Such K/D must not be used for rays
beyond `checked_theta_max_rad` (wider rectification output, other cameras with
larger image circles, extrapolated FOV). Robust Optimization still rejects it.
The output ROI uses the same balance=0 new K as the current quality remap; if
OpenCV's estimate for that K is degenerate the output angle approaches pi/2
and the result fails closed. No hardware FOV is inferred from either verdict.
Circular fisheye sensors whose corners lie outside the image circle need an
explicit `raw_roi`; the default full sensor rectangle may reject them.

## Compatibility and limits

The fitter objectives, flags, initializations, stopping rules, frame selection,
Train/Test split, recommendation weights and acceptance policy are unchanged.
A native numerical fit formerly called successful may now be physically rejected;
its numeric K/D and RMS remain available for diagnostics and cannot be exported
through the existing successful-result YAML path. Old files are not overwritten.
Existing cache entries are checked in memory without rewriting old cache files.
A cache entry already rejected remains rejected; there is no automatic upgrade.

Every report carries `interpretation`: PASS establishes necessary geometric
conditions only; it is not a calibration quality PASS or a vehicle approval.
PASS establishes necessary geometric conditions only. It does not establish
small residuals, parameter observability, correct board specification, measured
hardware FOV or vehicle suitability. New future rectification/ROI consumers
must supply their actual coordinate geometry for a separate diagnostic.

Tests: `tests/test_physical_validity.py` includes the Fisheye usage/hemisphere
split (fold outside usage passes base and fails Robust, fold inside rays or at
the usage boundary fails, unreachable raw corner fails), and the frozen 80/60 dataset pole,
normal old-bag K/D, common factors, near-zero/repeated roots, boundary roots,
radial and tangential folds, exact-Jacobian finite differences, mask/ROI rules,
repeatability, no mutation, native/robust gates and invalid YAML blocking.
