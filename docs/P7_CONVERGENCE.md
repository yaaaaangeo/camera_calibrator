# P7 perspective solver convergence completion

`calibration.models.common.complete_perspective_convergence` is applied after the
native `cv2.calibrateCameraExtended` call for Brown-Conrady and Rational
(`_calibrate_extended_pinhole_core`) and for Brown/Rational initial-condition
samples in `calibration.repeatability`. Pinhole, Fisheye and Object-Releasing
paths are unchanged.

## Why

Measured on stored Train frames (no Test use):

| Case | Native (100, 1e-6) | Completed |
|---|---:|---:|
| Old ROSbag Train28 Rational, no guess | 3.0008px | 1.6391px |
| Old ROSbag Train28 Brown, no guess | 3.6524px | 2.7139px |
| rosbag_image Train27 Rational, perturbed `CALIB_USE_INTRINSIC_GUESS` | 2.3641px | 1.4336px |

The unconverged and converged Train28 maps differ by up to ~140px on the sensor.
A seeded start stopped at the same point with 5000 iterations; a restart from
its result reached the no-guess optimum. A restart can also diverge (Train28
Brown 28 -> 47px), so it is never accepted without an objective decrease.

## Rule

Which candidates are used depends on the number of views:

- `<= DENSE_CONVERGENCE_MAX_VIEWS` (40) views, or flags the sparse solver does
  not parameterize: the original OpenCV candidates (`method: opencv_long`,
  below). This keeps every P7-validated solution unchanged.
- more views: sparse candidates (`method: sparse_lm`).

Sparse candidates, solved by a sparse Schur-complement Levenberg-Marquardt on
the same objective and parameterization as OpenCV
(`calibration/models/sparse_refine.py`, Jacobi-scaled reduced system):

1. `continue_sparse`: continue from the native state (K, D, per-view rvec/tvec);
2. `restart_sparse`: keep the native K/D, re-initialize per-view poses
   (`solvePnP` + `solvePnPRefineLM`), then minimize.

OpenCV candidates (`method: opencv_long`), both with
`(EPS+MAX_ITER, 5000, 1e-14)`:

1. same start (same initial K/D or none, same flags);
2. restart from the native K/D with `CALIB_USE_INTRINSIC_GUESS`.

The lowest finite candidate replaces the native result only when its RMS on the
same observations is below `native_rms * (1 - 1e-4)`. The tolerance is a
numerical-noise guard, not a quality threshold. Otherwise the native tuple is
returned unchanged, so already-converged results are bitwise identical.
`optimization_audit['convergence']` records the method, the native RMS, the
candidates and the adopted one. An adopted sparse candidate returns the
`calibrateCameraExtended` tuple layout; std deviations use OpenCV's normal
approximation (`sigma^2 = SSE / (2N - n_params)`) and match OpenCV within 0.1%
on stored data.

Only the solver objective on its own fit set is compared. Test, hold-out,
validation, recommendation and Confidence values are never consulted.
Physical validity (P6) is applied afterwards as before.

### Why sparse (2026-11-09)

`cv2.calibrateCameraExtended` solves a dense system over the intrinsics and all
6 x n_views pose parameters every iteration, so its cost is cubic in the view
count. Measured on 960x540 wide-angle rosbag data, single thread: 146 views,
2.3s per iteration; one Brown/Rational fit with the OpenCV candidates took
15-25 min (236 views), and a 3-model CLI run did not finish in 2h. The sparse
solver inverts only 6x6 blocks per view, so an iteration is linear in views.

Equivalence on the stored P7 data (A old ROSbag, B 15_35_43, C rosbag_image;
Full and Train; Brown and Rational; 12 fits): every adopted result equals the
OpenCV-candidate result to the reported digits.

| Case | Native | OpenCV candidates (adopted) | Sparse (adopted) | Sparse time |
|---|---:|---:|---:|---:|
| A Train Brown | 3.6524 | 2.71394 (same_start_long) | 2.71394 (continue_sparse) | 1.3s |
| A Train Rational | 3.0008 | 1.63915 (restart_long) | 1.63915 (restart_sparse) | 1.7s |
| C Full Rational | 2.0548 | 2.02424 (same_start_long) | 2.02424 (continue_sparse) | 0.5s |
| other 9 fits | converged | native | native | 0.1-0.4s |

Both solvers stop at true local minima, but Rational is not identified by the
data and the two solver paths can reach different minima: synthetic self-check
seed 3 reached 0.262171px (OpenCV, physically valid) vs 0.262331px (sparse,
zero gradient, OpenCV restart stays there, physically invalid); seed 42 the
other way (sparse 0.26420 < OpenCV 0.26437). This is why small view counts keep
the OpenCV candidates.

Wide-angle data (960x540): front 236 views Rational 0.5046 native kept in
0.13s (OpenCV candidates: 926-1090s, also kept native); right 146 views Brown
5.69 -> 5.29px in 9.7s, Rational 8.60 -> 8.08px in 6.2s.

## Effect

- Stored Full-fit K/D/RMS reproduce bitwise for 8 of 9 stored model results
  (old ROSbag, 15_35_43 and rosbag_image; Brown/Rational/Fisheye). The
  rosbag_image Rational Full-fit was unconverged (2.0548 -> 2.0242px). It is
  physically invalid before and after.
- Hold-out Train-only fits that were unconverged (old ROSbag) change when they
  are recomputed. Stored results are not rewritten.
- Robust Optimization keeps the legacy perspective policy, so its final
  perspective K/D are those of the native wrapper. Seeded candidates now converge.
- Rational physical validity can flip under completion (synthetic self-check
  seed 3 FAIL->PASS, seed 42 PASS->FAIL). Rational outside the observed ray
  radius is not identified by data, and an early stop is not a safeguard.
- Cost: two extra candidate solves per Brown/Rational fit, also when already
  converged. With the original OpenCV candidates (4K, 27-35 frames): +0.8-7.4s
  per fit, worst +16.6s, and cubic growth with view count (see "Why sparse").
  With the sparse candidates: 0.1-0.4s when converged, up to a few seconds
  when the native result stopped early, linear in view count. Every
  Brown/Rational fit (hold-out, k-fold, outlier passes, robust candidates) pays
  this.

Tests: `tests/test_convergence_completion.py`.
