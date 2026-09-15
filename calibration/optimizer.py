"""Leak-safe post-OpenCV calibration refinement using SciPy least squares.

The optimizer only receives the frozen training IDs while it is fitting.  The
hold-out IDs are passed to the evaluator *after* the final K/D has been chosen;
the evaluator estimates poses only and never feeds a value back into fitting.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from calibration.models.common import (
    collect_calibration_inputs,
    compute_regional_error,
    distortion_coeff_labels,
    project_points_for_model,
    regional_edge_average,
    validate_finite_calibration_output,
)
from calibration.observability import attach_observability_report
from calibration.residual_stats import compute_residual_stats
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    OptimizerMetricSet,
    OptimizerResult,
    OptimizerSettings,
    OptimizerStageResult,
    OptimizerStartResult,
    PatternConfig,
)
from calibration.validation import _evaluate_on_test, refit_on_train_split


PIPELINE_STEPS = (
    "OpenCV Initial Calibration",
    "Multi-start",
    "Best Initial Solution",
    "Robust Bundle Adjustment",
    "Huber Loss",
    "Staged Parameter Release",
    "Final Joint Optimization",
    "Hold-out Evaluation",
)


class OptimizationCancelled(RuntimeError):
    pass


def _subset(dataset: Dataset, ids: Iterable[str]) -> Dataset:
    wanted = set(ids)
    return Dataset(frames=[f for f in dataset.frames if f.image_info.image_id in wanted])


def parameter_definition(model: CameraModelType, distortion_count: int) -> dict[str, list[list[str]] | list[str]]:
    """Single source of truth for OpenCV coefficient order and release stages."""
    distortion = [] if model == CameraModelType.PINHOLE else distortion_coeff_labels(model, distortion_count)
    if model == CameraModelType.FISHEYE:
        stages = [["fx", "fy", "cx", "cy"], ["k1", "k2"], ["k3", "k4"]]
    elif model == CameraModelType.EXTENDED_PINHOLE:
        stages = [
            ["fx", "fy", "cx", "cy"], ["k1", "k2"],
            ["p1", "p2", "k3"], ["k4", "k5", "k6"],
        ]
    elif model == CameraModelType.BROWN_CONRADY:
        stages = [["fx", "fy", "cx", "cy"], ["k1", "k2"], ["p1", "p2", "k3"]]
    else:
        stages = [["fx", "fy", "cx", "cy"]]
    return {"intrinsics": ["fx", "fy", "cx", "cy"], "distortion": distortion, "stages": stages}


def _observations(dataset: Dataset, excluded_frame_ids: set[str] | None = None):
    frames, objects, images = collect_calibration_inputs(dataset)
    excluded_frame_ids = excluded_frame_ids or set()
    kept = [(f, o, i) for f, o, i in zip(frames, objects, images)
            if f.image_info.image_id not in excluded_frame_ids]
    return (
        [x[0] for x in kept],
        [np.asarray(x[1], dtype=np.float64) for x in kept],
        [np.asarray(x[2], dtype=np.float64) for x in kept],
    )


def _pack(result: CalibrationResult) -> tuple[np.ndarray, int]:
    K = np.asarray(result.camera_matrix, dtype=np.float64)
    D = (
        np.empty(0, dtype=np.float64)
        if result.model_name == CameraModelType.PINHOLE
        else np.asarray(result.distortion, dtype=np.float64).reshape(-1)
    )
    head = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D.tolist()]
    poses: list[float] = []
    for rv, tv in zip(result.rvecs, result.tvecs):
        poses.extend(np.asarray(rv, dtype=float).reshape(3).tolist())
        poses.extend(np.asarray(tv, dtype=float).reshape(3).tolist())
    return np.asarray(head + poses, dtype=np.float64), len(D)


def _unpack(x: np.ndarray, d_count: int, pose_count: int):
    K = np.array([[x[0], 0.0, x[2]], [0.0, x[1], x[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = x[4:4 + d_count].reshape(-1, 1).copy()
    pose = x[4 + d_count:].reshape(pose_count, 6)
    rvecs = [row[:3].reshape(3, 1).copy() for row in pose]
    tvecs = [row[3:].reshape(3, 1).copy() for row in pose]
    return K, D, rvecs, tvecs


def _labels(model: CameraModelType, d_count: int, pose_count: int) -> list[str]:
    labels = ["fx", "fy", "cx", "cy"] + ([] if model == CameraModelType.PINHOLE else distortion_coeff_labels(model, d_count))
    labels.extend(f"pose[{i}].{p}" for i in range(pose_count) for p in ("rx", "ry", "rz", "tx", "ty", "tz"))
    return labels


def _bounds(x: np.ndarray, d_count: int, pose_count: int, image_size: tuple[int, int]):
    w, h = image_size
    dim = float(max(w, h))
    low = np.full(x.size, -np.inf); high = np.full(x.size, np.inf)
    low[:4] = [0.05 * dim, 0.05 * dim, -0.5 * w, -0.5 * h]
    high[:4] = [10.0 * dim, 10.0 * dim, 1.5 * w, 1.5 * h]
    low[4:4 + d_count] = -10.0; high[4:4 + d_count] = 10.0
    # Keep poses finite without imposing assumptions about target units.
    pose_start = 4 + d_count
    low[pose_start:] = -1.0e6; high[pose_start:] = 1.0e6
    return low, high


def _raw_residual(x, d_count, objects, images, model, cancel_check):
    if cancel_check and cancel_check():
        raise OptimizationCancelled("Optimization cancelled")
    K, D, rvecs, tvecs = _unpack(x, d_count, len(objects))
    chunks = []
    for obj, img, rv, tv in zip(objects, images, rvecs, tvecs):
        projected = project_points_for_model(obj, rv, tv, K, D, model)
        chunks.append((np.asarray(img, dtype=float).reshape(-1, 2) - projected).reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty(0)


def _rms(residual: np.ndarray) -> float:
    pairs = np.asarray(residual, dtype=float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(pairs * pairs, axis=1))))


def _robust_objective(residual: np.ndarray, settings: OptimizerSettings, scale: float) -> float:
    z = np.asarray(residual, dtype=float) / max(scale, 1.0)
    threshold = max(settings.huber_delta / max(scale, 1.0), np.finfo(float).eps)
    q = (z / threshold) ** 2
    loss = settings.robust_loss.lower().replace("-", "_")
    if loss == "linear":
        rho = z ** 2
    elif loss == "huber":
        rho = np.where(q <= 1.0, z ** 2, threshold ** 2 * (2.0 * np.sqrt(q) - 1.0))
    elif loss == "soft_l1":
        rho = 2.0 * threshold ** 2 * (np.sqrt(1.0 + q) - 1.0)
    elif loss == "cauchy":
        rho = threshold ** 2 * np.log1p(q)
    else:
        raise ValueError(f"Unsupported robust loss: {settings.robust_loss}")
    return float(np.mean(rho))


def _sanity(x, d_count, pose_count, image_size) -> str | None:
    if not np.all(np.isfinite(x)):
        return "non-finite parameters"
    K, D, rvecs, tvecs = _unpack(x, d_count, pose_count)
    invalid = validate_finite_calibration_output(K, D)
    if invalid:
        return invalid
    w, h = image_size
    if not (-0.5 * w <= K[0, 2] <= 1.5 * w and -0.5 * h <= K[1, 2] <= 1.5 * h):
        return "principal point outside sanity bounds"
    if np.max(np.abs(D), initial=0.0) > 10.0:
        return "unreasonable distortion"
    if any(not np.all(np.isfinite(v)) for v in [*rvecs, *tvecs]):
        return "non-finite pose"
    return None


def _active_names(settings: OptimizerSettings, released: set[str], labels: list[str]) -> set[str]:
    names: set[str] = set()
    if settings.optimize_focal_length:
        names.update(("fx", "fy"))
    if settings.optimize_principal_point:
        names.update(("cx", "cy"))
    if settings.optimize_distortion:
        names.update(n for n in released if n not in ("fx", "fy", "cx", "cy"))
    if settings.optimize_extrinsics:
        names.update(n for n in labels if n.startswith("pose["))
    return names


def _least_squares_stage(x, active_names, labels, bounds, residual_fn, settings, max_nfev,
                         point_counts, normalization_scale):
    indices = np.asarray([i for i, name in enumerate(labels) if name in active_names], dtype=int)
    if indices.size == 0:
        return x.copy(), None
    base = x.copy()
    lo, hi = bounds
    # Residuals are optimized in focal-normalized image units.  This keeps a
    # 1 px Huber threshold equivalent across resolutions while the UI remains
    # intuitive (huber_delta is entered in pixels).
    scale = max(float(normalization_scale), 1.0)

    def fun(active):
        full = base.copy(); full[indices] = active
        return residual_fn(full) / scale

    # Bundle-adjustment Jacobians are block sparse: every observation depends
    # on global K/D and only on its own frame pose.  Supplying this structure
    # avoids a dense finite-difference pass over 6*N pose columns per step.
    row_count = 2 * sum(point_counts)
    sparsity = lil_matrix((row_count, indices.size), dtype=int)
    global_cols = [j for j, source in enumerate(indices) if not labels[source].startswith("pose[")]
    if global_cols:
        sparsity[:, global_cols] = 1
    row = 0
    for frame_index, count in enumerate(point_counts):
        rows = slice(row, row + 2 * count)
        prefix = f"pose[{frame_index}]."
        pose_cols = [j for j, source in enumerate(indices) if labels[source].startswith(prefix)]
        if pose_cols:
            sparsity[rows, pose_cols] = 1
        row += 2 * count

    loss = settings.robust_loss.lower().replace("-", "_")
    if loss not in {"linear", "huber", "soft_l1", "cauchy"}:
        raise ValueError(f"Unsupported robust loss: {settings.robust_loss}")
    result = least_squares(
        fun, base[indices], bounds=(lo[indices], hi[indices]), loss=loss,
        f_scale=max(settings.huber_delta / scale, np.finfo(float).eps),
        x_scale="jac", max_nfev=max_nfev, xtol=1e-7, ftol=1e-7, gtol=1e-7,
        jac_sparsity=sparsity.tocsr(),
    )
    out = base.copy(); out[indices] = result.x
    return out, result


def _candidate_vectors(
    x: np.ndarray, d_count: int, image_size: tuple[int, int], count: int,
    pinhole_seed: CalibrationResult | None = None,
    brown_seed: CalibrationResult | None = None,
):
    w, h = image_size
    candidates = [("OpenCV", x.copy())]
    if pinhole_seed is not None and pinhole_seed.success and pinhole_seed.camera_matrix is not None:
        pinhole = x.copy()
        K = pinhole_seed.camera_matrix
        pinhole[:4] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        pinhole[4:4 + d_count] = 0.0
        candidates.append(("Pinhole K / zero distortion", pinhole))
    if brown_seed is not None and brown_seed.success and brown_seed.camera_matrix is not None:
        brown = x.copy()
        K = brown_seed.camera_matrix
        brown[:4] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        brown[4:4 + d_count] = 0.0
        # Brown and Rational share the first five OpenCV coefficient slots.
        if d_count >= 5 and brown_seed.distortion is not None:
            count_shared = min(5, d_count, int(brown_seed.distortion.size))
            brown[4:4 + count_shared] = brown_seed.distortion.reshape(-1)[:count_shared]
        candidates.append(("Brown-derived K/D", brown))
    centered = x.copy(); centered[2:4] = [w / 2.0, h / 2.0]
    candidates.append(("Centered principal point", centered))
    for pct in (0.03, -0.03, 0.06, -0.06):
        perturbed = x.copy(); perturbed[:2] *= 1.0 + pct
        candidates.append((f"Focal {pct:+.0%}", perturbed))
    return candidates[:max(1, count)]


def _build_calibration(model, x, d_count, frames, objects, images, initial, image_size):
    K, D, rvecs, tvecs = _unpack(x, d_count, len(frames))
    result_distortion = np.zeros((5, 1), dtype=np.float64) if model == CameraModelType.PINHOLE else D
    point_errors: list[float] = []
    per_frame: dict[str, float] = {}
    xs: list[float] = []; ys: list[float] = []
    for frame, obj, img, rv, tv in zip(frames, objects, images, rvecs, tvecs):
        detected = np.asarray(img, dtype=float).reshape(-1, 2)
        projected = project_points_for_model(obj, rv, tv, K, D, model)
        errors = np.linalg.norm(detected - projected, axis=1)
        point_errors.extend(errors.tolist()); xs.extend(detected[:, 0]); ys.extend(detected[:, 1])
        per_frame[frame.image_info.image_id] = float(np.sqrt(np.mean(errors ** 2)))
    stats = compute_residual_stats(point_errors)
    regional = compute_regional_error(xs, ys, point_errors, image_size)
    result = CalibrationResult(
        model_name=model, camera_matrix=K, distortion=result_distortion, rvecs=rvecs, tvecs=tvecs,
        rms_error=stats.rmse, per_frame_error=per_frame, residual_stats=stats,
        regional_error=regional, success=True,
        input_frame_count=len(frames) + len(initial.excluded_frame_ids),
        used_frame_count=len(frames), calibration_method=initial.calibration_method,
        excluded_frame_ids=list(initial.excluded_frame_ids),
        exclusion_reason=initial.exclusion_reason,
    )
    return result


def _metric_set(cal: CalibrationResult, validation=None) -> OptimizerMetricSet:
    pu = cal.param_uncertainty_bootstrap or cal.param_uncertainty
    obs = cal.observability
    stats = validation.test_residual_stats if validation else None
    return OptimizerMetricSet(
        train_rms=cal.rms_error,
        test_rms=validation.test_rms if validation else None,
        test_p95=stats.p95 if stats else None,
        test_p99=stats.p99 if stats else None,
        edge_rms=validation.edge_rms if validation else regional_edge_average(cal.regional_error),
        stability=pu.overall_stability if pu else None,
        observability=obs.observability_score if obs else None,
    )


def _recommend(before: OptimizerMetricSet, after: OptimizerMetricSet) -> tuple[str, list[str]]:
    reasons: list[str] = []
    def change(name, a, b):
        if a is None or b is None or a == 0:
            return None
        pct = (b - a) / a * 100.0
        reasons.append(f"{name} {'improved' if pct < 0 else 'worsened'} by {abs(pct):.1f}%")
        return pct
    train = change("Train RMS", before.train_rms, after.train_rms)
    test = change("Hold-out RMS", before.test_rms, after.test_rms)
    p95 = change("Hold-out P95", before.test_p95, after.test_p95)
    p99 = change("Hold-out P99", before.test_p99, after.test_p99)
    edge = change("Edge RMS", before.edge_rms, after.edge_rms)
    stability_drop = None
    if before.stability is not None and after.stability is not None:
        stability_drop = before.stability - after.stability
        reasons.append(
            f"Stability {'improved' if stability_drop < 0 else 'worsened'} by {abs(stability_drop):.1f} points"
        )
    elif before.stability is not None:
        reasons.append("Optimized stability was not re-estimated; no bootstrap value was fabricated")
    observability_drop = None
    if before.observability is not None and after.observability is not None:
        observability_drop = before.observability - after.observability
        reasons.append(
            f"Observability {'improved' if observability_drop < 0 else 'worsened'} by {abs(observability_drop):.1f} points"
        )
    if test is None:
        reasons.append("Frozen hold-out evidence is unavailable")
        return "Neutral / Marginal", reasons
    if (test > 2.0 or (p95 is not None and p95 > 5.0)
            or (p99 is not None and p99 > 5.0) or (edge is not None and edge > 5.0)
            or (stability_drop is not None and stability_drop > 5.0)
            or (observability_drop is not None and observability_drop > 10.0)):
        reasons.append("Possible overfitting detected")
        return "Not Recommended", reasons
    stability_known_enough = before.stability is None or after.stability is not None
    if (train is not None and train < -1.0 and test < -1.0
            and (p95 is None or p95 <= 1.0) and (p99 is None or p99 <= 1.0)
            and (edge is None or edge <= 1.0) and stability_known_enough):
        return "Recommended", reasons
    return "Neutral / Marginal", reasons


class BaseCalibrationOptimizer:
    def optimize(self, *args, **kwargs) -> OptimizerResult:  # pragma: no cover - interface
        raise NotImplementedError


def apply_optimized_calibration(
    current: CalibrationResult, optimizer_result: OptimizerResult
) -> CalibrationResult:
    """Explicit apply operation; running optimization never calls this."""
    if not optimizer_result.success or optimizer_result.optimized_calibration is None:
        raise ValueError("No successful optimized calibration to apply")
    if not optimizer_result.applied:
        optimizer_result.pre_apply_calibration = deepcopy(current)
    optimizer_result.applied = True
    return deepcopy(optimizer_result.optimized_calibration)


def restore_original_calibration(optimizer_result: OptimizerResult) -> CalibrationResult:
    if optimizer_result.pre_apply_calibration is None:
        raise ValueError("No pre-apply OpenCV calibration snapshot")
    optimizer_result.applied = False
    return deepcopy(optimizer_result.pre_apply_calibration)


class ScipyCalibrationOptimizer(BaseCalibrationOptimizer):
    def optimize(
        self, dataset: Dataset, camera_config: CameraConfig, pattern_config: PatternConfig,
        model: CameraModelType, train_ids: list[str], holdout_ids: list[str],
        settings: OptimizerSettings | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[str, str], None] | None = None,
    ) -> OptimizerResult:
        settings = settings or OptimizerSettings()
        status = {step: "pending" for step in PIPELINE_STEPS}
        output = OptimizerResult(model_name=model, settings=deepcopy(settings),
                                 train_frame_ids=list(train_ids), holdout_frame_ids=list(holdout_ids),
                                 pipeline_status=status, input_training_frame_ids=list(train_ids))
        def mark(step, state):
            status[step] = state
            if progress: progress(step, state)
        try:
            mark("OpenCV Initial Calibration", "running")
            # Crucial: reproduce OpenCV on frozen TRAIN only.  The full-data result
            # is intentionally not accepted as an initialization.
            pinhole_seed = None
            if model == CameraModelType.FISHEYE or (
                settings.multi_start and model != CameraModelType.PINHOLE
            ):
                pinhole_seed = refit_on_train_split(
                    dataset, camera_config, CameraModelType.PINHOLE, train_ids
                )
            fisheye_guess = pinhole_seed if model == CameraModelType.FISHEYE else None
            initial = refit_on_train_split(
                dataset, camera_config, model, train_ids,
                fisheye_initial_guess=fisheye_guess,
            )
            if not initial.success:
                raise RuntimeError(initial.error_message or "OpenCV train-only calibration failed")
            output.original_calibration = deepcopy(initial)
            mark("OpenCV Initial Calibration", "completed")

            train_dataset = _subset(dataset, train_ids)
            excluded = set(initial.excluded_frame_ids)
            frames, objects, images = _observations(train_dataset, excluded)
            if len(frames) != len(initial.rvecs):
                raise RuntimeError("OpenCV pose/frame accounting mismatch")
            output.used_training_frame_ids = [f.image_info.image_id for f in frames]
            output.excluded_training_frame_ids = list(initial.excluded_frame_ids)
            output.exclusion_reason = initial.exclusion_reason
            x0, d_count = _pack(initial)
            normalization_scale = max(float((x0[0] + x0[1]) * 0.5), 1.0)
            labels = _labels(model, d_count, len(frames))
            bounds = _bounds(x0, d_count, len(frames), (camera_config.width, camera_config.height))
            raw = lambda x: _raw_residual(x, d_count, objects, images, model, cancel_check)
            definition = parameter_definition(model, d_count)

            brown_seed = None
            if settings.multi_start and model in (
                CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE
            ):
                brown_seed = refit_on_train_split(
                    dataset, camera_config, CameraModelType.BROWN_CONRADY, train_ids
                )

            mark("Multi-start", "running")
            candidates = _candidate_vectors(x0, d_count, (camera_config.width, camera_config.height),
                                            settings.num_starts if settings.multi_start else 1,
                                            pinhole_seed=pinhole_seed, brown_seed=brown_seed)
            released_first = set(definition["stages"][0])
            if definition["distortion"]:
                released_first.update(definition["distortion"][:2])
            preliminary_names = _active_names(settings, released_first, labels)
            best = None; best_obj = float("inf"); best_start_index = -1
            for name, candidate in candidates:
                try:
                    candidate = np.clip(candidate, bounds[0] + 1e-12, bounds[1] - 1e-12)
                    fitted, ls = _least_squares_stage(
                        candidate, preliminary_names, labels, bounds, raw, settings,
                        settings.max_iterations_per_stage,
                        [len(obj) for obj in objects],
                        normalization_scale,
                    )
                    # The OpenCV candidate is already a successfully converged
                    # calibration. If its short preliminary refinement reaches
                    # the cap, retain that certified seed instead of relabeling
                    # a capped SciPy iterate as converged.
                    if name == "OpenCV" and ls is not None and not ls.success:
                        fitted = candidate.copy()
                    residual = raw(fitted)
                    # Multi-start selection uses the configured TRAIN objective,
                    # never hold-out metrics.  SciPy's cost already includes the
                    # chosen robust rho function and f_scale.
                    objective = _robust_objective(residual, settings, normalization_scale)
                    reason = _sanity(fitted, d_count, len(frames), (camera_config.width, camera_config.height))
                    converged = (
                        reason is None and ls is not None
                        and (bool(ls.success) or name == "OpenCV")
                    )
                    output.starts.append(OptimizerStartResult(
                        name=name, converged=converged, objective=objective if np.isfinite(objective) else None,
                        message=reason or (ls.message if ls is not None else "no active parameters"),
                    ))
                    # Convergence is explicitly checked and a capped preliminary
                    # solve receives a penalty. It may still win when its robust
                    # train objective is materially better; the subsequent full
                    # staged/final solves must then certify the final solution.
                    selection_score = objective * (1.0 if converged else 1.05)
                    if reason is None and np.isfinite(selection_score) and selection_score < best_obj:
                        best, best_obj = fitted, selection_score
                        best_start_index = len(output.starts) - 1
                except OptimizationCancelled:
                    raise
                except Exception as exc:  # one bad start must not destroy the original result
                    output.starts.append(OptimizerStartResult(name=name, message=str(exc)))
            if best is None:
                raise RuntimeError("All multi-start candidates failed sanity/convergence checks")
            output.starts[best_start_index].selected = True
            mark("Multi-start", "completed"); mark("Best Initial Solution", "completed")

            mark("Robust Bundle Adjustment", "running")
            mark("Huber Loss", "completed" if settings.robust_loss == "huber" else "completed")
            x = best
            released: set[str] = set()
            stage_defs = definition["stages"] if settings.staged else [definition["intrinsics"] + definition["distortion"]]
            mark("Staged Parameter Release", "running")
            for index, released_now in enumerate(stage_defs, 1):
                released.update(released_now)
                names = _active_names(settings, released, labels)
                before_rms = _rms(raw(x))
                x, ls = _least_squares_stage(
                    x, names, labels, bounds, raw, settings,
                    settings.max_iterations_per_stage, [len(obj) for obj in objects],
                    normalization_scale,
                )
                after_rms = _rms(raw(x))
                output.stages.append(OptimizerStageResult(
                    name=f"Stage {index}", active_parameters=sorted(n for n in names if not n.startswith("pose[")) +
                    (["extrinsics"] if any(n.startswith("pose[") for n in names) else []),
                    iterations=int(ls.nfev if ls is not None else 0), rms_before=before_rms, rms_after=after_rms,
                    termination=str(ls.message if ls is not None else "no active parameters"),
                    success=bool(ls.success) if ls is not None else True,
                ))
                sanity = _sanity(x, d_count, len(frames), (camera_config.width, camera_config.height))
                if sanity: raise RuntimeError(f"Stage {index}: {sanity}")
            mark("Staged Parameter Release", "completed")

            mark("Final Joint Optimization", "running")
            all_released = set(definition["intrinsics"] + definition["distortion"])
            names = _active_names(settings, all_released, labels)
            before_rms = _rms(raw(x))
            x, ls = _least_squares_stage(
                x, names, labels, bounds, raw, settings,
                settings.final_joint_iterations, [len(obj) for obj in objects],
                normalization_scale,
            )
            after_rms = _rms(raw(x))
            output.stages.append(OptimizerStageResult(
                name="Final Joint", active_parameters=sorted(n for n in names if not n.startswith("pose[")) +
                (["extrinsics"] if any(n.startswith("pose[") for n in names) else []),
                iterations=int(ls.nfev if ls is not None else 0), rms_before=before_rms, rms_after=after_rms,
                termination=str(ls.message if ls is not None else "no active parameters"),
                success=bool(ls.success) if ls is not None else True,
            ))
            sanity = _sanity(x, d_count, len(frames), (camera_config.width, camera_config.height))
            if sanity: raise RuntimeError(f"Final joint: {sanity}")
            mark("Final Joint Optimization", "completed"); mark("Robust Bundle Adjustment", "completed")

            opencv_initial = initial
            optimized = _build_calibration(model, x, d_count, frames, objects, images, opencv_initial,
                                           (camera_config.width, camera_config.height))
            initial = _build_calibration(model, x0, d_count, frames, objects, images, opencv_initial,
                                         (camera_config.width, camera_config.height))
            initial.param_uncertainty = deepcopy(opencv_initial.param_uncertainty)
            initial.param_uncertainty_bootstrap = deepcopy(opencv_initial.param_uncertainty_bootstrap)
            output.original_calibration = initial
            try:
                attach_observability_report(initial, train_dataset)
                attach_observability_report(optimized, train_dataset)
            except Exception:
                pass  # diagnostics must never invalidate a sound calibration

            mark("Hold-out Evaluation", "running")
            before_val = _evaluate_on_test(dataset, camera_config, pattern_config, model,
                                           train_ids, holdout_ids, initial)
            after_val = _evaluate_on_test(dataset, camera_config, pattern_config, model,
                                          train_ids, holdout_ids, optimized)
            # A pose failure must not make Before/After use different hold-out
            # observations. Re-evaluate both on the identical successful-frame
            # intersection (K/D remain frozen in both calls).
            failed_union = set(before_val.failed_test_frame_ids) | set(after_val.failed_test_frame_ids)
            fair_holdout_ids = [frame_id for frame_id in holdout_ids if frame_id not in failed_union]
            if fair_holdout_ids != list(holdout_ids):
                before_val = _evaluate_on_test(
                    dataset, camera_config, pattern_config, model,
                    train_ids, fair_holdout_ids, initial,
                )
                after_val = _evaluate_on_test(
                    dataset, camera_config, pattern_config, model,
                    train_ids, fair_holdout_ids, optimized,
                )
            output.before_metrics = _metric_set(initial, before_val)
            output.after_metrics = _metric_set(optimized, after_val)
            output.optimized_calibration = optimized
            output.recommendation, output.reasons = _recommend(output.before_metrics, output.after_metrics)
            if fair_holdout_ids != list(holdout_ids):
                output.reasons.append(
                    f"Before/After restricted to the same {len(fair_holdout_ids)}/{len(holdout_ids)} "
                    "hold-out frames because pose estimation failed"
                )
            output.success = True
            mark("Hold-out Evaluation", "completed")
            return output
        except OptimizationCancelled:
            output.cancelled = True; output.error_message = "Optimization cancelled"
        except Exception as exc:
            output.error_message = str(exc)
        for step, state in list(status.items()):
            if state == "running": status[step] = "failed"
        return output
