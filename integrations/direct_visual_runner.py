"""
camera_calibrator.integrations.direct_visual_runner
========================================================

Runs the upstream `direct_visual_lidar_calibration` ROS package as an
external process pipeline:

    preprocess -> find_matches_superglue.py -> initial_guess_auto
        -> [optional] calibrate
        -> calib.json
        -> camera_lidar.targetless_prior.load_direct_visual_calib
        -> TargetlessPrior

This module NEVER ports or reimplements any of direct_visual's actual
calibration algorithm -- it only builds argv command lines, runs them as
external processes, and verifies/parses their output file with the
EXISTING camera_lidar.targetless_prior loader (never a new parser).

Dependency direction: camera_lidar/, geometry/, and evaluation/ must never
import this module or gain a ROS dependency -- only this module (and its
Qt worker wrapper in ui/worker.py) knows about ROS commands/environment.
The only thing that crosses back into camera_lidar/ is the resulting
TargetlessPrior, exactly like a manually-loaded calib.json (see
camera_lidar/types.py's TargetlessPrior docstring: it is used ONLY to seed
camera_lidar.guided_roi's LiDAR search region, never as a solver
constraint).

Deliberately Qt-free: this module has no PySide6 import and no QProcess
use, so every function/class here is directly unit-testable (including
`run_direct_visual_pipeline`, which drives the whole stage sequence)
without a Qt event loop or a live ROS install -- tests inject fake
`run_stage_fn` / `check_environment_fn` / `load_prior_fn` callables. The
Qt-facing wrapper (ui.worker.DirectVisualBootstrapWorker) is a thin
QObject that calls `run_direct_visual_pipeline` and translates its
callbacks into Qt signals, mirroring how every other worker in
ui/worker.py wraps a plain calibration/camera_lidar function.

Cancellation reuses this codebase's existing `cancel_check: Callable[[],
bool]` convention (see camera_lidar/lidar_detector.py, camera_lidar/
pipeline.py) instead of introducing a second, QProcess-specific lifecycle
model -- one process is running at a time, driven by subprocess.Popen with
incremental output polling, terminated (then killed if unresponsive) when
cancel_check() goes true.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from camera_lidar.targetless_prior import load_direct_visual_calib
from camera_lidar.types import TargetlessPrior

_VALID_ROS_VERSIONS = frozenset({"auto", "1", "2"})
_VALID_LIDAR_TYPES = frozenset({"spinning", "non_repetitive"})
_VALID_MODES = frozenset({"coarse", "full"})

# The exact upstream CLI flag names for topics/rotation are the one
# genuinely uncertain piece of this integration (direct_visual's own
# published command reference only pins down the positional bag/output
# dirs, the `-d` spinning-LiDAR flag, and the *names* of the camera
# intrinsic flags) -- centralized here, in ONE place, so a user pinned to
# a specific direct_visual_lidar_calibration release can correct the flag
# spelling in one spot if their installed version differs, without
# touching UI code or the ROS1/ROS2 command builder below.
_IMAGE_TOPIC_FLAG = "--image_topic"
_POINTS_TOPIC_FLAG = "--points_topic"
_CAMERA_INFO_TOPIC_FLAG = "--camera_info_topic"
_CAMERA_ROTATE_FLAG = "--camera_rotate_deg"
_LIDAR_ROTATE_FLAG = "--lidar_rotate_deg"


class RunnerStage(Enum):
    IDLE = "idle"
    ENV_CHECK = "env_check"
    PREPROCESS = "preprocess"
    MATCHING = "matching"
    INITIAL_GUESS = "initial_guess"
    FINE_CALIBRATION = "fine_calibration"
    VALIDATING_RESULT = "validating_result"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_COARSE_STAGES = (RunnerStage.PREPROCESS, RunnerStage.MATCHING, RunnerStage.INITIAL_GUESS)
_FULL_STAGES = _COARSE_STAGES + (RunnerStage.FINE_CALIBRATION,)

_PROGRAM_BY_STAGE = {
    RunnerStage.PREPROCESS: "preprocess",
    RunnerStage.MATCHING: "find_matches_superglue.py",
    RunnerStage.INITIAL_GUESS: "initial_guess_auto",
    RunnerStage.FINE_CALIBRATION: "calibrate",
}


class DirectVisualFailureReason(Enum):
    ENVIRONMENT_NOT_READY = "environment_not_ready"
    PREPROCESS_FAILED = "preprocess_failed"
    MATCHING_FAILED = "matching_failed"
    INITIAL_GUESS_FAILED = "initial_guess_failed"
    FINE_CALIBRATION_FAILED = "fine_calibration_failed"
    RESULT_FILE_MISSING = "result_file_missing"
    RESULT_INVALID = "result_invalid"
    CANCELLED = "cancelled"


_FAILURE_REASON_BY_STAGE = {
    RunnerStage.PREPROCESS: DirectVisualFailureReason.PREPROCESS_FAILED,
    RunnerStage.MATCHING: DirectVisualFailureReason.MATCHING_FAILED,
    RunnerStage.INITIAL_GUESS: DirectVisualFailureReason.INITIAL_GUESS_FAILED,
    RunnerStage.FINE_CALIBRATION: DirectVisualFailureReason.FINE_CALIBRATION_FAILED,
}


@dataclass
class DirectVisualConfig:
    """Configuration for one Targetless Bootstrap run. camera_matrix/
    distortion are populated from the app's OWN already-loaded camera
    intrinsic calibration (calibration.calibration_io.StandardCalibration)
    -- never re-entered by the user -- see integrations.direct_visual_runner
    module docstring / camera_lidar_workspace.py's bootstrap UI."""
    ros_version: str = "auto"          # "auto" | "1" | "2"

    input_bag_path: str = ""
    output_path: str = ""

    image_topic: str = ""
    points_topic: str = ""
    camera_info_topic: str = ""

    lidar_type: str = "spinning"       # "spinning" | "non_repetitive"

    camera_model: str = "plumb_bob"
    camera_matrix: Optional[np.ndarray] = None    # (3,3), from StandardCalibration
    distortion: Optional[np.ndarray] = None       # (N,), from StandardCalibration

    rotate_camera_deg: int = 0
    rotate_lidar_deg: int = 0

    mode: str = "coarse"               # "coarse" | "full"

    ros_setup_path: str = ""
    workspace_setup_path: str = ""

    def __post_init__(self) -> None:
        if self.ros_version not in _VALID_ROS_VERSIONS:
            raise ValueError(f"ros_version must be one of {sorted(_VALID_ROS_VERSIONS)}, got {self.ros_version!r}")
        if self.lidar_type not in _VALID_LIDAR_TYPES:
            raise ValueError(f"lidar_type must be one of {sorted(_VALID_LIDAR_TYPES)}, got {self.lidar_type!r}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}")


@dataclass
class EnvironmentCheckResult:
    passed: bool
    problems: list = field(default_factory=list)


@dataclass
class StageResult:
    success: bool
    exit_code: Optional[int]
    output: str
    cancelled: bool = False


@dataclass
class DirectVisualBootstrapResult:
    success: bool
    prior: Optional[TargetlessPrior] = None
    failure_reason: Optional[DirectVisualFailureReason] = None
    failure_message: str = ""
    cancelled: bool = False


def resolve_ros_version(config: DirectVisualConfig, environ: Optional[dict] = None) -> str:
    """"auto" resolves via the ROS_VERSION environment variable (set by a
    sourced ROS setup.bash) -- raises ValueError (a configuration error,
    never a silent guess) if it can't be determined that way."""
    if config.ros_version in ("1", "2"):
        return config.ros_version

    environ = environ if environ is not None else os.environ
    env_version = environ.get("ROS_VERSION")
    if env_version in ("1", "2"):
        return env_version

    raise ValueError(
        "ros_version=\"auto\" but the ROS_VERSION environment variable is not set to 1 or 2 -- "
        "select ROS1 or ROS2 explicitly in the Targetless Bootstrap ROS setting."
    )


def build_ros_command(
    config: DirectVisualConfig, program: str, args: list, environ: Optional[dict] = None,
) -> list:
    """Builds the argv list for one direct_visual_lidar_calibration stage.
    Never string-concatenates a shell command from user input -- `program`/
    `args`/setup paths are passed as separate argv elements (or, when ROS
    setup sourcing is needed, safely shlex-quoted into the one `bash -lc`
    script that does the sourcing)."""
    ros_version = resolve_ros_version(config, environ=environ)
    if ros_version == "1":
        inner = ["rosrun", "direct_visual_lidar_calibration", program, *args]
    else:
        inner = ["ros2", "run", "direct_visual_lidar_calibration", program, *args]

    setup_paths = [p for p in (config.ros_setup_path, config.workspace_setup_path) if p]
    if not setup_paths:
        return inner

    source_cmds = " && ".join(f"source {shlex.quote(p)}" for p in setup_paths)
    inner_cmd = " ".join(shlex.quote(part) for part in inner)
    script = f"{source_cmds} && {inner_cmd}"
    return ["bash", "-lc", script]


def _intrinsics_vector(camera_matrix: np.ndarray) -> list:
    return [float(camera_matrix[0, 0]), float(camera_matrix[1, 1]), float(camera_matrix[0, 2]), float(camera_matrix[1, 2])]


def _format_floats(values: list) -> str:
    return ",".join(repr(v) for v in values)


def build_preprocess_args(config: DirectVisualConfig) -> list:
    """<INPUT_BAG_DIR> <OUTPUT_PREPROCESSED_DIR> [-d] [topic/intrinsic/rotate flags]."""
    args = [config.input_bag_path, config.output_path]
    if config.lidar_type == "spinning":
        args.append("-d")

    if config.image_topic:
        args += [_IMAGE_TOPIC_FLAG, config.image_topic]
    if config.points_topic:
        args += [_POINTS_TOPIC_FLAG, config.points_topic]
    if config.camera_info_topic:
        args += [_CAMERA_INFO_TOPIC_FLAG, config.camera_info_topic]

    if config.camera_matrix is not None and config.distortion is not None:
        args += [
            "--camera_model", config.camera_model,
            "--camera_intrinsics", _format_floats(_intrinsics_vector(config.camera_matrix)),
            "--camera_distortion_coeffs", _format_floats(np.asarray(config.distortion).reshape(-1).tolist()),
        ]

    if config.rotate_camera_deg:
        args += [_CAMERA_ROTATE_FLAG, str(config.rotate_camera_deg)]
    if config.rotate_lidar_deg:
        args += [_LIDAR_ROTATE_FLAG, str(config.rotate_lidar_deg)]

    return args


def _stage_program_and_args(stage: RunnerStage, config: DirectVisualConfig):
    program = _PROGRAM_BY_STAGE[stage]
    if stage == RunnerStage.PREPROCESS:
        return program, build_preprocess_args(config)
    return program, [config.output_path]


def check_environment(
    config: DirectVisualConfig,
    *,
    platform_system: Callable[[], str] = platform.system,
    which: Callable[[str], Optional[str]] = shutil.which,
    environ: Optional[dict] = None,
) -> EnvironmentCheckResult:
    """Cheap, non-spawning readiness checks (no subprocess is started
    here) -- a missing `direct_visual_lidar_calibration` ROS *package*
    (as opposed to a missing rosrun/ros2 *command*) is intentionally not
    probed here (that would require actually running a package-list
    command) and instead surfaces naturally as a PREPROCESS_FAILED with
    the real command's own "package not found" stderr."""
    problems: list = []
    environ = environ if environ is not None else os.environ

    if platform_system() != "Linux":
        problems.append("Automatic direct_visual execution requires a Linux ROS environment.")
        return EnvironmentCheckResult(passed=False, problems=problems)

    ros_version = None
    try:
        ros_version = resolve_ros_version(config, environ=environ)
    except ValueError as e:
        problems.append(str(e))

    if ros_version is not None:
        ros_command = "rosrun" if ros_version == "1" else "ros2"
        has_setup = bool(config.ros_setup_path or config.workspace_setup_path)
        if which(ros_command) is None and not has_setup:
            problems.append(
                f"'{ros_command}' command not found on PATH, and no ROS setup path is configured."
            )

    if not config.input_bag_path or not os.path.exists(config.input_bag_path):
        problems.append(f"Input bag path does not exist: {config.input_bag_path!r}")

    if not config.output_path:
        problems.append("Output path is not set.")
    else:
        try:
            os.makedirs(config.output_path, exist_ok=True)
        except OSError as e:
            problems.append(f"Cannot create output directory {config.output_path!r}: {e}")

    if config.camera_matrix is None or config.distortion is None:
        problems.append("Camera intrinsic calibration is not set.")

    if not config.image_topic:
        problems.append("Image topic is not set.")
    if not config.points_topic:
        problems.append("PointCloud topic is not set.")

    return EnvironmentCheckResult(passed=not problems, problems=problems)


def verify_preprocess_output(output_path: str) -> bool:
    """preprocess exit code 0 alone is not treated as success -- at least
    one expected output artifact (.ply point cloud or .png image dump)
    must actually exist in the output directory."""
    output_dir = Path(output_path)
    if not output_dir.is_dir():
        return False
    return any(output_dir.rglob("*.ply")) or any(output_dir.rglob("*.png"))


def run_stage(
    command: list,
    *,
    on_output: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.2,
    popen_factory: Callable[..., "subprocess.Popen"] = subprocess.Popen,
) -> StageResult:
    """Runs one external command, streaming its combined stdout/stderr
    line-by-line to `on_output`, polling `cancel_check` between lines so a
    long-running stage can be interrupted -- terminate() first, kill() if
    it doesn't exit promptly. `popen_factory` is the single seam tests
    replace with a fake process (see tests/test_direct_visual_runner.py)
    to run the whole stage sequence without a real direct_visual/ROS
    install."""
    process = popen_factory(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_lines: list = []
    try:
        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                stripped = line.rstrip("\n")
                output_lines.append(stripped)
                if on_output is not None:
                    on_output(stripped)
                continue
            if process.poll() is not None:
                break
            if cancel_check is not None and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return StageResult(
                    success=False, exit_code=process.returncode, output="\n".join(output_lines), cancelled=True,
                )
            time.sleep(poll_interval_s)
    finally:
        if process.stdout is not None:
            process.stdout.close()

    exit_code = process.returncode
    return StageResult(success=(exit_code == 0), exit_code=exit_code, output="\n".join(output_lines))


def run_direct_visual_pipeline(
    config: DirectVisualConfig,
    on_progress: Optional[Callable[[str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_stage_started: Optional[Callable[[RunnerStage], None]] = None,
    on_stage_finished: Optional[Callable[[RunnerStage], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    run_stage_fn: Callable[..., StageResult] = run_stage,
    check_environment_fn: Callable[[DirectVisualConfig], EnvironmentCheckResult] = check_environment,
    load_prior_fn: Callable[..., TargetlessPrior] = load_direct_visual_calib,
) -> DirectVisualBootstrapResult:
    """Runs ENV_CHECK -> PREPROCESS -> MATCHING -> INITIAL_GUESS -> (FULL
    mode only) FINE_CALIBRATION -> VALIDATING_RESULT, stopping at the
    first failure/cancellation. Plain function (no Qt) -- see module
    docstring; ui.worker.DirectVisualBootstrapWorker wraps this in Qt
    signals for the GUI. Never touches any "current prior" state itself
    -- callers decide what to do with a failed/cancelled result (the
    existing prior, if any, is simply left alone since this function
    never reaches into UI state)."""

    def _cancelled() -> bool:
        return cancel_check() if cancel_check is not None else False

    if on_stage_started is not None:
        on_stage_started(RunnerStage.ENV_CHECK)
    env_result = check_environment_fn(config)
    if not env_result.passed:
        return DirectVisualBootstrapResult(
            success=False, failure_reason=DirectVisualFailureReason.ENVIRONMENT_NOT_READY,
            failure_message="\n".join(env_result.problems),
        )
    if on_stage_finished is not None:
        on_stage_finished(RunnerStage.ENV_CHECK)

    stages = _FULL_STAGES if config.mode == "full" else _COARSE_STAGES
    total = len(stages)

    for index, stage in enumerate(stages, start=1):
        if _cancelled():
            return DirectVisualBootstrapResult(success=False, cancelled=True)

        if on_progress is not None:
            on_progress(f"Stage {index}/{total}: {stage.value}")
        if on_stage_started is not None:
            on_stage_started(stage)

        program, args = _stage_program_and_args(stage, config)
        command = build_ros_command(config, program, args)

        result = run_stage_fn(command, on_output=on_log, cancel_check=_cancelled)

        if result.cancelled:
            return DirectVisualBootstrapResult(success=False, cancelled=True)

        if not result.success:
            return DirectVisualBootstrapResult(
                success=False, failure_reason=_FAILURE_REASON_BY_STAGE[stage],
                failure_message=result.output[-2000:],
            )

        if on_stage_finished is not None:
            on_stage_finished(stage)

        if stage == RunnerStage.PREPROCESS and not verify_preprocess_output(config.output_path):
            return DirectVisualBootstrapResult(
                success=False, failure_reason=DirectVisualFailureReason.RESULT_FILE_MISSING,
                failure_message="preprocess exited 0 but produced no .ply/.png output.",
            )

    if _cancelled():
        return DirectVisualBootstrapResult(success=False, cancelled=True)

    if on_progress is not None:
        on_progress("Validating result...")
    if on_stage_started is not None:
        on_stage_started(RunnerStage.VALIDATING_RESULT)

    calib_json = os.path.join(config.output_path, "calib.json")
    # FULL mode must use the actually-refined "final" transform -- never
    # silently fall back to the coarse initial guess if refinement's own
    # result key is missing/invalid (see module docstring / spec §20).
    source = "final" if config.mode == "full" else "auto_initial"
    try:
        prior = load_prior_fn(calib_json, source=source)
    except Exception as e:  # noqa: BLE001 -- surfaced as a typed failure_reason, not re-raised
        return DirectVisualBootstrapResult(
            success=False, failure_reason=DirectVisualFailureReason.RESULT_INVALID, failure_message=str(e),
        )

    if on_stage_finished is not None:
        on_stage_finished(RunnerStage.VALIDATING_RESULT)

    return DirectVisualBootstrapResult(success=True, prior=prior)
