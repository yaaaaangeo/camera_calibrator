"""
Tests for integrations.direct_visual_runner -- the external
direct_visual_lidar_calibration process pipeline.

No real ROS/direct_visual install is needed: command-builder tests check
the argv list directly, and pipeline tests inject fake run_stage_fn /
check_environment_fn / load_prior_fn callables (run_direct_visual_pipeline
is plain Python, no Qt) so the whole stage sequence -- success, every
failure mode, and cancellation -- can be exercised deterministically.

Load-bearing checks, per the design spec:
  - ROS1 uses rosrun, ROS2 uses `ros2 run`; "auto" resolves via the
    ROS_VERSION env var or raises a configuration error, never guesses.
  - spinning LiDAR gets `-d` on preprocess; non_repetitive does not.
  - COARSE mode NEVER invokes `calibrate`; FULL mode runs all 4 stages in
    order and uses source="final" (never silently falling back to the
    coarse initial guess).
  - a failed/cancelled run never touches "the current prior" -- this pure
    function has no such state to touch in the first place, and the UI
    wiring (ui/camera_lidar_workspace.py) only calls
    self._apply_targetless_prior() on the `succeeded` signal.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from integrations.direct_visual_runner import (
    DirectVisualBootstrapResult,
    DirectVisualConfig,
    DirectVisualFailureReason,
    EnvironmentCheckResult,
    RunnerStage,
    StageResult,
    build_matching_args,
    build_preprocess_args,
    build_ros_command,
    check_environment,
    resolve_ros_version,
    run_direct_visual_pipeline,
    run_stage,
    verify_preprocess_output,
)
from camera_lidar.types import TargetlessPrior


def _config(**overrides) -> DirectVisualConfig:
    # ros_version pinned explicitly (not "auto") so these tests never depend
    # on whether ROS_VERSION happens to be set in the real test environment.
    defaults = dict(
        ros_version="2",
        input_bag_path="/data/bag", output_path="/data/out",
        image_topic="/camera/image_raw", points_topic="/points_raw",
    )
    defaults.update(overrides)
    return DirectVisualConfig(**defaults)


# ---------------------------------------------------------------------------
# DirectVisualConfig validation (hardening, mirrors GuidedROIConfig)
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_ros_version():
    with pytest.raises(ValueError):
        _config(ros_version="3")


def test_config_rejects_invalid_lidar_type():
    with pytest.raises(ValueError):
        _config(lidar_type="rotating")


def test_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        _config(mode="ultra")


@pytest.mark.parametrize("rotate_camera_deg", [1, 45, 91, 360, -90])
def test_config_rejects_invalid_camera_rotation(rotate_camera_deg):
    with pytest.raises(ValueError):
        _config(rotate_camera_deg=rotate_camera_deg)


@pytest.mark.parametrize("rotate_lidar_deg", [1, 45, 91, 360, -90])
def test_config_rejects_invalid_lidar_rotation(rotate_lidar_deg):
    with pytest.raises(ValueError):
        _config(rotate_lidar_deg=rotate_lidar_deg)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_config_accepts_valid_rotation_steps(rotation):
    config = _config(rotate_camera_deg=rotation, rotate_lidar_deg=rotation)
    assert config.rotate_camera_deg == rotation
    assert config.rotate_lidar_deg == rotation


# ---------------------------------------------------------------------------
# resolve_ros_version
# ---------------------------------------------------------------------------

def test_resolve_ros_version_explicit_1():
    assert resolve_ros_version(_config(ros_version="1")) == "1"


def test_resolve_ros_version_explicit_2():
    assert resolve_ros_version(_config(ros_version="2")) == "2"


def test_resolve_ros_version_auto_uses_env_var():
    config = _config(ros_version="auto")
    assert resolve_ros_version(config, environ={"ROS_VERSION": "2"}) == "2"
    assert resolve_ros_version(config, environ={"ROS_VERSION": "1"}) == "1"


def test_resolve_ros_version_auto_raises_without_env_var():
    config = _config(ros_version="auto")
    with pytest.raises(ValueError):
        resolve_ros_version(config, environ={})


def test_resolve_ros_version_auto_infers_ros2_from_setup_path():
    config = _config(ros_version="auto", ros_setup_path="/opt/ros/humble/setup.bash")
    assert resolve_ros_version(config, environ={}) == "2"


def test_resolve_ros_version_auto_infers_ros1_from_workspace_setup_path():
    config = _config(ros_version="auto", workspace_setup_path="/opt/ros/noetic/setup.bash")
    assert resolve_ros_version(config, environ={}) == "1"


def test_resolve_ros_version_env_var_takes_priority_over_setup_path_inference():
    # ROS_VERSION in the environment always wins over a setup-path guess.
    config = _config(ros_version="auto", ros_setup_path="/opt/ros/noetic/setup.bash")
    assert resolve_ros_version(config, environ={"ROS_VERSION": "2"}) == "2"


def test_resolve_ros_version_auto_raises_when_setup_path_gives_no_hint():
    config = _config(ros_version="auto", ros_setup_path="/home/user/my_ros_ws/setup.bash")
    with pytest.raises(ValueError):
        resolve_ros_version(config, environ={})


# ---------------------------------------------------------------------------
# build_ros_command / build_preprocess_args (§31)
# ---------------------------------------------------------------------------

def test_ros1_uses_rosrun():
    command = build_ros_command(_config(ros_version="1"), "initial_guess_auto", ["/out"])
    assert command == ["rosrun", "direct_visual_lidar_calibration", "initial_guess_auto", "/out"]


def test_ros2_uses_ros2_run():
    command = build_ros_command(_config(ros_version="2"), "initial_guess_auto", ["/out"])
    assert command == ["ros2", "run", "direct_visual_lidar_calibration", "initial_guess_auto", "/out"]


def test_preprocess_spinning_lidar_includes_dash_d():
    args = build_preprocess_args(_config(lidar_type="spinning"))
    assert "-d" in args


def test_preprocess_non_repetitive_lidar_excludes_dash_d():
    args = build_preprocess_args(_config(lidar_type="non_repetitive"))
    assert "-d" not in args


def test_preprocess_args_include_positional_bag_and_output_dirs_first():
    args = build_preprocess_args(_config(input_bag_path="/data/bag", output_path="/data/out"))
    assert args[0] == "/data/bag"
    assert args[1] == "/data/out"


def test_preprocess_args_include_camera_intrinsics_when_set():
    config = _config(camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), distortion=np.zeros(5))
    args = build_preprocess_args(config)
    assert "--camera_intrinsics" in args
    assert "--camera_distortion_coeffs" in args
    assert "--camera_model" in args


def test_preprocess_args_omit_camera_intrinsics_when_unset():
    args = build_preprocess_args(_config(camera_matrix=None, distortion=None))
    assert "--camera_intrinsics" not in args


# ---------------------------------------------------------------------------
# Rotation option placement (§18-19, §34): --rotate_camera/--rotate_lidar
# belong to find_matches_superglue.py (MATCHING stage) ONLY -- never
# preprocess, and never under the old (wrong) --camera_rotate_deg /
# --lidar_rotate_deg names.
# ---------------------------------------------------------------------------

def test_preprocess_args_never_contain_rotation_flags():
    config = _config(rotate_camera_deg=90, rotate_lidar_deg=180)
    args = build_preprocess_args(config)
    assert "--rotate_camera" not in args
    assert "--rotate_lidar" not in args
    assert "--camera_rotate_deg" not in args
    assert "--lidar_rotate_deg" not in args


def test_matching_args_contain_correct_rotation_flags():
    config = _config(rotate_camera_deg=90, rotate_lidar_deg=180)
    args = build_matching_args(config)
    assert args == [config.output_path, "--rotate_camera", "90", "--rotate_lidar", "180"]


def test_matching_args_always_include_both_rotation_flags_even_at_zero():
    # Upstream's own example passes --rotate_lidar 0 explicitly rather than
    # omitting it -- both flags must always be present, not conditional.
    config = _config(rotate_camera_deg=0, rotate_lidar_deg=0)
    args = build_matching_args(config)
    assert "--rotate_camera" in args
    assert "--rotate_lidar" in args
    assert args[args.index("--rotate_camera") + 1] == "0"
    assert args[args.index("--rotate_lidar") + 1] == "0"


def test_old_wrong_rotation_flag_names_appear_nowhere():
    config = _config(rotate_camera_deg=270, rotate_lidar_deg=90)
    all_args = build_preprocess_args(config) + build_matching_args(config)
    assert "--camera_rotate_deg" not in all_args
    assert "--lidar_rotate_deg" not in all_args


def test_command_wraps_setup_sourcing_in_bash_lc():
    config = _config(ros_setup_path="/opt/ros/humble/setup.bash", workspace_setup_path="~/ws/install/setup.bash")
    command = build_ros_command(config, "initial_guess_auto", ["/out"])
    assert command[0] == "bash"
    assert command[1] == "-lc"
    script = command[2]
    assert "source" in script
    assert "/opt/ros/humble/setup.bash" in script
    assert "ros2 run direct_visual_lidar_calibration initial_guess_auto /out" in script


def test_command_without_setup_paths_is_plain_argv_no_shell():
    command = build_ros_command(_config(), "initial_guess_auto", ["/out"])
    assert command[0] != "bash"


def test_command_safely_quotes_paths_with_spaces():
    config = _config(ros_setup_path="/opt/ros/my setup/setup.bash")
    command = build_ros_command(config, "preprocess", ["/data/my bag", "/data/out"])
    # The whole point: a space in a path must not silently split into two
    # argv-visible tokens or break out of its quoting.
    script = command[2]
    assert "/opt/ros/my setup/setup.bash" not in script.replace("'/opt/ros/my setup/setup.bash'", "")


def test_build_preprocess_args_passed_through_build_ros_command_unmodified():
    config = _config(input_bag_path="/data/my bag")
    args = build_preprocess_args(config)
    command = build_ros_command(config, "preprocess", args)
    assert "/data/my bag" in command  # argv element, not shell-split


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------

def test_check_environment_fails_on_non_linux():
    result = check_environment(_config(), platform_system=lambda: "Windows")
    assert not result.passed
    assert any("Linux" in p for p in result.problems)


def test_check_environment_fails_when_bag_path_missing(tmp_path):
    config = _config(input_bag_path=str(tmp_path / "does_not_exist"), output_path=str(tmp_path / "out"))
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert not result.passed


# ---------------------------------------------------------------------------
# Input must be a bag DIRECTORY, never a single bag file (§5, §24, §35).
# ---------------------------------------------------------------------------

def test_check_environment_fails_when_bag_path_is_a_single_file(tmp_path):
    bag_file = tmp_path / "scene01.bag"
    bag_file.write_bytes(b"not a real bag, just a file")
    config = _config(input_bag_path=str(bag_file), output_path=str(tmp_path / "out"))
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert not result.passed
    assert any("directory" in p.lower() for p in result.problems)


def test_check_environment_fails_when_bag_path_is_a_plain_file_not_bag_extension(tmp_path):
    plain_file = tmp_path / "readme.txt"
    plain_file.write_text("hello")
    config = _config(input_bag_path=str(plain_file), output_path=str(tmp_path / "out"))
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert not result.passed


def test_check_environment_passes_with_a_real_directory_of_bags(tmp_path):
    bag_dir = tmp_path / "calibration_bags"
    bag_dir.mkdir()
    (bag_dir / "scene01.bag").write_bytes(b"fake")
    (bag_dir / "scene02.bag").write_bytes(b"fake")
    config = _config(
        input_bag_path=str(bag_dir), output_path=str(tmp_path / "out"), ros_version="2",
        camera_matrix=np.eye(3), distortion=np.zeros(5),
    )
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert result.passed, result.problems


def test_check_environment_passes_when_everything_ready(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    config = _config(
        input_bag_path=str(bag_dir), output_path=str(tmp_path / "out"), ros_version="2",
        camera_matrix=np.eye(3), distortion=np.zeros(5),
    )
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert result.passed, result.problems


def test_check_environment_fails_when_camera_intrinsic_missing(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    config = _config(
        input_bag_path=str(bag_dir), output_path=str(tmp_path / "out"), ros_version="2",
        camera_matrix=None, distortion=None,
    )
    result = check_environment(config, platform_system=lambda: "Linux", which=lambda _: "/usr/bin/ros2")
    assert not result.passed


# ---------------------------------------------------------------------------
# verify_preprocess_output
# ---------------------------------------------------------------------------

def test_verify_preprocess_output_true_with_ply(tmp_path):
    (tmp_path / "cloud.ply").write_text("fake")
    assert verify_preprocess_output(str(tmp_path)) is True


def test_verify_preprocess_output_false_when_empty(tmp_path):
    assert verify_preprocess_output(str(tmp_path)) is False


def test_verify_preprocess_output_false_when_dir_missing(tmp_path):
    assert verify_preprocess_output(str(tmp_path / "nope")) is False


# ---------------------------------------------------------------------------
# run_stage -- fake subprocess.Popen, no real process spawned
# ---------------------------------------------------------------------------

class _FakePopen:
    def __init__(self, lines=(), exit_code=0, never_finish=False):
        self._lines = list(lines)
        self._exit_code = exit_code
        self._never_finish = never_finish
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = self

    def readline(self):
        if self._lines:
            return self._lines.pop(0) + "\n"
        if not self._never_finish and self.returncode is None:
            self.returncode = self._exit_code
        return ""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.terminated and self.returncode is None:
            self.returncode = -15
        return self.returncode

    def close(self):
        pass


def test_run_stage_success_captures_output():
    fake = _FakePopen(lines=["hello", "world"], exit_code=0)
    logged = []
    result = run_stage(["echo", "hi"], on_output=logged.append, popen_factory=lambda *a, **k: fake)
    assert result.success
    assert result.exit_code == 0
    assert logged == ["hello", "world"]


def test_run_stage_failure_nonzero_exit():
    fake = _FakePopen(lines=["boom"], exit_code=1)
    result = run_stage(["false"], popen_factory=lambda *a, **k: fake)
    assert not result.success
    assert result.exit_code == 1


def test_run_stage_cancel_terminates_process():
    fake = _FakePopen(lines=[], never_finish=True)
    result = run_stage(
        ["sleep", "1000"], cancel_check=lambda: True, poll_interval_s=0.0,
        popen_factory=lambda *a, **k: fake,
    )
    assert result.cancelled
    assert fake.terminated


# ---------------------------------------------------------------------------
# run_direct_visual_pipeline -- stage orchestration (§32-36)
# ---------------------------------------------------------------------------

class _StageRecorder:
    """Fake run_stage_fn: returns success for every stage unless told
    otherwise, and records every command it was called with (in order)."""

    def __init__(self, fail_command_substring=None):
        self.calls = []
        self.fail_command_substring = fail_command_substring

    def __call__(self, command, on_output=None, cancel_check=None):
        self.calls.append(command)
        joined = " ".join(command) if isinstance(command, list) else str(command)
        if self.fail_command_substring and self.fail_command_substring in joined:
            return StageResult(success=False, exit_code=1, output="synthetic failure")
        return StageResult(success=True, exit_code=0, output="ok")


def _always_passing_env_check(config):
    return EnvironmentCheckResult(passed=True, problems=[])


def _fake_load_prior(path, source):
    return TargetlessPrior(T_lidar_from_camera=np.eye(4), source_path=path, source_key=(
        "T_lidar_camera" if source == "final" else "init_T_lidar_camera_auto"
    ))


def _run(config, run_stage_fn, load_prior_fn=_fake_load_prior, verify_preprocess=True, cancel_check=None, **extra):
    # These tests use fake bag/output paths that don't exist on disk, so
    # verify_preprocess_output's real filesystem check must always be
    # stubbed (never left as the real implementation) -- its return value
    # is exactly what `verify_preprocess` controls here.
    import integrations.direct_visual_runner as runner_module
    orig_verify = runner_module.verify_preprocess_output
    runner_module.verify_preprocess_output = lambda output_path: verify_preprocess
    try:
        return run_direct_visual_pipeline(
            config,
            run_stage_fn=run_stage_fn,
            check_environment_fn=_always_passing_env_check,
            load_prior_fn=load_prior_fn,
            cancel_check=cancel_check,
            **extra,
        )
    finally:
        runner_module.verify_preprocess_output = orig_verify


def test_pipeline_coarse_mode_succeeds_and_never_runs_calibrate():
    recorder = _StageRecorder()
    result = _run(_config(mode="coarse"), recorder)

    assert result.success, result.failure_message
    assert result.prior.source_key == "init_T_lidar_camera_auto"
    joined_calls = [" ".join(c) for c in recorder.calls]
    assert not any("calibrate" in c.split() for c in joined_calls)
    assert len(recorder.calls) == 3  # preprocess, matching, initial_guess only


def test_pipeline_full_mode_runs_all_four_stages_in_order():
    recorder = _StageRecorder()
    result = _run(_config(mode="full", rotate_camera_deg=90, rotate_lidar_deg=180), recorder)

    assert result.success, result.failure_message
    assert len(recorder.calls) == 4
    # Each stage's program name appears, in the exact expected order.
    joined = [" ".join(c) for c in recorder.calls]
    assert "preprocess" in joined[0]
    assert "find_matches_superglue.py" in joined[1]
    assert "initial_guess_auto" in joined[2]
    assert "calibrate" in joined[3]
    assert result.prior.source_key == "T_lidar_camera"

    # Rotation flags: correct names, on the MATCHING command only.
    assert "--rotate_camera 90" in joined[1]
    assert "--rotate_lidar 180" in joined[1]
    assert "--camera_rotate_deg" not in joined[0]
    assert "--lidar_rotate_deg" not in joined[0]
    for stage_command in joined:
        assert "--camera_rotate_deg" not in stage_command
        assert "--lidar_rotate_deg" not in stage_command
    assert "--rotate_camera" not in joined[0]  # never on preprocess


def test_pipeline_full_mode_uses_final_source_not_auto_initial():
    seen_sources = []

    def load_prior_fn(path, source):
        seen_sources.append(source)
        return _fake_load_prior(path, source)

    result = _run(_config(mode="full"), _StageRecorder(), load_prior_fn=load_prior_fn)
    assert result.success
    assert seen_sources == ["final"]


def test_pipeline_coarse_mode_uses_auto_initial_source():
    seen_sources = []

    def load_prior_fn(path, source):
        seen_sources.append(source)
        return _fake_load_prior(path, source)

    result = _run(_config(mode="coarse"), _StageRecorder(), load_prior_fn=load_prior_fn)
    assert result.success
    assert seen_sources == ["auto_initial"]


@pytest.mark.parametrize("failing_program,expected_reason", [
    ("preprocess", DirectVisualFailureReason.PREPROCESS_FAILED),
    ("find_matches_superglue.py", DirectVisualFailureReason.MATCHING_FAILED),
    ("initial_guess_auto", DirectVisualFailureReason.INITIAL_GUESS_FAILED),
])
def test_pipeline_stage_failure_maps_to_correct_reason(failing_program, expected_reason):
    recorder = _StageRecorder(fail_command_substring=failing_program)
    result = _run(_config(mode="coarse"), recorder)

    assert not result.success
    assert result.failure_reason == expected_reason
    assert result.prior is None


def test_pipeline_calibrate_failure_maps_to_fine_calibration_failed():
    recorder = _StageRecorder(fail_command_substring="calibrate")
    result = _run(_config(mode="full"), recorder)

    assert not result.success
    assert result.failure_reason == DirectVisualFailureReason.FINE_CALIBRATION_FAILED
    # calibrate is the LAST stage -- confirm it was actually reached, not
    # short-circuited earlier by an unrelated bug.
    assert len(recorder.calls) == 4


def test_pipeline_preprocess_output_missing_fails_even_with_exit_code_zero():
    recorder = _StageRecorder()  # every stage "succeeds" (exit 0)
    result = _run(_config(mode="coarse"), recorder, verify_preprocess=False)

    assert not result.success
    assert result.failure_reason == DirectVisualFailureReason.RESULT_FILE_MISSING
    # Must stop right after preprocess -- matching/initial_guess never run.
    assert len(recorder.calls) == 1


def test_pipeline_invalid_calib_json_maps_to_result_invalid():
    def failing_load_prior(path, source):
        raise ValueError("calib.json contains a non-finite value")

    result = _run(_config(mode="coarse"), _StageRecorder(), load_prior_fn=failing_load_prior)

    assert not result.success
    assert result.failure_reason == DirectVisualFailureReason.RESULT_INVALID
    assert "non-finite" in result.failure_message


def test_pipeline_environment_not_ready_runs_no_stages():
    recorder = _StageRecorder()

    def failing_env_check(config):
        return EnvironmentCheckResult(passed=False, problems=["Input bag path does not exist"])

    result = run_direct_visual_pipeline(
        _config(), run_stage_fn=recorder, check_environment_fn=failing_env_check,
    )

    assert not result.success
    assert result.failure_reason == DirectVisualFailureReason.ENVIRONMENT_NOT_READY
    assert recorder.calls == []


def test_pipeline_cancel_before_first_stage_runs_nothing():
    recorder = _StageRecorder()
    result = _run(_config(mode="coarse"), recorder, cancel_check=lambda: True)

    assert result.cancelled
    assert not result.success
    assert result.prior is None
    assert recorder.calls == []


def test_pipeline_cancel_mid_run_stops_remaining_stages():
    class _CancelAfterFirstCall:
        def __init__(self):
            self.calls = []

        def __call__(self, command, on_output=None, cancel_check=None):
            self.calls.append(command)
            if len(self.calls) == 1:
                return StageResult(success=True, exit_code=0, output="ok")
            return StageResult(success=False, exit_code=None, output="", cancelled=True)

    recorder = _CancelAfterFirstCall()
    result = _run(_config(mode="coarse"), recorder)

    assert result.cancelled
    assert not result.success
    # Stopped as soon as the (fake) in-flight stage reported cancellation --
    # never proceeded to a 3rd stage.
    assert len(recorder.calls) == 2
