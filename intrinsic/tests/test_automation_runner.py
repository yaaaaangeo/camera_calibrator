"""automation.runner / run_manifest: 입력 오류, 실패 기록, 덮어쓰기 방지.

실제 calibration 계산 없이(자식 프로세스 호출을 대체) 빠르게 도는 테스트들이다.
실제 계산 경로는 tests/test_automation_e2e.py(slow)가 검증한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import automation.runner as runner
from automation import run_manifest as rm
from automation.cli import main as automation_main
from automation.identity import IdentityError, build_identity
from automation.profile import resolve_profile
from tests._automation_test_utils import base_profile, write_blank_images


def _profile(tmp_path: Path, *, selection: bool = False, **overrides):
    raw = base_profile(image_selection={"enabled": selection}, **overrides)
    raw["output"] = {"root": str(tmp_path / "outputs"), "exports": ["opencv", "ros", "report", "json", "csv"]}
    return resolve_profile(raw, source_path="test")


def _identity():
    return build_identity("TEST-CAM", "SN0001")


def _sessions(tmp_path: Path) -> list[Path]:
    root = tmp_path / "outputs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _manifest(session: str) -> dict:
    return json.loads((Path(session) / "run_manifest.json").read_text(encoding="utf-8"))


def _quiet(_msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Input errors: 계산을 시작하지 않고 세션도 만들지 않는다
# ---------------------------------------------------------------------------


def test_missing_input_folder_is_rejected_without_session(tmp_path):
    outcome = runner.run_automation(_profile(tmp_path), _identity(), [str(tmp_path / "nope")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "존재하지 않습니다" in outcome.message
    assert _sessions(tmp_path) == []


def test_folder_without_images_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    outcome = runner.run_automation(_profile(tmp_path), _identity(), [str(tmp_path / "empty")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "이미지를 찾지 못했습니다" in outcome.message
    assert _sessions(tmp_path) == []


def test_mixed_resolution_is_rejected(tmp_path):
    write_blank_images(tmp_path / "imgs", 2, size=(64, 48), prefix="a")
    write_blank_images(tmp_path / "imgs", 1, size=(80, 60), prefix="b")
    outcome = runner.run_automation(_profile(tmp_path), _identity(), [str(tmp_path / "imgs")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "섞여" in outcome.message


def test_profile_resolution_mismatch_is_rejected(tmp_path):
    write_blank_images(tmp_path / "imgs", 2, size=(64, 48))
    profile = _profile(tmp_path, camera={"model": "TEST-CAM", "width": 1280, "height": 720})
    outcome = runner.run_automation(profile, _identity(), [str(tmp_path / "imgs")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "1280x720" in outcome.message and "64x48" in outcome.message
    assert _sessions(tmp_path) == []


def test_duplicate_image_stems_are_rejected(tmp_path):
    write_blank_images(tmp_path / "a", 1, prefix="same")
    write_blank_images(tmp_path / "b", 1, prefix="same")
    outcome = runner.run_automation(
        _profile(tmp_path), _identity(), [str(tmp_path / "a"), str(tmp_path / "b")], log=_quiet,
    )
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "stem" in outcome.message


def test_unreadable_image_is_rejected(tmp_path):
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "broken.png").write_bytes(b"not an image")
    outcome = runner.run_automation(_profile(tmp_path), _identity(), [str(tmp_path / "imgs")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_INPUT_ERROR
    assert "읽을 수 없습니다" in outcome.message


@pytest.mark.parametrize("model, serial", [("", "SN1"), ("CAM", ""), ("CAM", "../etc"), ("CAM X", "SN1")])
def test_identity_format_is_validated(model, serial):
    with pytest.raises(IdentityError):
        build_identity(model, serial)


# ---------------------------------------------------------------------------
# Failures after the session exists are recorded, never reported as completed
# ---------------------------------------------------------------------------


def _run_with_child(tmp_path, monkeypatch, child, *, selection=False):
    write_blank_images(tmp_path / "imgs", 3)
    monkeypatch.setattr(runner, "_run_cli_child", child)
    return runner.run_automation(
        _profile(tmp_path, selection=selection), _identity(), [str(tmp_path / "imgs")], log=_quiet,
    )


def _assert_failed(outcome, stage: str, fragment: str) -> dict:
    assert outcome.exit_code == runner.EXIT_RUN_FAILED
    assert outcome.status == "failed" and outcome.verdict == "FAIL"
    manifest = _manifest(outcome.session_dir)
    assert manifest["status"] == "failed"
    assert manifest["finished_at"] is not None
    assert manifest["failure"]["stage"] == stage
    assert fragment in manifest["failure"]["error"]
    assert not any(s["status"] == "running" for s in manifest["stages"])
    acceptance = json.loads((Path(outcome.session_dir) / "automation" / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["verdict"] == "FAIL" and acceptance["run_id"] == manifest["run_id"]
    return manifest


def test_exception_during_calculation_is_recorded(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash")

    manifest = _assert_failed(_run_with_child(tmp_path, monkeypatch, boom), "calibration", "simulated crash")
    stages = {s["name"]: s["status"] for s in manifest["stages"]}
    assert stages["input_fingerprint"] == "completed"
    assert stages["calibration"] == "failed"
    # 실행 설정은 실패해도 남아 있어 재현 가능해야 한다
    assert manifest["calibration"]["argv"][0] == "--images"
    assert manifest["inputs"]["fingerprint"]["count"] == 3


def test_cli_nonzero_exit_is_recorded(tmp_path, monkeypatch):
    outcome = _run_with_child(tmp_path, monkeypatch, lambda *a, **k: 2)
    _assert_failed(outcome, "calibration", "종료 코드 2")


def test_cli_success_without_results_is_not_marked_completed(tmp_path, monkeypatch):
    outcome = _run_with_child(tmp_path, monkeypatch, lambda *a, **k: 0)
    _assert_failed(outcome, "result_collection", "calibration.ccproj")


def test_all_detections_failed_is_recorded(tmp_path, monkeypatch):
    def must_not_run(*_a, **_k):
        raise AssertionError("CLI must not start when detection failed")

    outcome = _run_with_child(tmp_path, monkeypatch, must_not_run, selection=True)
    _assert_failed(outcome, "image_selection", "패턴 검출에 실패")


def test_output_write_failure_is_recorded(tmp_path, monkeypatch):
    def failing_write(options, path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(runner, "write_cli_config", failing_write)
    outcome = _run_with_child(tmp_path, monkeypatch, lambda *a, **k: 0)
    _assert_failed(outcome, "calibration", "disk full")


def test_interrupted_run_is_recorded_as_failed(tmp_path, monkeypatch):
    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    outcome = _run_with_child(tmp_path, monkeypatch, interrupt)
    assert outcome.exit_code == 130
    manifest = _manifest(outcome.session_dir)
    assert manifest["status"] == "failed"
    assert "interrupted" in manifest["failure"]["error"]


def test_unwritable_output_root_is_reported(tmp_path):
    write_blank_images(tmp_path / "imgs", 2)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file", encoding="utf-8")
    raw = base_profile(image_selection={"enabled": False})
    raw["output"] = {"root": str(blocker)}
    profile = resolve_profile(raw, source_path="test")
    outcome = runner.run_automation(profile, _identity(), [str(tmp_path / "imgs")], log=_quiet)
    assert outcome.exit_code == runner.EXIT_RUN_FAILED
    assert outcome.session_dir is None
    assert "결과 세션을 만들 수 없습니다" in outcome.message


def test_failed_runs_never_overwrite_each_other(tmp_path, monkeypatch):
    first = _run_with_child(tmp_path, monkeypatch, lambda *a, **k: 2)
    first_manifest = (Path(first.session_dir) / "run_manifest.json").read_bytes()
    second = runner.run_automation(
        _profile(tmp_path), _identity(), [str(tmp_path / "imgs")], log=_quiet,
    )
    assert first.run_id != second.run_id
    assert first.session_dir != second.session_dir
    assert (Path(first.session_dir) / "run_manifest.json").read_bytes() == first_manifest
    assert len(_sessions(tmp_path)) == 2


def test_input_images_are_not_modified(tmp_path, monkeypatch):
    paths = write_blank_images(tmp_path / "imgs", 3)
    before = {p.name: rm.sha256_file(p) for p in paths}
    monkeypatch.setattr(runner, "_run_cli_child", lambda *a, **k: 2)
    runner.run_automation(_profile(tmp_path), _identity(), [str(tmp_path / "imgs")], log=_quiet)
    after = {p.name: rm.sha256_file(p) for p in sorted((tmp_path / "imgs").iterdir())}
    assert after == before


# ---------------------------------------------------------------------------
# run_manifest helpers
# ---------------------------------------------------------------------------


def test_git_info_keeps_porcelain_status_columns(monkeypatch):
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("rev-parse", "--abbrev-ref", "HEAD"): "windshield\n",
        ("status", "--porcelain", "--untracked-files=no"): " M app/cli.py\nM  docs/x.md\n",
    }
    monkeypatch.setattr(rm, "_git", lambda args, cwd: outputs[tuple(args)])
    info = rm.git_info()
    assert info["commit_sha"] == "abc123" and info["branch"] == "windshield"
    assert info["dirty"] is True
    assert info["modified_tracked_files"] == ["app/cli.py", "docs/x.md"]


def test_git_unavailable_is_reported_not_faked(monkeypatch):
    def no_git(args, cwd):
        raise FileNotFoundError("git")

    monkeypatch.setattr(rm, "_git", no_git)
    assert rm.git_info() == {"available": False, "error": "git"}


def test_git_is_not_queried_outside_a_checkout(tmp_path, monkeypatch):
    def must_not_call(args, cwd):
        raise AssertionError("git must not run for an installed copy")

    monkeypatch.setattr(rm, "_git", must_not_call)
    info = rm.git_info(tmp_path)
    assert info["available"] is False and "not a git checkout" in info["error"]


def test_atomic_json_writes_standard_json_without_infinity(tmp_path):
    path = rm.atomic_write_json(tmp_path / "x.json", {"score": math.inf, "nan": float("nan"), "ok": 1.5})
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text) == {"score": None, "nan": None, "ok": 1.5}
    assert not (tmp_path / "x.json.tmp").exists()


def test_input_fingerprint_is_location_independent(tmp_path):
    a = write_blank_images(tmp_path / "a", 2)
    b = write_blank_images(tmp_path / "b", 2)
    fa = rm.input_fingerprint([str(p) for p in a])
    fb = rm.input_fingerprint([str(p) for p in reversed(b)])
    assert fa["aggregate_sha256"] == fb["aggregate_sha256"]


# ---------------------------------------------------------------------------
# automation command line
# ---------------------------------------------------------------------------


def test_check_command_accepts_default_profile(capsys):
    assert automation_main(["check", "--profile", "default", "--camera-model", "SG3S-ISX031C"]) == 0


def test_run_command_rejects_bad_profile_before_calculation(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pattern: {type: charuco}\n", encoding="utf-8")
    code = automation_main(["run", "--profile", str(bad), "--images", str(tmp_path),
                            "--serial", "SN1", "--camera-model", "CAM"])
    assert code == 1
    assert "계산을 시작하지 않았습니다" in capsys.readouterr().err


def test_run_command_requires_camera_model(tmp_path, capsys):
    code = automation_main(["run", "--profile", "default", "--images", str(tmp_path), "--serial", "SN1"])
    assert code == 1
    assert "카메라 모델명" in capsys.readouterr().err
