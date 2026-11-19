"""app.cli --session-dir 와 OutputManager.reserve_session_directory.

모두 계산 전에 끝나는 경로라 빠르다 (이미지 경로를 일부러 없는 것으로 줘서
Session 생성 직후 종료 코드 1로 끝나게 한다). 실제 계산까지 지정 Session에
저장되는지는 tests/test_automation_e2e.py(slow)가 확인한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import export.output_manager as om
from app.cli import build_arg_parser, main
from export.output_manager import OutputManager, existing_session_outputs

PATTERN = ["--squares-x", "7", "--squares-y", "5", "--square-size", "0.04", "--marker-size", "0.03", "--quiet"]


def _missing_images(tmp_path: Path) -> list[str]:
    return ["--images", str(tmp_path / "no_such_images")]


@pytest.fixture
def default_root(tmp_path, monkeypatch) -> Path:
    """기본 Output root를 테스트 폴더로 돌려 사용자 홈을 건드리지 않는다."""
    root = tmp_path / "default_root"
    monkeypatch.setattr(om, "DEFAULT_OUTPUT_ROOT", root)
    return root


def test_option_exists_and_defaults_to_none():
    args = build_arg_parser().parse_args([])
    assert args.session_dir is None
    assert args.output_dir is None


def test_session_dir_creates_session_exactly_at_path(tmp_path, default_root):
    session = tmp_path / "exact" / "my_session"
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    assert code == 1  # 이미지가 없어서 계산 전에 종료
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == "my_session"
    assert (session / "intrinsic").is_dir() and (session / "reports").is_dir()
    assert (session / om.SESSION_CLAIM_FILENAME).is_file()
    assert not default_root.exists()  # 기본 위치에는 아무 것도 만들지 않음


def test_existing_session_is_not_overwritten(tmp_path, default_root, capsys):
    session = tmp_path / "s"
    main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    before = (session / "manifest.json").read_bytes()
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    assert code == 1
    assert "덮어쓰지 않습니다" in capsys.readouterr().err
    assert (session / "manifest.json").read_bytes() == before


def test_folder_with_previous_results_is_refused(tmp_path, default_root, capsys):
    session = tmp_path / "s"
    (session / "reports").mkdir(parents=True)
    (session / "reports" / "report.html").write_text("old", encoding="utf-8")
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    assert code == 1
    assert "reports/report.html" in capsys.readouterr().err
    assert (session / "reports" / "report.html").read_text(encoding="utf-8") == "old"
    assert not (session / "manifest.json").exists()


def test_concurrent_claim_is_refused(tmp_path, default_root, capsys):
    session = tmp_path / "s"
    session.mkdir()
    (session / om.SESSION_CLAIM_FILENAME).write_text("pid=1\n", encoding="utf-8")
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    assert code == 1
    assert not (session / "manifest.json").exists()


def test_prepared_folder_without_results_is_accepted(tmp_path, default_root):
    """자동화 계층이 먼저 쓴 run_manifest.json / automation/ 은 결과로 보지 않는다."""
    session = tmp_path / "s"
    (session / "automation").mkdir(parents=True)
    (session / "run_manifest.json").write_text("{}", encoding="utf-8")
    (session / "automation" / "cli_config.yaml").write_text("x: 1", encoding="utf-8")
    assert existing_session_outputs(session) == []
    main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)])
    assert (session / "manifest.json").is_file()
    assert (session / "run_manifest.json").read_text(encoding="utf-8") == "{}"


def test_path_that_is_a_file_is_refused(tmp_path, default_root, capsys):
    target = tmp_path / "file.txt"
    target.write_text("keep", encoding="utf-8")
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(target)])
    assert code == 1
    assert "폴더가 아닙니다" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "keep"


def test_unwritable_location_is_reported(tmp_path, default_root, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code = main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(blocker / "child")])
    assert code == 1
    assert "--session-dir" in capsys.readouterr().err


def test_session_dir_and_output_dir_are_mutually_exclusive(tmp_path, default_root):
    with pytest.raises(SystemExit):
        main([*_missing_images(tmp_path), *PATTERN,
              "--session-dir", str(tmp_path / "s"), "--output-dir", str(tmp_path / "o")])
    assert not (tmp_path / "s").exists()


def test_session_dir_via_config_file(tmp_path, default_root):
    session = tmp_path / "from_config"
    config = tmp_path / "c.yaml"
    config.write_text(f"session_dir: '{session.as_posix()}'\n", encoding="utf-8")
    main(["--config", str(config), *_missing_images(tmp_path), *PATTERN])
    assert (session / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# 기존 동작 유지
# ---------------------------------------------------------------------------


def test_default_behaviour_still_creates_timestamped_session_under_default_root(tmp_path, default_root):
    code = main([*_missing_images(tmp_path), *PATTERN, "--sensor-name", "cam"])
    assert code == 1
    sessions = [p for p in default_root.iterdir() if p.is_dir()]
    assert len(sessions) == 1 and sessions[0].name.endswith("_cam")
    assert (sessions[0] / "manifest.json").is_file()
    assert not (sessions[0] / om.SESSION_CLAIM_FILENAME).exists()


def test_output_dir_still_uses_flat_legacy_layout(tmp_path, default_root):
    out = tmp_path / "flat"
    main([*_missing_images(tmp_path), *PATTERN, "--output-dir", str(out)])
    assert not default_root.exists()
    assert not (out / "manifest.json").exists()


# ---------------------------------------------------------------------------
# OutputManager additions
# ---------------------------------------------------------------------------


def test_reserve_session_directory_is_unique_and_empty(tmp_path):
    fixed = datetime(2026, 11, 1, 9, 30, 0)
    manager = OutputManager(tmp_path, now=lambda: fixed)
    first = manager.reserve_session_directory("cam A")
    second = manager.reserve_session_directory("cam A")
    assert first.name == "2026-11-01_093000_cam_A"
    assert second.name == "2026-11-01_093000_cam_A_02"
    assert list(first.iterdir()) == [] and list(second.iterdir()) == []
    assert manager.session_dir is None  # 이 객체의 세션으로 연결하지 않음


def test_ensure_session_naming_unchanged(tmp_path):
    fixed = datetime(2026, 11, 1, 9, 30, 0)
    manager = OutputManager(tmp_path, now=lambda: fixed)
    session = manager.ensure_session("cam")
    assert session.name == "2026-11-01_093000_cam"
    other = OutputManager(tmp_path, now=lambda: fixed).ensure_session("cam")
    assert other.name == "2026-11-01_093000_cam_02"


def test_reserved_directory_occupied_by_file_is_refused(tmp_path, default_root, capsys):
    session = tmp_path / "occupied"
    session.mkdir()
    (session / "reports").write_text("preserve", encoding="utf-8")
    assert main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(session)]) == 1
    assert "덮어쓰지 않습니다" in capsys.readouterr().err
    assert (session / "reports").read_text(encoding="utf-8") == "preserve"
    assert not (session / om.SESSION_CLAIM_FILENAME).exists()


def test_inspection_permission_error_is_reported(tmp_path, default_root, monkeypatch, capsys):
    import app.cli as cli

    def denied(_path):
        raise PermissionError("inspection denied")

    monkeypatch.setattr(cli, "existing_session_outputs", denied)
    assert main([*_missing_images(tmp_path), *PATTERN, "--session-dir", str(tmp_path / "s")]) == 1
    assert "inspection denied" in capsys.readouterr().err


def test_default_session_creation_error_returns_cli_error(tmp_path, default_root, capsys):
    default_root.write_text("preserve", encoding="utf-8")
    assert main([*_missing_images(tmp_path), *PATTERN]) == 1
    assert "Output Session을 만들 수 없습니다" in capsys.readouterr().err
    assert default_root.read_text(encoding="utf-8") == "preserve"


def test_linked_output_directory_is_refused(tmp_path):
    import os
    import subprocess

    session = tmp_path / "linked"
    session.mkdir()
    target = tmp_path / "source"
    target.mkdir()
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(session / "reports"), str(target)], check=True, capture_output=True)
    else:
        (session / "reports").symlink_to(target, target_is_directory=True)
    assert "reports" in existing_session_outputs(session)
    assert list(target.iterdir()) == []
