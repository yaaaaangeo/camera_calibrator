"""
tests/test_git_upload.py
============================

[Git 업로드] 버튼이 쓰는 export.git_upload 를 로컬 bare 저장소(네트워크 없음)로 검증한다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from export import git_upload as gu

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git 이 없음")


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gu, "CACHE_ROOT", tmp_path / "cache")
    for key, value in {"GIT_AUTHOR_NAME": "tester", "GIT_AUTHOR_EMAIL": "tester@example.invalid",
                       "GIT_COMMITTER_NAME": "tester", "GIT_COMMITTER_EMAIL": "tester@example.invalid",
                       "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "gitconfig").write_text("[init]\n\tdefaultBranch = main\n")
    remote = tmp_path / "results.git"
    _git("init", "--bare", "--quiet", str(remote))
    return tmp_path, str(remote)


def _session(root: Path, name: str = "2026-11-18_101010") -> Path:
    d = root / "out" / name
    (d / "intrinsic").mkdir(parents=True)
    (d / "intrinsic" / "camera_info.yaml").write_text("fx: 1\n")
    (d / "manifest.json").write_text("{}\n")
    return d


def _files_on(remote: str, branch: str, tmp: Path) -> set[str]:
    check = tmp / "check"
    if check.exists():
        shutil.rmtree(check)
    _git("clone", "--quiet", "--branch", branch, remote, str(check))
    return {p.relative_to(check).as_posix() for p in check.rglob("*") if p.is_file() and ".git" not in p.parts}


def test_upload_intrinsic_session_then_unchanged_reupload_makes_no_commit(env):
    tmp, remote = env
    session = _session(tmp)
    plan, message = gu.plan_intrinsic(session, "front cam")
    assert message == f"intrinsic: front_cam {session.name}"
    cfg = gu.GitUploadConfig(remote=remote, branch="results")

    first = gu.upload(cfg, plan, message)
    assert not first.nothing_to_commit and first.commit
    # 결과값(YAML)만 - manifest/리포트는 올리지 않는다
    assert _files_on(remote, "results", tmp) == {f"intrinsic/no_vehicle/front_cam/{session.name}/camera_info.yaml"}
    again = gu.upload(cfg, plan, message)
    assert again.nothing_to_commit


def test_results_branch_does_not_inherit_code_files(env):
    """저장소에 코드(main)가 있어도 결과 브랜치는 결과 파일만 가진다 (orphan)."""
    tmp, remote = env
    work = tmp / "code"
    _git("clone", "--quiet", remote, str(work))
    (work / "app.py").write_text("print('code')\n")
    _git("add", ".", cwd=work)
    _git("commit", "--quiet", "-m", "code", cwd=work)
    _git("push", "--quiet", "origin", "HEAD:main", cwd=work)

    plan, message = gu.plan_intrinsic(_session(tmp), "cam")
    gu.upload(gu.GitUploadConfig(remote=remote, branch="calibration-results"), plan, message)
    files = _files_on(remote, "calibration-results", tmp)
    assert "app.py" not in files and any(f.startswith("intrinsic/no_vehicle/cam/") for f in files)
    assert "app.py" in _files_on(remote, "main", tmp)  # main 은 그대로


def test_push_rejected_by_someone_elses_newer_commit_is_rebased_and_retried(env):
    tmp, remote = env
    cfg = gu.GitUploadConfig(remote=remote, branch="results")
    plan_a, msg_a = gu.plan_intrinsic(_session(tmp, "2026-01-01_000000"), "a")
    gu.upload(cfg, plan_a, msg_a)
    gu.upload(cfg, *gu.plan_intrinsic(_session(tmp, "2026-01-02_000000"), "b"))  # 캐시가 최신이 됨

    # 다른 PC 가 먼저 하나 더 올린다
    other = tmp / "other"
    _git("clone", "--quiet", "--branch", "results", remote, str(other))
    (other / "note.txt").write_text("other\n")
    _git("add", ".", cwd=other)
    _git("commit", "--quiet", "-m", "other", cwd=other)
    _git("push", "--quiet", "origin", "HEAD:results", cwd=other)

    # 캐시는 fetch 후 원격 최신으로 맞춰지므로 그대로 올라가고, 다른 사람 커밋도 보존된다
    result = gu.upload(cfg, *gu.plan_intrinsic(_session(tmp, "2026-01-03_000000"), "c"))
    assert not result.nothing_to_commit
    files = _files_on(remote, "results", tmp)
    assert "note.txt" in files
    assert {"a", "b", "c"} <= {f.split("/")[2] for f in files if f.startswith("intrinsic/")}


def test_extrinsic_plan_has_only_result_values_of_saved_cameras(tmp_path):
    car = tmp_path / "calib_result" / "CAR 1"
    session = car / "2026-11-18_221716"
    for rel in ("front/extrinsic.txt", "front/result.yaml", "front/overlay.png", "front/boards.yaml",
                "front/report.txt", "left/extrinsic.txt", "left/result.yaml", "session.yaml",
                "capture/capture.mcap"):
        (session / rel).parent.mkdir(parents=True, exist_ok=True)
        (session / rel).write_text("x\n")
    (car / "vehicle.yaml").write_text("name: CAR 1\n")

    plan, message = gu.plan_extrinsic(session, "CAR 1", ["front"])  # left 는 이번에 저장 안 함
    assert {f.dest for f in plan.files} == {
        "extrinsic/CAR_1/2026-11-18_221716/front/extrinsic.txt",
        "extrinsic/CAR_1/2026-11-18_221716/front/result.yaml",
    }
    assert message == "extrinsic: CAR_1 2026-11-18_221716 (front)"


def test_intrinsic_plan_has_only_result_yamls(tmp_path):
    session = tmp_path / "2026-11-18_101010_cam"
    for rel in ("intrinsic/camera_brown_conrady.yaml", "intrinsic/camera_info.yaml", "intrinsic/kalibr/x.yaml",
                "reports/report.html", "project/calibration.ccproj", "manifest.json"):
        (session / rel).parent.mkdir(parents=True, exist_ok=True)
        (session / rel).write_text("x\n")
    plan, _ = gu.plan_intrinsic(session, "cam")
    assert [f.dest for f in plan.files] == [
        "intrinsic/no_vehicle/cam/2026-11-18_101010_cam/camera_brown_conrady.yaml",
        "intrinsic/no_vehicle/cam/2026-11-18_101010_cam/camera_info.yaml",
    ]


def test_oversized_result_file_is_listed_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(gu, "MAX_FILE_MB", 1 / 1024)
    session = tmp_path / "s"
    (session / "intrinsic").mkdir(parents=True)
    (session / "intrinsic" / "camera_info.yaml").write_text("a\n")
    (session / "intrinsic" / "camera_rational.yaml").write_bytes(b"0" * 2048)
    plan, _ = gu.plan_intrinsic(session, "c")
    assert [f.dest for f in plan.files] == ["intrinsic/no_vehicle/c/s/camera_info.yaml"]
    assert plan.skipped == ["intrinsic/no_vehicle/c/s/camera_rational.yaml"]


def test_unreachable_remote_reports_a_readable_error(env):
    tmp, _remote = env
    plan, message = gu.plan_intrinsic(_session(tmp), "cam")
    with pytest.raises(gu.GitUploadError) as err:
        gu.upload(gu.GitUploadConfig(remote=str(tmp / "missing.git"), branch="main"), plan, message)
    assert "git clone" in str(err.value)


def test_config_round_trip(tmp_path):
    path = tmp_path / "cfg.json"
    assert gu.load_config(path) == gu.GitUploadConfig()
    gu.save_config(gu.GitUploadConfig(remote="git@example.invalid:r.git", branch="res"), path)
    assert gu.load_config(path) == gu.GitUploadConfig(remote="git@example.invalid:r.git", branch="res")


def test_push_rejected_between_commit_and_push_is_rebased(env):
    """커밋한 직후 다른 PC 가 먼저 push 해서 거절돼도 rebase 후 다시 올린다."""
    tmp, remote = env
    cfg = gu.GitUploadConfig(remote=remote, branch="results")
    gu.upload(cfg, *gu.plan_intrinsic(_session(tmp, "2026-01-01_000000"), "a"))
    other = tmp / "other"
    _git("clone", "--quiet", "--branch", "results", remote, str(other))

    def race(message: str) -> None:
        if message.startswith("커밋"):
            (other / "note.txt").write_text("other\n")
            _git("add", ".", cwd=other)
            _git("commit", "--quiet", "-m", "other", cwd=other)
            _git("push", "--quiet", "origin", "HEAD:results", cwd=other)

    log: list[str] = []
    result = gu.upload(cfg, *gu.plan_intrinsic(_session(tmp, "2026-01-02_000000"), "b"),
                       log=lambda m: (log.append(m), race(m)))
    assert any("다시 시도" in m for m in log)
    files = _files_on(remote, "results", tmp)
    assert "note.txt" in files and any(f.startswith("intrinsic/no_vehicle/b/") for f in files)
    assert result.commit == _git("rev-parse", "--short", "results", cwd=remote).strip()


def test_author_from_dialog_is_used_when_git_has_no_identity(env, monkeypatch):
    tmp, remote = env
    for key in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(key)
    monkeypatch.setenv("EMAIL", "")
    cfg = gu.GitUploadConfig(remote=remote, branch="results", author_name="홍길동", author_email="hong@example.invalid")
    gu.upload(cfg, *gu.plan_intrinsic(_session(tmp), "cam"))
    assert _git("--git-dir", remote, "log", "-1", "--format=%an <%ae>", "results").strip() == "홍길동 <hong@example.invalid>"


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_upload_dialog_uses_saved_settings_and_rebuilds_plan(qapp, tmp_path, monkeypatch):
    """설정 경로는 호출 시점에 읽고, 설정을 바꾸면 올라갈 파일 목록을 다시 만든다."""
    from ui.git_upload_dialog import GitUploadDialog

    monkeypatch.setattr(gu, "CONFIG_PATH", tmp_path / "cfg.json")
    monkeypatch.setattr(gu, "git_identity", lambda: ("", ""))
    gu.save_config(gu.GitUploadConfig(remote="git@example.invalid:r.git", branch="res", author_name="a",
                                      author_email="a@example.invalid"))
    session = tmp_path / "2026-11-18_101010_cam"
    for rel in ("intrinsic/camera_info.yaml", "intrinsic/camera_brown_conrady.yaml", "reports/report.html"):
        (session / rel).parent.mkdir(parents=True, exist_ok=True)
        (session / rel).write_text("x\n")
    dialog = GitUploadDialog(lambda config: gu.plan_intrinsic(session, "cam", config))
    try:
        assert "git@example.invalid:r.git" in dialog.target_label.text()
        assert len(dialog.plan.files) == 2 and dialog.upload_button.isEnabled()
        dialog.config = gu.GitUploadConfig(intrinsic_files=["report_html"], intrinsic_path="cal/{date}/{camera}/{session}")
        dialog._rebuild()
        assert [f.dest for f in dialog.plan.files] == ["cal/2026-11-18/cam/2026-11-18_101010_cam/report.html"]
        dialog.message_edit.setText("내 메시지")
        dialog.message_edit.setProperty("auto", False)
        dialog._rebuild()
        assert dialog.message_edit.text() == "내 메시지"  # 직접 쓴 메시지는 덮어쓰지 않는다
    finally:
        dialog.close()


def test_settings_dialog_validates_and_saves(qapp, tmp_path, monkeypatch):
    from ui.git_upload_dialog import GitSettingsDialog

    monkeypatch.setattr(gu, "CONFIG_PATH", tmp_path / "cfg.json")
    monkeypatch.setattr(gu, "git_identity", lambda: ("", ""))
    dialog = GitSettingsDialog()
    try:
        dialog.remote_edit.setText("git@example.invalid:r.git")
        dialog.extrinsic_path_edit.setText("extrinsic/{vehicle}/{camera}")  # {session} 없음
        dialog._save()
        assert "{session}" in dialog.status_label.text() and not (tmp_path / "cfg.json").exists()
        dialog.extrinsic_path_edit.setText("ext/{vehicle}/{date}/{session}/{camera}")
        dialog.extrinsic_checks["overlay.png"].setChecked(True)
        dialog._save()
        saved = gu.load_config()
        assert saved.remote == "git@example.invalid:r.git"
        assert saved.extrinsic_path == "ext/{vehicle}/{date}/{session}/{camera}"
        assert saved.extrinsic_files == ["extrinsic.txt", "result.yaml", "overlay.png"]
    finally:
        dialog.close()


def test_path_template_rules():
    fields = gu.EXTRINSIC_PATH_FIELDS
    assert gu.check_path_template("extrinsic/{vehicle}/{session}/{camera}", fields) == ""
    assert "모르는" in gu.check_path_template("x/{car}/{session}", fields)
    assert "{session}" in gu.check_path_template("x/{vehicle}", fields)
    assert ".." in gu.check_path_template("../{session}", fields)
    assert gu.render_path("a/{vehicle}//{session}/", vehicle="CAR 1", session="s") == "a/CAR_1/s"


def test_recommended_only_and_session_yaml_once(tmp_path):
    session = tmp_path / "2026-11-18_101010"
    for rel in ("intrinsic/camera_brown_conrady.yaml", "intrinsic/camera_rational.yaml", "intrinsic/camera_info.yaml"):
        (session / rel).parent.mkdir(parents=True, exist_ok=True)
        (session / rel).write_text("x\n")
    cfg = gu.GitUploadConfig(intrinsic_recommended_only=True)
    plan, _ = gu.plan_intrinsic(session, "c", cfg, recommended="extended_pinhole")
    assert [f.dest.rsplit("/", 1)[1] for f in plan.files] == ["camera_rational.yaml", "camera_info.yaml"]

    room = tmp_path / "car" / "2026-11-18_101010"
    for rel in ("a/extrinsic.txt", "b/extrinsic.txt", "session.yaml"):
        (room / rel).parent.mkdir(parents=True, exist_ok=True)
        (room / rel).write_text("x\n")
    cfg = gu.GitUploadConfig(extrinsic_files=["extrinsic.txt", "session.yaml"])
    plan, _ = gu.plan_extrinsic(room, "car", ["a", "b"], cfg)
    assert sorted(f.dest for f in plan.files) == [
        "extrinsic/car/2026-11-18_101010/a/extrinsic.txt",
        "extrinsic/car/2026-11-18_101010/b/extrinsic.txt",
        "extrinsic/car/2026-11-18_101010/session.yaml",
    ]
