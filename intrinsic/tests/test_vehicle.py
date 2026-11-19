"""
tests/test_vehicle.py
=====================

홈 화면의 차량 선택 (shared.vehicle) - Camera Intrinsic 과 CAMERA–LiDAR 가 같은 차량 프로필을
쓰는지, 결과 저장소(Git)로 차량 정보가 다른 PC 로 넘어가는지.
(conftest 의 autouse 픽스처가 차량 폴더와 렌즈 프리셋 파일을 임시 폴더로 바꿔 둔다)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from export import git_upload as gu
from shared import camera as shared_camera
from shared import vehicle as sv

ROOM = Path(__file__).resolve().parents[2] / "extrinsic"
if str(ROOM) not in sys.path:
    sys.path.append(str(ROOM))

K = np.array([[1100.0, 0, 955.5], [0, 1098.0, 541.25], [0, 0, 1]])


def _lens():
    return shared_camera.lens_from_kd(K, [-0.28, 0.1, 0.001, -0.002, -0.03], "brown_conrady", 1920, 1080)


def test_intrinsic_camera_names_and_lens_are_shared_with_room_profile(monkeypatch):
    from gui.core import project, session

    monkeypatch.setattr(project, "RESULT_DIR", sv.RESULT_DIR)
    monkeypatch.setattr(session, "RESULT_DIR", sv.RESULT_DIR)
    # CAMERA–LiDAR 가 먼저 차량을 만들고 카메라 하나를 저장했다
    prof = session.VehicleProfile(name="torres03", lidar_topics=["/points"],
                                  cameras={"/cam/front/image": session.CameraEntry("front", "")})
    prof.save()
    assert sv.list_vehicles() == ["torres03"]
    assert sv.camera_names("torres03") == ["front"]

    # Camera Intrinsic 이 새 카메라를 쓰고, front 의 렌즈를 등록한다
    assert sv.add_camera("torres03", "rear") is True
    assert sv.add_camera("torres03", "rear") is False
    sv.set_lens("torres03", "front", "torres03_front")
    assert sv.camera_names("torres03") == ["front", "rear"]
    assert sv.lens_of("torres03", "front") == "torres03_front"

    # CAMERA–LiDAR 가 다시 읽으면 그 렌즈를 고르고, 저장해도 인트린직 쪽 항목이 남는다
    prof = session.load_profile("torres03")
    assert prof.lens_for("/cam/front/image") == "torres03_front"
    assert prof.lens_for("/cam/rear/image", "rear") == ""
    prof.cameras["/cam/rear/image"] = session.CameraEntry("rear", "")
    prof.save()
    assert sv.camera_names("torres03") == ["front", "rear"]
    assert sv.load("torres03")["lenses"]["front"] == "torres03_front"


def test_no_vehicle_has_no_cameras_and_does_not_write():
    assert sv.camera_names(sv.NO_VEHICLE) == []
    assert sv.add_camera(sv.NO_VEHICLE, "econ120") is False
    assert not sv.RESULT_DIR.exists()


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def remote(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git 이 없음")
    monkeypatch.setattr(gu, "CACHE_ROOT", tmp_path / "cache")
    for key, value in {"GIT_AUTHOR_NAME": "tester", "GIT_AUTHOR_EMAIL": "tester@example.invalid",
                       "GIT_COMMITTER_NAME": "tester", "GIT_COMMITTER_EMAIL": "tester@example.invalid",
                       "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "gitconfig").write_text("[init]\n\tdefaultBranch = main\n")
    path = tmp_path / "results.git"
    _git("init", "--bare", "--quiet", str(path))
    return str(path)


def _session(root: Path) -> Path:
    d = root / "out" / "2026-11-19_101010"
    (d / "intrinsic").mkdir(parents=True)
    (d / "intrinsic" / "camera_info.yaml").write_text("fx: 1\n")
    return d


def test_vehicle_travels_to_another_pc_through_the_results_repo(tmp_path, remote, monkeypatch):
    # PC 1: 차량 torres03, 카메라 front 의 인트린직 결과를 렌즈로 등록하고 Git 업로드
    sv.ensure("torres03")
    sv.add_camera("torres03", "front")
    shared_camera.add_lens_preset("torres03_front", _lens(), "Camera Intrinsic")
    shared_camera.add_lens_preset("unrelated", _lens(), "")
    sv.set_lens("torres03", "front", "torres03_front")
    cfg = gu.GitUploadConfig(remote=remote, branch="results")
    plan, message = gu.plan_intrinsic(_session(tmp_path), "front", cfg, vehicle="torres03")
    dests = {f.dest for f in plan.files}
    assert dests == {"intrinsic/torres03/front/2026-11-19_101010/camera_info.yaml",
                     "vehicles/torres03/vehicle.yaml", "vehicles/torres03/lenses.yaml"}
    assert message == "intrinsic: torres03/front 2026-11-19_101010"
    gu.upload(cfg, plan, message)

    # PC 2: 아무 차량도 없는 새 PC
    pc2 = tmp_path / "pc2"
    monkeypatch.setattr(sv, "RESULT_DIR", pc2 / "calib_result")
    monkeypatch.setattr(shared_camera, "LENS_PRESETS_PATH", pc2 / "camera_intrinsics.yaml")
    monkeypatch.setattr(gu, "CACHE_ROOT", pc2 / "cache")
    (found,) = gu.fetch_vehicles(cfg)
    assert (found.name, found.status) == ("torres03", "새 차량")
    assert [e["name"] for e in found.lenses] == ["torres03_front"]  # 이 차량이 쓰는 렌즈만
    gu.import_vehicle(found)
    assert sv.list_vehicles() == ["torres03"]
    assert sv.lens_of("torres03", "front") == "torres03_front"
    preset = {e["name"]: e for e in shared_camera.load_preset_entries()}["torres03_front"]
    assert preset["fx"] == pytest.approx(1100.0) and preset["k1"] == pytest.approx(-0.28)
    assert gu.fetch_vehicles(cfg)[0].status == "같음"

    # PC 2 에만 있는 카메라는 다시 가져와도 남고, 바꾸기 전 파일은 .bak 으로 남는다
    sv.add_camera("torres03", "rear")
    gu.import_vehicle(gu.fetch_vehicles(cfg)[0])
    assert sv.camera_names("torres03") == ["front", "rear"]
    assert sv.profile_path("torres03").with_name("vehicle.yaml.bak").is_file()


def test_no_vehicle_goes_under_no_vehicle_without_vehicle_files(tmp_path):
    plan, message = gu.plan_intrinsic(_session(tmp_path), "econ120", gu.GitUploadConfig(), vehicle="")
    assert {f.dest for f in plan.files} == {"intrinsic/no_vehicle/econ120/2026-11-19_101010/camera_info.yaml"}
    assert message == "intrinsic: econ120 2026-11-19_101010"


def test_vehicle_files_can_be_left_out(tmp_path):
    sv.ensure("car")
    cfg = gu.GitUploadConfig(upload_vehicle=False)
    plan, _ = gu.plan_intrinsic(_session(tmp_path), "front", cfg, vehicle="car")
    assert not any(f.dest.startswith("vehicles/") for f in plan.files)


def test_old_intrinsic_template_without_vehicle_still_works(tmp_path):
    cfg = gu.GitUploadConfig(intrinsic_path="intrinsic/{camera}/{session}", upload_vehicle=False)
    plan, _ = gu.plan_intrinsic(_session(tmp_path), "front", cfg, vehicle="car")
    assert {f.dest for f in plan.files} == {"intrinsic/front/2026-11-19_101010/camera_info.yaml"}


# ------------------------------------------------------------------------- 화면


def test_home_requires_a_vehicle_choice_and_no_vehicle_is_intrinsic_only(qapp):
    from ui.calibration_home_view import CalibrationHomeView

    sv.ensure("torres03")
    home = CalibrationHomeView()
    assert not home.is_vehicle_chosen()
    assert not home.intrinsic_start.isEnabled() and not home.room_start.isEnabled()
    assert [home.vehicle_combo.itemText(i) for i in range(home.vehicle_combo.count())] == [sv.NO_VEHICLE_LABEL, "torres03"]

    home.vehicle_combo.setCurrentIndex(0)
    assert home.vehicle() == sv.NO_VEHICLE
    assert home.intrinsic_start.isEnabled() and not home.room_start.isEnabled()

    home.vehicle_combo.setEditText("new car")
    assert home.intrinsic_start.isEnabled() and home.room_start.isEnabled()
    assert home.confirm_vehicle() == "new_car"
    assert sv.profile_path("new_car").is_file()


def test_intrinsic_camera_name_lists_the_vehicle_cameras(qapp):
    from ui.main_window import MainWindow

    sv.ensure("torres03")
    sv.add_camera("torres03", "front")
    sv.add_camera("torres03", "rear")
    sv.set_lens("torres03", "front", "torres03_front")
    win = MainWindow()
    try:
        win.home_view.vehicle_combo.setEditText("torres03")
        win._show_intrinsic_workspace()
        assert win.vehicle == "torres03"
        combo = win.sensor_name_combo
        assert [combo.itemText(i) for i in range(combo.count())] == ["front", "rear"]
        combo.setCurrentIndex(0)
        assert win._session_camera_name() == "front"
        assert "torres03_front" in win.camera_lens_label.text()

        # 새 이름은 저장할 때 차량에 추가된다
        win.sensor_name_edit.setText("side")
        win._remember_vehicle_camera()
        assert "side" in sv.camera_names("torres03")

        # 차량 없음으로 바꾸면 목록이 비고 자유 입력
        win._show_home()
        win.home_view.vehicle_combo.setCurrentIndex(0)
        win._show_intrinsic_workspace()
        assert win.vehicle == "" and win.sensor_name_combo.count() == 0
    finally:
        win.close()
        win.deleteLater()


def test_profile_file_is_plain_yaml(tmp_path):
    sv.set_lens("car", "front", "p")
    data = yaml.safe_load(sv.profile_path("car").read_text())
    assert data == {"vehicle": "car", "lenses": {"front": "p"}}


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
