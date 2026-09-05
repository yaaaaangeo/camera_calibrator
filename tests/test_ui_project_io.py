"""
tests/test_ui_project_io.py
================================

ui/main_window.py의 프로젝트 저장/불러오기 - QFileDialog는 headless 환경에서
띄울 수 없으므로, _on_save_project/_on_load_project 내부 로직(상태 복원)을
직접 검증한다. QFileDialog 자체를 monkeypatch해서 실제 버튼 클릭 -> 다이얼로그
경로 반환 -> 핸들러 실행까지의 흐름도 함께 확인한다.

PySide6가 없는 환경에서는 전체 파일을 스킵한다.
"""

from __future__ import annotations

import glob

import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication, QFileDialog

from calibration.compare import run_all_models
from calibration.frame_quality import compute_frame_quality_scores
from calibration.models.common import infer_image_size
from calibration.project_io import save_project
from calibration.quality import analyze_dataset_quality
from calibration.recommender import compute_model_scores
from calibration.types import CalibrationProject
from calibration.validation import validate_all_models

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _monkeypatch_load_dialog(path: str) -> None:
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))


def _monkeypatch_save_dialog(path: str) -> None:
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ""))


@pytest.fixture(scope="module")
def saved_project_path(synthetic_distorted_dataset_dir, camera_config, pattern_config, tmp_path_factory):
    """실제 파이프라인을 돌려서 만든 .ccproj 파일 경로.

    module 스코프 - 이 파일의 여러 테스트가 같은 저장된 프로젝트를 읽기만
    하고(불러오기 로직 검증이 목적이지, 매번 새로 계산하는 게 목적이 아님)
    쓰지는 않으므로 공유해도 안전하다. detect_dataset()으로 이미지에서
    새로 만든 전용 Dataset이라(세션 공유 fixture를 참조하지 않음)
    다른 파일과의 상태 오염 걱정도 없다.

    module 스코프 fixture는 함수 스코프 전용인 tmp_path를 쓸 수 없어서
    tmp_path_factory로 바꿨다.
    """
    from calibration.detector import detect_dataset

    paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
    dataset = detect_dataset(paths, pattern_config)
    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    validation_results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.25)
    scores = compute_model_scores(calibration_results, validation_results)

    project = CalibrationProject(
        project_name="UI 테스트", camera_config=camera_config, pattern_config=pattern_config,
        dataset=dataset, calibration_results=calibration_results,
        validation_results=validation_results, model_scores=scores,
    )
    out_dir = tmp_path_factory.mktemp("ui_project_io")
    path = str(out_dir / "ui_test.ccproj")
    save_project(project, path)
    return path


def test_load_project_restores_widget_values(qapp, saved_project_path, camera_config, pattern_config):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        _monkeypatch_load_dialog(saved_project_path)
        win._on_load_project()

        assert win.width_spin.value() == camera_config.width
        assert win.height_spin.value() == camera_config.height
        assert win.squares_x_spin.value() == pattern_config.squares_x
        assert win.squares_y_spin.value() == pattern_config.squares_y
        assert abs(win.square_size_spin.value() - pattern_config.square_size * 1000.0) < 1e-6
        assert win.dictionary_combo.currentText() == pattern_config.dictionary
    finally:
        win.close()


def test_load_project_restores_dataset_and_results(qapp, saved_project_path):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        _monkeypatch_load_dialog(saved_project_path)
        win._on_load_project()

        assert win.dataset is not None
        assert win.dataset.num_total == 16
        assert len(win.image_paths) == 16
        assert len(win.calibration_results) == 3
        assert win.run_button.isEnabled()
        assert win.dataset_view.table.rowCount() == 16
    finally:
        win.close()


def test_save_then_load_round_trip_through_ui(
    qapp, synthetic_distorted_dataset_dir, camera_config, pattern_config, tmp_path
):
    """save -> load를 UI 핸들러로 왕복시켜서 위젯 상태까지 일관되는지 확인."""
    from ui.main_window import MainWindow
    from calibration.detector import detect_dataset

    win = MainWindow()
    try:
        paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
        win.dataset = detect_dataset(paths, pattern_config)
        win.camera_config = camera_config
        win.pattern_config = pattern_config
        results = run_all_models(win.dataset, camera_config)
        win.calibration_results = {r.model_name: r for r in results}

        save_path = str(tmp_path / "roundtrip.ccproj")
        _monkeypatch_save_dialog(save_path)
        win._on_save_project()

        import os
        assert os.path.exists(save_path)

        win2 = MainWindow()
        try:
            _monkeypatch_load_dialog(save_path)
            win2._on_load_project()
            assert win2.dataset.num_total == win.dataset.num_total
            assert win2.camera_config.width == camera_config.width
        finally:
            win2.close()
    finally:
        win.close()


def test_load_project_warns_on_missing_images(qapp, saved_project_path, monkeypatch, tmp_path):
    """이미지가 없어졌어도 크래시하지 않고 QMessageBox.warning으로 알리는지.

    주의: saved_project_path가 참조하는 이미지는 세션 전체가 공유하는
    synthetic_distorted_dataset_dir 안에 있다 - 옮긴 뒤 반드시 원래
    자리로 복구해야 한다 (안 그러면 이 테스트 이후에 도는 다른 모든
    테스트 파일이 이미지가 없어진 채로 실행되는 심각한 오염이 생긴다).
    finally에서 예외가 나도 최대한 복구를 시도한다.
    """
    import os
    import shutil
    from ui.main_window import MainWindow
    from calibration.project_io import load_project

    project, _ = load_project(saved_project_path)
    moved_dir = tmp_path / "moved_away"
    moved_dir.mkdir()
    original_locations = []
    for f in project.dataset.frames:
        if os.path.exists(f.image_info.path):
            dest = str(moved_dir / os.path.basename(f.image_info.path))
            shutil.move(f.image_info.path, dest)
            original_locations.append((dest, f.image_info.path))

    win = MainWindow()
    warned = {"called": False}

    def fake_warning(*args, **kwargs):
        warned["called"] = True

    monkeypatch.setattr("ui.main_window.QMessageBox.warning", fake_warning)
    try:
        _monkeypatch_load_dialog(saved_project_path)
        win._on_load_project()  # 크래시하지 않아야 함
        assert warned["called"], "이미지가 없어졌으면 경고가 떠야 함"
        assert win.dataset is not None  # 데이터 자체는 살아있어야 함
    finally:
        win.close()
        for dest, original in original_locations:
            if os.path.exists(dest):
                shutil.move(dest, original)
