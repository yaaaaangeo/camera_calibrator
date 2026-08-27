"""
tests/test_json_csv_export.py
==================================

설계 문서 11번 - JSON/CSV export. 실제 파이프라인 결과로 두 export가
정확한 값을 담고, 파일로 잘 저장되는지 검증한다. json_safe()가 중첩된
dataclass(RegionalError, RadialErrorProfile 등)까지 재귀적으로 순수
JSON 타입으로 풀어내는지가 핵심 - 이 프로젝트 자체를 모르는 외부
스크립트가 읽을 파일이라 numpy 배열/Enum/dataclass가 그대로 남아있으면 안 된다.
"""

from __future__ import annotations

import copy
import csv
import json

import numpy as np
import pytest

from calibration.compare import run_all_models
from calibration.frame_quality import compute_frame_quality_scores
from calibration.json_utils import json_safe
from calibration.models.common import infer_image_size
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.recommender import compute_final_result, compute_model_scores
from calibration.types import CameraModelType, RadialBin, RadialErrorProfile, RegionalError
from calibration.validation import validate_all_models
from export.csv_export import dataset_to_rows, export_csv
from export.json_export import build_export_dict, export_json

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_pipeline_result(synthetic_dataset, camera_config, pattern_config):
    """검출~FinalResult까지 실제로 다 돌린 결과 - 이 파일의 여러 테스트가 공유."""
    dataset = copy.deepcopy(synthetic_dataset)
    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    validation_results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.25)
    scores = compute_model_scores(calibration_results, validation_results)
    chosen = next((s.model_name for s in scores if s.is_recommended), list(calibration_results)[0])

    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None
    final_result = compute_final_result(
        chosen, calibration_results, validation_results,
        dataset_coverage_pct=coverage_pct, scores=scores,
    )

    return {
        "dataset": dataset, "calibration_results": calibration_results,
        "validation_results": validation_results, "scores": scores,
        "chosen": chosen, "final_result": final_result,
    }


# ---------------------------------------------------------------------------
# json_safe() 자체 - 중첩 dataclass 처리 (이번에 실제로 잡은 버그의 회귀 테스트)
# ---------------------------------------------------------------------------

def test_json_safe_flattens_nested_dataclass():
    """RegionalError 같은 중첩 dataclass 인스턴스를 미리 asdict()로 안
    풀어줘도 json_safe()가 알아서 순수 dict로 바꿔야 한다 - 안 그러면
    json.dump()가 TypeError를 던진다(실제로 이 프로젝트에서 겪은 버그).
    """
    obj = RegionalError(center=0.3, left=0.5, right=None, top=0.4, bottom=0.6, corner=0.7)
    safe = json_safe(obj, ndarray_wrapper=False)
    assert isinstance(safe, dict)
    assert safe["center"] == 0.3
    assert safe["right"] is None
    json.dumps(safe)


def test_json_safe_flattens_doubly_nested_dataclass():
    """RadialErrorProfile.bins처럼 dataclass 안에 dataclass 리스트이 또
    들어있는 2단 중첩도 처리돼야 한다.
    """
    profile = RadialErrorProfile(
        bins=[RadialBin(radius_min=0, radius_max=100, mean_error=0.5, num_points=10)],
        max_radius=500.0,
    )
    safe = json_safe(profile, ndarray_wrapper=False)
    assert isinstance(safe["bins"], list)
    assert isinstance(safe["bins"][0], dict)
    assert safe["bins"][0]["mean_error"] == 0.5
    json.dumps(safe)


def test_json_safe_no_wrapper_mode_gives_plain_nested_list():
    """ndarray_wrapper=False면 __ndarray__ 래퍼 없이 순수 중첩 리스트여야
    한다 (project_io.py의 .ccproj용 True 모드와 구분).
    """
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    safe = json_safe(arr, ndarray_wrapper=False)
    assert safe == [[1.0, 2.0], [3.0, 4.0]]
    assert "__ndarray__" not in str(safe)


def test_json_safe_wrapper_mode_still_works_for_project_io():
    """project_io.py가 기대하는 기존 동작(True가 기본값)은 안 바뀌어야 한다."""
    arr = np.array([1.0, 2.0])
    safe = json_safe(arr)
    assert safe["__ndarray__"] is True
    assert safe["data"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def test_build_export_dict_contains_expected_top_level_keys(full_pipeline_result, camera_config, pattern_config):
    d = build_export_dict(
        camera_config, pattern_config, full_pipeline_result["dataset"],
        full_pipeline_result["calibration_results"], full_pipeline_result["validation_results"],
        full_pipeline_result["chosen"], final_result=full_pipeline_result["final_result"],
        model_scores=full_pipeline_result["scores"],
    )
    for key in (
        "camera",
        "pattern",
        "dataset",
        "chosen_model",
        "models",
        "cross_validation",
        "bootstrap_stability",
        "final_result",
        "final_calibration_summary",
        "model_scores",
    ):
        assert key in d
    assert "failure_reasons" in d["dataset"]
    assert "holdout" in d["cross_validation"]
    assert d["final_calibration_summary"]["chosen_model"] == full_pipeline_result["chosen"].value


def test_export_json_writes_valid_json_file(full_pipeline_result, camera_config, pattern_config, tmp_path):
    path = str(tmp_path / "calibration.json")
    result_path = export_json(
        camera_config, pattern_config, full_pipeline_result["dataset"],
        full_pipeline_result["calibration_results"], full_pipeline_result["validation_results"],
        full_pipeline_result["chosen"], path,
        final_result=full_pipeline_result["final_result"], model_scores=full_pipeline_result["scores"],
    )
    assert result_path == path

    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["chosen_model"] == full_pipeline_result["chosen"].value
    assert loaded["camera"]["width"] == camera_config.width
    assert loaded["pattern"]["squares_x"] == pattern_config.squares_x


def test_export_json_camera_matrix_values_match_source(full_pipeline_result, camera_config, pattern_config, tmp_path):
    """export된 camera_matrix가 원본 CalibrationResult의 값과 정확히
    일치해야 한다 (JSON을 외부 도구가 신뢰하고 쓸 수 있으려면 필수).
    """
    chosen = full_pipeline_result["chosen"]
    cal = full_pipeline_result["calibration_results"][chosen]
    assert cal.success, "테스트 전제: 추천된 모델은 성공한 상태여야 함"

    path = str(tmp_path / "calibration.json")
    export_json(
        camera_config, pattern_config, full_pipeline_result["dataset"],
        full_pipeline_result["calibration_results"], full_pipeline_result["validation_results"],
        chosen, path, final_result=full_pipeline_result["final_result"],
    )
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)

    K_exported = loaded["models"][chosen.value]["camera_matrix"]
    assert isinstance(K_exported, list) and isinstance(K_exported[0], list)
    assert abs(K_exported[0][0] - float(cal.camera_matrix[0][0])) < 1e-6
    assert abs(K_exported[2][2] - float(cal.camera_matrix[2][2])) < 1e-9


def test_export_json_includes_radial_profile_and_validation(full_pipeline_result, camera_config, pattern_config, tmp_path):
    chosen = full_pipeline_result["chosen"]
    path = str(tmp_path / "calibration.json")
    export_json(
        camera_config, pattern_config, full_pipeline_result["dataset"],
        full_pipeline_result["calibration_results"], full_pipeline_result["validation_results"],
        chosen, path, final_result=full_pipeline_result["final_result"], model_scores=full_pipeline_result["scores"],
    )
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)

    entry = loaded["models"][chosen.value]
    assert "radial_error_profile" in entry
    assert isinstance(entry["radial_error_profile"]["bins"], list)
    assert "validation" in entry
    assert "residual_stats" in entry
    assert "parameter_uncertainty_bootstrap" in entry
    assert "final_result" in loaded
    assert "final_calibration_summary" in loaded
    assert "model_scores" in loaded and len(loaded["model_scores"]) == 3


def test_export_json_handles_failed_model_gracefully(camera_config, pattern_config, tmp_path):
    """실패한 모델(success=False)이 섞여 있어도 export 자체가 죽으면 안 된다."""
    from calibration.types import CalibrationResult, Dataset

    cal = {CameraModelType.PINHOLE: CalibrationResult(model_name=CameraModelType.PINHOLE, success=False, error_message="테스트 실패")}
    path = str(tmp_path / "calibration.json")
    export_json(camera_config, pattern_config, Dataset(frames=[]), cal, {}, CameraModelType.PINHOLE, path)

    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["models"][CameraModelType.PINHOLE.value]["success"] is False


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_dataset_to_rows_matches_frame_count(full_pipeline_result):
    dataset = full_pipeline_result["dataset"]
    rows = dataset_to_rows(dataset)
    assert len(rows) == dataset.num_total


def test_export_csv_writes_valid_csv_file(full_pipeline_result, tmp_path):
    dataset = full_pipeline_result["dataset"]
    path = str(tmp_path / "dataset.csv")
    result_path = export_csv(dataset, path)
    assert result_path == path

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == dataset.num_total
    assert "image_id" in rows[0]
    assert "quality_grade" in rows[0]
    assert "reprojection_error_px" in rows[0]


def test_export_csv_detected_frames_have_corner_counts(full_pipeline_result, tmp_path):
    dataset = full_pipeline_result["dataset"]
    path = str(tmp_path / "dataset.csv")
    export_csv(dataset, path)

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    detected_rows = [r for r in rows if r["status"] == "detected"]
    assert len(detected_rows) > 0
    for row in detected_rows:
        assert row["num_corners"] not in (None, "")
        assert int(row["num_corners"]) > 0


def test_export_csv_empty_dataset_writes_header_only(tmp_path):
    from calibration.types import Dataset

    path = str(tmp_path / "empty.csv")
    export_csv(Dataset(frames=[]), path)

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows == []
    with open(path, encoding="utf-8") as f:
        header_line = f.readline()
    assert "image_id" in header_line


# ---------------------------------------------------------------------------
# UI 핸들러 (main_window.py의 _on_export_json / _on_export_csv)
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_ui_export_json_handler_writes_file(qapp, full_pipeline_result, camera_config, pattern_config, tmp_path):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        win.dataset = full_pipeline_result["dataset"]
        win.camera_config = camera_config
        win.pattern_config = pattern_config
        win.calibration_results = full_pipeline_result["calibration_results"]
        win.validation_results = full_pipeline_result["validation_results"]
        win.scores = full_pipeline_result["scores"]

        out_path = str(tmp_path / "ui_export.json")
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_path, ""))

        win._on_export_json(full_pipeline_result["chosen"])

        assert (tmp_path / "ui_export.json").exists()
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["chosen_model"] == full_pipeline_result["chosen"].value
    finally:
        win.close()


def test_ui_export_csv_handler_writes_file(qapp, full_pipeline_result, camera_config, pattern_config, tmp_path):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        win.dataset = full_pipeline_result["dataset"]
        win.camera_config = camera_config
        win.pattern_config = pattern_config
        win.calibration_results = full_pipeline_result["calibration_results"]

        out_path = str(tmp_path / "ui_export.csv")
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_path, ""))

        win._on_export_csv(full_pipeline_result["chosen"])

        assert (tmp_path / "ui_export.csv").exists()
        content = (tmp_path / "ui_export.csv").read_text(encoding="utf-8")
        assert "image_id" in content.splitlines()[0]
    finally:
        win.close()


def test_ui_export_csv_works_even_if_selected_model_failed(qapp, full_pipeline_result, camera_config, pattern_config, tmp_path):
    """CSV는 데이터셋 통계라 특정 모델이 실패해도 export 가능해야 한다."""
    from ui.main_window import MainWindow
    from calibration.types import CalibrationResult

    win = MainWindow()
    try:
        win.dataset = full_pipeline_result["dataset"]
        win.camera_config = camera_config
        win.pattern_config = pattern_config
        # 일부러 실패한 모델만 넣어봄
        win.calibration_results = {
            CameraModelType.FISHEYE: CalibrationResult(model_name=CameraModelType.FISHEYE, success=False)
        }

        out_path = str(tmp_path / "ui_export_fail.csv")
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_path, ""))

        win._on_export_csv(CameraModelType.FISHEYE)

        assert (tmp_path / "ui_export_fail.csv").exists(), "CSV는 모델 실패와 무관하게 저장돼야 함"
    finally:
        win.close()
