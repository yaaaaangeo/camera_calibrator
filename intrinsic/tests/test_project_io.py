"""
tests/test_project_io.py
=============================

설계 문서 16번 - .ccproj 프로젝트 저장/불러오기. 실제 파이프라인(검출→Standard
4모델→검증→이상치 제거→FinalResult)을 전부 돌린 뒤 저장하고 다시 불러와서,
numpy 배열 dtype까지 포함해 모든 값이 정확히 왕복되는지 확인한다.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from calibration.compare import run_all_models
from calibration.frame_quality import compute_frame_quality_scores
from calibration.models.common import infer_image_size
from calibration.outlier import recalibrate_with_outlier_pruning
from calibration.project_io import (
    PROJECT_FORMAT_VERSION,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.recommender import compute_final_result, compute_model_scores
from calibration.types import CalibrationMethod, CalibrationProject, CalibrationResult, CameraModelType
from calibration.validation import validate_all_models

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_project(synthetic_dataset, camera_config, pattern_config):
    """검출~FinalResult까지 실제로 다 돌린 CalibrationProject 하나.

    module 스코프로 이 파일의 모든 테스트가 하나의 계산 결과를 공유한다
    (예전엔 함수 스코프라 테스트 7개가 매번 Standard 4모델+검증+이상치제거를 새로
    돌려서 setup에서만 테스트당 ~2.2초씩 낭비되고 있었다).

    synthetic_dataset은 세션 스코프라 다른 테스트 파일과도 공유되는데,
    recalibrate_with_outlier_pruning()은 프레임 상태를 그 자리에서 바꾼다
    (이상치로 판정되면 DISABLED_OUTLIER) - 원본 세션 fixture를 건드리면
    이 파일 이후에 도는 다른 테스트 파일이 "이상치가 이미 제거된" 데이터셋을
    받는 의도치 않은 상태 오염이 생길 수 있다. deepcopy로 이 파일 전용
    사본을 만들어 계산해서 그 위험을 없앤다.
    """
    dataset = copy.deepcopy(synthetic_dataset)
    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    validation_results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.25)
    scores = compute_model_scores(calibration_results, validation_results)
    recommended = next((s.model_name for s in scores if s.is_recommended), list(calibration_results)[0])

    ref_result, outlier_result = recalibrate_with_outlier_pruning(dataset, camera_config, recommended)
    calibration_results[recommended] = ref_result

    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None
    final_result = compute_final_result(
        recommended, calibration_results, validation_results,
        dataset_coverage_pct=coverage_pct, outlier_result=outlier_result, scores=scores,
    )

    return CalibrationProject(
        project_name="pytest 프로젝트",
        camera_config=camera_config,
        pattern_config=pattern_config,
        dataset=dataset,
        calibration_results=calibration_results,
        validation_results=validation_results,
        model_scores=scores,
        outlier_result=outlier_result,
        final_result=final_result,
    )


def test_save_creates_valid_json_file(full_project, tmp_path):
    path = str(tmp_path / "test.ccproj")
    result_path = save_project(full_project, path)
    assert result_path == path

    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["format_version"] == PROJECT_FORMAT_VERSION
    assert "project" in payload


def test_round_trip_preserves_basic_fields(full_project, tmp_path):
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)
    loaded, missing = load_project(path)

    assert not missing, "테스트 이미지가 그대로 있으니 누락이 없어야 함"
    assert loaded.project_name == full_project.project_name
    assert loaded.camera_config.width == full_project.camera_config.width
    assert loaded.camera_config.height == full_project.camera_config.height
    assert loaded.pattern_config.squares_x == full_project.pattern_config.squares_x
    assert loaded.pattern_config.dictionary == full_project.pattern_config.dictionary
    assert loaded.dataset.num_total == full_project.dataset.num_total
    assert loaded.dataset.num_detected == full_project.dataset.num_detected


def test_round_trip_preserves_object_releasing_result_separately(
    camera_config, pattern_config, tmp_path
):
    project = CalibrationProject(
        project_name="object releasing project",
        camera_config=camera_config,
        pattern_config=pattern_config,
        object_releasing_result=CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            calibration_method=CalibrationMethod.OBJECT_RELEASING,
            rms_error=0.123,
            success=True,
        ),
    )

    path = str(tmp_path / "object_releasing.ccproj")
    save_project(project, path)
    loaded, _ = load_project(path)

    assert loaded.calibration_results == {}
    assert loaded.object_releasing_result is not None
    assert loaded.object_releasing_result.model_name == CameraModelType.BROWN_CONRADY
    assert loaded.object_releasing_result.calibration_method == CalibrationMethod.OBJECT_RELEASING
    assert loaded.object_releasing_result.rms_error == 0.123


def test_round_trip_preserves_numpy_arrays_exactly(full_project, tmp_path):
    """numpy 배열은 JSON을 한 바퀴 돌면서 dtype이 바뀌기 쉽다 (특히
    float32<->float64) - 이게 틀리면 이후 계산 결과가 미묘하게 달라질 수 있어
    엄격하게 확인한다.
    """
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)
    loaded, _ = load_project(path)

    orig_frame = next(f for f in full_project.dataset.frames if f.detection and f.detection.success)
    loaded_frame = next(
        f for f in loaded.dataset.frames if f.image_info.image_id == orig_frame.image_info.image_id
    )
    assert np.allclose(loaded_frame.detection.corners, orig_frame.detection.corners, atol=1e-4)
    assert loaded_frame.detection.corners.dtype == np.float32
    assert np.array_equal(loaded_frame.detection.ids, orig_frame.detection.ids)
    assert loaded_frame.detection.ids.dtype == np.int32

    for model, orig_result in full_project.calibration_results.items():
        if not orig_result.success:
            continue
        loaded_result = loaded.calibration_results[model]
        assert np.allclose(loaded_result.camera_matrix, orig_result.camera_matrix)
        assert loaded_result.camera_matrix.dtype == np.float64
        assert np.allclose(loaded_result.distortion, orig_result.distortion)
        assert len(loaded_result.rvecs) == len(orig_result.rvecs)
        if orig_result.rvecs:
            assert np.allclose(loaded_result.rvecs[0], orig_result.rvecs[0])


def test_round_trip_preserves_validation_and_final_result(full_project, tmp_path):
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)
    loaded, _ = load_project(path)

    for model, orig_val in full_project.validation_results.items():
        loaded_val = loaded.validation_results[model]
        assert loaded_val.test_rms == orig_val.test_rms
        assert loaded_val.straightness_residual == orig_val.straightness_residual

    assert loaded.final_result.chosen_model == full_project.final_result.chosen_model
    assert loaded.final_result.overall_grade == full_project.final_result.overall_grade
    assert len(loaded.final_result.model_scores) == len(full_project.final_result.model_scores)

    assert loaded.outlier_result.threshold_used == full_project.outlier_result.threshold_used
    assert loaded.outlier_result.removed_frame_ids == full_project.outlier_result.removed_frame_ids


def test_load_detects_missing_images_without_crashing(full_project, tmp_path):
    """이미지가 옮겨지거나 지워졌어도 load_project는 예외를 던지지 않고
    missing 리스트로 알려줘야 한다 (설계 문서 9번과 같은 원칙).
    """
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)

    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    for frame in payload["project"]["dataset"]["frames"]:
        frame["image_info"]["path"] = "/nonexistent/" + frame["image_info"]["path"].split("/")[-1]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, missing = load_project(path)
    assert len(missing) == loaded.dataset.num_total
    assert loaded.dataset.num_total == full_project.dataset.num_total


def test_unsupported_format_version_raises_clear_error(full_project, tmp_path):
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)

    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["format_version"] = 999
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="지원하지 않는"):
        load_project(path)


def test_project_to_dict_and_from_dict_are_inverse(full_project):
    """파일 I/O 없이 순수 변환 함수 왕복만 확인 (더 빠른 단위 테스트)."""
    d = project_to_dict(full_project)
    restored = project_from_dict(d)
    assert restored.project_name == full_project.project_name
    assert restored.dataset.num_total == full_project.dataset.num_total
    assert set(restored.calibration_results.keys()) == set(full_project.calibration_results.keys())


def test_frame_quality_and_status_round_trip(full_project, tmp_path):
    path = str(tmp_path / "test.ccproj")
    save_project(full_project, path)
    loaded, _ = load_project(path)

    for orig_frame, loaded_frame in zip(full_project.dataset.frames, loaded.dataset.frames):
        assert orig_frame.status == loaded_frame.status
        if orig_frame.quality:
            assert loaded_frame.quality is not None
            assert abs(orig_frame.quality.overall_score - loaded_frame.quality.overall_score) < 1e-6
            assert orig_frame.quality.grade == loaded_frame.quality.grade
