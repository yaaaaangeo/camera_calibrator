"""
tests/test_pipeline_integration.py
=======================================

가장 중요한 테스트 파일. "각 모듈이 따로 동작하는 것"과 "전체 파이프라인이
실제로 이어붙었을 때 안 깨지는 것"은 다른 문제다 - 이 파일은 후자를 검증한다.

detect_dataset -> quality -> frame_quality -> Standard 4모델 계산 -> hold-out
validation -> 추천 -> outlier 제거 -> FinalResult -> OpenCV/ROS/HTML export까지
실제 합성 데이터(conftest.py의 synthetic_distorted_dataset_dir)로 전부 이어서
돌린다.

@pytest.mark.slow: ChArUco 렌더링 + Standard 4모델 계산이 들어가 있어 다른 테스트보다
느리다 (전체 스위트 1초 미만 vs 이 파일 혼자 수 초). CI에서 빠른 피드백이
필요하면 `pytest -m "not slow"`로 건너뛸 수 있다.
"""

from __future__ import annotations

import pytest

from calibration.compare import run_all_models
from calibration.frame_quality import compute_frame_quality_scores
from calibration.models.common import infer_image_size
from calibration.outlier import recalibrate_with_outlier_pruning
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.recommender import compute_final_result, compute_model_scores
from calibration.validation import validate_all_models
from export.opencv import export_opencv_yaml
from export.report import export_html_report
from export.ros import export_ros_camera_info

pytestmark = pytest.mark.slow


def test_detection_succeeds_on_synthetic_dataset(synthetic_dataset):
    """가장 기초적인 스모크 테스트: 합성 이미지에서 ChArUco가 실제로 검출되는가.
    이게 실패하면 아래 모든 통합 테스트가 무의미해진다.
    """
    assert synthetic_dataset.num_total == 16
    assert synthetic_dataset.num_detected >= 10, (
        f"검출 성공률이 너무 낮음: {synthetic_dataset.num_detected}/{synthetic_dataset.num_total}"
    )


def test_full_pipeline_end_to_end(synthetic_dataset, camera_config, pattern_config, tmp_path):
    """설계 문서 15번 파이프라인 전체를 실제로 이어서 돈다:
    Detection -> Quality Gate -> Standard 4모델 -> Model Evaluation -> Outlier -> Validation
    -> Final Result -> Export(OpenCV/ROS/HTML) 까지.
    """
    dataset = synthetic_dataset

    # 1. Dataset Quality Gate
    warnings = analyze_dataset_quality(dataset, camera_config)
    assert isinstance(warnings, list)
    assert dataset.coverage_grid, "coverage_grid가 계산되지 않음"

    # 2. Frame Quality Score (1단계, 재투영 오차 반영 전)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)
    scored = [f for f in dataset.frames if f.quality is not None]
    assert len(scored) == dataset.num_detected

    # 3. Calibration Engine - Standard 4모델 동시 계산
    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    assert len(calibration_results) == 4
    assert any(r.success for r in results), "Standard 4모델이 전부 실패함 - 합성 데이터 문제 가능성"

    for m, r in calibration_results.items():
        if r.success:
            assert r.camera_matrix is not None
            assert r.rms_error is not None and r.rms_error > 0
            assert r.radial_profile is not None and len(r.radial_profile.bins) > 0, (
                f"{m.value}: radial_profile이 비어있음"
            )

    # Frame Quality Score 2단계 (재투영 오차 반영)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    # 4. Model Evaluation - Hold-out Validation (+ Straightness)
    validation_results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.25)
    assert len(validation_results) == 4
    for m, v in validation_results.items():
        if v.success and v.test_frame_ids:
            assert v.test_rms is not None

    # 5. Model Score 기반 추천 ("RMS 최소=정답" 금지 원칙 검증 - recommender.py 내부에서
    #    이미 검증되지만, 여기서는 "추천이 하나만 나오는지"를 확인)
    scores = compute_model_scores(calibration_results, validation_results)
    recommended = [s for s in scores if s.is_recommended]
    assert len(recommended) == 1, "추천 모델은 정확히 하나여야 함"
    chosen_model = recommended[0].model_name

    # 6. Outlier Analysis -> Re-calibration
    ref_result, outlier_result = recalibrate_with_outlier_pruning(dataset, camera_config, chosen_model)
    assert ref_result.success or ref_result.error_message is not None
    assert outlier_result.iterations <= 3, "max_iterations=3 제한이 지켜지지 않음"

    # 7. Final Result 조립
    coverage_pct = coverage_percentage(dataset.coverage_grid)
    final_result = compute_final_result(
        chosen_model, calibration_results, validation_results,
        dataset_coverage_pct=coverage_pct, outlier_result=outlier_result, scores=scores,
    )
    assert final_result.chosen_model == chosen_model
    assert final_result.overall_grade is not None

    # 8. Export - OpenCV / ROS / HTML 리포트, 전부 예외 없이 파일이 만들어져야 함
    if calibration_results[chosen_model].success:
        opencv_path = export_opencv_yaml(
            calibration_results[chosen_model], camera_config, pattern_config,
            str(tmp_path / "camera.yaml"),
        )
        ros_path = export_ros_camera_info(
            calibration_results[chosen_model], camera_config, str(tmp_path / "camera_info.yaml")
        )
        html_path = export_html_report(
            "pytest-e2e", camera_config, pattern_config, dataset,
            calibration_results, validation_results, final_result, str(tmp_path / "report.html"),
        )

        import os
        for p in (opencv_path, ros_path, html_path):
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0


def test_recommended_model_beats_naive_lowest_train_rms(synthetic_dataset, camera_config, pattern_config):
    """설계 문서 8번의 핵심 경고 - "Train RMS가 가장 낮은 모델 = 무조건 정답"이
    아니라는 걸 실제로 확인한다. 이 테스트는 추천 시스템이 Test RMS/Edge RMS도
    반영해서 "단순 Train RMS 최소" 전략과 다른 결정을 내릴 수 있는 구조인지를
    간접적으로 검증한다 (매 실행마다 다른 모델이 이길 필요는 없지만, 최소한
    scoring이 train RMS만으로 계산되지 않는다는 걸 확인).
    """
    results = run_all_models(synthetic_dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    validation_results = validate_all_models(synthetic_dataset, camera_config, pattern_config, test_ratio=0.25)
    scores = compute_model_scores(calibration_results, validation_results)

    # Score가 전부 rms_error만으로 계산됐다면 순위가 train_rms 순위와 100% 같을 텐데,
    # 실제로는 test/edge/complexity가 섞여있어야 한다 - 최소한 스코어 자체가
    # train RMS와 다른 값이어야 한다 (완전히 같은 숫자면 다른 요소를 안 쓴다는 뜻).
    for s in scores:
        cal = calibration_results.get(s.model_name)
        if cal and cal.success:
            assert s.score != cal.rms_error, (
                "Model Score가 Train RMS와 완전히 같음 - Test/Edge/Complexity가 "
                "반영 안 되고 있을 가능성"
            )
