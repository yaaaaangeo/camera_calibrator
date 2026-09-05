"""
tests/test_report.py
=========================

설계 문서 12번 - HTML 종합 리포트. 필수 섹션이 다 들어가는지, HTML 태그가
안 깨지는지(간단한 개수 균형 체크), 실제로 파일로 저장되는지 확인한다.
"""

from __future__ import annotations

import re

from calibration.recommender import compute_final_result
from calibration.types import CalibrationResult, CameraModelType, ValidationResult
from export.report import export_html_report, generate_html_report


def _make_success_calibration_results(camera_config):
    import numpy as np
    K = np.array([[1000.0, 0, camera_config.width / 2], [0, 1000.0, camera_config.height / 2], [0, 0, 1]])
    D = np.array([-0.1, 0.02, 0.0, 0.0, 0.0])
    results = {}
    for m in (CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE):
        results[m] = CalibrationResult(
            model_name=m, camera_matrix=K, distortion=D,
            rms_error=0.4, per_frame_error={"img_00": 0.3, "img_01": 0.5}, success=True,
        )
    return results


def _make_validation_results():
    results = {}
    for m in (CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE):
        results[m] = ValidationResult(
            train_frame_ids=["img_00"], test_frame_ids=["img_01"],
            train_rms=0.4, test_rms=0.5, edge_rms=0.6, straightness_residual=0.3, success=True,
        )
    return results


def test_html_report_contains_all_sections(camera_config, pattern_config):
    from calibration.types import Dataset

    cal = _make_success_calibration_results(camera_config)
    val = _make_validation_results()
    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, cal, val)

    html = generate_html_report(
        "테스트 프로젝트", camera_config, pattern_config, Dataset(frames=[]), cal, val, final
    )

    for expected in [
        "Camera Calibration Report",
        "1. Camera",
        "2. Dataset",
        "3. Dataset Quality",
        "4. Cross Validation",
        "5. Model Comparison",
        "6. Chosen Model Detail",
        "7. Bootstrap Stability",
        "8. Sanity Check",
        "9. Outlier",
        "Final Calibration Summary",
        "Overall Quality",
        "Rational",
    ]:
        assert expected in html, f"리포트에 '{expected}' 섹션이 없음"


def test_html_tags_are_balanced(camera_config, pattern_config):
    from calibration.types import Dataset

    cal = _make_success_calibration_results(camera_config)
    val = _make_validation_results()
    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, cal, val)
    html = generate_html_report("proj", camera_config, pattern_config, Dataset(frames=[]), cal, val, final)

    for tag in ["table", "tr", "td", "th", "div"]:
        opens = len(re.findall(f"<{tag}[ >]", html))
        closes = len(re.findall(f"</{tag}>", html))
        assert opens == closes, f"<{tag}> 태그 개수가 안 맞음: open={opens} close={closes}"


def test_export_html_report_writes_file(tmp_path, camera_config, pattern_config):
    from calibration.types import Dataset

    cal = _make_success_calibration_results(camera_config)
    val = _make_validation_results()
    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, cal, val)

    out_path = str(tmp_path / "report.html")
    result_path = export_html_report(
        "proj", camera_config, pattern_config, Dataset(frames=[]), cal, val, final, out_path
    )
    assert result_path == out_path

    import os
    assert os.path.exists(out_path)
    content = open(out_path, encoding="utf-8").read()
    assert len(content) > 500


def test_report_handles_failed_calibration_gracefully(camera_config, pattern_config):
    """선택된 모델의 캘리브레이션이 실패했어도 리포트 생성 자체가 죽으면 안 된다."""
    from calibration.types import Dataset

    cal = {CameraModelType.PINHOLE: CalibrationResult(model_name=CameraModelType.PINHOLE, success=False, error_message="테스트 실패")}
    final = compute_final_result(CameraModelType.PINHOLE, cal, {})

    html = generate_html_report("proj", camera_config, pattern_config, Dataset(frames=[]), cal, {}, final)
    assert "Camera Calibration Report" in html
