from __future__ import annotations

import cv2
import numpy as np

from calibration.compare import run_all_models
from calibration.detector import clear_image_preprocess_cache, detect_dataset, detect_image_file
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)


def _tiny_dataset() -> Dataset:
    info = ImageInfo("img0", "/fake/img0.jpg", 640, 480)
    det = DetectionResult(
        "img0",
        success=True,
        corners=np.zeros((4, 1, 2), dtype=np.float32),
        object_points=np.zeros((4, 1, 3), dtype=np.float32),
        num_corners=4,
    )
    return Dataset(frames=[Frame(info, det, status=FrameStatus.DETECTED)])


def _result(model: CameraModelType) -> CalibrationResult:
    return CalibrationResult(
        model_name=model,
        success=True,
        rms_error=0.1,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
    )


def test_parallel_model_calibration_path_uses_pinhole_and_extended_workers(monkeypatch):
    calls: list[str] = []

    def pinhole(dataset, camera):
        calls.append("pinhole")
        return _result(CameraModelType.PINHOLE)

    def extended(dataset, camera, use_rational_model=False):
        calls.append(f"extended:{use_rational_model}")
        return _result(CameraModelType.EXTENDED_PINHOLE)

    def fisheye(dataset, camera, initial_guess=None, **kwargs):
        calls.append(f"fisheye:{initial_guess.model_name.value}")
        return _result(CameraModelType.FISHEYE)

    monkeypatch.setattr("calibration.compare.calibrate_pinhole", pinhole)
    monkeypatch.setattr("calibration.compare.calibrate_extended_pinhole", extended)
    monkeypatch.setattr("calibration.compare.calibrate_fisheye", fisheye)
    monkeypatch.setattr("calibration.compare.attach_observability_report", lambda *a, **k: None)
    monkeypatch.setattr("calibration.compare.attach_undistortion_quality_report", lambda *a, **k: None)

    results = run_all_models(
        _tiny_dataset(),
        CameraConfig(width=640, height=480),
        use_rational_model=True,
        estimate_fisheye_uncertainty=False,
        model_jobs=2,
        # 이 테스트는 pinhole/extended/fisheye 워커 병렬 실행 경로만 검증한다 -
        # Brown-Conrady까지 포함하면 monkeypatch 안 된 실제 calibrate_brown_conrady가
        # 가짜 데이터로 불려서 무관한 실패를 낼 수 있으므로 모델을 명시적으로 제한한다.
        models=[CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE],
    )

    assert [r.model_name for r in results] == [
        CameraModelType.PINHOLE,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    ]
    assert sorted(calls[:2]) == ["extended:True", "pinhole"]
    assert calls[-1] == "fisheye:pinhole"


def test_persistent_model_result_cache_reuses_previous_run(monkeypatch, tmp_path):
    counts = {"pinhole": 0, "extended": 0, "fisheye": 0}

    def pinhole(dataset, camera):
        counts["pinhole"] += 1
        return _result(CameraModelType.PINHOLE)

    def extended(dataset, camera, use_rational_model=False):
        counts["extended"] += 1
        return _result(CameraModelType.EXTENDED_PINHOLE)

    def fisheye(dataset, camera, initial_guess=None, **kwargs):
        counts["fisheye"] += 1
        return _result(CameraModelType.FISHEYE)

    monkeypatch.setattr("calibration.compare.calibrate_pinhole", pinhole)
    monkeypatch.setattr("calibration.compare.calibrate_extended_pinhole", extended)
    monkeypatch.setattr("calibration.compare.calibrate_fisheye", fisheye)
    monkeypatch.setattr("calibration.compare.attach_observability_report", lambda *a, **k: None)
    monkeypatch.setattr("calibration.compare.attach_undistortion_quality_report", lambda *a, **k: None)

    kwargs = dict(
        dataset=_tiny_dataset(),
        camera_config=CameraConfig(width=640, height=480),
        estimate_fisheye_uncertainty=False,
        persistent_cache_dir=tmp_path / "cache",
    )
    first = run_all_models(**kwargs)
    second = run_all_models(**kwargs)

    assert [r.model_name for r in first] == [r.model_name for r in second]
    assert counts == {"pinhole": 1, "extended": 1, "fisheye": 1}


def test_image_preprocessing_cache_reuses_file_quality_metadata(monkeypatch, tmp_path):
    clear_image_preprocess_cache()
    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), np.full((32, 48, 3), 127, dtype=np.uint8))
    calls = {"phash": 0}

    def fake_phash(gray):
        calls["phash"] += 1
        return "cached-hash"

    def fake_detect(img, image_id):
        return DetectionResult(image_id, success=False, num_corners=0)

    monkeypatch.setattr("calibration.detector.compute_phash", fake_phash)

    info1, _ = detect_image_file(str(image_path), fake_detect)
    info2, _ = detect_image_file(str(image_path), fake_detect)

    assert calls["phash"] == 1
    assert info1.phash == "cached-hash"
    assert info2.phash == "cached-hash"
    assert info1.sharpness == info2.sharpness


def test_parallel_detection_caps_default_worker_count_for_large_inputs(monkeypatch):
    observed = []
    monkeypatch.setattr("calibration.detector.os.cpu_count", lambda: 32)
    monkeypatch.setattr(
        "calibration.detector._detect_dataset_parallel",
        lambda paths, pattern, workers: observed.append(workers) or [],
    )
    pattern = PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=7,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )

    detect_dataset([f"image_{i}.jpg" for i in range(100)], pattern, parallel=True)

    assert observed == [4]
