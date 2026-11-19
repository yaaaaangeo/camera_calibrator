from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.cli import _bootstrap_flags_for_model
from calibration.models.common import MIN_FRAMES_REQUIRED
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    PatternType,
    ValidationResult,
)


_DISTORTION_SIZE = {
    CameraModelType.PINHOLE: 5,
    CameraModelType.BROWN_CONRADY: 5,
    CameraModelType.EXTENDED_PINHOLE: 8,
    CameraModelType.FISHEYE: 4,
}


def _camera_config() -> CameraConfig:
    return CameraConfig(width=640, height=480)


def _pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=7,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )


def _result(model: CameraModelType) -> CalibrationResult:
    return CalibrationResult(
        model_name=model,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion=np.zeros((_DISTORTION_SIZE[model], 1), dtype=np.float64),
        rms_error=0.1,
        per_frame_error={},
        success=True,
    )


def _validation_result() -> ValidationResult:
    return ValidationResult(
        train_frame_ids=[],
        test_frame_ids=[],
        train_rms=0.1,
        test_rms=0.1,
        success=True,
    )


def _spy(monkeypatch, module, calls: list[str], model: CameraModelType) -> None:
    mapping = {
        "calibrate_pinhole": CameraModelType.PINHOLE,
        "calibrate_brown_conrady": CameraModelType.BROWN_CONRADY,
        "calibrate_extended_pinhole": CameraModelType.EXTENDED_PINHOLE,
        "calibrate_fisheye": CameraModelType.FISHEYE,
    }

    for name, returned_model in mapping.items():
        def fake(dataset, camera_config, *args, _name=name, _model=returned_model, **kwargs):
            calls.append(_name)
            return _result(_model)

        monkeypatch.setattr(module, name, fake)


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_initial_calibration_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.compare as compare

    calls: list[str] = []
    _spy(monkeypatch, compare, calls, model)

    results = compare.run_all_models(Dataset(frames=[]), _camera_config(), models=[model])

    assert calls == [expected]
    assert forbidden not in calls
    assert results[0].model_name == model
    assert results[0].distortion.size == _DISTORTION_SIZE[model]


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_frame_outlier_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.outlier as outlier

    calls: list[str] = []
    _spy(monkeypatch, outlier, calls, model)

    result, _ = outlier.recalibrate_with_outlier_pruning(
        Dataset(frames=[]),
        _camera_config(),
        model,
        max_iterations=1,
    )

    assert calls == [expected]
    assert forbidden not in calls
    assert result.model_name == model


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_corner_outlier_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.outlier as outlier

    calls: list[str] = []
    _spy(monkeypatch, outlier, calls, model)

    result, _ = outlier.recalibrate_with_corner_outlier_pruning(
        Dataset(frames=[]),
        _camera_config(),
        model,
        max_iterations=1,
    )

    assert calls == [expected]
    assert forbidden not in calls
    assert result.model_name == model


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_holdout_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.validation as validation

    calls: list[str] = []
    _spy(monkeypatch, validation, calls, model)
    monkeypatch.setattr(validation, "_evaluate_on_test", lambda *args, **kwargs: _validation_result())

    result = validation.validate_holdout(
        Dataset(frames=[]),
        _camera_config(),
        _pattern_config(),
        model,
        [f"train_{i}" for i in range(MIN_FRAMES_REQUIRED)],
        ["test_0"],
    )

    assert calls == [expected]
    assert forbidden not in calls
    assert result.success


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_kfold_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.kfold as kfold
    import calibration.validation as validation

    calls: list[str] = []
    _spy(monkeypatch, validation, calls, model)
    monkeypatch.setattr(validation, "_evaluate_on_test", lambda *args, **kwargs: _validation_result())
    monkeypatch.setattr(
        kfold,
        "split_k_folds",
        lambda dataset, camera_config, k, seed: [
            [f"fold0_{i}" for i in range(MIN_FRAMES_REQUIRED)],
            [f"fold1_{i}" for i in range(MIN_FRAMES_REQUIRED)],
        ],
    )

    result = kfold.compute_kfold_validation(
        Dataset(frames=[]),
        _camera_config(),
        _pattern_config(),
        model,
        k=2,
        n_jobs=1,
        cache=None,
    )

    assert calls == [expected, expected]
    assert forbidden not in calls
    assert result.n_successful_folds == 2


@pytest.mark.parametrize(
    ("model", "expected", "forbidden"),
    [
        (CameraModelType.BROWN_CONRADY, "calibrate_brown_conrady", "calibrate_extended_pinhole"),
        (CameraModelType.EXTENDED_PINHOLE, "calibrate_extended_pinhole", "calibrate_brown_conrady"),
    ],
)
def test_repeated_kfold_dispatch_keeps_brown_and_rational_distinct(monkeypatch, model, expected, forbidden):
    import calibration.kfold as kfold
    import calibration.validation as validation

    calls: list[str] = []
    _spy(monkeypatch, validation, calls, model)
    monkeypatch.setattr(validation, "_evaluate_on_test", lambda *args, **kwargs: _validation_result())
    monkeypatch.setattr(
        kfold,
        "split_k_folds",
        lambda dataset, camera_config, k, seed: [
            [f"fold0_{i}" for i in range(MIN_FRAMES_REQUIRED)],
            [f"fold1_{i}" for i in range(MIN_FRAMES_REQUIRED)],
        ],
    )

    result = kfold.compute_repeated_kfold(
        Dataset(frames=[]),
        _camera_config(),
        _pattern_config(),
        model,
        k=2,
        n_repeats=2,
        n_jobs=1,
        cache=None,
    )

    assert calls == [expected, expected, expected, expected]
    assert forbidden not in calls
    assert result.n_successful_runs == 4


@pytest.mark.parametrize(
    ("model", "module_name", "expected", "forbidden"),
    [
        (
            CameraModelType.BROWN_CONRADY,
            "calibration.models.brown_conrady",
            "calibrate_brown_conrady",
            "calibrate_extended_pinhole",
        ),
        (
            CameraModelType.EXTENDED_PINHOLE,
            "calibration.models.extended_pinhole",
            "calibrate_extended_pinhole",
            "calibrate_brown_conrady",
        ),
    ],
)
def test_repeatability_order_samples_dispatch_to_model_solver(monkeypatch, model, module_name, expected, forbidden):
    import importlib

    import calibration.repeatability as repeatability

    calls: list[str] = []
    target_module = importlib.import_module(module_name)

    def fake(dataset, camera_config, *args, **kwargs):
        calls.append(expected)
        return _result(model)

    monkeypatch.setattr(target_module, expected, fake)

    result = repeatability.compute_repeatability(
        Dataset(frames=[]),
        _camera_config(),
        model,
        n_runs=3,
        n_jobs=1,
        vary_initial_conditions=False,
    )

    assert calls == [expected, expected, expected]
    assert forbidden not in calls
    assert result.n_successful == 3


def test_model_score_parameter_counts_keep_brown_5_and_rational_8():
    from calibration.recommender import parameter_count_for_model

    assert parameter_count_for_model(CameraModelType.PINHOLE) == 4
    assert parameter_count_for_model(CameraModelType.BROWN_CONRADY) == 9
    assert parameter_count_for_model(CameraModelType.EXTENDED_PINHOLE) == 12
    assert parameter_count_for_model(CameraModelType.FISHEYE) == 8


def test_cli_bootstrap_flags_keep_rational_model_active():
    assert _bootstrap_flags_for_model(CameraModelType.BROWN_CONRADY) & cv2.CALIB_RATIONAL_MODEL == 0
    assert _bootstrap_flags_for_model(CameraModelType.EXTENDED_PINHOLE) & cv2.CALIB_RATIONAL_MODEL
