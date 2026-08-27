"""
Workflow-facing stereo calibration service.

This keeps UI/worker code from depending on every low-level stereo helper
directly. The controller is intentionally thin: heavy math still lives in
calibration.stereo and image detection still lives in calibration.detector.
"""

from __future__ import annotations

from calibration.calibration_io import StandardCalibration
from calibration.detector import detect_dataset
from calibration.stereo import (
    StereoCalibrationResult,
    StereoPairObservation,
    build_stereo_pairs_from_datasets,
    calibrate_stereo,
)
from calibration.types import Dataset, PatternConfig


class StereoController:
    def detect_pairs(
        self,
        camera1_paths: list[str],
        camera2_paths: list[str],
        pattern_config: PatternConfig,
    ) -> tuple[list[StereoPairObservation], Dataset, Dataset]:
        ds1 = detect_dataset(camera1_paths, pattern_config, parallel=len(camera1_paths) > 8)
        ds2 = detect_dataset(camera2_paths, pattern_config, parallel=len(camera2_paths) > 8)
        return build_stereo_pairs_from_datasets(ds1, ds2), ds1, ds2

    def calibrate(
        self,
        pairs: list[StereoPairObservation],
        camera1: StandardCalibration,
        camera2: StandardCalibration,
        image_size: tuple[int, int],
        *,
        audit_mode: str = "full",
    ) -> StereoCalibrationResult:
        return calibrate_stereo(pairs, camera1, camera2, image_size, audit_mode=audit_mode)
