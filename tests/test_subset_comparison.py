import csv
import json

import numpy as np
import pytest

import calibration.subset_comparison as subject
from calibration.calibration_io import StandardCalibration
from calibration.subset_comparison import (
    ComparisonTolerance, DataLeakageError, SplitManifest,
    classify_metric, compare_subset_calibrations, validate_split_manifest,
    write_subset_comparison_outputs,
)
from calibration.types import (
    CameraConfig, CameraModelType, Dataset, DetectionResult, Frame, FrameStatus,
    ImageInfo, PatternConfig, PatternType,
)


def _cal(label="model"):
    return StandardCalibration(
        label=label, camera_matrix=np.eye(3), distortion=np.zeros((4, 1)),
        model_name=CameraModelType.FISHEYE, width=640, height=480,
        metadata={"rms_reprojection_error": 0.5},
    )


def _manifest():
    return SplitManifest(["a", "b"], ["a"], ["h1", "h2"], seed=42)


def _pattern():
    return PatternConfig(PatternType.CHARUCO, 7, 5, .04, .03, "DICT_5X5_100")


def _evaluation(values, total=2):
    rows = {
        frame_id: {"rms": value, "median": value, "p95": value, "p99": value,
                   "max": value, "edge_rms": value, "straightness": value}
        for frame_id, value in values.items()
    }
    return {
        "metrics": {}, "per_frame": rows, "failures": [], "total_frames": total,
        "evaluated_frames": len(rows), "failed_detection_frames": total - len(rows),
        "failed_pose_estimation_frames": 0, "success_rate": len(rows) / total,
        "train_rms": .5,
    }


def test_metric_statuses_and_identical_delta():
    tolerance = ComparisonTolerance()
    assert classify_metric(1.0, 1.0, tolerance)["delta"] == 0
    assert classify_metric(1.0, .8, tolerance)["status"] == "Improved"
    assert classify_metric(1.0, 1.09, tolerance)["status"] == "Maintained"
    assert classify_metric(1.0, 1.11, tolerance)["status"] == "Regressed"


def test_overlap_fails_before_evaluation():
    manifest = SplitManifest(["train", "hold"], ["train"], ["hold"])
    with pytest.raises(DataLeakageError):
        validate_split_manifest(manifest)


def test_pairwise_uses_only_common_success_frames(monkeypatch):
    results = iter([_evaluation({"h1": 1.0, "h2": 100.0}), _evaluation({"h1": .5})])
    monkeypatch.setattr(subject, "_evaluate_model", lambda *args, **kwargs: next(results))
    result = compare_subset_calibrations(
        Dataset(), _cal("baseline"), _cal("candidate"), _manifest(),
        CameraConfig(640, 480), _pattern(), ComparisonTolerance(minimum_common_frames=1),
    )
    assert result["paired"]["common_frame_count"] == 1
    assert result["comparison"]["holdout_rms"]["baseline"] == 1.0
    assert result["comparison"]["holdout_rms"]["candidate"] == .5


def test_json_is_strict_and_csv_is_rounded(tmp_path):
    result = {
        "verdict": "PASS", "verdict_reasons": ["ok"],
        "provenance": {"counts": {"baseline_training": 2, "subset_training": 1, "holdout": 1}, "split_seed": 1, "baseline_model": {"path": None, "sha256": None}, "candidate_model": {"path": None, "sha256": None}},
        "comparison": {"holdout_rms": {"baseline": 1.23456, "candidate": float("nan"), "delta": None, "relative_change_pct": None, "tolerance": .1, "status": "Unavailable", "winner": "N/A"}},
        "baseline": {"evaluated_frames": 0, "total_frames": 1, "success_rate": 0, "failures": []},
        "candidate": {"evaluated_frames": 0, "total_frames": 1, "success_rate": 0, "failures": []},
        "paired": {"common_frame_count": 0, "evidence": "insufficient evidence"},
        "per_frame": [],
    }
    paths = write_subset_comparison_outputs(result, tmp_path)
    loaded = json.loads((tmp_path / "subset_comparison_summary.json").read_text())
    assert loaded["comparison"]["holdout_rms"]["candidate"] is None
    with open(paths["summary_csv"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["baseline"] == "1.235"


def test_fisheye_projection_path_and_detection_failure_are_recorded(monkeypatch):
    detected = Frame(
        ImageInfo("h1", "h1.png", 640, 480),
        DetectionResult(
            "h1", True,
            corners=np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], np.float32),
            object_points=np.array([[[0, 0, 0]], [[1, 0, 0]], [[1, 1, 0]], [[0, 1, 0]]], np.float32),
            ids=np.arange(4).reshape(-1, 1), num_corners=4,
        ),
        status=FrameStatus.DETECTED,
    )
    failed = Frame(
        ImageInfo("h2", "h2.png", 640, 480),
        DetectionResult("h2", False, failure_reason="no corners"),
        status=FrameStatus.DETECTION_FAILED,
    )
    seen = []
    monkeypatch.setattr(subject, "solve_pnp_for_model_robust", lambda *args: (True, np.zeros((3, 1)), np.ones((3, 1)), "fisheye"))

    def project(obj, rvec, tvec, camera_matrix, distortion, model):
        seen.append(model)
        return detected.detection.corners.reshape(-1, 2).astype(float)

    monkeypatch.setattr(subject, "project_points_for_model", project)
    result = subject._evaluate_model(
        Dataset(frames=[detected, failed]), ["h1", "h2"], _cal(),
        CameraConfig(640, 480), _pattern(),
    )
    assert seen == [CameraModelType.FISHEYE]
    assert result["failed_detection_frames"] == 1
    assert result["failures"][0]["reason"] == "no corners"
