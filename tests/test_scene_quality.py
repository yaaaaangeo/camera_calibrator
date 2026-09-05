from __future__ import annotations

import numpy as np

from calibration.scene_quality import (
    QUALITY_WEIGHTS,
    SUBSET_WEIGHTS,
    compute_scene_quality_analysis,
    recommend_best_subset,
)
from calibration.types import (
    CalibrationProject, CalibrationResult, CameraConfig, CameraModelType, Dataset, DetectionResult,
    Frame, FrameStatus, ImageInfo, PatternConfig, PatternType,
    SceneQualityAnalysis, SceneQualityEntry, SubsetCalibrationResult,
)


def _frame(fid, center, area, tilt, corners, sharpness):
    cx, cy = center
    points = np.array([
        [cx - 10, cy - 10], [cx + 10, cy - 10],
        [cx - 10, cy + 10], [cx + 10, cy + 10],
    ], dtype=np.float32).reshape(-1, 1, 2)
    return Frame(
        image_info=ImageInfo(fid, f"/fake/{fid}.png", 1000, 800, sharpness=sharpness),
        detection=DetectionResult(
            image_id=fid, success=True, corners=points, num_corners=corners,
            board_center_px=center, board_area_ratio=area, board_tilt_deg=tilt,
        ),
        status=FrameStatus.DETECTED,
    )


def test_scene_quality_combines_rms_detection_and_relative_sharpness():
    dataset = Dataset(frames=[
        _frame("balanced", (500, 400), .25, 5, 24, 500),
        _frame("low_rms_blurry", (510, 405), .25, 5, 24, 10),
        _frame("few_corners", (200, 200), .20, 30, 5, 450),
    ])
    result = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY, success=True,
        per_frame_error={"balanced": .4, "low_rms_blurry": .2, "few_corners": .5},
    )
    pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=.04)
    analysis = compute_scene_quality_analysis(dataset, result, pattern)
    by_id = {scene.frame_id: scene for scene in analysis.scenes}

    assert sum(QUALITY_WEIGHTS.values()) == 1.0
    assert by_id["low_rms_blurry"].reprojection_score > by_id["balanced"].reprojection_score
    assert by_id["balanced"].sharpness_score > by_id["low_rms_blurry"].sharpness_score
    assert by_id["balanced"].detection_score > by_id["few_corners"].detection_score
    assert [scene.rank for scene in analysis.scenes] == [1, 2, 3]


def test_best_subset_rewards_pose_and_coverage_not_only_top_ranking():
    frames = [
        _frame("center_a", (500, 400), .25, 0, 24, 500),
        _frame("center_b", (505, 405), .25, 1, 24, 490),
        _frame("left_tilt", (100, 120), .10, 45, 22, 420),
        _frame("right_close", (900, 680), .50, -40, 22, 410),
    ]
    dataset = Dataset(frames=frames)
    result = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY, success=True,
        per_frame_error={"center_a": .2, "center_b": .21, "left_tilt": .35, "right_close": .36},
    )
    pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=.04)
    analysis = compute_scene_quality_analysis(dataset, result, pattern)
    selected = recommend_best_subset(dataset, analysis, CameraConfig(1000, 800), 3)

    assert sum(SUBSET_WEIGHTS.values()) == 1.0
    assert len(selected) == 3
    assert "center_a" in selected
    assert {"left_tilt", "right_close"} & set(selected)
    assert selected != [scene.frame_id for scene in analysis.scenes[:3]]


def test_scene_analysis_and_subset_result_round_trip(tmp_path):
    from calibration.project_io import load_project, save_project

    pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=.04)
    subset_calibration = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY, success=True, rms_error=.42,
    )
    project = CalibrationProject(
        project_name="scene-ranking",
        camera_config=CameraConfig(1000, 800),
        pattern_config=pattern,
        scene_quality_analysis=SceneQualityAnalysis(
            model_name=CameraModelType.BROWN_CONRADY,
            scenes=[SceneQualityEntry("scene-1", rank=1, quality_score=91.2)],
        ),
        subset_calibration_result=SubsetCalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            selected_frame_ids=["scene-1"],
            calibration_result=subset_calibration,
            coverage_percentage=75.0,
            original_coverage_percentage=93.75,
            warnings=["coverage warning"],
        ),
    )
    path = tmp_path / "scene-ranking.ccproj"
    save_project(project, str(path))
    loaded, _missing = load_project(str(path))

    assert loaded.scene_quality_analysis.scenes[0].quality_score == 91.2
    assert loaded.subset_calibration_result.selected_frame_ids == ["scene-1"]
    assert loaded.subset_calibration_result.calibration_result.rms_error == .42
    assert loaded.subset_calibration_result.original_coverage_percentage == 93.75


def test_subset_opencv_export_records_subset_metadata(tmp_path):
    import cv2
    from export.opencv import export_opencv_yaml

    result = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion=np.zeros((5, 1), dtype=np.float64),
        rms_error=0.42,
        success=True,
    )
    pattern = PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=.04,
    )
    path = tmp_path / "subset.yaml"
    export_opencv_yaml(
        result, CameraConfig(1000, 800), pattern, str(path),
        calibration_source="best_subset",
        selected_frame_ids=["scene-1", "scene-2", "scene-3"],
    )

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert storage.getNode("calibration_source").string() == "best_subset"
        assert int(storage.getNode("subset_scene_count").real()) == 3
        assert storage.getNode("subset_scene_ids").string() == "scene-1,scene-2,scene-3"
    finally:
        storage.release()
