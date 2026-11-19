"""OpenCV YAML export must not fail when frame-ID lists exceed the FileStorage
string limit (~4KB). Real failure: rosbag front camera, ~160 IDs like
"bag_0001_t1793943447.673" -> "The written string is too long"."""
import cv2
import numpy as np

from calibration.calibration_io import load_opencv_calibration
from calibration.types import CalibrationResult, CameraConfig, CameraModelType, PatternConfig, PatternType
from export.opencv import export_opencv_yaml, read_id_list

PATTERN = PatternConfig(type=PatternType.CHARUCO, squares_x=10, squares_y=7, square_size=.08, marker_size=.06,
                        dictionary="DICT_5X5_100")


def _result(ids):
    return CalibrationResult(model_name=CameraModelType.FISHEYE, camera_matrix=np.array([[440., 0, 480], [0, 440, 270], [0, 0, 1]]),
                             distortion=np.zeros((4, 1)), rms_error=0.3, success=True,
                             per_frame_error={i: 0.3 for i in ids})


def test_long_frame_id_lists_export_and_roundtrip(tmp_path):
    ids = [f"bag_{i:04d}_t1793943{i:03d}.673" for i in range(400)]
    assert len(",".join(ids)) > 4096
    path = tmp_path / "long.yaml"
    export_opencv_yaml(_result(ids), CameraConfig(960, 540), PATTERN, str(path), selected_frame_ids=ids)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert read_id_list(fs.getNode("fit_frame_ids")) == ids
        assert read_id_list(fs.getNode("subset_scene_ids")) == ids
        assert int(fs.getNode("subset_scene_count").real()) == len(ids)
    finally:
        fs.release()
    loaded = load_opencv_calibration(str(path))
    assert loaded.metadata["subset_scene_ids"] == ids


def test_short_frame_id_lists_keep_comma_string_format(tmp_path):
    ids = ["scene-1", "scene-2", "scene-3"]
    path = tmp_path / "short.yaml"
    export_opencv_yaml(_result(ids), CameraConfig(960, 540), PATTERN, str(path), selected_frame_ids=ids)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert fs.getNode("fit_frame_ids").string() == "scene-1,scene-2,scene-3"
        assert fs.getNode("subset_scene_ids").string() == "scene-1,scene-2,scene-3"
        assert read_id_list(fs.getNode("fit_frame_ids")) == ids
    finally:
        fs.release()
