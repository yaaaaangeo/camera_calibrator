"""A model can reject frames without mutating the shared input dataset."""
import copy
import cv2
import numpy as np
from calibration.observability import residual_vector, compute_observability_report
from calibration.types import CameraModelType
from tests.test_observability import _dataset_and_result


def subset_fisheye_result():
    data, result = _dataset_and_result()
    result.model_name = CameraModelType.FISHEYE
    result.distortion = np.zeros((4, 1))
    for frame, rv, tv in zip(data.frames, result.rvecs, result.tvecs):
        frame.detection.corners = cv2.fisheye.projectPoints(
            frame.detection.object_points.astype(np.float64), rv, tv,
            result.camera_matrix, result.distortion)[0].astype(np.float32)
    result.rvecs = result.rvecs[1:]
    result.tvecs = result.tvecs[1:]
    result.per_frame_error = {f.image_info.image_id: 0. for f in data.frames[1:]}
    return data, result


def test_observability_uses_model_retained_frames_for_pose_alignment():
    data, result = subset_fisheye_result()
    before = copy.deepcopy(data)
    residual = residual_vector(result, data)
    assert residual.size == 80  # 2 retained views * 20 corners * xy
    assert np.max(abs(residual)) < 1e-3
    assert len(data.frames) == 3
    for a, b in zip(data.frames, before.frames):
        np.testing.assert_array_equal(a.detection.corners, b.detection.corners)


def test_observability_does_not_guess_unknown_subset_pose_alignment():
    data, result = subset_fisheye_result()
    result.per_frame_error = {}  # legacy artifact without retained frame identities
    report = compute_observability_report(result, data)
    assert report.num_points == 0
    assert report.observability_score == 0
    assert report.warnings
