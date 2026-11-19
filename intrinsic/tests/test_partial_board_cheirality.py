"""A board origin is a coordinate choice; observed-point cheirality is physical."""
import copy

import cv2
import numpy as np
import pytest

from calibration.robust_optimization import admissible
from calibration.types import (
    CalibrationResult, CameraModelType, Dataset, DetectionResult, Frame, ImageInfo,
)


def partial_board():
    # Five non-collinear visible corners near the far end of a tilted board.
    obj = np.array([[.4,.5,0], [.5,.5,0], [.6,.5,0], [.5,.6,0], [.6,.6,0]], np.float32).reshape(-1,1,3)
    rv = np.array([[.65],[0.],[0.]])
    tv = np.array([[-.5],[-.35],[-.09]])
    k = np.array([[800.,0,640.],[0,800.,360.],[0,0,1.]])
    d = np.zeros((5,1))
    pixels = cv2.projectPoints(obj,rv,tv,k,d)[0]
    det = DetectionResult('partial',True,corners=pixels,object_points=obj,ids=np.array([39,40,41,49,50]),num_corners=5)
    data = Dataset(frames=[Frame(ImageInfo('partial','synthetic',1280,720),det)])
    result = CalibrationResult(model_name=CameraModelType.BROWN_CONRADY,camera_matrix=k,distortion=d,
        rvecs=[rv],tvecs=[tv],per_frame_error={'partial':0.},rms_error=0.,success=True)
    return data,result


def test_valid_visible_partial_board_with_negative_origin_depth():
    data,result = partial_board()
    xyz=data.frames[0].detection.object_points.reshape(-1,3)@cv2.Rodrigues(result.rvecs[0])[0].T+result.tvecs[0].T
    assert result.tvecs[0][2,0]<0 and np.all(xyz[:,2]>0)
    assert admissible(result,data=data)


def test_admissibility_is_invariant_to_board_coordinate_origin():
    data,result = partial_board()
    shifted,rebased = copy.deepcopy(data),copy.deepcopy(result)
    offset=np.array([.5,.5,0.])
    shifted.frames[0].detection.object_points=(data.frames[0].detection.object_points-offset).astype(np.float32)
    rebased.tvecs[0]=result.tvecs[0]+cv2.Rodrigues(result.rvecs[0])[0]@offset.reshape(3,1)
    assert rebased.tvecs[0][2,0]>0
    original=cv2.projectPoints(data.frames[0].detection.object_points,result.rvecs[0],result.tvecs[0],result.camera_matrix,result.distortion)[0]
    other=cv2.projectPoints(shifted.frames[0].detection.object_points,rebased.rvecs[0],rebased.tvecs[0],rebased.camera_matrix,rebased.distortion)[0]
    np.testing.assert_allclose(original,other,atol=2e-4,rtol=0)
    assert admissible(result,data=data) == admissible(rebased,data=shifted) == True


def test_positive_origin_does_not_allow_a_corner_behind_camera():
    data,result=partial_board()
    result.rvecs=[np.array([[-.65],[0.],[0.]])]
    result.tvecs=[np.array([[0.],[0.],[.32]])]
    xyz=data.frames[0].detection.object_points.reshape(-1,3)@cv2.Rodrigues(result.rvecs[0])[0].T+result.tvecs[0].T
    assert result.tvecs[0][2,0]>0 and np.any(xyz[:,2]<=0) and np.any(xyz[:,2]>0)
    assert not admissible(result,data=data)


@pytest.mark.parametrize('invalid',['nonfinite_translation','nonfinite_rotation','missing_pose','all_behind'])
def test_invalid_observed_pose_still_rejected(invalid):
    data,result=partial_board()
    if invalid=='nonfinite_translation':result.tvecs[0][0,0]=np.nan
    elif invalid=='nonfinite_rotation':result.rvecs[0][0,0]=np.nan
    elif invalid=='missing_pose':result.tvecs=[]
    else:result.tvecs[0][2,0]=-10.
    assert not admissible(result,data=data)


def test_without_correspondences_negative_origin_remains_conservative():
    _,result=partial_board()
    assert not admissible(result)


def test_positive_depth_does_not_override_reprojection_quality_gate():
    from automation.acceptance import AcceptanceEvidence, evaluate_acceptance, parse_policy

    data, result = partial_board()
    result.tvecs[0][0, 0] += .1  # Wrong pose, still entirely in front of camera.
    det = data.frames[0].detection
    projected = cv2.projectPoints(det.object_points, result.rvecs[0], result.tvecs[0],
                                 result.camera_matrix, result.distortion)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected-det.corners)**2, axis=2))))
    assert admissible(result, data=data) and rms > 100
    evidence = AcceptanceEvidence(
        chosen_model='brown_conrady', recommended_model='brown_conrady',
        calibration_success=True, validation_success=True,
        train_frame_ids=['train'], test_frame_ids=['test'],
        metrics={'train_rms_px': rms},
    )
    # An example test policy only; no production threshold is introduced.
    policy = parse_policy({'thresholds': {'train_rms_px': {'max': 1., 'required': True}}})
    assert evaluate_acceptance(evidence, policy).verdict == 'FAIL'
