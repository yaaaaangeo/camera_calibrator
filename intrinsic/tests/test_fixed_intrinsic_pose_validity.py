"""Fixed-K evaluation must report unusable views, not score invalid poses."""
import copy
import cv2
import numpy as np
import pytest
from calibration.types import CameraConfig, CameraModelType as M, Dataset, Frame, ImageInfo, DetectionResult, PatternConfig, PatternType
from calibration.external_compare import _evaluate_side

K=np.array([[800.,0,640.],[0,800.,360.],[0,0,1.]])
D=np.zeros((5,1))
PATTERN=PatternConfig(PatternType.CHARUCO,10,7,.1,.075,'DICT_5X5_100')


def frame(fid='view', obj=None):
    if obj is None:obj=np.array([[0,0,0],[.2,0,0],[.2,.1,0],[0,.1,0],[.1,.05,0]],float)
    obj=np.asarray(obj,dtype=np.float32).reshape(-1,1,3)
    pixels=cv2.projectPoints(obj,np.array([.2,.1,0.]),np.array([0.,0.,1.]),K,D)[0]
    return Frame(ImageInfo(fid,'synthetic',1280,720),DetectionResult(fid,True,corners=pixels,object_points=obj,num_corners=len(obj)))


def evaluate(frames):
    return _evaluate_side(Dataset(frames=frames),CameraConfig(1280,720),PATTERN,
                          [f.image_info.image_id for f in frames],K.copy(),D.copy(),M.BROWN_CONRADY,'frozen')


def test_collinear_view_is_recorded_as_unevaluable():
    f=frame(obj=[[x,0,0] for x in np.linspace(0,.5,6)])
    side=evaluate([f])
    assert not side.success and side.test_rms is None
    assert side.failed_frame_reasons['view']=='object_points_collinear'
    assert side.num_evaluation_frames==1 and side.num_evaluated_frames==0


def test_negative_depth_native_pose_cannot_be_scored(monkeypatch):
    monkeypatch.setattr(cv2,'solvePnP',lambda *a,**k:(True,np.zeros((3,1)),np.array([[0.],[0.],[-1.]])))
    monkeypatch.setattr(cv2,'solvePnPGeneric',lambda *a,**k:(0,(),(),None))
    side=evaluate([frame()])
    assert not side.success and not side.per_frame_error
    assert side.pose_diagnostics['view']['attempts'][0]['rejection']=='nonpositive_observed_depth'


def test_partial_board_negative_origin_is_allowed_if_observed_depths_positive():
    f=frame(obj=[[.4,.5,0],[.5,.5,0],[.6,.5,0],[.5,.6,0],[.6,.6,0]])
    rv=np.array([.65,0.,0.]);tv=np.array([-.5,-.35,-.09])
    f.detection.corners=cv2.projectPoints(f.detection.object_points,rv,tv,K,D)[0]
    side=evaluate([f])
    assert side.success and side.test_rms<.001
    assert side.pose_diagnostics['view']['selected_pose']['origin_depth']<0
    assert min(side.pose_diagnostics['view']['selected_pose']['corner_depths'])>0


def test_success_rate_and_conditional_rms_have_distinct_denominators():
    good=frame('good');bad=frame('bad',[[x,0,0] for x in np.linspace(0,.5,6)])
    side=evaluate([good,bad])
    assert side.success and side.evaluation_status=='partial'
    assert side.num_evaluation_frames==2 and side.num_evaluated_frames==1
    assert side.evaluation_success_rate==.5 and side.test_rms<.001
    assert set(side.failed_frame_reasons)=={'bad'}


def test_nondetection_is_reported_not_silently_omitted():
    f=frame();f.detection.success=False;f.disable('detection_failed', outlier=False)
    side=evaluate([f])
    assert side.failed_frame_reasons['view']=='detection_failed'


def test_numerically_near_collinear_view_is_not_rescued_by_low_rms():
    obj=[[x,1e-7*(i%2),0] for i,x in enumerate(np.linspace(0,.5,6))]
    side=evaluate([frame(obj=obj)])
    assert not side.success
    assert side.failed_frame_reasons['view']=='object_points_near_collinear'


def test_intrinsics_remain_fixed_and_calibration_functions_are_never_called(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Intrinsic fitting called during fixed-K evaluation')
    monkeypatch.setattr(cv2,'calibrateCamera',forbidden)
    monkeypatch.setattr(cv2,'calibrateCameraExtended',forbidden)
    monkeypatch.setattr(cv2.fisheye,'calibrate',forbidden)
    before=(K.tobytes(),D.tobytes());f=frame();pixels=f.detection.corners.copy()
    side=evaluate([f])
    assert side.success and before==(K.tobytes(),D.tobytes())
    np.testing.assert_array_equal(pixels,f.detection.corners)


def test_planar_retry_recovers_only_after_native_invalidity(monkeypatch):
    f=frame()
    native=cv2.solvePnP
    ok,rv,tv=native(f.detection.object_points,f.detection.corners,K,D)
    assert ok
    monkeypatch.setattr(cv2,'solvePnP',lambda *a,**k:(False,None,None))
    monkeypatch.setattr(cv2,'solvePnPGeneric',lambda *a,**k:(1,(rv,),(tv,),None))
    side=evaluate([f])
    assert side.success and side.test_rms<.001
    assert side.pose_diagnostics['view']['retry_used'] is True


def test_high_native_rms_does_not_trigger_replacement_or_exclusion(monkeypatch):
    rv=np.array([[.2],[.1],[0.]]);tv=np.array([[.3],[0.],[1.]])
    monkeypatch.setattr(cv2,'solvePnP',lambda *a,**k:(True,rv,tv))
    side=evaluate([frame()])
    assert side.success and side.test_rms>100
    assert side.pose_diagnostics['view']['selected_pose']['method']=='native'
    assert side.pose_diagnostics['view']['retry_used'] is False


def test_geometry_conditioning_is_scale_and_origin_invariant():
    from calibration.fixed_intrinsic_pose import geometry_diagnostics
    f=frame();obj=f.detection.object_points.astype(float);img=f.detection.corners.astype(float)
    a,reason=geometry_diagnostics(obj,img)
    b,other=geometry_diagnostics(obj*1000+123.,img*2+100.)
    assert reason is None and other is None
    assert b['object_minor_major_ratio']==pytest.approx(a['object_minor_major_ratio'])
    assert b['image_minor_major_ratio']==pytest.approx(a['image_minor_major_ratio'])


def test_indistinguishable_distinct_retry_solutions_remain_unevaluable(monkeypatch):
    import calibration.fixed_intrinsic_pose as poses
    f=frame()
    monkeypatch.setattr(cv2,'solvePnP',lambda *a,**k:(False,None,None))
    monkeypatch.setattr(cv2,'solvePnPGeneric',lambda *a,**k:(2,(np.zeros((3,1)),np.array([[.2],[0.],[0.]])),(np.array([[0.],[0.],[1.]]),np.array([[0.],[0.],[1.]])),None))
    monkeypatch.setattr(poses,'project_points_for_model',lambda *a,**k:f.detection.corners.reshape(-1,2).copy())
    side=evaluate([f])
    assert not side.success and side.failed_frame_reasons['view']=='ambiguous_retry_solutions'


def test_nonfinite_projection_is_not_a_successful_numeric_score(monkeypatch):
    import calibration.fixed_intrinsic_pose as poses
    f=frame()
    monkeypatch.setattr(poses,'project_points_for_model',lambda *a,**k:np.full((5,2),np.nan))
    side=evaluate([f])
    assert not side.success and side.test_rms is None
    assert any(a.get('rejection')=='nonfinite_projection' for a in side.pose_diagnostics['view']['attempts'])


def test_explicitly_requested_disabled_view_has_a_reason_and_counts_in_denominator():
    good=frame('good');disabled=frame('disabled');disabled.disable('diagnostic request', outlier=False)
    side=evaluate([good,disabled])
    assert side.num_evaluation_frames==2 and side.num_evaluated_frames==1
    assert side.failed_frame_reasons['disabled']=='frame_disabled'
    assert side.evaluation_success_rate==.5
