"""P6 reproductions: continuous map certificates, never intrinsic refits."""
import copy
import numpy as np
import pytest
from calibration.types import CalibrationResult,CameraConfig,CameraModelType as M
from calibration.physical_validity import distortion_diagnostic as check, enforce_distortion_validity

K=np.array([[1900.8001711926072,0,1916.239704166391],
            [0,1905.562962179699,1082.7633220029925],[0,0,1.]])
D=[-.03663684455251561,-.357270800990237,.0009533611426422413,.00024206221790270638,
   -.023425604272298037,.319623501382767,-.4565608422269512,-.10792682638905104]
CAM=CameraConfig(3840,2160)

def result(d=None,k=None,model=M.EXTENDED_PINHOLE):
    return CalibrationResult(model_name=model,camera_matrix=K.copy() if k is None else k,
        distortion=np.array(D if d is None else d,dtype=float),success=True,rms_error=2.0547646250060847)


def test_current_rational_pole_outside_raw_domain_inside_observed_domain():
    r=result(); oldraw=max(np.hypot((x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1]) for x in (0,3840) for y in (0,2160))
    a=check(r,CAM,observed_xyz=[[1.912806327826,0,1]])
    assert oldraw==pytest.approx(1.16067605794)
    assert a['status']=='FAIL'
    assert a['denominator_roots_radius'][0]==pytest.approx(1.244265211702593)
    assert a['nearest_root_separation']==pytest.approx(9.32396e-7,abs=1e-10)
    assert a['checked_radius']>a['denominator_roots_radius'][0]>oldraw


def test_raw_inverse_envelope_is_not_misrepresented_as_observed_rays():
    a=check(result(),CAM)
    # This model has a tiny valid central inverse branch; actual fitted rays
    # beyond its pole are indispensable evidence and cannot be inferred by
    # silently trusting iterative undistortPoints.
    assert a['raw_inverse_envelope_radius']>a['raw_normalized_radius_max']
    assert a['observed_points']==0


@pytest.mark.parametrize('d',[[0]*8,[-.12,.015,.0001,-.0002,.001,.02,.001,.0001]])
def test_normal_rational_not_falsely_rejected(d):
    a=check(result(d),CAM)
    assert a['status']=='PASS',a
    assert a['raw_inverse_envelope_radius']>0
    assert a['inverse_boundary_polynomial_residual']>a['inverse_boundary_roundoff_bound']
    assert a['jacobian_certificate']


@pytest.mark.parametrize('delta',[0,1e-12,1e-6])
def test_common_or_nearly_common_factor_never_cancelled(delta):
    d=[-1+delta,0,0,0,0,-1,0,0]
    a=check(result(d),CAM,observed_xyz=[[1.1,0,1]])
    assert a['status']=='FAIL' and a['denominator_roots_radius'][0]==pytest.approx(1)


@pytest.mark.parametrize('r',[1.,1.+1e-12,1.-1e-12])
def test_pole_on_or_numerically_at_domain_boundary(r):
    k=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    a=check(result([0,0,0,0,0,-1/r**2,0,0],k),observed_xyz=[[1,0,1]])
    assert a['status']=='FAIL'


def test_double_denominator_root_between_sample_points():
    # Q=(1-r²/a²)^2 touches zero, without a sign change.
    a=1.23456789
    d=[0,0,0,0,0,-2/a**2,1/a**4,0]
    assert check(result(d),CAM,observed_xyz=[[1.8,0,1]])['status']=='FAIL'


def test_arbitrarily_narrow_negative_denominator_interval_not_sampled():
    a,b=1.23456789,1.23456801
    d=[0,0,0,0,0,-1/a**2-1/b**2,1/(a*b)**2,0]
    assert check(result(d),CAM,observed_xyz=[[1.8,0,1]])['status']=='FAIL'


def test_radial_fold_without_denominator_pole():
    a=check(result([-.4,0,0,0,0,0,0,0]),CAM)
    assert a['status']=='FAIL'


def test_tangential_jacobian_fold_without_radial_fold():
    a=check(result([0,0,0,1.,0,0,0,0]),observed_xyz=[[1.,0,1]])
    assert a['status']=='FAIL'
    assert 'tangential_2d' in a['reasons'][0]


def test_fisheye_legacy_hemisphere_not_relaxed_to_observed_fov():
    d=[-.0165858973823961,-.02674492862773239,.02149287035939189,-.008047640149521668]
    a=check(result(d,model=M.FISHEYE),CAM,observed_xyz=[[np.tan(np.deg2rad(62.50168)),0,1]])
    assert a['status']=='FAIL'
    assert np.rad2deg(a['angular_derivative_roots_rad'][0])==pytest.approx(86.70115)
    assert np.rad2deg(a['observed_theta_max_rad'])==pytest.approx(62.50168)


def test_valid_fisheye_and_positive_depth():
    assert check(result([0]*4,model=M.FISHEYE),CAM)['status']=='PASS'
    assert check(result([0]*8),CAM,observed_xyz=[[0,0,-1]])['status']=='FAIL'


def test_output_roi_and_its_own_matrix_extend_domain():
    r=result([0,0,0,0,0,-1/4,0,0]); k=K.copy();k[:2,:2]/=3
    a=check(r,CAM,output_matrix=k,output_roi=(0,0,3840,2160))
    assert a['status']=='FAIL' and a['output_ideal_radius_max']>2
    assert check(result([0]*8),CAM,output_roi=(0,0,10,10))['status']=='FAIL'


def test_mask_never_hides_a_bad_corner_or_shrinks_default_roi():
    # No mask argument: a hood/coverage mask cannot authorize geometric cropping.
    a=check(result(),CAM,observed_xyz=[[1.913,0,1]])
    assert a['status']=='FAIL' and a['raw_roi']==[0,0,3840,2160]
    assert 'does not crop' in a['mask_policy']


def test_diagnostics_reproducible_nonmutating_and_gate_preserves_numbers(monkeypatch):
    r=result(); original=copy.deepcopy(r)
    a=check(r,CAM,observed_xyz=[[1.913,0,1]])
    assert all(check(r,CAM,observed_xyz=[[1.913,0,1]])==a for _ in range(4))
    assert r.success and not r.optimization_audit
    monkeypatch.setattr('calibration.physical_validity.fitted_rays',lambda *_:np.array([[1.913,0,1]]))
    enforce_distortion_validity(r,CAM,None)
    assert not r.success and r.optimization_audit['physical_validity']['status']=='FAIL'
    np.testing.assert_array_equal(r.camera_matrix,original.camera_matrix)
    np.testing.assert_array_equal(r.distortion,original.distortion)
    assert r.rms_error==original.rms_error


def test_jacobian_matches_opencv_finite_differences():
    import cv2
    from calibration.physical_validity import rational_jacobian
    d=np.array([-.1,.01,.002,-.003,.001,.01,.001,0.])
    xy=np.array([.7,-.4]); eps=1e-6
    def forward(v):
        return cv2.projectPoints(np.array([[v[0],v[1],1.]]),np.zeros(3),np.zeros(3),np.eye(3),d)[0].ravel()
    numeric=np.column_stack([(forward(xy+eps*axis)-forward(xy-eps*axis))/(2*eps) for axis in np.eye(2)])
    np.testing.assert_allclose(rational_jacobian(xy,d),numeric,atol=1e-9)


def test_smaller_explicit_raw_crop_cannot_hide_retained_rays():
    a=check(result(),CAM,raw_roi=(1800,900,200,200),observed_xyz=[[1.913,0,1]])
    assert a['status']=='FAIL' and a['checked_radius']>=1.913


def test_existing_bag_rational_control_remains_valid():
    # Frozen pre-P6 numeric output, not refitted or rounded for this test.
    k=np.array([[1940.1202506851255,0,1929.5235509169643],
                [0,1946.4167473432765,1098.281895334406],[0,0,1.]])
    d=[5.716850102921429,1.9079779135785635,-4.099408822446131e-05,
       -.00022860349558336525,.046501087080368875,6.136442158667708,
       3.8676870851212386,.3414811570151213]
    assert check(result(d,k),CAM)['status']=='PASS'


def observed_data(r):
    import cv2
    from calibration.types import Dataset,Frame,ImageInfo,DetectionResult
    obj=np.array([[0,0,0],[.5,0,0],[0,.5,0],[1.913,0,0],[1.5,.1,0]],np.float32).reshape(-1,1,3)
    r.rvecs=[np.zeros(3)];r.tvecs=[np.array([0.,0.,1.])];r.per_frame_error={'observed':r.rms_error}
    uv=cv2.projectPoints(obj,r.rvecs[0],r.tvecs[0],r.camera_matrix,r.distortion)[0]
    det=DetectionResult('observed',True,corners=uv,object_points=obj,ids=np.arange(5),num_corners=5)
    return Dataset(frames=[Frame(ImageInfo('observed','synthetic',3840,2160),det)])


def test_actual_admissibility_and_native_gate_reject_reproduction(monkeypatch,tmp_path):
    from calibration.robust_optimization import admissible
    from calibration.models.extended_pinhole import calibrate_extended_pinhole
    from export.opencv import export_opencv_yaml
    r=result();data=observed_data(r)
    assert not admissible(r,CAM,data)
    monkeypatch.setattr('calibration.models.extended_pinhole._calibrate_extended_pinhole_core',lambda *a,**kw:r)
    gated=calibrate_extended_pinhole(data,CAM)
    assert not gated.success and gated.rms_error==2.0547646250060847
    with pytest.raises(ValueError):export_opencv_yaml(gated,CAM,None,str(tmp_path/'invalid.yaml'))
    assert not (tmp_path/'invalid.yaml').exists()
    from automation.acceptance import AcceptanceEvidence,evaluate_acceptance,parse_policy
    evidence=AcceptanceEvidence(chosen_model='extended_pinhole',recommended_model='extended_pinhole',
        calibration_success=gated.success,validation_success=True,
        train_frame_ids=['train'],test_frame_ids=['test'],metrics={'train_rms_px':0.1})
    policy=parse_policy({'thresholds':{'train_rms_px':{'max':1.,'required':True}}})
    assert evaluate_acceptance(evidence,policy).verdict=='FAIL'


def test_resolution_mismatch_fails_closed():
    r=result([0]*8);data=observed_data(r)
    assert check(r,CameraConfig(1920,1080),data)['reasons']==['camera_frame_resolution_mismatch']


def test_pre_p6_cache_hit_rechecked_without_rewriting_cache(monkeypatch,tmp_path):
    from calibration import compare
    r=result();data=observed_data(r)
    class Cache:
        def get(self,key):return [r]
        def set(self,*args):raise AssertionError('Historical cache must not be overwritten')
    monkeypatch.setattr(compare,'PersistentResultCache',lambda *a,**kw:Cache())
    monkeypatch.setattr(compare,'model_results_cache_key',lambda *a,**kw:'frozen')
    rows=compare.run_all_models(data,CAM,persistent_cache_dir=tmp_path,models=[M.EXTENDED_PINHOLE])
    assert not rows[0].success and rows[0].optimization_audit['physical_validity']['status']=='FAIL'


def test_fisheye_monotone_but_raw_sensor_outside_front_inverse_domain():
    k=np.array([[400.,0,1920.],[0,400.,1080.],[0,0,1.]])
    a=check(result([0]*4,k,model=M.FISHEYE),CAM)
    assert a['status']=='FAIL'
    assert a['reasons']==['raw_roi_outside_front_hemisphere_inverse_domain']


def test_robust_final_rejection_keeps_continuous_diagnostic(monkeypatch):
    import calibration.robust_optimization as robust
    r=result();data=observed_data(r)
    monkeypatch.setattr(robust,'fit_strategy',lambda *a,**kw:copy.deepcopy(r))
    # Avoid sharing a previous in-process synthetic test's development cache.
    monkeypatch.setattr(robust,'_DEVELOPMENT_CACHE',{})
    gated=robust.robust_fit(data,CAM,M.EXTENDED_PINHOLE,data)
    assert not gated.success
    assert gated.optimization_audit['final_physical_check']=='FAIL'
    assert gated.optimization_audit['physical_validity']['status']=='FAIL'
    assert gated.optimization_audit['physical_validity']['denominator_roots_radius'][0]==pytest.approx(1.244265211702593)


# (b) Base calibration evaluates Fisheye on its usage domain; Robust Optimization
# keeps the legacy [0,pi/2] gate unchanged.
FK=np.array([[1892.1525747585413,0,1924.9897715247241],
             [0,1897.7029519524917,1102.1574478041052],[0,0,1.]])
FD=[-.0165858973823961,-.02674492862773239,.02149287035939189,-.008047640149521668]
OBS62=[[np.tan(np.deg2rad(62.50168)),0,1]]


def test_fisheye_fold_outside_usage_domain_passes_base_but_not_hemisphere():
    r=result(FD,FK,model=M.FISHEYE)
    hemi=check(r,CAM,observed_xyz=OBS62)
    usage=check(r,CAM,observed_xyz=OBS62,fisheye_domain='usage')
    assert hemi['status']=='FAIL' and hemi['reasons']==['fisheye_angular_fold_or_uncertified']
    assert usage['status']=='PASS',usage
    fold=usage['angular_derivative_roots_rad'][0]
    assert usage['checked_theta_max_rad']<fold and np.rad2deg(fold)==pytest.approx(86.70115)
    assert usage['checked_theta_max_rad']>=max(usage['observed_theta_max_rad'],usage['raw_inverse_theta_rad'],
                                               usage['output_theta_max_rad'])
    assert usage['fold_outside_usage_domain_only'] is True
    assert 'not calibration quality PASS' in usage['interpretation']


def test_fisheye_fold_inside_observed_rays_fails_base():
    r=result(FD,FK,model=M.FISHEYE)
    a=check(r,CAM,observed_xyz=[[np.tan(np.deg2rad(87.)),0,1]],fisheye_domain='usage')
    assert a['status']=='FAIL' and a['reasons']==['fisheye_angular_fold_in_usage_domain_or_uncertified']


def test_fisheye_fold_at_usage_boundary_fails_base():
    r=result(FD,FK,model=M.FISHEYE)
    fold=check(r,CAM,fisheye_domain='usage')['angular_derivative_roots_rad'][0]
    a=check(r,CAM,observed_xyz=[[np.tan(fold),0,1]],fisheye_domain='usage')
    assert a['status']=='FAIL'


def test_fisheye_raw_corner_unreachable_before_fold_fails_base():
    # Native fit from test_fisheye_import_safety: theta_d peaks at 0.907 rad
    # (55.4 deg) but the 1280x720 sensor corner needs 1.065 -> no inverse ray.
    k=np.array([[695.2327684130341,-.0139053851573842,634.5441759691159],
                [0,695.7680517567067,357.2436806233866],[0,0,1.]])
    d=[.046278525510332236,-.009482486248902917,.13026892954455224,-.2661183504625318]
    a=check(result(d,k,model=M.FISHEYE),CameraConfig(1280,720),observed_xyz=[[.844128033623355,0,1]],
            fisheye_domain='usage')
    assert a['status']=='FAIL' and a['reasons']==['raw_roi_not_reached_before_fold_or_front_hemisphere']


def test_fisheye_base_usage_domain_still_requires_front_hemisphere():
    k=np.array([[400.,0,1920.],[0,400.,1080.],[0,0,1.]])
    a=check(result([0]*4,k,model=M.FISHEYE),CAM,fisheye_domain='usage')
    assert a['status']=='FAIL'


def test_base_gate_uses_usage_domain_and_robust_keeps_hemisphere(monkeypatch):
    from calibration.robust_optimization import admissible
    r=result(FD,FK,model=M.FISHEYE)
    monkeypatch.setattr('calibration.physical_validity.fitted_rays',lambda *_:np.array(OBS62,float))
    gated=enforce_distortion_validity(copy.deepcopy(r),CAM,None)
    assert gated.success and gated.optimization_audit['physical_validity']['fisheye_domain']=='usage'
    robust=copy.deepcopy(r)
    assert not admissible(robust,CAM,None)
    assert robust.optimization_audit['physical_validity']['fisheye_domain']=='hemisphere'
    assert robust.optimization_audit['physical_validity']['checked_theta_max_rad']==pytest.approx(np.pi/2)


def test_rational_is_independent_of_fisheye_domain():
    for kw in ({},{'fisheye_domain':'usage'}):
        a=check(result(),CAM,observed_xyz=[[1.912806327826,0,1]],**kw)
        assert a['status']=='FAIL' and a['denominator_roots_radius'][0]==pytest.approx(1.244265211702593)
