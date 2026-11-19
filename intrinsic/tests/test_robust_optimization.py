import copy
import numpy as np
import pytest
from calibration.types import CalibrationResult, CameraModelType as M
from calibration.initialization import compatible_seed
from calibration.robust_optimization import robust_fit, admissible
from calibration.validation import split_train_test,_subset_dataset,validate_holdout
from calibration.project_io import project_to_dict,project_from_dict


def result(model,d):
 return CalibrationResult(model_name=model,camera_matrix=np.diag([800.,800.,1.]),distortion=np.array(d,dtype=float).reshape(-1,1),success=True)


def test_distortion_compatibility():
 brown=result(M.BROWN_CONRADY,[1,2,3,4,5]);_,d=compatible_seed(brown,M.EXTENDED_PINHOLE)
 np.testing.assert_array_equal(d.ravel(),[1,2,3,4,5,0,0,0])
 fish=result(M.FISHEYE,[1,2,3,4]);_,d=compatible_seed(fish,M.BROWN_CONRADY)
 assert not d.any()
 rational=result(M.EXTENDED_PINHOLE,list(range(8)));_,d=compatible_seed(rational,M.BROWN_CONRADY)
 assert not d.any()  # denominator cannot be silently discarded
 _,d=compatible_seed(brown,M.PINHOLE);assert not d.any()


def test_seed_rejects_invalid():
 r=result(M.BROWN_CONRADY,[0]*5);r.camera_matrix[0,0]=-1
 with pytest.raises(ValueError):compatible_seed(r,M.BROWN_CONRADY)
 assert not admissible(r)


def test_rational_pole_rejected(camera_config):
 r=result(M.EXTENDED_PINHOLE,[0,0,0,0,0,-1,0,0])
 assert not admissible(r,camera_config)


def test_outer_test_never_reaches_optimizer(monkeypatch,synthetic_dataset,camera_config):
 import calibration.robust_optimization as opt
 train_ids,test_ids=split_train_test(synthetic_dataset,camera_config)
 dev=_subset_dataset(synthetic_dataset,train_ids)
 seen=[]
 def fake(data,camera,model,strategy,seed_data):
  ids={f.image_info.image_id for f in data.enabled_frames}
  seeds={f.image_info.image_id for f in seed_data.enabled_frames}
  assert ids.isdisjoint(test_ids) and seeds.isdisjoint(test_ids)
  seen.append((ids,seeds));return CalibrationResult(model_name=model,success=False,error_message='injected')
 monkeypatch.setattr(opt,'fit_strategy',fake)
 r=robust_fit(dev,camera_config,M.EXTENDED_PINHOLE,dev)
 assert seen and not r.success and r.optimization_audit['fallback']
 assert set(r.optimization_audit['selection_train_ids']).isdisjoint(r.optimization_audit['selection_validation_ids'])


def test_train_parameters_persist(synthetic_dataset,camera_config,pattern_config):
 from calibration.types import CalibrationProject
 train,test=split_train_test(synthetic_dataset,camera_config)
 r=validate_holdout(copy.deepcopy(synthetic_dataset),camera_config,pattern_config,M.BROWN_CONRADY,train,test)
 assert r.success and r.train_camera_matrix.shape==(3,3)
 project=CalibrationProject(project_name="p5",camera_config=camera_config,pattern_config=pattern_config,dataset=synthetic_dataset,
                            validation_results={M.BROWN_CONRADY:r})
 restored=project_from_dict(project_to_dict(project)).validation_results[M.BROWN_CONRADY]
 np.testing.assert_array_equal(r.train_camera_matrix,restored.train_camera_matrix)
 np.testing.assert_array_equal(r.train_distortion,restored.train_distortion)


def test_synthetic_default_unchanged(synthetic_dataset,camera_config):
 from calibration.models.extended_pinhole import calibrate_extended_pinhole
 import cv2
 cv2.setNumThreads(1)
 a=calibrate_extended_pinhole(copy.deepcopy(synthetic_dataset),camera_config)
 b=calibrate_extended_pinhole(copy.deepcopy(synthetic_dataset),camera_config,initial_guess=None)
 # P6: identical native numbers and identical physical verdict (this narrow
 # synthetic Rational folds inside the sensor ROI, so both are rejected).
 assert a.success==b.success and a.rms_error==b.rms_error and a.error_message==b.error_message
 np.testing.assert_array_equal(a.camera_matrix,b.camera_matrix)
 np.testing.assert_array_equal(a.distortion,b.distortion)

@pytest.mark.parametrize('model',[M.BROWN_CONRADY,M.FISHEYE])
def test_known_intrinsics_control(model):
 import cv2
 from calibration.types import CameraConfig,Dataset,Frame,ImageInfo,DetectionResult
 from calibration.detector import _compute_board_geometry
 from calibration.validation import _test_reprojection_errors
 from tests._automation_test_utils import _sample_poses,TRUE_K,TRUE_D
 cv2.setNumThreads(1)
 obj=np.array([[x*.04,y*.04,0.] for y in range(1,5) for x in range(1,7)],np.float32).reshape(-1,1,3)
 camera=CameraConfig(1280,720);frames=[]
 true_d=TRUE_D if model==M.BROWN_CONRADY else np.array([-.04,.01,-.002,.0003])
 for i,(rotation,translation) in enumerate(_sample_poses(36,214)):
  rv=cv2.Rodrigues(rotation)[0]
  pts=(cv2.projectPoints(obj,rv,translation,TRUE_K,true_d)[0] if model==M.BROWN_CONRADY
       else cv2.fisheye.projectPoints(obj,rv,translation,TRUE_K,true_d)[0])
  area,xy,tilt,edge=_compute_board_geometry(pts,(720,1280));fid=f'known_{model.value}_{i}'
  det=DetectionResult(fid,True,corners=pts,object_points=obj.copy(),num_corners=len(obj),board_area_ratio=area,board_center_px=xy,board_tilt_deg=tilt,min_edge_margin_px=edge)
  frames.append(Frame(ImageInfo(fid,'POINT_CONTROL',1280,720),det))
 data=Dataset(frames=frames);train_ids,test_ids=split_train_test(data,camera)
 dev=_subset_dataset(data,train_ids)
 r=robust_fit(dev,camera,model,dev)
 assert r.success
 np.testing.assert_allclose(r.camera_matrix,TRUE_K,atol=.02,rtol=0)
 np.testing.assert_allclose(r.distortion.reshape(-1),true_d,atol=.002,rtol=0)
 per,failed,*_=_test_reprojection_errors(_subset_dataset(data,test_ids).enabled_frames,r.camera_matrix,r.distortion,model)
 assert not failed and max(per.values())<.01
 assert set(r.optimization_audit['selection_train_ids']).isdisjoint(test_ids)
 assert set(r.optimization_audit['selection_validation_ids']).isdisjoint(test_ids)


def test_validation_selects_instead_of_minimum_fit_rms(monkeypatch,synthetic_dataset,camera_config):
 import calibration.robust_optimization as opt
 import calibration.validation as validation
 opt._DEVELOPMENT_CACHE.clear()
 train_ids,_=split_train_test(synthetic_dataset,camera_config);dev=_subset_dataset(synthetic_dataset,train_ids)
 def fake(data,camera,model,strategy,seed_data):
  # P6: physically valid fake (principal point near the sensor centre); the
  # validation error is the principal point offset from that centre.
  r=result(M.FISHEYE,[0]*4)
  r.camera_matrix[0,2]=camera.width/2+(1 if strategy=='cold' else 2)
  r.camera_matrix[1,2]=camera.height/2
  r.rms_error=100 if strategy=='cold' else .01
  return r
 def errors(frames,k,*args):
  return {f.image_info.image_id:float(k[0,2]-camera_config.width/2) for f in frames},[],[],[],[],{}
 monkeypatch.setattr(opt,'fit_strategy',fake);monkeypatch.setattr(validation,'_test_reprojection_errors',errors)
 r=robust_fit(dev,camera_config,M.FISHEYE,dev)
 assert r.optimization_audit['selected_strategy']=='cold'
 opt._DEVELOPMENT_CACHE.clear()


def test_inadmissible_legacy_fallback_cannot_succeed(monkeypatch,synthetic_dataset,camera_config):
 import calibration.robust_optimization as opt
 opt._DEVELOPMENT_CACHE.clear()
 def invalid(*args,**kwargs):
  r=result(M.FISHEYE,[0]*4);r.camera_matrix[0,0]=-10
  return r
 monkeypatch.setattr(opt,'fit_strategy',invalid)
 r=robust_fit(synthetic_dataset,camera_config,M.FISHEYE,synthetic_dataset)
 assert not r.success and r.optimization_audit['final_physical_check']=='FAIL'
 assert 'no physically admissible' in r.error_message
 opt._DEVELOPMENT_CACHE.clear()
