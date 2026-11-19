import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.evaluate_frozen_intrinsics import assert_disjoint,evaluate


def test_overlap_guard_checks_decoded_pixels_across_encodings(tmp_path):
 pixels=np.full((12,16,3),40,np.uint8)
 source=tmp_path/'source.png';copied=tmp_path/'copied.bmp'
 cv2.imwrite(str(source),pixels);cv2.imwrite(str(copied),pixels)
 with pytest.raises(ValueError,match='duplicates development'):
  assert_disjoint([source],[copied],(16,12))
 alpha=tmp_path/'alpha.png';cv2.imwrite(str(alpha),cv2.cvtColor(pixels,cv2.COLOR_BGR2BGRA))
 with pytest.raises(ValueError,match='duplicates development'):
  assert_disjoint([source],[alpha],(16,12))
 with pytest.raises(ValueError,match='resolution mismatch'):
  assert_disjoint([source],[copied],(32,24))


def test_same_recording_is_not_independent(tmp_path):
 with pytest.raises(ValueError,match='distinct new capture'):
  evaluate('not-read','not-read','not-read',tmp_path,tmp_path/'new','rosbag2_2026_10_23-15_38_30')


def test_frozen_evaluation_has_no_intrinsic_fit(tmp_path,monkeypatch):
 from tests._automation_test_utils import render_consistent_charuco_dataset,TRUE_K,TRUE_D,SQUARES_X,SQUARES_Y,SQUARE_SIZE,MARKER_SIZE,DICTIONARY
 from calibration.types import CalibrationProject,CameraConfig,PatternConfig,PatternType,CalibrationResult,CameraModelType as M
 from calibration.detector import detect_dataset
 from calibration.project_io import save_project
 from export.opencv import export_opencv_yaml
 from scripts.evaluate_frozen_intrinsics import digest
 cv2.setNumThreads(1)
 train=tmp_path/'train';fresh=tmp_path/'fresh'
 render_consistent_charuco_dataset(train,16,7);render_consistent_charuco_dataset(fresh,16,71)
 pattern=PatternConfig(PatternType.CHARUCO,SQUARES_X,SQUARES_Y,SQUARE_SIZE,marker_size=MARKER_SIZE,dictionary=DICTIONARY)
 camera=CameraConfig(1280,720)
 data=detect_dataset([str(p) for p in train.glob('*.png')],pattern,parallel=False)
 source=tmp_path/'source.ccproj';save_project(CalibrationProject('synthetic',camera,pattern,data),str(source))
 paths=[]
 for name,delta in [('reference',30.),('candidate',0.)]:
  k=TRUE_K.copy();k[0,0]+=delta
  r=CalibrationResult(M.BROWN_CONRADY,camera_matrix=k,distortion=TRUE_D.reshape(-1,1),rms_error=0.,success=True)
  path=tmp_path/(name+'.yaml');export_opencv_yaml(r,camera,pattern,str(path));paths.append(path)
 locked={str(p):digest(p) for p in [source,*paths]}
 def forbidden(*args,**kwargs):pytest.fail('Intrinsic calibration called while evaluating frozen K/D')
 monkeypatch.setattr(cv2,'calibrateCamera',forbidden);monkeypatch.setattr(cv2,'calibrateCameraExtended',forbidden);monkeypatch.setattr(cv2.fisheye,'calibrate',forbidden)
 output=tmp_path/'evaluation'
 report=evaluate(source,*paths,fresh,output,'SYNTHETIC-new-seed71')
 assert report['intrinsics_refitted'] is False and report['approval']=='NOT_APPROVED'
 np.testing.assert_array_equal(report['frozen_parameters']['candidate']['K'],TRUE_K)
 np.testing.assert_array_equal(np.array(report['frozen_parameters']['candidate']['D']).reshape(-1),TRUE_D.reshape(-1))
 assert report['result']['evaluation_source']=='independent_benchmark'
 assert all(digest(p)==h for p,h in locked.items())
 assert (output/'independent_result.json').is_file()
 with pytest.raises(ValueError,match='new output'):
  evaluate(source,*paths,fresh,output,'SYNTHETIC-new-seed71')
