"""Occlusion-aware actions must not silently change grading or confidence."""
from dataclasses import asdict
import numpy as np
from calibration.types import CalibrationResult,ValidationResult,CameraModelType as M,CoverageCell
from calibration.diagnosis import diagnose_calibration
from calibration.recommender import compute_final_result


def inputs():
 cal=CalibrationResult(model_name=M.FISHEYE,success=True,camera_matrix=np.eye(3),distortion=np.zeros((4,1)),rms_error=1.5)
 val=ValidationResult(train_rms=1.5,test_rms=1.6,edge_rms=1.4,straightness_residual=.4)
 grid=[CoverageCell(row=r,col=c,corner_count=0 if r==3 else 20,coverage_score=0 if r==3 else 1) for r in range(4) for c in range(4)]
 obs={'grid':[{'row':r,'col':c,'visible_pixels':0 if r==3 else 100,'visible_fraction':0 if r==3 else 1,
               'sufficient':r<2 or (r==2 and c==3),'coverage_score':1 if r<2 else .2,'corner_count':20} for r in range(4) for c in range(4)]}
 return cal,val,grid,obs


def test_masked_cells_are_not_capture_targets():
 cal,val,grid,obs=inputs()
 old=diagnose_calibration(cal,val,68.75,grid)
 new=diagnose_calibration(cal,val,68.75,grid,observable_coverage=obs)
 assert any(r.code=='capture_bottom' for r in old.capture_recommendations)
 assert {r.code for r in new.capture_recommendations}=={'capture_observable_r2_c0','capture_observable_r2_c1','capture_observable_r2_c2'}
 assert all('unmasked' in r.action for r in new.capture_recommendations)
 assert [(p.code,p.severity,p.evidence) for p in old.patterns]==[(p.code,p.severity,p.evidence) for p in new.patterns]
 assert all('do not capture' in p.recommendation for p in new.patterns if p.code in ('limited_coverage','coverage_location_gaps'))


def test_guidance_does_not_change_grade_confidence_or_parameters():
 cal,val,grid,obs=inputs()
 before=compute_final_result(M.FISHEYE,{M.FISHEYE:cal},{M.FISHEYE:val},68.75,coverage_grid=grid)
 after=compute_final_result(M.FISHEYE,{M.FISHEYE:cal},{M.FISHEYE:val},68.75,coverage_grid=grid,capture_observable_coverage=obs)
 assert before.overall_grade==after.overall_grade
 assert asdict(before.confidence)==asdict(after.confidence)
 assert before.calibration is after.calibration and before.validation is after.validation


def test_without_mask_legacy_guidance_is_unchanged():
 cal,val,grid,_=inputs()
 assert asdict(diagnose_calibration(cal,val,68.75,grid))==asdict(diagnose_calibration(cal,val,68.75,grid,observable_coverage=None))
