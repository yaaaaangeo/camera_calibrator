from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from calibration.types import (CalibrationResult,ValidationResult,CameraModelType as M,QualityGrade,
 ResidualStats,ParameterUncertainty,ObservabilityReport,UndistortionQualityReport,ModelScore,DiagnosisSeverity)
from calibration.recommender import compute_final_confidence


def recipe():
 cal=CalibrationResult(M.FISHEYE,camera_matrix=np.eye(3),distortion=np.zeros((4,1)),rms_error=1.5386657428960624,success=True,
  param_uncertainty=ParameterUncertainty(overall_stability=68.9561860530732),observability=ObservabilityReport(observability_score=0.),
  undistortion_quality=UndistortionQualityReport(quality_score=99.8591338734568))
 val=ValidationResult(train_rms=1.5164065248285348,test_rms=1.6136043045226272,edge_rms=1.4136072022351427,
  straightness_residual=.4003287934106072,test_residual_stats=ResidualStats(p95=3.5825318808008975),success=True)
 scores=[ModelScore(M.FISHEYE,score=.14,selection_confidence=62.8)]
 diag=SimpleNamespace(patterns=[SimpleNamespace(severity=DiagnosisSeverity.WARNING) for _ in range(5)])
 return cal,val,scores,diag


def test_confidence_104_reproduces_existing_rounded_weighted_recipe():
 cal,val,scores,diag=recipe()
 result=compute_final_confidence(cal,val,QualityGrade.WARNING,68.75,scores,diag)
 assert result.score==10.4 and result.level=='REJECT'
 assert result.components['train_rms']==26.4  # Full-fit input, not val.train_rms
 assert result.components['test_p95']==0.0 and result.components['observability']==0.0
 assert '30 points' in result.warnings[0]


def test_historical_failure_and_exclusion_count_are_not_direct_penalties():
 cal,val,scores,diag=recipe()
 baseline=compute_final_confidence(cal,val,QualityGrade.WARNING,68.75,scores,diag)
 cal.warning_message='Historical native failure; one frame excluded'
 cal.optimization_audit={'failed_candidates':[{'error':'previous failure'}],'excluded_frames':['frame5']}
 assert compute_final_confidence(cal,val,QualityGrade.WARNING,68.75,scores,diag)==baseline
 failed=replace(cal,success=False,error_message='current fit failure')
 result=compute_final_confidence(failed,val,QualityGrade.WARNING,68.75,scores,diag)
 assert result.score==0 and result.level=='REJECT'


def test_missing_observability_is_distinct_from_measured_zero():
 cal,val,scores,diag=recipe()
 measured=compute_final_confidence(cal,val,QualityGrade.WARNING,68.75,scores,diag)
 missing=compute_final_confidence(replace(cal,observability=None),val,QualityGrade.WARNING,68.75,scores,diag)
 assert measured.components['observability']==0
 assert 'observability' not in missing.components


def test_failed_or_absent_holdout_omits_metrics_and_reports_insufficient_evidence():
 cal,val,scores,diag=recipe()
 for evidence in [None,replace(val,success=False,error_message='pose evaluation failed')]:
  result=compute_final_confidence(cal,evidence,QualityGrade.WARNING,68.75,scores,diag)
  assert not {'test_rms','test_p95','edge_rms','straightness'}&set(result.components)
  assert 'Hold-out validation is missing or failed.' in result.warnings
