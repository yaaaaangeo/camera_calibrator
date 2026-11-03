"""Regression: model-selection confidence and final confidence are distinct values."""
import numpy as np
from calibration.types import (CalibrationResult,ValidationResult,CameraModelType as M,CameraConfig,
 PatternConfig,PatternType,CalibrationConfidenceReport,ModelScore)
from calibration.recommender import compute_final_result
from export.report import generate_html_report
from tests.test_enhanced_report_export import _dataset


def test_confidence_labels_and_decimal_precision():
 camera=CameraConfig(640,480)
 pattern=PatternConfig(PatternType.CHESSBOARD,5,4,.04)
 cal=CalibrationResult(M.PINHOLE,camera_matrix=np.eye(3),distortion=np.zeros((5,1)),rms_error=1.5,success=True)
 val=ValidationResult(train_rms=1.4,test_rms=1.6,edge_rms=1.4,straightness_residual=.4)
 score=ModelScore(model_name=M.PINHOLE,score=.1,is_recommended=True,selection_confidence=62.8,selection_confidence_level='LOW')
 final=compute_final_result(M.PINHOLE,{M.PINHOLE:cal},{M.PINHOLE:val},scores=[score])
 final.confidence=CalibrationConfidenceReport(score=10.4,level='REJECT',components={'train_rms':26.4})
 html=generate_html_report('audit',camera,pattern,_dataset(),{M.PINHOLE:cal},{M.PINHOLE:val},final)
 assert html.count('10.4/100 - REJECT')>=2
 assert 'Model selection Confidence' in html and 'LOW 62.8%' in html
 assert 'Full-fit RMS score (export K/D)' in html
