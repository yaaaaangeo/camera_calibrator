"""Known-K projected-point control under unchanged P2 review thresholds."""
from dataclasses import asdict
import json

import cv2
import numpy as np
import pytest

from automation.live.assessment import assess_observability
from automation.live.config import ReadinessConfig
from automation.live.readiness import preflight_observability
from calibration.detector import _compute_board_geometry
from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye
from calibration.observability import compute_observability_report
from calibration.types import CameraConfig, Dataset, Frame, ImageInfo, DetectionResult
from calibration.validation import split_train_test
from tests._automation_test_utils import _sample_poses, TRUE_K, TRUE_D


@pytest.fixture(scope='module')
def control(tmp_path_factory):
    cv2.setNumThreads(1)
    obj = np.array([[x*.04, y*.04, 0.] for y in range(1,5) for x in range(1,7)], np.float32).reshape(-1,1,3)
    camera = CameraConfig(1280,720)
    frames = []
    for i,(rotation,translation) in enumerate(_sample_poses(36,214)):
        rv=cv2.Rodrigues(rotation)[0]
        pts=cv2.projectPoints(obj,rv,translation,TRUE_K,TRUE_D)[0]
        area,xy,tilt,edge=_compute_board_geometry(pts,(720,1280));fid=f'projected_{i:03d}'
        det=DetectionResult(fid,True,corners=pts,object_points=obj.copy(),num_corners=len(obj),
            board_area_ratio=area,board_center_px=xy,board_tilt_deg=tilt,min_edge_margin_px=edge)
        frames.append(Frame(ImageInfo(fid,'POINT_CONTROL',1280,720),det))
    data=Dataset(frames=frames)
    train_ids,test_ids=split_train_test(data,camera,.25,42)
    train=Dataset(frames=[f for f in frames if f.image_info.image_id in train_ids])
    results={name:fn(train,camera) for name,fn in [('brown',calibrate_brown_conrady),
        ('rational',calibrate_extended_pinhole),('fisheye',calibrate_fisheye)]}
    reports={name:asdict(compute_observability_report(result,train)) for name,result in results.items()}
    root=tmp_path_factory.mktemp('known_k_control')
    summary={name:{'success':r.success,'rms':r.rms_error,
        'K':r.camera_matrix.tolist() if r.success else None,'D':r.distortion.tolist() if r.success else None,
        'observability':reports[name], 'policy':assess_observability(reports[name],ReadinessConfig())}
        for name,r in results.items()}
    (root/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return data,camera,train_ids,results,reports


def test_near_ground_truth_brown_is_computable_but_requires_review(control):
    _,_,_,results,reports=control
    brown=results['brown'];assert brown.success
    np.testing.assert_allclose(brown.camera_matrix,TRUE_K,atol=.01,rtol=0)
    np.testing.assert_allclose(brown.distortion.ravel(),TRUE_D,atol=.001,rtol=0)
    assert reports['brown']['observability_score'] < 50.
    assessment=assess_observability(reports['brown'],ReadinessConfig())
    assert assessment['ready'] and assessment['review_required']


def test_preflight_uses_train_only_and_default_review_threshold(control,monkeypatch):
    data,camera,train_ids,results,_=control
    def fit(train,*_a,**_kw):
        assert [f.image_info.image_id for f in train.frames]==[f.image_info.image_id for f in data.frames if f.image_info.image_id in train_ids]
        return results['brown']
    monkeypatch.setattr('calibration.models.brown_conrady.calibrate_brown_conrady',fit)
    assessment=preflight_observability(data,camera,train_ids,ReadinessConfig())
    assert assessment['ready'] and assessment['review_required'] and assessment['warnings']


@pytest.mark.parametrize('model,columns',[('brown',9),('rational',12),('fisheye',8)])
def test_model_specific_diagnostics_are_retained(control,model,columns):
    _,_,_,results,reports=control
    assert results[model].success
    assert reports[model]['jacobian_cols']==columns
    assert reports[model]['rank'] <= columns
    assert reports[model]['max_abs_correlation'] is not None
