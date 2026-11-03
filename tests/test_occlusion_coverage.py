import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from calibration.occlusion import resolve_mask, observable_coverage, validate_binding, mask_metadata
from calibration.quality import compute_coverage_grid, coverage_percentage
from calibration.types import CameraConfig, Dataset, Frame, ImageInfo, DetectionResult
from automation.identity import build_identity
from automation.live.config import resolve_live_profile


def mask(y=12):
    return {'schema_version':1,'version':'test-v1','source':'synthetic known occluder',
            'camera':{'model':'TEST','serial':'S1'},'width':16,'height':16,
            'coordinate_system':'native_distorted_pixel_edges',
            'polygons':[[[0,y],[16,y],[16,16],[0,16]]]}


def data(points=None):
    if points is None:points=[(2+4*c,2+4*r) for r in range(3) for c in range(4)]
    return Dataset(frames=[Frame(ImageInfo('f','f.png',16,16),DetectionResult(
        image_id='f',success=True,corners=np.array(points,dtype=np.float32).reshape(-1,1,2),num_corners=len(points)))])


def test_full_frame_unchanged_and_visible_area_is_separate():
    d=data();camera=CameraConfig(16,16)
    before=coverage_percentage(compute_coverage_grid(d,camera))
    result=observable_coverage(d,camera,resolve_mask(mask()))
    assert before==75 and coverage_percentage(compute_coverage_grid(d,camera))==75
    assert result['coverage_pct']==100 and result['visible_area_fraction']==.75
    assert result['required_corner_cells']==[[0,0],[0,3],[2,0],[2,3]]
    assert not result['missing_corner_cells'] and not result['accuracy_validated']


def test_crossing_cells_retain_their_visible_area_and_require_observations():
    result=observable_coverage(data([(2+4*c,2+4*r) for r in range(2) for c in range(4)]),CameraConfig(16,16),resolve_mask(mask(10)))
    assert result['coverage_pct']==80
    assert all(c['visible_fraction']==.5 for c in result['grid'][8:12])
    assert all(not c['sufficient'] for c in result['grid'][8:12])
    assert result['missing_corner_cells']==[[2,0],[2,3]]


def test_masked_corners_do_not_inflate_observable_counts():
    result=observable_coverage(data([(2,2),(2,14)]),CameraConfig(16,16),resolve_mask(mask()))
    assert result['visible_corner_count']==1


def test_overlapping_polygons_are_unioned_and_winding_does_not_matter():
    m=mask();m['polygons'] += [list(reversed(m['polygons'][0]))]
    assert observable_coverage(data(),CameraConfig(16,16),resolve_mask(m))['visible_area_fraction']==.75


@pytest.mark.parametrize('change',[
    {'width':True}, {'height':0}, {'schema_version':2}, {'version':''}, {'source':''},
    {'coordinate_system':'normalized'}, {'camera':{'model':'T'}}, {'polygons':[]},
    {'polygons':[[[0,0],[16,0],[16,16],[0,16]]]},
    {'polygons':[[[0,0],[17,0],[0,16]]]},
    {'polygons':[[[0,0],[float('nan'),1],[0,16]]]},
    {'polygons':[[[0,0],[1,1],[2,2]]]},
    {'polygons':[[[0,0],[16,16],[0,16],[16,0]]]},
])
def test_invalid_masks_fail_closed(change):
    m=mask();m.update(change)
    with pytest.raises(ValueError):resolve_mask(m)


def test_binding_and_resolution_do_not_implicitly_scale():
    m=resolve_mask(mask())
    validate_binding(m,16,16,build_identity('TEST','S1'))
    with pytest.raises(ValueError):validate_binding(m,32,32)
    with pytest.raises(ValueError):validate_binding(m,identity=build_identity('TEST','S2'))
    with pytest.raises(ValueError):observable_coverage(data(),CameraConfig(32,32),m)


def test_profile_fingerprint_binds_mask_version_and_provenance():
    p=resolve_live_profile({'live_profile_version':1})
    m=mask()
    a=resolve_live_profile({'live_profile_version':1,'coverage_mask':m})
    b=copy.deepcopy(m);b['version']='test-v2'
    c=resolve_live_profile({'live_profile_version':1,'coverage_mask':b})
    assert len({p.fingerprint,a.fingerprint,c.fingerprint})==3
    assert mask_metadata(a.coverage_mask)['sha256']!=mask_metadata(c.coverage_mask)['sha256']


def test_controller_records_mask_and_rejects_resolution_before_capture(tmp_path):
    from automation.live.controller import LiveController, State
    p=resolve_live_profile({'live_profile_version':1,'coverage_mask':mask()})
    with pytest.raises(ValueError):LiveController(p,build_identity('OTHER','S1'),{},output_root=tmp_path/'bad')
    c=LiveController(p,build_identity('TEST','S1'),{},output_root=tmp_path/'ok')
    try:
        saved=json.loads((c.session/'p2_manifest.json').read_text(encoding='utf8'))
        assert saved['coverage_mask']==mask_metadata(p.coverage_mask)
        assert c.accept_frame(np.zeros((32,32,3),dtype=np.uint8))=='mask_resolution_mismatch'
        assert c.state==State.ERROR and not c.paths
    finally:c.close()


def test_readiness_uses_mask_only_for_spatial_gate_and_keeps_holdout(monkeypatch):
    from types import SimpleNamespace
    import automation.live.readiness as module
    from calibration.types import DiversityScores, HoldoutEvidenceGate
    frames=[]
    for i in range(8):
        f=copy.deepcopy(data().frames[0]);f.image_info.image_id=str(i);f.image_info.path=f'{i}.png'
        f.detection.board_center_px=((i%3+.5)*16/3,(i//3+.5)*16/3)
        frames.append(f)
    dataset=Dataset(frames=frames)
    def analyze(d,c):
        d.coverage_grid=compute_coverage_grid(d,c);d.diversity=DiversityScores(1.,1.,1.,1.)
        return ['Capture hidden bottom cells']
    monkeypatch.setattr(module,'analyze_dataset_quality',analyze)
    monkeypatch.setattr(module,'compute_frame_quality_scores',lambda *a,**k:None)
    monkeypatch.setattr(module,'select_calibration_images',lambda *a:SimpleNamespace(
        selected_paths=[f.image_info.path for f in frames],counts={}))
    monkeypatch.setattr(module,'split_train_test',lambda *a: (['0','1','2','3'],['6','7']))
    monkeypatch.setattr(module,'evaluate_holdout_evidence',lambda *a,**k:HoldoutEvidenceGate(status='sufficient'))
    plain=resolve_live_profile({'live_profile_version':1,'readiness':{'min_images':3}})
    masked=resolve_live_profile({'live_profile_version':1,'readiness':{'min_images':3},'coverage_mask':mask()})
    a,_=module.check_readiness(dataset,CameraConfig(16,16),None,plain)
    b,_=module.check_readiness(dataset,CameraConfig(16,16),None,masked)
    assert not a['ready'] and b['ready']
    assert b['full_frame_coverage_pct']==a['coverage_pct']==75
    assert b['coverage_pct']==100 and not b['coaching']
    assert a['holdout']==b['holdout'] and a['train_ids']==b['train_ids']
    # A mask cannot bypass hold-out failure or trigger a fit here.
    monkeypatch.setattr(module,'evaluate_holdout_evidence',lambda *a,**k:HoldoutEvidenceGate(status='insufficient_evidence',reasons=['holdout failed']))
    b,_=module.check_readiness(dataset,CameraConfig(16,16),None,masked)
    assert not b['ready'] and 'holdout failed' in b['reasons']


def test_changed_mask_cannot_resume_existing_session(tmp_path):
    from automation.live.controller import LiveController
    p=resolve_live_profile({'live_profile_version':1,'coverage_mask':mask()})
    c=LiveController(p,build_identity('TEST','S1'),{},output_root=tmp_path)
    session=c.session;c.close()
    m=mask();m['version']='new-mount-v2'
    changed=resolve_live_profile({'live_profile_version':1,'coverage_mask':m})
    with pytest.raises(ValueError,match='identical profile'):
        LiveController(changed,build_identity('TEST','S1'),{},resume=session)


def test_preflight_checks_mask_binding_without_writing(tmp_path):
    from tests.test_experiment import profile
    from tests._publish_test_utils import snapshot
    from automation.experiment.preflight import preflight
    m=mask();m.update(width=640,height=480,polygons=[[[0,360],[640,360],[640,480],[0,480]]])
    m['camera']['serial']='SN1'
    exp,_=profile(tmp_path,live={'live_profile_version':1,'coverage_mask':m})
    before=snapshot(tmp_path)
    checks=preflight(exp,dry_run=True)['checks']
    assert any(c['check']=='coverage_mask' and c['status']=='PASS' for c in checks)
    assert snapshot(tmp_path)==before
    exp.live.coverage_mask['camera']['serial']='OTHER'
    assert preflight(exp,dry_run=True)['status']=='FAIL'


def test_gui_distinguishes_full_frame_and_observable(tmp_path,pattern_config):
    from PySide6.QtWidgets import QApplication
    from ui.automatic_calibration_dialog import AutomaticCalibrationDialog
    app=QApplication.instance() or QApplication([])
    dialog=AutomaticCalibrationDialog(tmp_path,pattern_config)
    try:
        dialog.display({'state':'CAPTURING','readiness':{'coverage_basis':'observable',
            'full_frame_coverage_pct':75.,'coverage_pct':92.37,'center_cells':6,'diversity':{}}})
        text=dialog.details.toPlainText()
        assert 'Full-frame Coverage 75.0%' in text and 'Observable Coverage 92.4%' in text
        assert '정확도 검증 아님' in text
    finally:dialog.close()
