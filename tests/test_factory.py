import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from automation.factory.provider import FileProvider, FactoryError, ORDERS, SensingSDKProvider
from automation.factory.compare import compare_project, compare_session, seeded_fit
from calibration.types import (CalibrationProject, CameraConfig, CameraModelType, Dataset,
    DetectionResult, Frame, FrameStatus, ImageInfo, PatternConfig, PatternType, ValidationResult)


def envelope():
    return dict(schema_version=1, camera_model='SG3S-ISX031C', serial_number='SN1',
        width=640, height=480, distortion_model='brown_conrady', coefficient_order=ORDERS['brown_conrady'],
        K=[[800.,0,320],[0,805.,240],[0,0,1]], D=[-.12,.04,.001,-.002,.01],
        source='Synthetic control, NOT manufacturer data', projection_convention='opencv',
        coordinate_system=dict(reference_resolution=[640,480],roi=[0,0,640,480],binning=[1,1],orientation='identity'))


def load(tmp_path, value=None):
    p=tmp_path/'reference.json';p.write_text(json.dumps(value or envelope()),encoding='utf8')
    return FileProvider(p).lookup('SG3S-ISX031C','SN1')


def project(noise=0):
    m=envelope();K=np.array(m['K']);D=np.array(m['D'])
    obj=np.array([[x*.04,y*.04,0] for y in range(6) for x in range(8)],np.float32).reshape(-1,1,3)
    rng=np.random.default_rng(47);frames=[]
    for i in range(18):
        r=rng.uniform(-.5,.5,3);t=np.array([rng.uniform(-.16,.01),rng.uniform(-.1,.01),rng.uniform(.65,1.2)])
        img,_=cv2.projectPoints(obj,r,t,K,D)
        img=(img+rng.normal(0,noise,img.shape)).astype(np.float32)
        fid=f'f{i}'
        frames.append(Frame(ImageInfo(fid,f'{fid}.png',640,480),DetectionResult(fid,True,corners=img,object_points=obj.copy(),num_corners=len(obj)),status=FrameStatus.DETECTED))
    model=CameraModelType.BROWN_CONRADY
    return CalibrationProject('control',CameraConfig(640,480),PatternConfig(PatternType.CHESSBOARD,8,6,.04),Dataset(frames=frames),
        validation_results={model:ValidationResult(train_frame_ids=[f'f{i}' for i in range(12)],test_frame_ids=[f'f{i}' for i in range(12,18)])})


@pytest.mark.parametrize('key,value',[
    ('schema_version',2),('serial_number','OTHER'),('camera_model','OTHER'),('width',0),('width',640.),
    ('source',''),('distortion_model','unknown'),('coefficient_order',['k1','k2','k3','p1','p2']),
    ('K',[[0,0,320],[0,805,240],[0,0,1]]),('K',[[800,1,320],[0,805,240],[0,0,1]]),
    ('K',[[800,0,640],[0,805,240],[0,0,1]]),('D',[0]*4),('D',[0,0,0,0,float('nan')]),
    ('D',[0,0,0,0,float('inf')]),('K',[[1]]),('coordinate_system',None),
])
def test_reject_invalid_reference(tmp_path,key,value):
    m=envelope();m[key]=value
    with pytest.raises(FactoryError):load(tmp_path,m)


@pytest.mark.parametrize('key,value',[('roi',[1,0,639,480]),('binning',[2,2]),('orientation','flip'),('reference_resolution',[1280,960])])
def test_reject_coordinate_conversion(tmp_path,key,value):
    m=envelope();m['coordinate_system'][key]=value
    with pytest.raises(FactoryError):load(tmp_path,m)


def test_json_and_yaml(tmp_path):
    r=load(tmp_path);np.testing.assert_array_equal(r.D,envelope()['D'])
    path=tmp_path/'intrinsic.yaml';fs=cv2.FileStorage(str(path),cv2.FILE_STORAGE_WRITE)
    for key,value in [('camera_matrix',r.K),('distortion_coefficients',r.D),('image_width',640),('image_height',480),('calibration_model','brown_conrady')]:fs.write(key,value)
    fs.release();m=envelope();del m['K'];del m['D'];m['intrinsic_file']='intrinsic.yaml'
    other=load(tmp_path,m);np.testing.assert_array_equal(other.K,r.K);np.testing.assert_array_equal(other.D,r.D)


def test_sdk_unavailable():
    with pytest.raises(FactoryError,match='SDK_UNAVAILABLE'):SensingSDKProvider().lookup('SG8-ISX028C','SN1')


def test_fixed_local_seeded_known_truth_and_no_mutation(tmp_path):
    p=project();r=load(tmp_path);before=copy.deepcopy(p);K=r.K.copy();D=r.D.copy()
    report=compare_project(p,r,CameraModelType.BROWN_CONRADY,seeded=True)
    assert not report['failures'];assert report['conclusion']=='UNCLEAR'
    assert set(report['candidates'])=={'Local','Factory-fixed','Factory-seeded'}
    for side in report['candidates'].values():
        assert side['success'] and side['test_rms']<.001
        assert set(side['per_frame_error'])==set(report['test_frame_ids'])
        assert side['residual_stats']['p99']<.001
    np.testing.assert_array_equal(report['candidates']['Factory-fixed']['camera_matrix'],K)
    np.testing.assert_array_equal(report['candidates']['Factory-fixed']['distortion'],D)
    np.testing.assert_array_equal(r.K,K);np.testing.assert_array_equal(r.D,D)
    for a,b in zip(p.dataset.frames,before.dataset.frames):
        np.testing.assert_array_equal(a.detection.corners,b.detection.corners)
        assert a.reprojection_error==b.reprojection_error and a.status==b.status


def test_train_only_independent_of_test_corners(tmp_path):
    p=project();r=load(tmp_path)
    first=seeded_fit(p.dataset,p.camera_config,[f'f{i}' for i in range(12)],r)
    for f in p.dataset.frames[12:]:f.detection.corners[:]+=100
    second=seeded_fit(p.dataset,p.camera_config,[f'f{i}' for i in range(12)],r)
    for a,b in zip(first,second):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


@pytest.mark.parametrize('kind',['overlap','missing','duplicate','alias'])
def test_no_split_leakage(tmp_path,kind):
    p=project();v=p.validation_results[CameraModelType.BROWN_CONRADY]
    if kind=='overlap':v.test_frame_ids[0]='f0'
    if kind=='missing':v.test_frame_ids[0]='absent'
    if kind=='duplicate':v.test_frame_ids.append('f12')
    if kind=='alias':p.dataset.frames[12].detection=copy.deepcopy(p.dataset.frames[0].detection)
    with pytest.raises(FactoryError):compare_project(p,load(tmp_path),CameraModelType.BROWN_CONRADY)


def test_bad_factory_local_better(tmp_path):
    p=project();m=envelope();m['D']=[.5,-.3,.02,-.02,.2]
    report=compare_project(p,load(tmp_path,m),CameraModelType.BROWN_CONRADY)
    assert report['candidates']['Local']['test_rms']<.001
    assert report['candidates']['Factory-fixed']['test_rms']>.1
    assert report['pairwise']['Local']['bootstrap']['improvement_ci_low']>0
    # These center-only views provide no edge evidence: do not overclaim.
    assert report['conclusion']=='UNCLEAR'


def test_failed_seed_retained_incomparable(tmp_path,monkeypatch):
    def fail(*args):raise FactoryError('deliberate optimizer failure')
    monkeypatch.setattr('automation.factory.compare.seeded_fit',fail)
    r=compare_project(project(),load(tmp_path),CameraModelType.BROWN_CONRADY,seeded=True)
    assert r['conclusion']=='INCOMPARABLE';assert 'Factory-seeded' in r['failures']
    assert r['candidates']['Factory-fixed']['success']


def test_seed_uses_both_k_and_d_train_only(tmp_path,monkeypatch):
    r=load(tmp_path);p=project()
    original=cv2.calibrateCamera
    def capture(obj,img,size,K,D,**kw):
        assert len(obj)==12;np.testing.assert_array_equal(K,r.K);np.testing.assert_array_equal(D,r.D)
        assert kw['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS
        return original(obj,img,size,K,D,**kw)
    monkeypatch.setattr(cv2,'calibrateCamera',capture)
    seeded_fit(p.dataset,p.camera_config,[f'f{i}' for i in range(12)],r)


@pytest.mark.parametrize('model',['extended_pinhole','fisheye'])
def test_model_specific_projection_and_seed(tmp_path,model):
    m=envelope();m['distortion_model']=model;m['coefficient_order']=ORDERS[model]
    m['D']=([-.12,.04,.001,-.002,.01,.002,.001,.0001] if model=='extended_pinhole' else [-.1,.02,.001,.0001])
    r=load(tmp_path,m);p=project();rng=np.random.default_rng(47)
    for f in p.dataset.frames:
        rv=rng.uniform(-.5,.5,3);tv=np.array([rng.uniform(-.16,.01),rng.uniform(-.1,.01),rng.uniform(.65,1.2)])
        obj=f.detection.object_points.astype(float)
        if model=='fisheye':img,_=cv2.fisheye.projectPoints(obj,rv,tv,r.K,r.D)
        else:img,_=cv2.projectPoints(obj,rv,tv,r.K,r.D)
        f.detection.corners=img.astype(np.float32)
        rng.normal(0,0,img.shape)  # preserve deterministic pose sequence
    p.validation_results={r.model:p.validation_results[CameraModelType.BROWN_CONRADY]}
    report=compare_project(p,r,r.model,seeded=True)
    assert report['candidates']['Factory-fixed']['test_rms']<.001
    assert report['candidates']['Factory-seeded']['success']
    assert report['candidates']['Factory-seeded']['test_rms']<.001


def session_fixture(tmp_path):
    from tests._publish_test_utils import make_session
    from calibration.project_io import save_project
    from calibration.types import FinalResult,CalibrationResult
    s=make_session(tmp_path/'session',serial='SN1',camera_model='SG3S-ISX031C')
    (s/'project').mkdir();p=project();model=CameraModelType.BROWN_CONRADY
    p.final_result=FinalResult(model,CalibrationResult(model,success=True))
    save_project(p,str(s/'project/calibration.ccproj'))
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(envelope()),encoding='utf8')
    return s,ref


def test_session_reports_readonly_and_local_bundle_history(tmp_path):
    from tests._publish_test_utils import snapshot
    from automation.publish.bundle import build_bundle,verify_bundle
    from automation.publish.history import list_history
    s,ref=session_fixture(tmp_path);before=snapshot(s)
    out,report=compare_session(s,ref,coordinates=envelope()['coordinate_system'],seeded=True,output_root=tmp_path/'comparisons')
    assert snapshot(s)==before
    assert (out/'factory_comparison.html').is_file()
    dest=tmp_path/'results'/'items'/'one'/'bundle'
    m=build_bundle(s,dest,full_frame=True,factory_report=out/'factory_comparison.json')
    assert verify_bundle(dest)==m
    assert m['summary']['factory_comparison']['conclusion']==report['conclusion']
    assert list_history(tmp_path/'results','SN1')[0]['factory_comparison']['authenticated'] is False
    ordinary=build_bundle(s,tmp_path/'ordinary',full_frame=True)
    assert ordinary['content_hash']!=m['content_hash']
    assert snapshot(s)==before


def test_bundle_rejects_other_session_factory_report(tmp_path):
    from automation.publish.bundle import build_bundle
    from automation.publish.base import PublishError
    s,ref=session_fixture(tmp_path)
    out,report=compare_session(s,ref,coordinates=envelope()['coordinate_system'])
    report['session_manifest_sha256']='0'*64
    target=out/'factory_comparison.json';target.write_text(json.dumps(report),encoding='utf8')
    with pytest.raises(PublishError,match='FACTORY_SESSION_MISMATCH'):
        build_bundle(s,tmp_path/'bundle',factory_report=target)


def test_cli_lookup_and_sdk_error(tmp_path,capsys):
    from automation.cli import main
    p=tmp_path/'reference.json';p.write_text(json.dumps(envelope()),encoding='utf8')
    assert main(['factory','lookup','--serial','SN1','--camera-model','SG3S-ISX031C','--factory',str(p)])==0
    assert json.loads(capsys.readouterr().out)['authenticated'] is False
    assert main(['factory','lookup','--serial','SN1','--camera-model','SG8-ISX028C'])==1
    assert 'SDK_UNAVAILABLE' in capsys.readouterr().err


def test_post_hook_failure_does_not_change_p1_result(tmp_path,monkeypatch,capsys):
    from automation.cli import main
    from automation.runner import RunOutcome
    s,_=session_fixture(tmp_path)
    monkeypatch.setattr('automation.runner.run_automation',lambda *a,**k:RunOutcome(3,'RUN-1','completed','WARN',str(s),'review'))
    assert main(['run','--profile','default','--images','x','--serial','SN1','--camera-model','SG3S-ISX031C','--factory','missing'])==3
    assert 'local_result_preserved' in capsys.readouterr().out


def test_true_factory_beats_noisy_train_fit_without_overclaim(tmp_path):
    p=project();rng=np.random.default_rng(99)
    for f in p.dataset.frames[:12]:f.detection.corners+=rng.normal(0,1,f.detection.corners.shape).astype(np.float32)
    report=compare_project(p,load(tmp_path),CameraModelType.BROWN_CONRADY)
    assert report['candidates']['Factory-fixed']['test_rms']<.001
    assert report['candidates']['Local']['test_rms']>.01
    assert report['conclusion']=='UNCLEAR'  # missing edge evidence


@pytest.mark.parametrize('winner',['Factory-fixed','Local','Factory-seeded'])
def test_conclusion_requires_paired_and_edge_evidence(tmp_path,monkeypatch,winner):
    from calibration.external_compare import ComparisonSide
    def evaluate(dataset,camera,pattern,test,K,D,model,label):
        value=.1 if label==winner else 1.
        from calibration.external_compare import PointErrorDetail
        details=[PointErrorDetail(fid,i,0,0,value) for fid in test for i in range(48)]
        return ComparisonSide(label,model_name=model,camera_matrix=K,distortion=D,test_rms=value,
            edge_rms=value,straightness_residual=value,per_frame_error={fid:value for fid in test},
            point_error_details=details,success=True)
    monkeypatch.setattr('automation.factory.compare._evaluate_side',evaluate)
    result=compare_project(project(),load(tmp_path),CameraModelType.BROWN_CONRADY,seeded=True)
    assert result['conclusion']=={'Factory-fixed':'FACTORY_SUPERIOR','Local':'LOCAL_SUPERIOR','Factory-seeded':'SEEDED_SUPERIOR'}[winner]


def test_partial_test_failure_never_ranks_survivors(tmp_path,monkeypatch):
    from automation.factory import compare
    original=compare._evaluate_side
    def partial(*a):
        side=original(*a)
        if side.label=='Factory-fixed':side.failed_frame_ids=[next(iter(side.per_frame_error))]
        return side
    monkeypatch.setattr(compare,'_evaluate_side',partial)
    result=compare_project(project(),load(tmp_path),CameraModelType.BROWN_CONRADY)
    assert result['conclusion']=='INCOMPARABLE' and 'Factory-fixed' in result['failures']


def test_cli_session_compare_and_output_restriction(tmp_path,capsys):
    from automation.cli import main
    s,ref=session_fixture(tmp_path);coords=tmp_path/'coordinates.json';coords.write_text(json.dumps(envelope()['coordinate_system']))
    assert main(['factory','compare','--session',str(s),'--factory',str(ref),'--coordinates',str(coords),'--output-root',str(tmp_path/'reports')])==0
    assert json.loads(capsys.readouterr().out)['conclusion']=='UNCLEAR'
    with pytest.raises(FactoryError,match='outside original Session'):
        compare_session(s,ref,coordinates=envelope()['coordinate_system'],output_root=s/'new')


def test_broken_file_and_missing_parameters(tmp_path):
    path=tmp_path/'bad';path.write_text('{')
    with pytest.raises(FactoryError):FileProvider(path).lookup('x','y')
    value=envelope();del value['K']
    with pytest.raises(FactoryError):load(tmp_path,value)


def test_reject_undeclared_vendor_projection(tmp_path):
    value=envelope();del value['projection_convention']
    with pytest.raises(FactoryError,match='projection convention'):load(tmp_path,value)


def test_factory_bundle_dedup_ignores_binding_timestamp_hash(tmp_path):
    from automation.publish.bundle import build_bundle
    s,ref=session_fixture(tmp_path)
    out,report=compare_session(s,ref,coordinates=envelope()['coordinate_system'])
    path=out/'factory_comparison.json'
    first=build_bundle(s,tmp_path/'first',factory_report=path)
    manifest=s/'run_manifest.json';value=json.loads(manifest.read_text());value['finished_at']='2026-11-02T04:00:00+00:00'
    manifest.write_text(json.dumps(value),encoding='utf8')
    report['session_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    path.write_text(json.dumps(report),encoding='utf8')
    second=build_bundle(s,tmp_path/'second',factory_report=path)
    assert first['content_hash']==second['content_hash']
    assert first['source_file_sha256']['factory_comparison.json']!=second['source_file_sha256']['factory_comparison.json']


def test_p2_factory_failure_preserves_completed_state(tmp_path,monkeypatch,capsys):
    import automation.live.cli as live
    from automation.live.controller import State
    from automation.cli import main
    s,_=session_fixture(tmp_path)
    class Controller:
        def __init__(self,*a,**k):
            self.state=State.COMPLETED;self.attempts=[{'outcome':{'session_dir':str(s)}}]
    monkeypatch.setattr(live,'LiveController',Controller)
    monkeypatch.setattr(live,'ImageReplay',lambda *a:None)
    monkeypatch.setattr(live,'run',lambda *a,**k:0)
    assert main(['live','--serial','SN1','--camera-model','SG3S-ISX031C','--replay','x','--factory','absent'])==0
    assert 'local_result_preserved' in capsys.readouterr().out


def test_optional_parameter_diagnostic_failure_is_explicit(tmp_path,monkeypatch):
    def fail(*a):raise RuntimeError('diagnostic unavailable')
    monkeypatch.setattr('automation.factory.compare.build_parameter_diagnostics',fail)
    report=compare_project(project(),load(tmp_path),CameraModelType.BROWN_CONRADY)
    assert not report['failures']
    assert all(v['parameter_diagnostics']['status']=='UNAVAILABLE' for v in report['candidates'].values())
