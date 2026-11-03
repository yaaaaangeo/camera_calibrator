from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from automation.identity import build_identity
from automation.live.config import load_live_profile, resolve_live_profile
from automation.live.controller import LiveController, State
from automation.live.readiness import check_readiness, preflight_observability
from calibration.types import DetectionResult, Dataset, CameraConfig


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def step(self, seconds=2): self.value += seconds


def profile(**sections):
    raw = {'live_profile_version': 1, **sections}
    return resolve_live_profile(raw)


def checker():
    return np.where((np.indices((120, 160)).sum(axis=0) // 12) % 2, 220, 50).astype(np.uint8)


def detection(dx=0, success=True):
    corners = np.array([[30+dx, 30], [60+dx, 30], [60+dx, 60], [30+dx, 60]], np.float32).reshape(-1, 1, 2)
    return DetectionResult('test', success, corners=corners,
                           object_points=np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], np.float32).reshape(-1,1,3),
                           ids=np.arange(4).reshape(-1,1), num_corners=4,
                           board_area_ratio=.1, board_center_px=(45+dx,45), board_tilt_deg=10.,
                           min_edge_margin_px=30., failure_reason=None if success else 'no board')


@pytest.fixture
def controller(tmp_path):
    clock = Clock()
    c = LiveController(profile(capture={'motion_blur_max': 1000.}), build_identity('TEST', 'SN'),
                       {'kind':'images', 'path':'test'}, output_root=tmp_path, clock=clock)
    yield c, clock
    c.close()


@pytest.mark.parametrize('section,key,value', [
    ('capture','interval_sec',float('nan')), ('capture','max_images',True),
    ('readiness','coverage_pct',101), ('readiness','check_every_new_frames',0),
    ('readiness','min_test_frames',0), ('readiness','min_images',36),
    ('recovery','max_calculations',0), ('capture','unknown',1)])
def test_invalid_profile_rejected(section,key,value):
    with pytest.raises(ValueError): profile(**{section:{key:value}})


def test_builtin_profile_and_p1_unchanged():
    from automation.profile import load_profile
    assert load_live_profile().p1.to_dict() | {'source_path':None} == load_profile('default').to_dict() | {'source_path':None}


def test_detect_failures_keep_capture_running(controller):
    c, _ = controller
    assert c.accept_frame(checker(), detection=detection(success=False)) == 'detection_failed'
    c.detect = lambda *_: (_ for _ in ()).throw(RuntimeError('detector failed'))
    assert c.accept_frame(checker()) == 'detection_error'
    assert c.state == State.CAPTURING and not c.paths


def test_novel_pose_duplicate_raw_preservation(controller):
    c, clock = controller
    image = checker(); before = image.copy()
    assert c.accept_frame(image, detection=detection()) == 'saved'
    clock.step()
    assert c.accept_frame(image, detection=detection()) == 'duplicate'
    clock.step()
    assert c.accept_frame(image, detection=detection(25)) == 'saved'
    assert np.array_equal(image, before)
    assert np.array_equal(cv2.imread(c.paths[0], cv2.IMREAD_GRAYSCALE), before)
    assert len(c.paths) == 2 and c.duplicates == 1


def test_interval_and_resolution_change(controller):
    c, _ = controller
    c.accept_frame(checker(), detection=detection())
    assert c.accept_frame(checker(), detection=detection(25)) == 'interval'
    assert c.accept_frame(np.zeros((121,160),np.uint8), detection=detection()) == 'resolution_changed'
    assert c.state == State.ERROR


@pytest.mark.parametrize('kind', ['blur', 'motion_blur', 'exposure_brightness', 'exposure_clipping', 'roi_contrast'])
def test_quality_rejections(controller,monkeypatch,kind):
    c, _ = controller
    image = checker()
    if kind == 'blur': image = cv2.GaussianBlur(image,(51,51),20)
    elif kind == 'motion_blur':
        c.profile.capture = replace(c.profile.capture,motion_blur_max=3.)
        monkeypatch.setattr('automation.live.controller.compute_motion_blur_score',lambda _:4.)
    elif kind == 'exposure_brightness': image = (image.astype(float)/20).astype(np.uint8)
    elif kind == 'exposure_clipping': image = np.where(image>100,255,170).astype(np.uint8)
    else:
        c.profile.p1.image_selection = replace(c.profile.p1.image_selection,blur_min_laplacian_var=0)
        image = np.full_like(image,110)
    if kind == 'exposure_brightness':
        c.profile.p1.image_selection = replace(c.profile.p1.image_selection,blur_min_laplacian_var=0)
        c.profile.capture = replace(c.profile.capture,roi_contrast_min=0)  # isolate brightness gate
    assert c.accept_frame(image,detection=detection()) == kind
    assert not c.paths


def test_stream_loss_and_recovery_are_bounded(controller):
    c, clock = controller
    clock.step(6)
    assert c.stream_check() and c.state == State.CONNECTING
    c.accept_frame(checker(), detection=detection())
    assert c.state == State.CAPTURING and c.connection == 'connected'
    c.profile.capture = replace(c.profile.capture,max_reconnects=1)
    clock.step(6)
    assert not c.stream_check() and c.state == State.ERROR


def test_checkpoint_lock_recovery_and_hash_validation(controller):
    c, _ = controller
    c.accept_frame(checker(), detection=detection())
    with pytest.raises(RuntimeError,match='already in use'):
        LiveController(c.profile,c.identity,c.source,resume=c.session)
    c.cancel(); c.close()
    recovered = LiveController(c.profile,c.identity,c.source,resume=c.session)
    assert recovered.paths == c.paths and recovered.processed == 1
    recovered.close()
    Path(c.paths[0]).write_bytes(b'changed')  # only test-owned capture; source remains untouched
    with pytest.raises(ValueError,match='changed'):
        LiveController(c.profile,c.identity,c.source,resume=c.session)


def test_mismatched_identity_cannot_resume(controller):
    c, _ = controller; c.close()
    with pytest.raises(ValueError,match='identity'):
        LiveController(c.profile,build_identity('OTHER','SN'),c.source,resume=c.session)


def prepare_task(c, kind='calibration'):
    c.accept_frame(checker(), detection=detection())
    c.readiness={'train_ids':[c.dataset.frames[0].image_info.image_id]}
    c.state = State.CALIBRATING if kind=='calibration' else State.CHECKING_READINESS
    c.calculations=1
    return c.request_task(kind)


@pytest.mark.parametrize('verdict,kind,recs,expected', [
    ('PASS', [], [], State.COMPLETED),
    ('WARN', ['criteria_missing'], ['more coverage'], State.COMPLETED),
    ('WARN', ['evidence'], ['more coverage'], State.CAPTURING),
    ('FAIL', ['quality'], ['more coverage'], State.CAPTURING),
    ('FAIL', ['invalid_result'], ['more coverage'], State.ERROR),
    ('FAIL', ['invalid_result', 'criteria_missing'], ['more coverage'], State.ERROR),
    ('WARN', ['quality'], [], State.COMPLETED)])
def test_result_followup_separates_capture_remedies(controller,verdict,kind,recs,expected):
    c,_=controller; task=prepare_task(c)
    response={'task_id':task['task_id'],'outcome':{'status':'completed','verdict':verdict,'message':'test'},
              'acceptance':{'criteria_complete':'criteria_missing' not in kind,'warn_kinds':kind if verdict=='WARN' else [],
                            'fail_kinds':kind if verdict=='FAIL' else [],'reasons':['test']},
              'capture_recommendations':recs}
    paths=list(c.paths)
    c.finish_task(response)
    assert c.state==expected and c.paths==paths
    if expected==State.CAPTURING: assert c.recaptures==1


def test_calculation_failure_is_not_blind_recapture(controller):
    c,_=controller; task=prepare_task(c)
    c.finish_task({'task_id':task['task_id'],'outcome':{'status':'failed','verdict':'FAIL','message':'process failed'}})
    assert c.state==State.ERROR and c.recaptures==0


def test_iteration_limits(controller):
    c,_=controller; task=prepare_task(c)
    c.recaptures=c.profile.recovery.max_recaptures
    c.finish_task({'task_id':task['task_id'],'outcome':{'status':'completed','verdict':'WARN','message':''},
                   'acceptance':{'criteria_complete':True,'warn_kinds':['quality']},
                   'capture_recommendations':['move board']})
    assert c.state==State.ERROR and c.guidance


def test_observability_low_and_stale_task(controller):
    c,_=controller; task=prepare_task(c,'observability')
    with pytest.raises(ValueError,match='Stale'):
        c.finish_task({'task_id':'wrong'})
    c.finish_task({'task_id':task['task_id'],'observability':{'ready':False,'reasons':['need tilt']}})
    assert c.state==State.CAPTURING and c.guidance==['need tilt']


def test_max_calculation_limit(controller):
    c,_=controller; task=prepare_task(c,'observability')
    c.calculations=c.profile.recovery.max_calculations
    c.finish_task({'task_id':task['task_id'],'observability':{'ready':True,'reasons':[]}})
    assert c.state==State.ERROR


def test_saved_observability_response_recovers_without_recomputation(controller):
    from automation.run_manifest import atomic_write_json
    c,_=controller; task=prepare_task(c,'observability')
    atomic_write_json(task['response'],{'task_id':task['task_id'],'observability':{'ready':False,'reasons':['need tilt']}})
    c.close()
    recovered=LiveController(c.profile,c.identity,c.source,resume=c.session)
    assert recovered.state==State.CAPTURING and recovered.guidance==['need tilt']
    recovered.close()


def test_save_failure_is_not_a_successful_capture(controller,monkeypatch):
    c,_=controller
    monkeypatch.setattr('automation.live.controller.cv2.imencode',lambda *_:(False,None))
    with pytest.raises(OSError): c.accept_frame(checker(),detection=detection())
    assert not c.paths


def test_readiness_is_not_count_only(controller):
    c,clock=controller
    for i in range(3):
        c.accept_frame(checker(),detection=detection(i*15)); clock.step()
    p=profile(readiness={'min_images':3})
    result,_=check_readiness(c.dataset,c.camera,c.pattern,p)
    assert not result['ready']
    assert result['holdout']['status']=='insufficient_evidence'
    assert len(result['reasons'])>=3


def test_observability_failed_preliminary_fit_is_not_ready(controller,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('calibration.models.brown_conrady.calibrate_brown_conrady',
                        lambda *_a,**_k:SimpleNamespace(success=False,error_message='bad fit'))
    c,_=controller
    result=preflight_observability(Dataset(),CameraConfig(160,120),[],c.profile.readiness)
    assert not result['ready'] and result['error']=='bad fit'


def test_ros_source_latest_only_and_reconnect(monkeypatch):
    from automation.live.sources import RosSource
    class Subscriber:
        def start(self,topic,msg_type,on_frame,on_error):
            for i in range(20): on_frame(np.full((2,2,3),i,np.uint8),i)
        def stop(self): pass
    monkeypatch.setattr('calibration.ros_live.LiveTopicSubscriber',Subscriber)
    source=RosSource('/cam','sensor_msgs/msg/Image'); source.start()
    image,stamp=source.poll()
    assert stamp==19 and image[0,0,0]==19
    assert source.buffer.stats().replaced==19 and source.poll() is None
    source.stop(); source.start(); assert source.poll() is not None; source.stop()


def test_final_input_can_check_below_cadence_once(controller,monkeypatch):
    c,clock=controller
    for i in range(3):
        c.accept_frame(checker(),detection=detection(i*15)); clock.step()
    c.profile.readiness=replace(c.profile.readiness,min_images=3,check_every_new_frames=5)
    c.last_checked_count=2; calls=[]
    def check(*_):
        calls.append(True)
        return {'ready':False,'reasons':['more edge data'],'coaching':[]},c.dataset
    monkeypatch.setattr('automation.live.controller.check_readiness',check)
    assert c.maybe_check() is None and not calls
    assert c.maybe_check(force=True) is None and len(calls)==1
    assert c.maybe_check(force=True) is None and len(calls)==1


def test_existing_rational_model_alias_is_supported():
    p=profile(p1={'calibration':{'models':['pinhole','brown_conrady','extended_pinhole','fisheye']}})
    assert p.p1.calibration['models'][-1]=='fisheye'
