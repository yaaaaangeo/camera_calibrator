"""Sampling/format/export readiness. No Git commit/push in this module."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from automation.live.sources import BagReplay
from automation.experiment.config import load_experiment
from automation.experiment.preflight import preflight
from tests.test_experiment import profile
from tests.test_rosbag_reader import synthetic_bag_path


def test_ros1_timestamp_fps_and_backward_compatibility(synthetic_bag_path):
    source=BagReplay(synthetic_bag_path,'/camera/image_raw',extract_fps=2)
    source.start();times=[]
    try:
        while not source.exhausted:
            frame=source.poll()
            if frame is not None:times.append(frame[1])
    finally:source.stop()
    np.testing.assert_allclose(times,[0,.5,1,1.5,2,2.5],rtol=0,atol=1e-9)


@pytest.mark.parametrize('fps',[0,-1,float('nan'),float('inf'),True,'2'])
def test_invalid_fps_rejected(tmp_path,fps):
    with pytest.raises(ValueError):BagReplay(tmp_path/'not-read.bag','/camera',extract_fps=fps)
    with pytest.raises(ValueError):profile(tmp_path,input={'kind':'bag','path':'not-read.bag','topic':'/camera','extract_fps':fps})


def write_ros2(path,plugin):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores,get_typestore
    store=get_typestore(Stores.ROS2_HUMBLE);t=store.types
    Image=t['sensor_msgs/msg/Image'];Header=t['std_msgs/msg/Header'];Time=t['builtin_interfaces/msg/Time']
    pixels=np.full((48,64,3),100,np.uint8)
    with Writer(path,version=8,storage_plugin=plugin) as writer:
        connection=writer.add_connection('/camera',Image.__msgtype__,typestore=store)
        for i in range(6):
            msg=Image(header=Header(stamp=Time(sec=0,nanosec=i*100000000),frame_id='c'),height=48,width=64,
                encoding='bgr8',is_bigendian=0,step=192,data=pixels.ravel())
            writer.write(connection,i*100000000,store.serialize_cdr(msg,Image.__msgtype__))


@pytest.mark.parametrize('kind',['db3','mcap'])
def test_ros2_storage_formats_topic_fps_and_native_resolution(tmp_path,kind):
    from rosbags.rosbag2.enums import StoragePlugin
    folder=tmp_path/'recording';write_ros2(folder,StoragePlugin.SQLITE3 if kind=='db3' else StoragePlugin.MCAP)
    chosen=next(folder.glob('*.'+kind))
    exp,_=profile(tmp_path,input={'kind':'bag','path':str(chosen),'topic':'/camera','extract_fps':2},
        live={'live_profile_version':1,'p1':{'camera':{'width':64,'height':48}}})
    assert Path(exp.source['path'])==(folder if kind=='db3' else chosen)
    report=preflight(exp,dry_run=True)
    assert report['status']=='WARN',report
    assert report['image_source_settings']['resize'] is False
    source=BagReplay(exp.source['path'],'/camera',extract_fps=2);source.start();frames=[]
    try:
        while not source.exhausted:
            f=source.poll()
            if f is not None:frames.append(f)
    finally:source.stop()
    assert len(frames)==2 and all(f[0].shape==(48,64,3) for f in frames)
    exp.live.p1.camera.update(width=128,height=96)
    assert preflight(exp)['status']=='FAIL'


def test_db3_without_metadata_is_not_guessed(tmp_path):
    file=tmp_path/'orphan.db3';file.write_bytes(b'not a recording')
    with pytest.raises(ValueError,match='metadata.yaml'):
        profile(tmp_path,input={'kind':'bag','path':str(file),'topic':'/camera'})


def test_selected_decode_error_is_not_silently_dropped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    source=BagReplay(tmp_path/'unused.bag','/camera')
    connection=SimpleNamespace(msgtype='sensor_msgs/msg/Image')
    source.messages=iter([(connection,0,b'bad')])
    source.reader=SimpleNamespace(deserialize=lambda *a:None)
    monkeypatch.setattr('calibration.ros_image_codec.decode_image_message',lambda *a:None)
    with pytest.raises(ValueError,match='not silently skipped'):source.poll()


def test_missing_factory_skips_fit_and_keeps_local_warn(tmp_path,monkeypatch):
    from tests.test_experiment import cached_experiment
    from automation.experiment.runner import run_experiment
    exp,_,_,_=cached_experiment(tmp_path)
    exp.factory={'reference':str(tmp_path/'missing.json'),'seeded':True}
    monkeypatch.setattr('automation.factory.service.post_compare',lambda *a,**k:pytest.fail('missing reference launched fit'))
    assert run_experiment(exp)==3
    state=json.loads((exp.folder/'experiment.json').read_text(encoding='utf8'))
    assert state['factory_skip']['reason']=='FACTORY_FILE_NOT_FOUND'
    assert state['publication']['status']=='PENDING' and state['calibration_verdict']=='WARN'


def test_transport_failure_retry_preserves_warn_without_compute_or_git_push(tmp_path,monkeypatch):
    from tests.test_experiment import cached_experiment
    from tests._publish_test_utils import snapshot
    from automation.experiment.runner import run_experiment
    from automation.publish.base import PublishError,PublishReceipt
    from automation.publish.bundle import verify_bundle
    exp,_,session,_=cached_experiment(tmp_path,enabled=True);before=snapshot(session)
    def fail(*a):raise PublishError('AUTHENTICATION_FAILED',False)
    monkeypatch.setattr('automation.publish.git_publisher.GitPublisher.publish',fail)
    monkeypatch.setattr('automation.live.cli.run',lambda *a,**k:pytest.fail('recalculated'))
    assert run_experiment(exp)==2
    assert snapshot(session)==before
    monkeypatch.setattr('automation.experiment.runner.fingerprint_inputs',lambda *a:pytest.fail('retry reads bag'))
    def simulated_publish(self,bundle):
        meta=verify_bundle(bundle)
        return PublishReceipt('f'*40,exp.publisher.remote,exp.publisher.branch,meta['run_path'],meta['content_hash'])
    monkeypatch.setattr('automation.publish.git_publisher.GitPublisher.publish',simulated_publish)
    assert run_experiment(exp,retry_only=True,reset_failed=True)==3
    state=json.loads((exp.folder/'experiment.json').read_text(encoding='utf8'))
    assert state['calibration_verdict']=='WARN' and state['publication']['status']=='PUBLISHED'
    assert snapshot(session)==before


@pytest.mark.slow
def test_synthetic_mcap_to_four_models_yaml_outbox_without_factory_or_push(tmp_path_factory):
    tmp_path=tmp_path_factory.mktemp('bag')
    from rosbags.rosbag2 import Writer
    from rosbags.rosbag2.enums import StoragePlugin
    from rosbags.typesys import Stores,get_typestore
    from tests._automation_test_utils import render_consistent_charuco_dataset
    from tests.test_p2_e2e import replay_profile
    from tests._publish_test_utils import snapshot
    from calibration.project_io import load_project
    from calibration.calibration_io import load_standard_calibration
    images=tmp_path/'images';render_consistent_charuco_dataset(images,24,7)
    before_images=snapshot(images);bag=tmp_path/'recording'
    store=get_typestore(Stores.ROS2_HUMBLE);t=store.types
    Message=t['sensor_msgs/msg/CompressedImage'];Header=t['std_msgs/msg/Header'];Time=t['builtin_interfaces/msg/Time']
    with Writer(bag,version=8,storage_plugin=StoragePlugin.MCAP) as writer:
        c=writer.add_connection('/camera',Message.__msgtype__,typestore=store)
        for i,p in enumerate(sorted(images.glob('*.png'))):
            msg=Message(header=Header(stamp=Time(sec=i,nanosec=0),frame_id='c'),format='png',data=np.frombuffer(p.read_bytes(),np.uint8).copy())
            writer.write(c,i*10**9,store.serialize_cdr(msg,Message.__msgtype__))
    before_bag=snapshot(bag)
    live=replay_profile();live['p1']={'camera':{'width':1280,'height':720}}
    raw={'experiment_version':1,'job_id':'standard-mcap','camera':{'model':'TEST','serial':'MCAP-READY'},
        'input':{'kind':'bag','path':str(next(bag.glob('*.mcap'))),'topic':'/camera','extract_fps':1,'replay_interval':.01},
        'live':live,'output':{'root':str(tmp_path/'output'),'min_free_gb':0},
        'publish':{'config':{'enabled':False}},'factory':{'seeded':True}}
    config=tmp_path/'experiment.yaml';config.write_text(yaml.safe_dump(raw),encoding='utf8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf8',QT_QPA_PLATFORM='offscreen')
    p=subprocess.run([sys.executable,'-m','automation','experiment','run','--profile',str(config)],
        capture_output=True,text=True,encoding='utf8',env=env,timeout=300)
    evidence=Path('.verification/rosbag-ready')
    (evidence/'mcap-e2e-stdout.log').write_text(p.stdout,encoding='utf8')
    (evidence/'mcap-e2e-stderr.log').write_text(p.stderr,encoding='utf8')
    assert p.returncode==3,(p.returncode,p.stdout[-3000:],p.stderr)
    state=json.loads((tmp_path/'output/experiments/standard-mcap/experiment.json').read_text(encoding='utf8'))
    assert state['status']=='COMPLETED' and state['calibration_verdict']=='WARN'
    assert state['factory_skip']=={'status':'SKIPPED','reason':'NO_FACTORY_REFERENCE','seeded_executed':False}
    assert state['publication']['status']=='PENDING' and state['publication']['attempts']==0
    session=Path(state['calibration_session']);project,_=load_project(str(session/'project/calibration.ccproj'))
    assert len(project.calibration_results)==4 and all(v.success for v in project.calibration_results.values())
    chosen=project.final_result.chosen_model
    assert chosen==next(v.model_name for v in project.model_scores if v.is_recommended)
    partitions={(tuple(v.train_frame_ids),tuple(v.test_frame_ids)) for v in project.validation_results.values()}
    assert len(partitions)==1
    for v in project.validation_results.values():
        assert v.test_rms is not None and not set(v.train_frame_ids)&set(v.test_frame_ids)
    stem='rational' if chosen.value=='extended_pinhole' else chosen.value
    for file in [session/f'intrinsic/camera_{stem}.yaml',session/'intrinsic/camera_info.yaml']:
        exported=load_standard_calibration(str(file));result=project.calibration_results[chosen]
        np.testing.assert_allclose(exported.camera_matrix,result.camera_matrix,rtol=0,atol=1e-6)
        np.testing.assert_allclose(exported.distortion.ravel(),result.distortion.ravel(),rtol=0,atol=1e-6)
        assert (exported.width,exported.height)==(1280,720)
    assert snapshot(images)==before_images and snapshot(bag)==before_bag
    (evidence/'mcap-e2e-result.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf8')
