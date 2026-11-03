"""One real synthetic ROS2 bag -> P2/P1/P4 -> local bare publication, once."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from tests._automation_test_utils import render_consistent_charuco_dataset,TRUE_K,TRUE_D,IMG_W,IMG_H
from tests.test_p2_e2e import replay_profile
from tests._publish_test_utils import bare,git,snapshot


@pytest.fixture(scope='module')
def integrated_run(tmp_path_factory):
    # Explicit verification reuse after a test-harness-only failure: validate the
    # completed input hash rather than unnecessarily running calibration again.
    if os.environ.get('CAMERA_EXPERIMENT_E2E_REUSE'):
        from automation.experiment.config import load_experiment
        from automation.experiment.runner import fingerprint_inputs
        root=Path(os.environ['CAMERA_EXPERIMENT_E2E_REUSE'])
        profile=root/'experiment.yaml'
        state=json.loads((root/'output/experiments/synthetic-bag/experiment.json').read_text(encoding='utf8'))
        assert state['input_sha256']==fingerprint_inputs(load_experiment(profile).source)
        return root,profile,root/'results.git',state
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore,Stores
    root=tmp_path_factory.mktemp('experiment_bag');images=root/'images'
    render_consistent_charuco_dataset(images,24,7);original=snapshot(images)
    store=get_typestore(Stores.ROS2_HUMBLE);types=store.types
    message=types['sensor_msgs/msg/CompressedImage'];header=types['std_msgs/msg/Header'];stamp=types['builtin_interfaces/msg/Time']
    bag=root/'synthetic_ros2'
    with Writer(bag,version=8) as writer:
        connection=writer.add_connection('/camera/image/compressed',message.__msgtype__,typestore=store)
        for i,path in enumerate(sorted(images.glob('*.png'))):
            msg=message(header=header(stamp=stamp(sec=i,nanosec=0),frame_id='camera'),format='png',data=np.frombuffer(path.read_bytes(),np.uint8).copy())
            writer.write(connection,i*10**9,store.serialize_cdr(msg,message.__msgtype__))
    assert len(list(images.glob('*.png')))==24
    bag_before=snapshot(bag);remote=bare(root/'results.git')
    coordinates={'reference_resolution':[IMG_W,IMG_H],'roi':[0,0,IMG_W,IMG_H],'binning':[1,1],'orientation':'identity'}
    ref=root/'reference.json';ref.write_text(json.dumps({'schema_version':1,'camera_model':'TEST','serial_number':'SYNTH-BAG',
        'width':IMG_W,'height':IMG_H,'distortion_model':'brown_conrady','projection_convention':'opencv',
        'coefficient_order':['k1','k2','p1','p2','k3'],'K':TRUE_K.tolist(),'D':TRUE_D.ravel().tolist(),
        'source':'Known-K synthetic control, NOT manufacturer data','coordinate_system':coordinates}))
    coords=root/'coordinates.json';coords.write_text(json.dumps(coordinates))
    config={'experiment_version':1,'job_id':'synthetic-bag','camera':{'model':'TEST','serial':'SYNTH-BAG'},
        'input':{'kind':'bag','path':str(bag),'topic':'/camera/image/compressed','replay_interval':.01},
        'live':replay_profile(),'output':{'root':str(root/'output'),'min_free_gb':0},
        'publish':{'results_repository':True,'full_frame':True,'config':{'enabled':True,'remote':str(remote)}},
        'factory':{'reference':str(ref),'coordinates':str(coords),'seeded':True}}
    profile=root/'experiment.yaml';profile.write_text(yaml.safe_dump(config),encoding='utf8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',QT_QPA_PLATFORM='offscreen')
    proc=subprocess.run([sys.executable,'-m','automation','experiment','run','--profile',str(profile)],
        capture_output=True,text=True,encoding='utf8',env=env,timeout=360)
    evidence=Path(os.environ.get('CAMERA_EXPERIMENT_E2E_EVIDENCE','.verification/integration'));evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'bag-pipeline-stdout.log').write_text(proc.stdout,encoding='utf8')
    (evidence/'bag-pipeline-stderr.log').write_text(proc.stderr,encoding='utf8')
    assert proc.returncode==3,(proc.returncode,proc.stdout[-4000:],proc.stderr)
    state=json.loads((root/'output/experiments/synthetic-bag/experiment.json').read_text(encoding='utf8'))
    (evidence/'bag-pipeline-result.json').write_text(json.dumps({'root':str(root),'returncode':proc.returncode,'state':state},indent=2),encoding='utf8')
    assert snapshot(images)==original and snapshot(bag)==bag_before
    return root,profile,remote,state


def test_bag_to_yaml_factory_and_local_git(integrated_run):
    root,profile,remote,state=integrated_run
    assert state['status']=='COMPLETED' and state['calibration_verdict']=='WARN'
    assert state['publication']['status']=='PUBLISHED'
    stages={entry['name']:entry for entry in state['stages']}
    assert stages['capture_readiness_calibration_export']['finished_at'] <= stages['publication']['started_at']
    assert all(entry['status']=='completed' for entry in state['p1_stage_timing'])
    assert state['factory']['status']=='COMPLETED',state['factory']
    p1=Path(state['calibration_session'])
    assert (p1/'intrinsic/camera_info.yaml').is_file()
    comparison=json.loads(Path(state['factory']['report']).read_text(encoding='utf8'))
    assert comparison['test_frame_ids'] and set(comparison['candidates'])=={'Factory-fixed','Local','Factory-seeded'}
    files=git('--git-dir='+str(remote),'ls-tree','-r','--name-only','calibration-results')
    assert 'factory_comparison.json' in files
    assert not any(x.endswith(('.png','.bag','.db3','.mcap','.ccproj')) for x in files.splitlines())
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'


def test_same_job_reuses_calculation_and_publication(integrated_run):
    root,profile,remote,state=integrated_run
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    proc=subprocess.run([sys.executable,'-m','automation','experiment','run','--profile',str(profile)],
        capture_output=True,text=True,encoding='utf8',env=env,timeout=90)
    (Path(os.environ.get('CAMERA_EXPERIMENT_E2E_EVIDENCE','.verification/integration'))/'bag-reuse.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf8')
    assert proc.returncode==3 and 'calibration_reused' in proc.stdout
    after=json.loads((root/'output/experiments/synthetic-bag/experiment.json').read_text(encoding='utf8'))
    assert after['calibration_session']==state['calibration_session']
    assert after['result_manifest_sha256']==state['result_manifest_sha256']
    assert after['publication']['receipt']['commit_sha']==state['publication']['receipt']['commit_sha']
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'
