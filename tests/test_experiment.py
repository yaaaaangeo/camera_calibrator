import copy
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest
import yaml

from automation.experiment.config import load_experiment
from automation.experiment.preflight import preflight
from automation.experiment.runner import run_experiment,fingerprint_inputs
from automation.publish.base import PublishError
from tests._publish_test_utils import bare,make_session,git,snapshot


def profile(tmp_path,**changes):
    images=tmp_path/'images';images.mkdir(exist_ok=True);cv2.imwrite(str(images/'one.png'),np.full((480,640,3),120,np.uint8))
    raw={'experiment_version':1,'job_id':'trial','camera':{'model':'TEST','serial':'SN1'},
        'input':{'kind':'images','path':str(images)},'live':{'live_profile_version':1},
        'output':{'root':str(tmp_path/'outputs'),'min_free_gb':0},'publish':{'config':{'enabled':False}}}
    raw.update(changes);path=tmp_path/'experiment.yaml';path.write_text(yaml.safe_dump(raw),encoding='utf8')
    return load_experiment(path),path


def test_preflight_readonly_and_dry_run_side_effect_free(tmp_path,monkeypatch,capsys):
    from automation.cli import main
    exp,path=profile(tmp_path);before=snapshot(tmp_path)
    monkeypatch.setattr('automation.experiment.runner.run_experiment',lambda *a,**k:pytest.fail('execution in dry run'))
    assert main(['experiment','run','--profile',str(path),'--dry-run','--network','--verify-write'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['dry_run'] and not result['network_probe'] and not result['write_probe']
    assert not result['calibration_will_run'] and snapshot(tmp_path)==before
    assert not exp.output.exists()
    assert all(c['status'] in {'PASS','WARN','FAIL'} for c in result['checks'])


@pytest.mark.parametrize('kind',['missing_input','space','permission','collision','pattern','git','source_remote'])
def test_preflight_errors(tmp_path,monkeypatch,kind):
    exp,path=profile(tmp_path)
    if kind=='missing_input':exp.source['path']=str(tmp_path/'absent')
    if kind=='space':monkeypatch.setattr('automation.experiment.preflight.shutil.disk_usage',lambda *a:type('Disk',(),{'free':0})());exp.raw['output']['min_free_gb']=1
    if kind=='permission':monkeypatch.setattr('automation.experiment.preflight.os.access',lambda *a:False)
    if kind=='collision':exp.folder.mkdir(parents=True)
    if kind=='pattern':monkeypatch.setattr('calibration.detector.build_detect_fn',lambda *a:(_ for _ in ()).throw(ValueError('invalid board')))
    if kind=='git':monkeypatch.setattr('automation.experiment.preflight.shutil.which',lambda *a:None)
    if kind=='source_remote':
        from dataclasses import replace
        exp.publisher=replace(exp.publisher,remote='https://github.com/yaaaaangeo/camera_calibrator.git')
    assert preflight(exp)['status']=='FAIL'


def test_bag_preflight_topic_and_decode(tmp_path):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore,Stores
    store=get_typestore(Stores.ROS2_HUMBLE);types=store.types
    image=types['sensor_msgs/msg/Image'];header=types['std_msgs/msg/Header'];stamp=types['builtin_interfaces/msg/Time']
    pixels=np.full((48,64,3),90,np.uint8);bag=tmp_path/'ros2'
    with Writer(bag,version=8) as writer:
        c=writer.add_connection('/camera',image.__msgtype__,typestore=store)
        msg=image(header=header(stamp=stamp(sec=0,nanosec=0),frame_id='c'),height=48,width=64,
            encoding='bgr8',is_bigendian=0,step=192,data=pixels.ravel())
        writer.write(c,0,store.serialize_cdr(msg,image.__msgtype__))
    exp,_=profile(tmp_path,input={'kind':'bag','path':str(bag),'topic':'/camera'})
    assert preflight(exp)['status']=='WARN'
    exp.source['topic']='/absent';assert preflight(exp)['status']=='FAIL'


def cached_experiment(tmp_path,enabled=False):
    remote=bare(tmp_path/'results.git')
    exp,path=profile(tmp_path,publish={'results_repository':True,'full_frame':True,'config':{'enabled':enabled,'remote':str(remote)}})
    session=make_session(tmp_path/'completed',camera_model='TEST',serial='SN1')
    from automation.publish.storage import write_json
    from automation.run_manifest import sha256_file
    exp.folder.mkdir(parents=True)
    state={'schema_version':1,'fingerprint':exp.fingerprint,'job_id':exp.job,'status':'COMPLETED','stages':[],
        'input_sha256':fingerprint_inputs(exp.source),'calibration_session':str(session),
        'calibration_verdict':'WARN','result_manifest_sha256':sha256_file(session/'run_manifest.json')}
    write_json(exp.folder/'experiment.json',state)
    return exp,path,session,remote


def test_cached_warn_publishes_bare_and_deduplicates_without_calculation(tmp_path,monkeypatch):
    exp,_,session,remote=cached_experiment(tmp_path,True);before=snapshot(session)
    monkeypatch.setattr('automation.live.cli.run',lambda *a,**k:pytest.fail('recalculation'))
    assert run_experiment(exp)==3
    assert run_experiment(exp)==3
    state=json.loads((exp.folder/'experiment.json').read_text())
    assert state['publication']['status']=='PUBLISHED' and state['calibration_verdict']=='WARN'
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'
    names=git('--git-dir='+str(remote),'ls-tree','-r','--name-only','calibration-results').splitlines()
    assert not any(p.endswith(('.png','.bag','.db3','.mcap','.ccproj')) for p in names)
    assert before==snapshot(session)


def test_publication_failure_retry_uses_outbox_without_input_or_compute(tmp_path,monkeypatch):
    from automation.publish.git_publisher import GitPublisher
    exp,_,session,remote=cached_experiment(tmp_path,True)
    original=GitPublisher.publish
    monkeypatch.setattr(GitPublisher,'publish',lambda *a:(_ for _ in ()).throw(PublishError('NETWORK',False)))
    assert run_experiment(exp)==2
    before=snapshot(session)
    monkeypatch.setattr(GitPublisher,'publish',original)
    monkeypatch.setattr('automation.experiment.runner.fingerprint_inputs',lambda *a:pytest.fail('retry reads input'))
    Path(exp.source['path']).rename(tmp_path/'archived-images')  # original path unavailable; profile binding unchanged
    assert run_experiment(exp,retry_only=True,reset_failed=True)==3
    assert before==snapshot(session)


def test_changed_input_refuses_completed_result_relabel(tmp_path):
    exp,_,_,_=cached_experiment(tmp_path)
    cv2.imwrite(str(Path(exp.source['path'])/'one.png'),np.zeros((480,640,3),np.uint8))
    assert run_experiment(exp)==2
    state=json.loads((exp.folder/'experiment.json').read_text())
    assert state['failure']['stage']=='input_fingerprint'


def test_disabled_publisher_queues_without_git_push(tmp_path,monkeypatch):
    exp,_,_,_=cached_experiment(tmp_path)
    monkeypatch.setattr('automation.publish.git_publisher.GitPublisher.publish',lambda *a:pytest.fail('disabled push'))
    assert run_experiment(exp)==3
    assert json.loads((exp.folder/'experiment.json').read_text())['publication']['status']=='PENDING'


def test_job_lock_and_fingerprint_guard(tmp_path):
    from automation.publish.storage import Lock
    exp,_,_,_=cached_experiment(tmp_path)
    with Lock(exp.folder/'.lock'):
        with pytest.raises(PublishError,match='BUSY'):run_experiment(exp)
    exp.source['topic']='/changed'
    assert preflight(exp)['status']=='FAIL'


@pytest.mark.parametrize('existing_results_branch',[False,True])
def test_results_tree_refuses_code_before_publisher_commit(tmp_path,existing_results_branch):
    from automation.publish.git_publisher import GitPublisher
    from automation.publish.config import PublishConfig
    from automation.publish.bundle import build_bundle
    remote=bare(tmp_path/'code.git');checkout=tmp_path/'fixture-code'
    git('init',checkout);(checkout/'pyproject.toml').write_text('[project]')
    git('add','.',cwd=checkout);git('commit','-m','local fixture',cwd=checkout)
    git('push',str(remote),'HEAD:refs/heads/main',cwd=checkout)
    git('--git-dir='+str(remote),'symbolic-ref','HEAD','refs/heads/main')
    if existing_results_branch:
        git('checkout','--orphan','result-fixture',cwd=checkout)
        git('rm','--cached','pyproject.toml',cwd=checkout)
        (checkout/'README.md').write_text('results fixture')
        git('add','README.md',cwd=checkout);git('commit','-m','results branch fixture',cwd=checkout)
        git('push',str(remote),'HEAD:refs/heads/calibration-results',cwd=checkout)
    bundle=tmp_path/'bundle';build_bundle(make_session(tmp_path/'s'),bundle)
    before=git('--git-dir='+str(remote),'rev-list','--count','--all')
    with pytest.raises(PublishError,match='NOT_RESULTS'):GitPublisher(PublishConfig(True,str(remote))).publish(bundle)
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')==before


def test_write_probe_and_outbox_input_overlap(tmp_path):
    from dataclasses import replace
    exp,_=profile(tmp_path)
    before=snapshot(tmp_path);result=preflight(exp,verify_write=True)
    assert next(c for c in result['checks'] if c['check']=='output_permission')['status']=='PASS'
    assert snapshot(tmp_path)==before
    exp.publisher=replace(exp.publisher,outbox=str(Path(exp.source['path'])/'outbox'))
    assert preflight(exp)['status']=='FAIL'


def test_interruption_after_calculation_reuses_result(tmp_path,monkeypatch):
    from automation.publish.outbox import Outbox
    exp,_,session,_=cached_experiment(tmp_path)
    original=Outbox.enqueue
    monkeypatch.setattr(Outbox,'enqueue',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert run_experiment(exp)==130
    state=json.loads((exp.folder/'experiment.json').read_text())
    assert state['status']=='INTERRUPTED' and state['calibration_session']==str(session)
    monkeypatch.setattr(Outbox,'enqueue',original)
    monkeypatch.setattr('automation.live.cli.run',lambda *a,**k:pytest.fail('recalculated'))
    assert run_experiment(exp)==3


def test_adoption_rejects_wrong_identity_without_modifying_session(tmp_path):
    exp,_=profile(tmp_path)
    session=make_session(tmp_path/'foreign',serial='OTHER',camera_model='TEST')
    before=snapshot(session)
    assert run_experiment(exp,reuse_session=session)==2
    state=json.loads((exp.folder/'experiment.json').read_text())
    assert state['failure']['stage']=='adopt_existing_result' and snapshot(session)==before


def test_diagnostics_use_existing_pipeline_timing(tmp_path):
    exp,_,_,_=cached_experiment(tmp_path)
    running=[]
    def observe(line):
        if json.loads(line).get('event')=='stage_started':
            running.append(json.loads((exp.folder/'experiment.json').read_text())['status'])
    assert run_experiment(exp,emit=observe)==3
    state=json.loads((exp.folder/'experiment.json').read_text())
    assert all(s['seconds']>=0 and s['status']=='COMPLETED' for s in state['stages'])
    assert 'preflight' in state and 'p1_stage_timing' in state
    assert running and all(value=='RUNNING' for value in running)


def test_changed_factory_repackages_without_recalculation(tmp_path,monkeypatch):
    from automation.publish.outbox import Outbox
    exp,_,_,_=cached_experiment(tmp_path)
    assert run_experiment(exp)==3
    reference=tmp_path/'reference.json';reference.write_text('{}')
    exp.factory={'reference':str(reference)}
    calls=[]
    monkeypatch.setattr('automation.factory.service.post_compare',lambda *a:{'status':'COMPLETED','report':'new-comparison.json'})
    def enqueue(self,session,**options):
        calls.append(options);return {'job_id':'f'*64,'status':'PENDING'}
    monkeypatch.setattr(Outbox,'enqueue',enqueue)
    monkeypatch.setattr(Outbox,'process',lambda *a,**k:{'job_id':'f'*64,'status':'PENDING'})
    assert run_experiment(exp)==3
    assert calls==[{'full_frame':True,'factory_report':'new-comparison.json'}]


def test_retry_dry_run_needs_no_input_and_has_no_side_effects(tmp_path,capsys):
    from automation.cli import main
    exp,path,_,_=cached_experiment(tmp_path)
    Path(exp.source['path']).rename(tmp_path/'archived')
    before=snapshot(tmp_path)
    assert main(['experiment','retry-publish','--profile',str(path),'--dry-run'])==0
    result=json.loads(capsys.readouterr().out)
    assert not result['calibration_will_run'] and 'Retry publication only' in result['plan'][-1]
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('readable',[True,False])
def test_git_read_probe_never_claims_push_authorization(tmp_path,monkeypatch,readable):
    from dataclasses import replace
    from types import SimpleNamespace
    exp,_=profile(tmp_path);exp.publisher=replace(exp.publisher,remote='https://github.com/EXAMPLE/results.git')
    calls=[]
    def git_read(*args,**kwargs):
        calls.append(args)
        assert 'push' not in args
        return SimpleNamespace(returncode=0 if '--get-url' in args or readable else 1,
            stdout=exp.publisher.remote if '--get-url' in args else '',stderr='PRIVATE_TOKEN_NEVER_PRINT')
    monkeypatch.setattr('automation.experiment.preflight.git_read',git_read)
    result=preflight(exp,network=True)
    check={c['check']:c for c in result['checks']}
    assert check['git_read_access']['status']==('PASS' if readable else 'WARN')
    assert check['git_push_authorization']['status']=='WARN'
    assert 'PRIVATE_TOKEN_NEVER_PRINT' not in json.dumps(result)


def test_capture_failure_resumes_same_p2_session(tmp_path,monkeypatch):
    from automation.live.controller import State
    exp,_=profile(tmp_path);completed=make_session(tmp_path/'completed',camera_model='TEST',serial='SN1')
    resumes=[]
    class Controller:
        def __init__(self,*args,output_root=None,resume=None,**kwargs):
            resumes.append(resume);self.session=Path(resume) if resume else Path(output_root)/'capture'
            self.session.mkdir(parents=True,exist_ok=True)
            self.state=State.ERROR if len(resumes)==1 else State.COMPLETED
            self.attempts=[] if self.state==State.ERROR else [{'outcome':{'session_dir':str(completed),'verdict':'WARN'}}]
            self.guidance=['interrupted source']
        def close(self):pass
    monkeypatch.setattr('automation.live.controller.LiveController',Controller)
    monkeypatch.setattr('automation.live.cli.run',lambda controller,*a,**k:2 if controller.state==State.ERROR else 0)
    assert run_experiment(exp)==2
    saved=json.loads((exp.folder/'experiment.json').read_text())['capture_session']
    assert run_experiment(exp)==3
    assert resumes==[None,saved]
