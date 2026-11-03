from dataclasses import replace
import json
from pathlib import Path
import pytest
from automation.publish.base import PublishError,PublishReceipt
from automation.publish.bundle import build_bundle
from automation.publish.config import PublishConfig
from automation.publish.outbox import Outbox
from automation.publish.git_publisher import GitPublisher
from automation.publish.history import list_history,compare_runs
from automation.publish.service import publish_session
from tests._publish_test_utils import make_session,bare,git,read,write,snapshot


class Clock:
    def __init__(self):self.value=100.
    def __call__(self):return self.value


class Publisher:
    def __init__(self,error=None):self.error=error;self.calls=0
    def publish(self,bundle):
        self.calls+=1
        if self.error:raise self.error
        meta=read(bundle/'bundle.json')
        return PublishReceipt('c'*40,'test-remote','results',meta['run_path'],meta['content_hash'])


def queued(tmp_path,**options):
    clock=Clock();cfg=PublishConfig(enabled=True,outbox=str(tmp_path/'outbox'),**options)
    box=Outbox(cfg.outbox,cfg,clock=clock);session=make_session(tmp_path/'session')
    state=box.enqueue(session,full_frame=True)
    return box,cfg,clock,session,state['job_id']


def test_network_retry_backoff_and_no_recalculation(tmp_path):
    box,cfg,clock,session,job=queued(tmp_path);before=snapshot(session)
    bad=Publisher(PublishError('NETWORK',True));state=box.process(job,bad)
    assert state['status']=='RETRY_WAIT' and state['next_attempt_at']==105.
    box.process(job,bad);assert bad.calls==1
    clock.value=105.;good=Publisher();state=box.process(job,good)
    assert state['status']=='PUBLISHED' and state['receipt']['commit_sha']=='c'*40
    box.process(job,good);assert good.calls==1 and snapshot(session)==before


def test_auth_failure_is_not_retried_until_explicit_reset(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path);bad=Publisher(PublishError('AUTHENTICATION_FAILED'))
    assert box.process(job,bad)['status']=='FAILED'
    clock.value=10000.;box.process(job,bad);assert bad.calls==1
    assert box.process(job,Publisher(),reset_failed=True)['status']=='PUBLISHED'


def test_retry_limit_and_exponential_delay(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path,max_attempts=3);bad=Publisher(PublishError('NETWORK',True))
    assert box.process(job,bad)['next_attempt_at']==105.
    clock.value=105.;assert box.process(job,bad)['next_attempt_at']==115.
    clock.value=115.;assert box.process(job,bad)['status']=='FAILED'
    clock.value=10000.;box.process(job,bad);assert bad.calls==3


def test_corrupt_state_recovers_last_failure_from_journal(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path);bad=Publisher(PublishError('AUTHENTICATION_FAILED'))
    box.process(job,bad);folder=box.root/'items'/job
    (folder/'state.json').write_text('{broken',encoding='utf-8');(folder/'state.backup.json').write_text('{broken',encoding='utf-8')
    fresh=Outbox(box.root,cfg,clock=clock)
    assert fresh.list()[0]['status']=='FAILED';fresh.process(job,bad);assert bad.calls==1


def test_push_then_crash_recovers_without_duplicate_commit(tmp_path):
    remote=bare(tmp_path/'r.git');box,cfg,clock,_,job=queued(tmp_path,remote=str(remote))
    class Crash(GitPublisher):
        def publish(self,bundle):
            super().publish(bundle)
            raise SystemExit('simulated process loss immediately after confirmed push')
    with pytest.raises(SystemExit):box.process(job,Crash(cfg))
    assert read(box.root/'items'/job/'state.json')['status']=='PUBLISHING'
    fresh=Outbox(box.root,cfg,clock=clock);state=fresh.process(job,GitPublisher(cfg))
    assert state['status']=='PUBLISHED' and state['receipt']['duplicate']
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'


def test_missing_all_state_records_requires_manual_review(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path)
    folder=box.root/'items'/job
    (folder/'state.json').unlink();(folder/'events.jsonl').unlink()
    state=Outbox(box.root,cfg,clock=clock).list()[0]
    assert state['status']=='FAILED' and not state['retryable']


def test_disabled_queues_without_git_and_execution_errors_are_local_only(tmp_path,monkeypatch):
    cfg=PublishConfig(outbox=str(tmp_path/'outbox'))
    monkeypatch.setattr(GitPublisher,'publish',lambda *_:pytest.fail('network attempted'))
    session=make_session(tmp_path/'s');output=publish_session(session,cfg)
    assert output['publication_status']=='PENDING'
    bad=make_session(tmp_path/'bad');m=read(bad/'run_manifest.json');m['status']='failed';write(bad/'run_manifest.json',m)
    assert publish_session(bad,cfg)['publication_status']=='FAILED'
    errors=list((tmp_path/'outbox/errors').glob('*.json'));assert len(errors)==1
    assert read(errors[0])['calibration_status']=='ERROR'


@pytest.mark.parametrize('verdict',['PASS','WARN','FAIL'])
def test_quality_verdict_is_preserved_independently_of_publication(tmp_path,verdict):
    cfg=PublishConfig(outbox=str(tmp_path/'o'));session=make_session(tmp_path/'s',verdict=verdict)
    box=Outbox(cfg.outbox,cfg);state=box.enqueue(session)
    bundle=read(box.root/'items'/state['job_id']/'bundle/summary.json')
    assert bundle['acceptance_verdict']==verdict and bundle['approval_status']=='UNREVIEWED'


def test_history_without_index_and_compatible_shared_ray_comparison(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path)
    other=make_session(tmp_path/'other',run_id='RUN-2',delta=2.)
    box.enqueue(other,full_frame=True)
    records=list_history(box.root,'SN123');assert len(records)==2
    comparison=compare_runs(*records)
    assert comparison['comparable'] and comparison['intrinsic_delta']['fx']==2.
    assert comparison['shared_ray_projection_px']['rms']>0
    assert 'not a quality verdict' in comparison['interpretation']
    assert list_history(box.root,'ABSENT')==[]


@pytest.mark.parametrize('change,reason',[
    ({'camera':{'camera_model':'OTHER','serial_number':'SN123'}},'CAMERA_IDENTITY_MISMATCH'),
    ({'resolution':[1280,720]},'RESOLUTION_MISMATCH'),
    ({'chosen_model':'fisheye'},'DISTORTION_MODEL_MISMATCH'),
    ({'coordinate_system':{'kind':'unknown'}},'ROI_UNCONFIRMED'),
    ({'coordinate_system':{'kind':'roi','roi':[20,0,640,480],'source':'user_declared'}},'ROI_MISMATCH')])
def test_incompatible_history_blocks_numeric_comparison(tmp_path,change,reason):
    build_bundle(make_session(tmp_path/'s'),tmp_path/'b',full_frame=True)
    left=read(tmp_path/'b/summary.json');right={**left,**change}
    result=compare_runs(left,right)
    assert not result['comparable'] and reason in result['limitations'] and 'intrinsic_delta' not in result


def test_pending_job_cannot_silently_switch_remote(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path,remote='https://github.com/org/results.git')
    other=Outbox(box.root,replace(cfg,remote='https://gitlab.com/org/results.git'),clock=clock)
    with pytest.raises(PublishError,match='TARGET_MISMATCH'):other.process(job,Publisher())


def test_invalid_utf8_journal_does_not_hide_valid_atomic_state(tmp_path):
    box,cfg,clock,_,job=queued(tmp_path)
    (box.root/'items'/job/'events.jsonl').write_bytes(b'\xff\xfe')
    assert box.list()[0]['status']=='PENDING'
