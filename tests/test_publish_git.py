from dataclasses import replace
from pathlib import Path
import pytest
from automation.publish.bundle import build_bundle
from automation.publish.git_publisher import GitPublisher
from automation.publish.config import PublishConfig
from automation.publish.base import PublishError
from tests._publish_test_utils import make_session,bare,git,read


def test_bare_publish_dedup_new_runs_serials_and_index(tmp_path):
    remote=bare(tmp_path/'remote.git');pub=GitPublisher(PublishConfig(enabled=True,remote=str(remote)))
    b=tmp_path/'b';m=build_bundle(make_session(tmp_path/'s'),b)
    first=pub.publish(b);again=pub.publish(b)
    assert again.duplicate and first.commit_sha==again.commit_sha
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'
    b2=tmp_path/'b2';build_bundle(make_session(tmp_path/'s2',run_id='RUN-2',delta=1.),b2);pub.publish(b2)
    b3=tmp_path/'b3';build_bundle(make_session(tmp_path/'s3',run_id='RUN-3',serial='OTHER'),b3);pub.publish(b3)
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='3'
    checkout=tmp_path/'read';git('clone','--branch','calibration-results',remote,checkout)
    assert len(read(checkout/('/'.join(m['run_path'].split('/')[:3]))/'index.json')['runs'])==2
    from automation.publish.history import list_history
    records=list_history(checkout,'SN123')
    assert len(records)==2 and all(r['publication_status']=='PUBLISHED' and r['publication_commit_sha'] for r in records)


def test_same_run_id_different_result_fails_without_overwrite(tmp_path):
    remote=bare(tmp_path/'r.git');pub=GitPublisher(PublishConfig(enabled=True,remote=str(remote)))
    build_bundle(make_session(tmp_path/'a'),tmp_path/'ab');pub.publish(tmp_path/'ab')
    build_bundle(make_session(tmp_path/'b',delta=2.),tmp_path/'bb')
    with pytest.raises(PublishError,match='RUN_ID_COLLISION'):pub.publish(tmp_path/'bb')
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='1'


def test_non_fast_forward_is_safe_retry_and_preserves_other_users_commit(tmp_path):
    remote=bare(tmp_path/'r.git');cfg=PublishConfig(enabled=True,remote=str(remote))
    build_bundle(make_session(tmp_path/'s'),tmp_path/'b');GitPublisher(cfg).publish(tmp_path/'b')
    build_bundle(make_session(tmp_path/'s2',run_id='RUN-2',delta=1),tmp_path/'b2')
    class Race(GitPublisher):
        raced=False
        def _git(self,repo,*args,**kwargs):
            assert '--force' not in args and '--force-with-lease' not in args
            if args and args[0]=='push' and not self.raced:
                self.raced=True;other=tmp_path/'other';git('clone','--branch',cfg.branch,remote,other)
                (other/'other-user.txt').write_text('keep me',encoding='utf-8')
                git('add','other-user.txt',cwd=other);git('commit','-m','concurrent change',cwd=other);git('push','origin',cfg.branch,cwd=other)
            return super()._git(repo,*args,**kwargs)
    with pytest.raises(PublishError) as error:Race(cfg).publish(tmp_path/'b2')
    assert error.value.retryable
    GitPublisher(cfg).publish(tmp_path/'b2')
    assert git('--git-dir='+str(remote),'show',cfg.branch+':other-user.txt')=='keep me'


def test_corrupt_index_does_not_lose_runs_and_next_upload_rebuilds(tmp_path):
    remote=bare(tmp_path/'r.git');cfg=PublishConfig(enabled=True,remote=str(remote));pub=GitPublisher(cfg)
    m=build_bundle(make_session(tmp_path/'a'),tmp_path/'ab');pub.publish(tmp_path/'ab')
    other=tmp_path/'other';git('clone','--branch',cfg.branch,remote,other)
    index=other/('/'.join(m['run_path'].split('/')[:3]))/'index.json';index.write_text('{bad',encoding='utf-8')
    git('add','.',cwd=other);git('commit','-m','broken index',cwd=other);git('push','origin',cfg.branch,cwd=other)
    build_bundle(make_session(tmp_path/'b',run_id='RUN-2',delta=1),tmp_path/'bb');pub.publish(tmp_path/'bb')
    import json
    fixed=json.loads(git('--git-dir='+str(remote),'show',cfg.branch+':'+index.relative_to(other).as_posix()))
    assert len(fixed['runs'])==2


def test_disabled_publisher_never_runs_git(tmp_path,monkeypatch):
    pub=GitPublisher(PublishConfig());monkeypatch.setattr(pub,'_git',lambda *_a,**_k:pytest.fail('git ran'))
    with pytest.raises(PublishError,match='DISABLED'):pub.publish(tmp_path/'not-needed')
