from dataclasses import dataclass
import json
from pathlib import Path
import pytest
from automation.cli import main
from automation.publish.config import PublishConfig
from automation.publish.service import post_publish
from tests._publish_test_utils import make_session,write,read


def test_existing_session_command_queues_and_history_is_offline(tmp_path,capsys):
    s=make_session(tmp_path/'s');cfg=tmp_path/'cfg.json';write(cfg,{'outbox':str(tmp_path/'outbox')})
    assert main(['publish','--session',str(s),'--publish-config',str(cfg)])==0
    assert json.loads(capsys.readouterr().out)['publication_status']=='PENDING'
    assert main(['history','--serial','SN123','--publish-config',str(cfg)])==0
    assert len(json.loads(capsys.readouterr().out)['runs'])==1
    assert main(['publish','retry','--publish-config',str(cfg)])==0


def test_run_opt_in_hook_failure_never_changes_calibration_exit(tmp_path,monkeypatch):
    import automation.runner as runner
    import automation.publish.service as service
    from automation.runner import RunOutcome
    s=make_session(tmp_path/'s');calls=[]
    monkeypatch.setattr(runner,'run_automation',lambda *_a,**_k:RunOutcome(3,'RUN-1','completed','WARN',str(s),'review'))
    monkeypatch.setattr(service,'post_publish',lambda *_a,**_k:calls.append(True) or {'publication_status':'FAILED','error':'NETWORK'})
    base=['run','--profile','default','--images','x','--serial','SN123','--camera-model','CAM']
    assert main(base)==3 and not calls
    assert main(base+['--publish'])==3 and calls==[True]


def test_p2_post_export_hook_is_opt_in_and_preserves_result(tmp_path,monkeypatch):
    import automation.live.cli as live
    import automation.publish.service as service
    from automation.live.controller import State
    session=make_session(tmp_path/'s');calls=[]
    class Controller:
        def __init__(self,*_a,**_k):
            self.state=State.COMPLETED;self.attempts=[{'outcome':{'session_dir':str(session)}}]
    monkeypatch.setattr(live,'LiveController',Controller)
    monkeypatch.setattr(live,'ImageReplay',lambda *_:None)
    monkeypatch.setattr(live,'run',lambda *_a,**_k:0)
    monkeypatch.setattr(service,'post_publish',lambda *_a,**_k:calls.append(True) or {'publication_status':'PENDING'})
    base=['live','--serial','SN123','--camera-model','CAM','--replay','x']
    assert main(base)==0 and not calls
    assert main(base+['--publish'])==0 and calls==[True]


def test_invalid_config_does_not_expose_secret(tmp_path,capsys):
    p=tmp_path/'bad.json';write(p,{'remote':'https://SECRET_TOKEN@github.com/x/y','enabled':True})
    assert main(['publish','retry','--publish-config',str(p)])==2
    assert 'SECRET_TOKEN' not in capsys.readouterr().out
