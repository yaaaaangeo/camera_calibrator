import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.test_p2_live import controller, checker, detection, prepare_task


def test_cancel_stops_active_task_and_keeps_captures(controller,monkeypatch):
    import automation.live.cli as cli
    c,_=controller; task=prepare_task(c); c.state=cli.State.CAPTURING
    class Source:
        exhausted=False
        def start(self): pass
        def poll(self): return None
        def clear(self): pass
        def stop(self): pass
    stopped=[]
    class Process:
        def __init__(self,_): pass
        def poll(self): return None
        def stop(self): stopped.append(True)
    def check():
        c.state=cli.State.CALIBRATING
        return task
    c.maybe_check=check
    monkeypatch.setattr(cli,'TaskProcess',Process)
    paths=list(c.paths)
    def status(line):
        if json.loads(line)['state']=='CALIBRATING': (c.session/'cancel.request').touch()
    assert cli.run(c,Source(),emit=status,poll_sec=.001,replay_interval=.001)==130
    assert stopped and c.paths==paths and all(Path(p).is_file() for p in paths)


def test_real_owned_process_tree_can_be_cancelled(tmp_path):
    from automation.live.process import TaskProcess,process_alive
    pid_file=tmp_path/'grandchild.pid'
    child_code="import time; time.sleep(60)"
    root_code=("from automation.live.windows_job import join_job_from_environment; join_job_from_environment(); "
               "import subprocess,sys,time,pathlib; "
               f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
               f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)")
    options=({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
             if os.name=='nt' else {'start_new_session':True})
    task=TaskProcess.__new__(TaskProcess)
    task.job=None
    env=dict(os.environ)
    if os.name=='nt':
        from automation.live.windows_job import WindowsJob
        task.job=WindowsJob()
        env['CAMERA_P2_JOB']=task.job.name
    task.log=open(tmp_path/'owned.log','w')
    task.process=subprocess.Popen([sys.executable,'-c',root_code],stdout=task.log,stderr=task.log,env=env,**options)
    if task.job is not None: task.job.assign(task.process)
    try:
        deadline=time.monotonic()+5
        while not pid_file.exists() and time.monotonic()<deadline: time.sleep(.05)
        assert pid_file.exists()
        grandchild=int(pid_file.read_text())
        assert process_alive(task.process.pid) and process_alive(grandchild)
        task.stop()
        deadline=time.monotonic()+3
        while process_alive(grandchild) and time.monotonic()<deadline: time.sleep(.05)
        assert task.process.poll() is not None and not process_alive(grandchild)
    finally:
        task.stop()


def test_interrupted_calculation_is_recoverable_and_counts_do_not_reset(controller):
    from automation.live.controller import LiveController,State
    c,_=controller; prepare_task(c); c.last_calculated_count=len(c.paths); c.checkpoint(); c.close()
    recovered=LiveController(c.profile,c.identity,c.source,resume=c.session)
    try:
        assert recovered.state==State.CONNECTING and recovered.recover_retry
        assert recovered.calculations==1 and recovered.paths==c.paths
    finally: recovered.close()


def test_runtime_task_timeout_is_error_without_recapture(controller,monkeypatch):
    from dataclasses import replace
    import automation.live.cli as cli
    c,_=controller; task=prepare_task(c); c.state=cli.State.CAPTURING
    c.profile.recovery=replace(c.profile.recovery,calculation_timeout_sec=.005)
    class Source:
        exhausted=False
        def start(self): pass
        def poll(self): return None
        def clear(self): pass
        def stop(self): pass
    stopped=[]
    class Process:
        def __init__(self,_): pass
        def poll(self): return None
        def stop(self): stopped.append(True)
    def check(): c.state=cli.State.CALIBRATING; return task
    c.maybe_check=check
    monkeypatch.setattr(cli,'TaskProcess',Process)
    assert cli.run(c,Source(),emit=lambda _:None,poll_sec=.001,replay_interval=.001)==2
    assert c.state==cli.State.ERROR and stopped and c.recaptures==0
