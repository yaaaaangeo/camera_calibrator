"""Actual P1/P2 CLI -> exports -> sanitized outbox -> LOCAL bare Git."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from tests._automation_test_utils import render_consistent_charuco_dataset,base_profile
from tests.test_p2_e2e import replay_profile
from tests._publish_test_utils import bare,git,write,snapshot
from calibration.project_io import load_project
from automation.publish.history import list_history

pytestmark=pytest.mark.slow


@pytest.fixture(scope='module')
def published_runs(tmp_path_factory):
    root=tmp_path_factory.mktemp('published_pipeline');images=root/'images'
    render_consistent_charuco_dataset(images,24,7);before=snapshot(images)
    remote=bare(root/'results.git');outbox=root/'outbox'
    cfg=root/'publisher.json';write(cfg,{'enabled':True,'remote':str(remote),'outbox':str(outbox)})
    p1=root/'p1.yaml';p1.write_text(yaml.safe_dump(base_profile()),encoding='utf-8')
    p2=root/'p2.yaml';p2.write_text(yaml.safe_dump(replay_profile()),encoding='utf-8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',QT_QPA_PLATFORM='offscreen')
    results={}
    for mode in ['p1','p2']:
        args=(['run','--profile',str(p1),'--images',str(images)] if mode=='p1' else
              ['live','--profile',str(p2),'--replay',str(images),'--replay-interval','.01'])
        p=subprocess.run([sys.executable,'-m','automation',*args,'--serial','P3-'+mode,
            '--camera-model','TEST-CAM','--output-root',str(root/mode),'--publish','--publish-config',str(cfg)],
            env=env,capture_output=True,text=True,encoding='utf-8',timeout=360)
        (root/(mode+'-stdout.log')).write_text(p.stdout,encoding='utf-8')
        (root/(mode+'-stderr.log')).write_text(p.stderr,encoding='utf-8')
        rows=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
        publication=next((r for r in rows if r.get('event')=='publication'),None)
        assert publication and publication['publication_status']=='PUBLISHED',(p.returncode,p.stdout[-3000:],p.stderr)
        assert p.returncode==(3 if mode=='p1' else 0)
        results[mode]={'publication':publication,'rows':rows}
    assert snapshot(images)==before
    write(root/'verification.json',results)
    return root,remote,outbox,results


@pytest.mark.parametrize('mode',['p1','p2'])
def test_real_calculation_to_local_git_and_history(published_runs,mode):
    root,remote,outbox,results=published_runs
    publication=results[mode]['publication'];receipt=publication['receipt']
    uploaded=json.loads(git('--git-dir='+str(remote),'show',receipt['commit_sha']+':'+receipt['run_path']+'/summary.json'))
    assert uploaded['acceptance_verdict']=='WARN'
    assert uploaded['approval_status']=='UNREVIEWED' and uploaded['deployment_status']=='NOT_DEPLOYED'
    records=list_history(outbox,'P3-'+mode);assert len(records)==1
    assert records[0]['publication_status']=='PUBLISHED'
    assert records[0]['publication_commit_sha']==receipt['commit_sha']
    manifest=next((root/mode).glob('**/run_manifest.json'))
    project,_=load_project(str(manifest.parent/'project/calibration.ccproj'))
    chosen=project.final_result.chosen_model
    assert uploaded['chosen_model']==chosen.value
    np.testing.assert_array_equal(uploaded['K'],project.calibration_results[chosen].camera_matrix)
    np.testing.assert_array_equal(uploaded['D'],project.calibration_results[chosen].distortion.ravel())
    assert git('--git-dir='+str(remote),'rev-list','--count','--all')=='2'
