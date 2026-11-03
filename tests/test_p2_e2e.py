"""Real subprocess/replay integration on known-K synthetic images, not hardware."""
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from tests._automation_test_utils import render_consistent_charuco_dataset
from automation.run_manifest import sha256_file
from calibration.project_io import load_project

pytestmark = pytest.mark.slow


def replay_profile():
    return {'live_profile_version':1,'capture':{'interval_sec':.001,'motion_blur_max':20.,'roi_contrast_min':1.},
            'readiness':{'min_images':16,'coverage_pct':0.,'min_center_cells':0,'require_corner_regions':False,
                         'distance_diversity':0.,'tilt_diversity':0.,'edge_coverage':0.,'max_duplicate_ratio':1.,
                         'min_test_frames':1,'min_test_corners':4,'test_coverage_pct':0.,'test_pose_diversity':0.,
                         'check_every_new_frames':1},
            'recovery':{'calculation_timeout_sec':240.}}


@pytest.fixture(scope='module')
def replay_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('p2_e2e')
    images=root/'images'; render_consistent_charuco_dataset(images,count=24,seed=7)
    before={p.name:sha256_file(p) for p in images.iterdir()}
    profile=root/'p2.yaml'; profile.write_text(yaml.safe_dump(replay_profile()),encoding='utf-8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    proc=subprocess.run([sys.executable,'-m','automation','live','--profile',str(profile),
                         '--replay',str(images),'--replay-interval','.01',
                         '--serial','REPLAY','--camera-model','TEST','--output-root',str(root/'runs')],
                        capture_output=True,text=True,encoding='utf-8',env=env,timeout=300)
    (root/'stdout.log').write_text(proc.stdout,encoding='utf-8')
    (root/'stderr.log').write_text(proc.stderr,encoding='utf-8')
    snapshots=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
    assert snapshots,proc.stderr
    final=snapshots[-1]
    assert proc.returncode==0,(proc.returncode,final,proc.stderr)
    assert final['state']=='COMPLETED',final
    return root,images,before,final,snapshots


def test_replay_triggers_existing_p1_and_preserves_sources(replay_run):
    root,images,before,final,snapshots=replay_run
    assert before=={p.name:sha256_file(p) for p in images.iterdir()}
    assert final['calculations']==1 and final['recaptures']==0
    assert final['attempts'][0]['outcome']['verdict']=='WARN'  # undefined operating thresholds
    assert any(s['state']=='CALIBRATING' for s in snapshots)
    session=Path(final['session'])
    manifest=json.loads((session/'p2_manifest.json').read_text(encoding='utf-8'))
    assert manifest['state']=='COMPLETED' and len(manifest['hashes'])>=16
    assert (session/'capture_events.jsonl').is_file()
    assert manifest['readiness']['observability']['ready']
    assert manifest['readiness']['observability']['seconds']>=0


def test_p2_matches_direct_cli_on_exact_captured_inputs(replay_run):
    from automation.runner import _run_cli_child
    root,_,_,final,_=replay_run
    p1=Path(final['attempts'][0]['outcome']['session_dir'])
    config=yaml.safe_load((p1/'automation/cli_config.yaml').read_text(encoding='utf-8'))
    direct=root/'direct'; direct.mkdir()
    config.update(session_dir=str(direct/'session'),save_project=str(direct/'calibration.ccproj'),
                  json_summary=str(direct/'summary.json'),log_file=str(direct/'debug.log'))
    config_path=direct/'config.yaml'; config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
    assert _run_cli_child(config_path,direct,direct/'output.log',240,1)==0
    auto,_=load_project(str(p1/'project/calibration.ccproj'))
    reference,_=load_project(str(direct/'calibration.ccproj'))
    assert auto.final_result.chosen_model==reference.final_result.chosen_model
    assert [s.model_name for s in auto.model_scores if s.is_recommended]==[s.model_name for s in reference.model_scores if s.is_recommended]
    for model,a in auto.calibration_results.items():
        b=reference.calibration_results[model]
        assert a.success==b.success
        if a.success:
            np.testing.assert_array_equal(a.camera_matrix,b.camera_matrix)
            np.testing.assert_array_equal(a.distortion,b.distortion)
            assert a.rms_error==b.rms_error
            av,bv=auto.validation_results[model],reference.validation_results[model]
            assert av.train_frame_ids==bv.train_frame_ids and av.test_frame_ids==bv.test_frame_ids
            assert av.test_rms==bv.test_rms and av.edge_rms==bv.edge_rms
