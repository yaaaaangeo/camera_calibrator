"""Optional local real-55-image replay. No ROS/hardware claim or threshold easing."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from automation.run_manifest import input_fingerprint
from calibration.project_io import load_project

pytestmark = pytest.mark.slow
IMAGES = Path(__file__).resolve().parents[1] / 'image'


@pytest.fixture(scope='module')
def real_gate_run(tmp_path_factory):
    paths = sorted(p for p in IMAGES.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(paths) != 55:
        pytest.skip('Local 55-image dataset unavailable; this is not hardware coverage.')
    root = tmp_path_factory.mktemp('real55_gate')
    before = input_fingerprint([str(p) for p in paths])
    # Same physical-board assumption and quality/readiness defaults as the
    # previous real-55 audit. Only replay/capture time is accelerated.
    profile = root/'profile.yaml'
    profile.write_text(yaml.safe_dump({'live_profile_version': 1,
        'p1': {'pattern': {'squares_x': 10, 'squares_y': 7}},
        'capture': {'interval_sec': .001}}), encoding='utf-8')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.run([sys.executable, '-m', 'automation', 'live', '--profile', str(profile),
        '--replay', str(IMAGES), '--replay-interval', '.01', '--serial', 'REAL55',
        '--camera-model', 'GATE-AUDIT', '--output-root', str(root/'runs')],
        capture_output=True, text=True, encoding='utf-8', env=env, timeout=360)
    (root/'stdout.log').write_text(proc.stdout, encoding='utf-8')
    (root/'stderr.log').write_text(proc.stderr, encoding='utf-8')
    snapshots = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
    assert snapshots, proc.stderr
    final = snapshots[-1]
    assert proc.returncode == 0 and final['state'] == 'COMPLETED', (final, proc.stderr)
    assert input_fingerprint([str(p) for p in paths]) == before
    return root, final


def test_real55_replay_computes_but_does_not_approve_low_observability(real_gate_run):
    root, final = real_gate_run
    assert final['calculations'] == 1 and final['recaptures'] == 0
    assert final['attempts'][0]['outcome']['verdict'] == 'WARN'
    manifest = json.loads((Path(final['session'])/'p2_manifest.json').read_text(encoding='utf-8'))
    obs = manifest['readiness']['observability']
    assert obs['ready'] and obs['review_required']
    assert obs['report']['observability_score'] < 50.
    assert obs['report']['rank'] == obs['report']['jacobian_cols'] == 9
    p2 = json.loads(Path(final['attempts'][0]['p2_acceptance_path']).read_text(encoding='utf-8'))
    assert p2['verdict'] == 'WARN' and p2['warnings']
    (root/'final-check.json').write_text(json.dumps(final, indent=2), encoding='utf-8')


def test_real_gate_matches_exact_input_direct_cli(real_gate_run):
    from automation.runner import _run_cli_child
    root, final = real_gate_run
    p1 = Path(final['attempts'][0]['outcome']['session_dir'])
    config = yaml.safe_load((p1/'automation/cli_config.yaml').read_text(encoding='utf-8'))
    direct = root/'direct'; direct.mkdir()
    config.update(session_dir=str(direct/'session'), save_project=str(direct/'calibration.ccproj'),
                  json_summary=str(direct/'summary.json'), log_file=str(direct/'debug.log'))
    spec = direct/'config.yaml'; spec.write_text(yaml.safe_dump(config), encoding='utf-8')
    assert _run_cli_child(spec, direct, direct/'output.log', 240, 1) == 0
    auto, _ = load_project(str(p1/'project/calibration.ccproj'))
    reference, _ = load_project(str(direct/'calibration.ccproj'))
    assert auto.final_result.chosen_model == reference.final_result.chosen_model
    for model, a in auto.calibration_results.items():
        b = reference.calibration_results[model]
        assert a.success == b.success
        if a.success:
            np.testing.assert_array_equal(a.camera_matrix, b.camera_matrix)
            np.testing.assert_array_equal(a.distortion, b.distortion)
            assert a.rms_error == b.rms_error
            av, bv = auto.validation_results[model], reference.validation_results[model]
            assert av.test_rms == bv.test_rms and av.edge_rms == bv.edge_rms
