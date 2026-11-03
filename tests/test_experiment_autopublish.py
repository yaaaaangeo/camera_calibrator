"""Publication ordering and opt-in retries; local bare Git only."""
import json
from dataclasses import replace

import pytest

from automation.experiment.config import load_experiment
from automation.experiment.runner import run_experiment
from automation.publish.base import PublishError
from automation.publish.bundle import verify_bundle
from automation.publish.git_publisher import GitPublisher
from tests.test_experiment import cached_experiment, profile
from tests._publish_test_utils import git, snapshot


def test_opt_in_retry_finishes_same_run_without_recalculation(tmp_path, monkeypatch):
    exp, path, session, remote = cached_experiment(tmp_path, enabled=True)
    exp.raw['publish']['retry_in_process'] = True
    exp.publisher = replace(exp.publisher, base_backoff_sec=0)
    original = GitPublisher.publish
    calls = []
    before = snapshot(session)

    def flaky(self, bundle):
        verify_bundle(bundle)
        calls.append(bundle)
        if len(calls) == 1:
            raise PublishError('GIT_TRANSPORT_FAILED', True)
        return original(self, bundle)

    monkeypatch.setattr(GitPublisher, 'publish', flaky)
    monkeypatch.setattr('automation.live.cli.run', lambda *a, **k: pytest.fail('recalculation'))
    assert run_experiment(exp) == 3
    state = json.loads((exp.folder/'experiment.json').read_text())
    assert state['status'] == 'COMPLETED'
    assert state['publication']['status'] == 'PUBLISHED'
    assert state['publication']['attempts'] == 2 and len(calls) == 2
    assert snapshot(session) == before
    assert run_experiment(exp) == 3 and len(calls) == 2
    assert git('--git-dir='+str(remote), 'rev-list', '--count', '--all') == '1'


@pytest.mark.parametrize('missing', ['reports/report.html', 'intrinsic/camera_info.yaml'])
def test_incomplete_export_never_calls_publisher(tmp_path, monkeypatch, missing):
    exp, _, session, remote = cached_experiment(tmp_path, enabled=True)
    (session/missing).unlink()  # disposable fixture, never real results
    monkeypatch.setattr(GitPublisher, 'publish', lambda *a: pytest.fail('incomplete publish'))
    assert run_experiment(exp) == 2
    state = json.loads((exp.folder/'experiment.json').read_text())
    assert state['failure']['stage'] == 'publication'
    assert git('--git-dir='+str(remote), 'rev-list', '--all', '--count') == '0'
    assert not list((exp.output/'outbox/items').glob('*/bundle'))


@pytest.mark.parametrize('retryable', [True, False])
def test_automatic_retry_is_bounded_and_keeps_bundle(tmp_path, monkeypatch, retryable):
    exp, _, session, _ = cached_experiment(tmp_path, enabled=True)
    exp.raw['publish']['retry_in_process'] = True
    exp.publisher = replace(exp.publisher, max_attempts=2, base_backoff_sec=0)
    before = snapshot(session)
    def fail(*a):
        raise PublishError('GIT_TRANSPORT_FAILED' if retryable else 'AUTHENTICATION_FAILED', retryable)
    monkeypatch.setattr(GitPublisher, 'publish', fail)
    assert run_experiment(exp) == 2
    state = json.loads((exp.folder/'experiment.json').read_text())
    assert state['publication']['status'] == 'FAILED'
    assert state['publication']['attempts'] == (2 if retryable else 1)
    assert snapshot(session) == before
    verify_bundle(exp.output/'outbox/items'/state['publication']['job_id']/'bundle')


def test_retry_setting_is_strict_boolean(tmp_path):
    import yaml
    _, path = profile(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['publish']['retry_in_process'] = True
    path.write_text(yaml.safe_dump(raw))
    assert load_experiment(path).raw['publish']['retry_in_process'] is True
    raw['publish']['retry_in_process'] = 'true'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment(path)


def test_cancel_during_retry_preserves_result_and_resumes(tmp_path, monkeypatch):
    exp, _, session, _ = cached_experiment(tmp_path, enabled=True)
    exp.raw['publish']['retry_in_process'] = True
    exp.publisher = replace(exp.publisher, base_backoff_sec=0)
    original = GitPublisher.publish
    before = snapshot(session)
    def fail(*a):
        raise PublishError('GIT_TRANSPORT_FAILED', True)
    monkeypatch.setattr(GitPublisher, 'publish', fail)
    def cancel(*a):
        raise KeyboardInterrupt()
    monkeypatch.setattr('automation.experiment.runner.sleep', cancel)
    assert run_experiment(exp) == 130
    state = json.loads((exp.folder/'experiment.json').read_text())
    assert state['status'] == 'INTERRUPTED' and state['publication']['status'] == 'RETRY_WAIT'
    monkeypatch.setattr(GitPublisher, 'publish', original)
    monkeypatch.setattr('automation.live.cli.run', lambda *a, **k: pytest.fail('recalculation'))
    assert run_experiment(exp, retry_only=True) == 3
    assert snapshot(session) == before


def test_calculation_failure_never_calls_publisher(tmp_path, monkeypatch):
    from automation.live.controller import State
    from tests._publish_test_utils import bare
    remote = bare(tmp_path/'results.git')
    exp, _ = profile(tmp_path, publish={'results_repository':True,
        'config':{'enabled':True,'remote':str(remote)}})
    class FailedController:
        def __init__(self, *a, output_root=None, **k):
            self.session = output_root/'failed-capture'
            self.session.mkdir(parents=True)
            self.state = State.ERROR
            self.attempts = []
            self.guidance = ['calculation or export failed']
        def close(self):
            pass
    monkeypatch.setattr('automation.live.controller.LiveController', FailedController)
    monkeypatch.setattr('automation.live.cli.run', lambda *a, **k: 2)
    monkeypatch.setattr(GitPublisher, 'publish', lambda *a: pytest.fail('failed calibration published'))
    assert run_experiment(exp) == 2
    state = json.loads((exp.folder/'experiment.json').read_text())
    assert state['failure']['stage'] == 'capture_readiness_calibration_export'
    assert 'publication' not in state
    assert git('--git-dir='+str(remote), 'rev-list', '--all', '--count') == '0'
