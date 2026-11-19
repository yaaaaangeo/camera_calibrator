from dataclasses import replace
import json
from pathlib import Path

import pytest

from automation.live.assessment import assess_observability, review_result, POLICY_VERSION
from automation.live.config import ReadinessConfig
from automation.live.controller import LiveController, State
from tests.test_p2_live import controller, prepare_task, checker, detection


def report(**overrides):
    return {'rank': 9, 'jacobian_cols': 9, 'jacobian_rows': 2614,
            'normalized_condition_number': 100.01120147554356,
            'observability_score': 3.5, 'max_abs_correlation': .9833169155317061,
            'top_correlations': [{'param_a': 'k2', 'param_b': 'k3', 'correlation': -.9833169155317061}],
            **overrides}


def test_recorded_real_low_score_is_review_not_compute_failure():
    result = assess_observability(report(), ReadinessConfig())
    assert result['ready'] and result['review_required'] and not result['blockers']
    assert len(result['warnings']) == 2
    assert ReadinessConfig().observability_min == 50.


@pytest.mark.parametrize('changes', [
    {'rank': 8}, {'jacobian_rows': 0}, {'normalized_condition_number': float('inf')},
    {'observability_score': None}, {'max_abs_correlation': None}, {}])
def test_feasibility_blocks_unidentifiable_or_unavailable_diagnostics(changes):
    value = {} if not changes else report(**changes)
    assert not assess_observability(value, ReadinessConfig())['ready']


@pytest.mark.parametrize('changes', [
    {'normalized_condition_number': 1e9}, {'max_abs_correlation': .99},
    {'observability_score': 49.9}])
def test_individual_quality_signals_warn_without_lowering_thresholds(changes):
    value = report(observability_score=90., max_abs_correlation=.5, **changes) if 'observability_score' not in changes and 'max_abs_correlation' not in changes else report(**changes)
    result = assess_observability(value, ReadinessConfig())
    assert result['ready'] and result['warnings']


@pytest.mark.parametrize('p1_verdict,expected', [('PASS', 'WARN'), ('WARN', 'WARN'), ('FAIL', 'FAIL')])
def test_overlay_never_promotes_p1_or_approves_low_observability(p1_verdict, expected):
    outcome = {'verdict': p1_verdict, 'status': 'completed'}
    acceptance = {'verdict': p1_verdict, 'criteria_complete': True}
    new, effective, overlay = review_result(outcome, acceptance, {'brown_conrady': report()},
                                            'brown_conrady', ReadinessConfig())
    assert new['verdict'] == effective['verdict'] == expected
    assert outcome['verdict'] == acceptance['verdict'] == p1_verdict
    assert overlay['p1_verdict'] == p1_verdict and overlay['review_required']


def test_chosen_rank_deficiency_fails_even_if_preflight_rank_option_disabled():
    outcome, acceptance, _ = review_result({'status': 'completed', 'verdict': 'PASS'}, {},
                                          {'brown_conrady': report(rank=8)}, 'brown_conrady',
                                          replace(ReadinessConfig(), require_full_rank=False))
    assert outcome['verdict'] == 'FAIL' and 'identifiability' in acceptance['fail_kinds']


def test_observability_only_warning_ends_in_review_without_recapture(controller):
    c, _ = controller; task = prepare_task(c)
    outcome, acceptance, overlay = review_result({'status': 'completed', 'verdict': 'PASS'},
        {'criteria_complete': True}, {'brown_conrady': report()}, 'brown_conrady', c.profile.readiness)
    c.finish_task({'task_id': task['task_id'], 'outcome': outcome, 'acceptance': acceptance,
                   'p2_acceptance': overlay, 'capture_recommendations': ['move board']})
    assert c.state == State.COMPLETED and c.recaptures == 0
    assert c.attempts[-1]['outcome']['verdict'] == 'WARN'


def test_same_selected_split_is_not_recomputed_after_more_captures(controller, monkeypatch):
    c, clock = controller; prepare_task(c); c.pending = None; c.state = State.CAPTURING
    c.profile.readiness = replace(c.profile.readiness, min_images=3)
    selected = list(c.paths)
    c.readiness.update(selected_paths=selected, test_ids=[])
    c.last_calculation_fingerprint = c.calculation_fingerprint()
    for i in range(1, 6):
        clock.step(); c.accept_frame(checker(), detection=detection(15*i))
    def check(*_):
        return {'ready': True, 'reasons': [], 'coaching': [], 'selected_paths': selected,
                'train_ids': c.readiness['train_ids'], 'test_ids': []}, c.dataset
    monkeypatch.setattr('automation.live.controller.check_readiness', check)
    assert c.maybe_check() is None and c.checks == 0 and c.pending is None
    saved = json.loads((c.session/'p2_manifest.json').read_text(encoding='utf-8'))
    assert saved['last_calculation_fingerprint'] == c.last_calculation_fingerprint


def test_cached_v1_preflight_is_reassessed_and_new_calculation_can_resume(controller):
    c, _ = controller; task = prepare_task(c, 'observability')
    Path(task['response']).write_text(json.dumps({'task_id': task['task_id'],
        'observability': {'ready': False, 'reasons': ['old score gate'], 'report': report()}}), encoding='utf-8')
    c.close()
    recovered = LiveController(c.profile, c.identity, c.source, resume=c.session)
    try:
        assert recovered.state == State.CALIBRATING and recovered.pending['kind'] == 'calibration'
        assert recovered.readiness['observability']['policy_version'] == POLICY_VERSION
        assert recovered.readiness['observability']['review_required']
    finally:
        recovered.close()


def test_healthy_result_can_pass_and_missing_chosen_report_cannot():
    good = report(observability_score=90., max_abs_correlation=.5)
    passed, _, _ = review_result({'status': 'completed', 'verdict': 'PASS'},
        {'criteria_complete': True}, {'brown_conrady': good}, 'brown_conrady', ReadinessConfig())
    failed, _, _ = review_result({'status': 'completed', 'verdict': 'PASS'},
        {'criteria_complete': True}, {}, 'brown_conrady', ReadinessConfig())
    assert passed['verdict'] == 'PASS' and failed['verdict'] == 'FAIL'


def test_cached_p1_only_pass_resumes_for_review_without_rewriting_response(controller):
    c, _ = controller; task = prepare_task(c)
    response = {'task_id': task['task_id'], 'outcome': {'status': 'completed', 'verdict': 'PASS'},
                'acceptance': {'criteria_complete': True}}
    path = Path(task['response']); path.write_text(json.dumps(response), encoding='utf-8')
    before = path.read_bytes(); c.close()
    recovered = LiveController(c.profile, c.identity, c.source, resume=c.session)
    try:
        assert recovered.state == State.COMPLETED
        assert recovered.attempts[-1]['outcome']['verdict'] == 'WARN'
        assert path.read_bytes() == before
    finally:
        recovered.close()


def test_runtime_starts_pending_calculation_created_by_recovery(controller,monkeypatch):
    import automation.live.cli as cli
    c, _ = controller; task = prepare_task(c); started = []
    class Source:
        exhausted = False
        def start(self): pass
        def stop(self): pass
        def clear(self): pass
    class Process:
        def __init__(self, request):
            started.append(request['task_id'])
            Path(request['response']).write_text(json.dumps({'task_id': request['task_id'],
                'outcome': {'status': 'completed', 'verdict': 'WARN', 'message': 'review'},
                'acceptance': {'criteria_complete': False}}), encoding='utf-8')
        def poll(self): return 0
        def close(self): pass
        def stop(self): pass
    monkeypatch.setattr(cli,'TaskProcess',Process)
    assert cli.run(c,Source(),emit=lambda _:None,poll_sec=.001)==0
    assert started==[task['task_id']] and c.state==State.COMPLETED


def test_low_score_keeps_location_coaching_during_compute_and_manual_review(controller):
    c, _ = controller; task = prepare_task(c, 'observability')
    c.readiness['coaching'] = ['Capture lower-left views.']
    calculation = c.finish_task({'task_id': task['task_id'], 'observability': {
        'ready': False, 'reasons': ['old gate'], 'report': report()}})
    assert calculation['kind'] == 'calibration' and c.state == State.CALIBRATING
    assert 'Capture lower-left views.' in c.guidance
    c.finish_task({'task_id': calculation['task_id'], 'outcome': {
        'status': 'completed', 'verdict': 'WARN'}, 'acceptance': {
        'criteria_complete': False, 'reasons': ['criteria missing']}})
    assert c.state == State.COMPLETED and 'Capture lower-left views.' in c.guidance
