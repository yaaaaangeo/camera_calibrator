"""Regression for actionable coverage coaching lost after a failed preflight."""
from dataclasses import replace
from pathlib import Path
from tests.test_p2_live import controller, prepare_task, checker, detection
from automation.live.controller import State


def test_failed_observability_keeps_specific_capture_coaching(controller):
    c, _ = controller
    task = prepare_task(c, 'observability')
    calculations_before = c.calculations
    c.readiness['coaching'] = ['Capture lower-left views.', 'Capture lower-right views.']
    c.finish_task({'task_id': task['task_id'], 'observability': {
        'ready': False, 'reasons': ['Strong radial parameter coupling.'],
        'report': {'observability_score': 3.5, 'rank': 8, 'jacobian_cols': 9,
                   'jacobian_rows': 80, 'normalized_condition_number': 100.,
                   'max_abs_correlation': .9833}}})
    assert c.state == State.CAPTURING
    assert c.guidance[-2:] == ['Capture lower-left views.', 'Capture lower-right views.']
    assert any('rank' in reason for reason in c.guidance)
    assert c.calculations == calculations_before
    assert not c.readiness['observability']['ready']


def test_quality_failure_collects_new_views_before_rechecking(controller, monkeypatch):
    c, clock = controller
    c.profile.readiness = replace(c.profile.readiness, min_images=3)
    task = prepare_task(c)
    c.last_calculated_count = len(c.paths)
    c.last_checked_count = len(c.paths)
    preserved = {path: Path(path).read_bytes() for path in c.paths}
    c.finish_task({'task_id': task['task_id'], 'outcome': {
        'status': 'completed', 'verdict': 'FAIL', 'message': 'quality'},
        'acceptance': {'criteria_complete': True, 'fail_kinds': ['quality']},
        'capture_recommendations': ['Capture lower views.']})
    calls = []

    def readiness(data, *_):
        calls.append(len(data.frames))
        return {'ready': True, 'reasons': [], 'coaching': [],
                'train_ids': [f.image_info.image_id for f in data.frames[:3]]}, data

    monkeypatch.setattr('automation.live.controller.check_readiness', readiness)
    assert c.state == State.CAPTURING and c.recaptures == 1
    for i in range(1, 5):
        clock.step()
        assert c.accept_frame(checker(), detection=detection(i * 15)) == 'saved'
        assert c.maybe_check() is None
    assert not calls
    clock.step()
    assert c.accept_frame(checker(), detection=detection(75)) == 'saved'
    preflight = c.maybe_check()
    assert calls == [6] and preflight['kind'] == 'observability'
    final_task = c.finish_task({'task_id': preflight['task_id'],
                               'observability': {'ready': True, 'reasons': []}})
    assert final_task['kind'] == 'calibration' and c.calculations == 2
    for path, contents in preserved.items():
        assert Path(path).read_bytes() == contents
