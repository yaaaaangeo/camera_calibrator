"""P2 feasibility vs quality policy. Never changes fitting or P1 acceptance."""
from __future__ import annotations

import copy
import math

POLICY_VERSION = 2


def assess_observability(report, config):
    """A local coefficient correlation is a review signal, not fit failure."""
    blockers, warnings = [], []
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    rank, cols = report.get('rank'), report.get('jacobian_cols')
    condition = report.get('normalized_condition_number', report.get('condition_number'))
    score, correlation = report.get('observability_score'), report.get('max_abs_correlation')
    if not finite(rank) or not finite(cols) or cols <= 0 or report.get('jacobian_rows', 0) <= 0:
        blockers.append('관측성 Jacobian을 평가할 수 없습니다. 데이터/진단 결과를 수동 검토하세요.')
    elif rank < cols:
        # Retain the existing configurable full-rank preflight requirement.
        (blockers if config.require_full_rank else warnings).append(f'파라미터 rank가 부족합니다 ({rank}/{cols}).')
    if not finite(condition):
        blockers.append('유한한 Condition Number를 얻지 못했습니다. 식별가능성을 검토하세요.')
    elif condition >= 1e8:  # existing observability warning boundary
        warnings.append(f'Condition Number가 높습니다 ({condition:.6g}). 결과 승인이 필요합니다.')
    if not finite(score):
        blockers.append('Observability Score를 산출하지 못했습니다.')
    elif score < config.observability_min:
        warnings.append(f'Observability Score {score:.1f} < 검토 기준 {config.observability_min:.1f}: 계산은 허용하되 PASS로 승인하지 않습니다.')
    if not finite(correlation):
        blockers.append('파라미터 상관계수를 평가할 수 없습니다.')
    elif correlation >= .98:  # existing observability warning boundary
        pairs = report.get('top_correlations') or []
        pair = f" ({pairs[0]['param_a']}/{pairs[0]['param_b']})" if pairs else ''
        warnings.append(f'파라미터 상관계수가 높습니다{pair}: {correlation:.6f}. 모델/예측 오차를 검토하세요.')
    return {'policy_version': POLICY_VERSION, 'ready': not blockers,
            'review_required': bool(blockers or warnings), 'blockers': blockers,
            'warnings': warnings, 'reasons': blockers or warnings, 'report': report}


def review_result(outcome, acceptance, model_reports, chosen_model, config, preflight=None):
    """P2 overlay: preserve P1 artifacts and never promote its WARN/FAIL."""
    effective = copy.deepcopy(acceptance or {})
    reviewed_outcome = copy.deepcopy(outcome)
    checks = {name: assess_observability(report or {}, config) for name, report in model_reports.items()}
    chosen = checks.get(chosen_model) or assess_observability({}, config)
    blockers = list(chosen['blockers'])
    # Full-result rank deficiency is never approved, even with the legacy
    # preflight require_full_rank=false option.
    report = chosen['report']
    if report.get('rank', 0) < report.get('jacobian_cols', 0) and not blockers:
        blockers.append('최종 선택 모델의 파라미터 rank가 부족합니다.')
    warnings = list(chosen['warnings'])
    if preflight:
        prior = assess_observability(preflight.get('report') or {}, config)
        warnings.extend('사전 train 진단: ' + r for r in prior['reasons'])
    warnings = list(dict.fromkeys(warnings))
    original = outcome.get('verdict')
    verdict = 'FAIL' if original == 'FAIL' or blockers else 'WARN' if original == 'WARN' or warnings else 'PASS'
    effective['verdict'] = verdict
    effective['reasons'] = list(dict.fromkeys(effective.get('reasons', []) + blockers + warnings))
    effective['fail_kinds'] = sorted(set(effective.get('fail_kinds', [])) | ({'identifiability'} if blockers else set()))
    effective['warn_kinds'] = sorted(set(effective.get('warn_kinds', [])) | ({'observability_review'} if warnings else set()))
    reviewed_outcome.update(verdict=verdict, message='; '.join(effective['reasons']))
    if outcome.get('status') == 'completed':
        reviewed_outcome['exit_code'] = {'PASS': 0, 'WARN': 3, 'FAIL': 4}[verdict]
    overlay = {'policy_version': POLICY_VERSION, 'p1_verdict': original, 'verdict': verdict,
               'chosen_model': chosen_model, 'model_observability': checks,
               'preflight': preflight, 'review_required': verdict != 'PASS',
               'blockers': blockers, 'warnings': warnings,
               'observability_only': original == 'PASS' and verdict == 'WARN'}
    return reviewed_outcome, effective, overlay
