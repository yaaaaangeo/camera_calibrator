"""
tests/test_compare_model_rankings.py
===========================================

설계 문서 17번 - "Outlier 제거 전후 효과 측정"의 마지막 항목,
"model ranking 변화".

주의: 이 프로젝트의 ModelScore.score는 "낮을수록 좋다"(오차 성격의 가중합 -
compute_model_scores docstring 참고). 아래 테스트 데이터도 전부 이 규칙을
따른다 - 가장 낮은 점수를 가진 모델이 1위이자 추천 모델이어야 한다.
"""

from __future__ import annotations

from calibration.recommender import compare_model_rankings
from calibration.types import CameraModelType, ModelScore


def _score(model, score, recommended=False) -> ModelScore:
    return ModelScore(model_name=model, score=score, is_recommended=recommended)


class TestCompareModelRankings:
    def test_no_change_when_order_and_recommendation_same(self):
        before = [
            _score(CameraModelType.FISHEYE, 0.30, recommended=True),
            _score(CameraModelType.EXTENDED_PINHOLE, 0.55),
            _score(CameraModelType.PINHOLE, 0.60),
        ]
        after = [
            _score(CameraModelType.FISHEYE, 0.29, recommended=True),
            _score(CameraModelType.EXTENDED_PINHOLE, 0.50),
            _score(CameraModelType.PINHOLE, 0.58),
        ]
        text = compare_model_rankings(before, after)
        assert "바뀌지 않았습니다" in text

    def test_detects_recommendation_change(self):
        before = [
            _score(CameraModelType.FISHEYE, 0.30, recommended=True),
            _score(CameraModelType.EXTENDED_PINHOLE, 0.59),
            _score(CameraModelType.PINHOLE, 0.61),
        ]
        after = [
            _score(CameraModelType.EXTENDED_PINHOLE, 0.29, recommended=True),
            _score(CameraModelType.FISHEYE, 0.58),
            _score(CameraModelType.PINHOLE, 0.62),
        ]
        text = compare_model_rankings(before, after)
        assert "바뀌었습니다" in text
        assert "Fisheye" in text and "Rational Pinhole" in text

    def test_includes_ranked_order_for_both_sides(self):
        before = [
            _score(CameraModelType.PINHOLE, 0.60),
            _score(CameraModelType.FISHEYE, 0.30, recommended=True),
            _score(CameraModelType.EXTENDED_PINHOLE, 0.55),
        ]
        after = list(before)
        text = compare_model_rankings(before, after)
        lines = text.splitlines()
        assert any(line.startswith("1\uc704") and "Fisheye" in line for line in lines)

    def test_lowest_score_ranks_first_not_highest(self):
        """회귀 방지용 - score는 낮을수록 좋으므로 오름차순 정렬이어야 한다
        (한때 내림차순으로 잘못 정렬했던 버그의 재발 방지)."""
        scores = [
            _score(CameraModelType.PINHOLE, 0.90),
            _score(CameraModelType.EXTENDED_PINHOLE, 0.10, recommended=True),
            _score(CameraModelType.FISHEYE, 0.50),
        ]
        text = compare_model_rankings(scores, scores)
        first_rank_line = next(line for line in text.splitlines() if line.startswith("1\uc704"))
        assert "Rational Pinhole" in first_rank_line

    def test_empty_inputs_return_message(self):
        assert "정보가 없습니다" in compare_model_rankings([], [])
        assert "정보가 없습니다" in compare_model_rankings([_score(CameraModelType.PINHOLE, 0.5)], [])

    def test_handles_different_length_lists(self):
        before = [
            _score(CameraModelType.FISHEYE, 0.3, recommended=True),
            _score(CameraModelType.PINHOLE, 0.6),
        ]
        after = [
            _score(CameraModelType.FISHEYE, 0.3, recommended=True),
        ]
        text = compare_model_rankings(before, after)
        assert "N/A" in text
