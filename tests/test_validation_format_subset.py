"""--models로 일부 모델만 계산했을 때 검증 표 출력이 KeyError 없이 동작하는지.

기존에는 표 함수가 4개 모델이 모두 있다고 가정해, `app.cli --models brown_conrady rational`
(--quiet 없이)이 Hold-out 직후 KeyError로 종료 코드 2가 됐다.
"""

from __future__ import annotations

from calibration.types import CameraModelType, ValidationResult
from calibration.validation import (
    format_straightness_comparison,
    format_train_test_residual_comparison,
    format_validation_table,
)

ALL = [CameraModelType.PINHOLE, CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]


def _results(models):
    return {m: ValidationResult(train_rms=0.2, test_rms=0.3, edge_rms=0.4) for m in models}


def test_tables_accept_model_subset():
    subset = _results([CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE])
    for formatter in (format_validation_table, format_train_test_residual_comparison, format_straightness_comparison):
        text = formatter(subset)
        assert isinstance(text, str) and text


def test_full_model_set_output_is_unchanged_in_shape():
    full = format_validation_table(_results(ALL))
    subset = format_validation_table(_results(ALL[1:3]))
    assert full.splitlines()[0].count("|") >= subset.splitlines()[0].count("|")
    assert len(full.splitlines()) == len(subset.splitlines())
