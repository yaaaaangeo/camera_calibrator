"""
tests/test_project_migration.py
====================================

P1-D: legacy .ccproj (format_version=1) -> 현재 포맷(2) 마이그레이션 검증.

Fixture는 scripts/generate_legacy_project_fixtures.py로 생성된
tests/assets/projects/*.ccproj를 사용한다 (재생성하려면 그 스크립트를 실행).

핵심 판별 규칙(project_io.migrate_v1_to_v2):
    v1 "extended_pinhole" + distortion 5계수  -> brown_conrady로 이름 변경
    v1 "extended_pinhole" + distortion 8계수 이상 -> 그대로 (오늘날 Rational과 동일 의미)
    "pinhole" -> 항상 그대로 (Ideal Pinhole)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.project_io import (
    PROJECT_FORMAT_VERSION,
    load_project,
    migrate_v1_to_v2,
    project_from_dict,
    save_project,
)
from calibration.types import CameraModelType

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "projects"


def test_fixtures_exist():
    for name in (
        "v1_pinhole.ccproj",
        "v1_extended_5coeff.ccproj",
        "v1_extended_rational.ccproj",
        "v1_extended_5coeff_mixed_refs.ccproj",
        "v2_project.ccproj",
    ):
        assert (ASSETS_DIR / name).exists(), f"missing fixture: {name}"


def test_v1_pinhole_loads_unchanged():
    project, missing = load_project(str(ASSETS_DIR / "v1_pinhole.ccproj"))
    assert CameraModelType.PINHOLE in project.calibration_results
    assert project.calibration_results[CameraModelType.PINHOLE].model_name == CameraModelType.PINHOLE


def test_v1_extended_5coeff_migrates_to_brown_conrady():
    project, _missing = load_project(str(ASSETS_DIR / "v1_extended_5coeff.ccproj"))

    assert CameraModelType.EXTENDED_PINHOLE not in project.calibration_results
    assert CameraModelType.BROWN_CONRADY in project.calibration_results
    result = project.calibration_results[CameraModelType.BROWN_CONRADY]
    assert result.model_name == CameraModelType.BROWN_CONRADY
    assert result.distortion is not None and len(result.distortion.reshape(-1)) == 5


def test_v1_extended_rational_stays_extended_pinhole():
    project, _missing = load_project(str(ASSETS_DIR / "v1_extended_rational.ccproj"))

    assert CameraModelType.BROWN_CONRADY not in project.calibration_results
    assert CameraModelType.EXTENDED_PINHOLE in project.calibration_results
    result = project.calibration_results[CameraModelType.EXTENDED_PINHOLE]
    assert result.model_name == CameraModelType.EXTENDED_PINHOLE
    assert result.distortion is not None and len(result.distortion.reshape(-1)) == 8


@pytest.mark.parametrize(
    "distortion",
    [
        [[0.1, 0.2, 0.0, 0.0, 0.01, 0.001, 0.002, 0.003]],
        [[0.1], [0.2], [0.0], [0.0], [0.01], [0.001], [0.002], [0.003]],
    ],
)
def test_v1_extended_rational_shape_is_flattened_before_classification(distortion):
    """OpenCV can serialize D as (1, 8) or (8, 1); migration must classify both as Rational."""
    with open(ASSETS_DIR / "v1_extended_5coeff.ccproj", "r", encoding="utf-8") as f:
        payload = json.load(f)
    entry = payload["project"]["calibration_results"]["extended_pinhole"]
    entry["distortion"] = {"__ndarray__": True, "dtype": "float64", "data": distortion}

    migrated = migrate_v1_to_v2(payload)

    assert "extended_pinhole" in migrated["project"]["calibration_results"]
    assert "brown_conrady" not in migrated["project"]["calibration_results"]


def test_v1_mixed_nested_refs_all_migrate_consistently():
    """Case C(사용자 스펙 P1 섹션 9) - extended_pinhole(5계수) 참조가
    calibration_results/validation_results뿐 아니라 model_scores,
    cross_dataset_results, final_result.chosen_model,
    final_result.calibration.model_name, final_result.model_scores,
    final_result.diagnosis.model_name 전부에 흩어져 있어도, 마이그레이션 후
    전부 일관되게 brown_conrady를 가리켜야 한다 - 어느 한 곳이라도
    extended_pinhole로 남아있으면 프로젝트 내부에서 모델 의미가 갈라진다.
    """
    project, _missing = load_project(str(ASSETS_DIR / "v1_extended_5coeff_mixed_refs.ccproj"))

    assert CameraModelType.BROWN_CONRADY in project.calibration_results
    assert CameraModelType.EXTENDED_PINHOLE not in project.calibration_results

    assert len(project.model_scores) == 2
    score_models = {s.model_name for s in project.model_scores}
    assert CameraModelType.EXTENDED_PINHOLE not in score_models
    assert CameraModelType.BROWN_CONRADY in score_models

    assert len(project.cross_dataset_results) == 1
    assert project.cross_dataset_results[0].model_name == CameraModelType.BROWN_CONRADY

    assert project.final_result is not None
    assert project.final_result.chosen_model == CameraModelType.BROWN_CONRADY
    assert project.final_result.calibration.model_name == CameraModelType.BROWN_CONRADY
    assert len(project.final_result.model_scores) == 1
    assert project.final_result.model_scores[0].model_name == CameraModelType.BROWN_CONRADY
    assert project.final_result.diagnosis is not None
    assert project.final_result.diagnosis.model_name == CameraModelType.BROWN_CONRADY


def test_v2_project_loads_without_migration(monkeypatch):
    """migrate_v1_to_v2가 v2 파일에는 아예 호출되지 않는지 spy로 확인."""
    import calibration.project_io as pio

    calls = {"n": 0}
    real = pio.migrate_v1_to_v2

    def counting(payload):
        calls["n"] += 1
        return real(payload)

    monkeypatch.setattr(pio, "migrate_v1_to_v2", counting)

    project, _missing = load_project(str(ASSETS_DIR / "v2_project.ccproj"))

    assert calls["n"] == 0
    assert CameraModelType.BROWN_CONRADY in project.calibration_results


def test_resaving_migrated_project_writes_current_format_version(tmp_path):
    project, _missing = load_project(str(ASSETS_DIR / "v1_extended_5coeff.ccproj"))
    out_path = tmp_path / "resaved.ccproj"

    save_project(project, str(out_path))

    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["format_version"] == PROJECT_FORMAT_VERSION == 2


def test_unsupported_format_version_still_raises():
    with open(ASSETS_DIR / "v2_project.ccproj", "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["format_version"] = 999

    with pytest.raises(ValueError, match="지원하지 않는"):
        project_from_dict(payload)


def test_migrate_v1_to_v2_unexpected_distortion_length_leaves_key_unchanged():
    """5도 8+도 아닌 이상한 길이는 함부로 추측해서 바꾸지 않는다."""
    with open(ASSETS_DIR / "v1_extended_5coeff.ccproj", "r", encoding="utf-8") as f:
        payload = json.load(f)
    entry = payload["project"]["calibration_results"]["extended_pinhole"]
    entry["distortion"] = {"__ndarray__": True, "dtype": "float64", "data": [0.1, 0.2, 0.3]}  # length 3

    migrated = migrate_v1_to_v2(payload)

    assert "extended_pinhole" in migrated["project"]["calibration_results"]
    assert "brown_conrady" not in migrated["project"]["calibration_results"]
