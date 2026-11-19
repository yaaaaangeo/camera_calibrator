"""End-to-end: 이미지 폴더 → 검출 → 선별 → 4모델 → Hold-out → 추천 → 판정 → 저장.

실제 자식 프로세스에서 기존 app.cli를 실행한다 (느림, 기존 test_cli.py와 같은
slow 마커). 무거운 실행은 모듈 fixture로 3번만 한다:
  run A, run B (같은 입력·같은 프로파일) + 기존 CLI 직접 실행 1번.

임계값 숫자는 이 테스트 데이터(물리적으로 일관된 합성 이미지)에서 판정
경로를 검증하기 위한 예시 값이며 운영 기준이 아니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from automation.acceptance import FAIL, PASS, collect_evidence, evaluate_acceptance, parse_policy
from automation.identity import build_identity
from automation.profile import resolve_profile
from automation import runner
from automation.run_manifest import sha256_file
from calibration.project_io import load_project
from tests._automation_test_utils import (
    DICTIONARY,
    MARKER_SIZE,
    SQUARE_SIZE,
    SQUARES_X,
    SQUARES_Y,
    TRUE_K,
    base_profile,
    pattern_section,
    render_consistent_charuco_dataset,
)

pytestmark = pytest.mark.slow

TEST_RATIO = 0.35
# 허용오차 근거: 기존 app.cli를 같은 입력으로 별도 프로세스에서 2번 실행했을 때
# (자동화 계층 없이) 실측한 실행 간 변동은 K 최대 7e-8 (Rational), Hold-out
# Test RMS 상대 최대 1.9e-6였다. 그보다 충분히 넓되 실제 회귀는 잡을 수 있는 값.
PARAM_RTOL, PARAM_ATOL = 1e-5, 1e-6
METRIC_REL = 1e-4
SEED = 42
EXAMPLE_ACCEPTANCE = {
    "thresholds": {
        "train_rms_px": {"max": 0.5, "required": True},
        "test_rms_px": {"max": 0.5, "required": True},
        "test_p95_px": {"max": 1.0, "required": True},
        "test_p99_px": {"max": None, "required": False},
        "edge_rms_px": {"max": 0.6, "required": True},
        "coverage_pct": {"min": 75.0, "required": True},
        "num_images_used": {"min": 15, "required": True},
    }
}


def _quiet(_msg: str) -> None:
    pass


def _profile(output_root: Path, **overrides):
    raw = base_profile(
        calibration={"test_ratio": TEST_RATIO, "seed": SEED, "jobs": 1},
        acceptance=EXAMPLE_ACCEPTANCE,
        **overrides,
    )
    raw["output"] = {"root": str(output_root), "exports": ["opencv", "ros", "report", "json", "csv"]}
    return resolve_profile(raw, source_path="e2e")


def _hash_dir(folder: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("automation_e2e_images")
    render_consistent_charuco_dataset(folder, count=24, seed=7)
    return folder


@pytest.fixture(scope="module")
def runs(image_dir, tmp_path_factory) -> dict:
    output_root = tmp_path_factory.mktemp("automation_e2e_outputs")
    profile = _profile(output_root)
    identity = build_identity("TEST-CAM", "SN-E2E-001")
    before = _hash_dir(image_dir)
    first = runner.run_automation(profile, identity, [str(image_dir)], log=_quiet)
    first_outputs = {
        f["path"]: f["sha256"]
        for f in json.loads((Path(first.session_dir) / "run_manifest.json").read_text(encoding="utf-8"))["outputs"]
    }
    second = runner.run_automation(profile, identity, [str(image_dir)], log=_quiet)
    return {
        "first": first,
        "second": second,
        "first_outputs": first_outputs,
        "inputs_before": before,
        "inputs_after": _hash_dir(image_dir),
        "output_root": output_root,
    }


def _manifest(outcome) -> dict:
    return json.loads((Path(outcome.session_dir) / "run_manifest.json").read_text(encoding="utf-8"))


def _project(outcome):
    project, _missing = load_project(str(Path(outcome.session_dir) / "project" / "calibration.ccproj"))
    return project


def test_run_completes_with_session_layout_and_records(runs):
    outcome = runs["first"]
    assert outcome.status == "completed", outcome.message
    session = Path(outcome.session_dir)
    m = _manifest(outcome)

    expected = [
        "manifest.json", "run_manifest.json",
        "project/calibration.ccproj",
        "intrinsic/camera_info.yaml",
        "reports/report.html", "reports/calibration.json", "reports/dataset.csv",
        "reports/image_selection_manifest.json",
        "automation/acceptance.json", "automation/run_summary.json",
        "automation/cli_config.yaml", "automation/cli_summary.json", "automation/cli_output.log",
    ]
    for relative in expected:
        assert (session / relative).is_file(), relative
    chosen = m["results"]["recommendation"]["chosen_model"]
    from export.output_manager import model_filename
    assert (session / "intrinsic" / f"camera_{model_filename(chosen)}.yaml").is_file()

    # 기존 manifest.json 형식 유지 + 자동화 산출물 기록
    session_manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert session_manifest["schema_version"] == 1
    for key in ("reports.json", "project.save_as", "automation.run_manifest",
                "automation.acceptance", "automation.run_summary", "image_selection.manifest"):
        assert key in session_manifest["exports"], key

    assert m["status"] == "completed"
    assert [s["name"] for s in m["stages"]] == [
        "input_fingerprint", "image_selection", "calibration", "result_collection", "acceptance", "finalize",
    ]
    assert all(s["status"] == "completed" for s in m["stages"])
    assert m["camera"]["serial_number"] == "SN-E2E-001"
    assert m["camera"]["camera_model"] == "TEST-CAM"
    assert (m["camera"]["image_width"], m["camera"]["image_height"]) == (1280, 720)
    assert len(m["profile"]["sha256"]) == 64
    assert m["code_version"]["available"] is True and len(m["code_version"]["commit_sha"]) == 40
    assert m["environment"]["libraries"]["cv2"]
    assert m["inputs"]["fingerprint"]["count"] == 24
    assert m["image_selection"]["enabled"] is True
    assert set(m["results"]["models"]) == {"pinhole", "brown_conrady", "extended_pinhole", "fisheye"}
    assert m["split"]["seed"] == SEED and m["split"]["test_ratio"] == TEST_RATIO
    assert m["split"]["shared_across_models"] is True and m["split"]["leakage_free"] is True
    assert m["split"]["test_frame_ids"]
    assert not set(m["split"]["train_frame_ids"]) & set(m["split"]["test_frame_ids"])
    assert m["acceptance"]["verdict"] == outcome.verdict
    listed = {f["path"] for f in m["outputs"]}
    assert "reports/calibration.json" in listed and "run_manifest.json" not in listed

    # 자식 프로세스에 넘긴 이미지 = 선별 결과
    import yaml

    config = yaml.safe_load((session / "automation" / "cli_config.yaml").read_text(encoding="utf-8"))
    selection = json.loads((session / "reports" / "image_selection_manifest.json").read_text(encoding="utf-8"))
    selected = [img["path"] for img in selection["images"] if img["selected"]]
    assert config["images"] == selected
    assert Path(config["session_dir"]) == session
    assert (session / ".session_claim").is_file()  # 기존 CLI --session-dir가 세션을 열었음


def test_example_thresholds_pass_on_consistent_synthetic_data(runs):
    outcome = runs["first"]
    acceptance = json.loads((Path(outcome.session_dir) / "automation" / "acceptance.json").read_text(encoding="utf-8"))
    assert outcome.verdict == PASS, acceptance["reasons"]
    assert outcome.exit_code == runner.EXIT_PASS
    assert acceptance["criteria_complete"] is True
    # 계산 자체도 정답 K 근처인지 (데이터가 물리적으로 일관됨을 확인)
    project = _project(outcome)
    chosen = project.final_result.chosen_model
    fx = project.calibration_results[chosen].camera_matrix[0, 0]
    assert abs(fx - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.02


def test_recommended_model_with_quality_below_standard_fails(runs):
    """추천이 성공한 같은 결과라도 더 엄격한 기준이면 FAIL - 추천과 판정은 별개."""
    project = _project(runs["first"])
    strict = parse_policy({"thresholds": {"test_rms_px": {"max": 0.01, "required": True}}})
    result = evaluate_acceptance(collect_evidence(project), strict)
    assert result.recommended_model is not None
    assert result.verdict == FAIL


def test_matches_direct_existing_cli_run(runs, image_dir, tmp_path):
    from app.cli import main as cli_main

    m = _manifest(runs["first"])
    assert m["image_selection"]["selected_count"] == 24  # 선별이 전부 고른 경우 = 폴더 전체와 같은 입력
    direct_ccproj = tmp_path / "direct.ccproj"
    code = cli_main([
        "--images", str(image_dir),
        "--pattern", "charuco",
        "--squares-x", str(SQUARES_X), "--squares-y", str(SQUARES_Y),
        "--square-size", str(SQUARE_SIZE), "--marker-size", str(MARKER_SIZE),
        "--dictionary", DICTIONARY,
        "--test-ratio", str(TEST_RATIO), "--seed", str(SEED),
        "--export", "opencv", "json",
        "--output-dir", str(tmp_path / "direct"),
        "--save-project", str(direct_ccproj),
        "--quiet",
    ])
    assert code == 0
    direct, _ = load_project(str(direct_ccproj))
    auto = _project(runs["first"])

    assert auto.final_result.chosen_model == direct.final_result.chosen_model
    assert ([s.model_name for s in auto.model_scores if s.is_recommended]
            == [s.model_name for s in direct.model_scores if s.is_recommended])
    assert set(auto.calibration_results) == set(direct.calibration_results)
    for model, a in auto.calibration_results.items():
        d = direct.calibration_results[model]
        assert a.success == d.success, model
        if not a.success:
            continue
        np.testing.assert_allclose(a.camera_matrix, d.camera_matrix, rtol=PARAM_RTOL, atol=PARAM_ATOL, err_msg=str(model))
        np.testing.assert_allclose(a.distortion, d.distortion, rtol=PARAM_RTOL, atol=PARAM_ATOL, err_msg=str(model))
        assert a.rms_error == pytest.approx(d.rms_error, rel=METRIC_REL)
        va, vd = auto.validation_results[model], direct.validation_results[model]
        assert va.train_frame_ids == vd.train_frame_ids
        assert va.test_frame_ids == vd.test_frame_ids
        if va.test_rms is not None:
            assert va.test_rms == pytest.approx(vd.test_rms, rel=METRIC_REL)


def test_repeat_run_is_reproducible(runs):
    first, second = runs["first"], runs["second"]
    assert second.status == "completed", second.message
    m1, m2 = _manifest(first), _manifest(second)
    assert m1["run_id"] != m2["run_id"]
    assert first.session_dir != second.session_dir
    assert m1["inputs"]["fingerprint"]["aggregate_sha256"] == m2["inputs"]["fingerprint"]["aggregate_sha256"]
    assert m1["dataset"]["detection_fingerprint_sha256"] == m2["dataset"]["detection_fingerprint_sha256"]
    assert m1["profile"]["sha256"] == m2["profile"]["sha256"]
    assert m1["split"]["train_frame_ids"] == m2["split"]["train_frame_ids"]
    assert m1["split"]["test_frame_ids"] == m2["split"]["test_frame_ids"]
    assert m1["calibration"]["effective_options"] == m2["calibration"]["effective_options"]
    assert first.verdict == second.verdict
    p1, p2 = _project(first), _project(second)
    for model, a in p1.calibration_results.items():
        b = p2.calibration_results[model]
        if a.success:
            np.testing.assert_allclose(a.camera_matrix, b.camera_matrix, rtol=PARAM_RTOL, atol=PARAM_ATOL)
            np.testing.assert_allclose(a.distortion, b.distortion, rtol=PARAM_RTOL, atol=PARAM_ATOL)


def test_second_run_did_not_overwrite_first(runs):
    session = Path(runs["first"].session_dir)
    assert runs["first_outputs"]
    for relative, digest in runs["first_outputs"].items():
        assert sha256_file(session / relative) == digest, relative


def test_source_images_are_untouched(runs):
    assert runs["inputs_after"] == runs["inputs_before"]


def test_wrong_dictionary_fails_at_detection_without_calculation(image_dir, tmp_path):
    pattern = pattern_section() | {"dictionary": "DICT_6X6_250"}
    profile = _profile(tmp_path / "out", pattern=pattern)
    outcome = runner.run_automation(profile, build_identity("TEST-CAM", "SN-E2E-002"), [str(image_dir)], log=_quiet)
    assert outcome.exit_code == runner.EXIT_RUN_FAILED
    m = _manifest(outcome)
    assert m["status"] == "failed" and m["failure"]["stage"] == "image_selection"
    assert not (Path(outcome.session_dir) / "project" / "calibration.ccproj").exists()


def test_single_thread_opencv_runs_are_bit_identical(image_dir, tmp_path):
    """execution.opencv_threads=1 이면 실행 간 변동(기본 스레드에서 실측 ~1e-6)이 사라진다."""
    raw_profile = base_profile(
        calibration={"test_ratio": TEST_RATIO, "seed": SEED, "jobs": 1, "models": ["brown_conrady", "rational"]},
    )
    raw_profile["output"] = {"root": str(tmp_path / "out")}
    raw_profile["execution"] = {"opencv_threads": 1}
    profile = resolve_profile(raw_profile, source_path="e2e")
    identity = build_identity("TEST-CAM", "SN-E2E-ST")
    first = runner.run_automation(profile, identity, [str(image_dir)], log=_quiet)
    second = runner.run_automation(profile, identity, [str(image_dir)], log=_quiet)
    assert first.status == second.status == "completed"
    assert _manifest(first)["calibration"]["opencv_threads"] == 1
    p1, p2 = _project(first), _project(second)
    for model, a in p1.calibration_results.items():
        b = p2.calibration_results[model]
        assert np.array_equal(a.camera_matrix, b.camera_matrix), model
        assert np.array_equal(a.distortion, b.distortion), model
        assert a.rms_error == b.rms_error
        assert p1.validation_results[model].test_rms == p2.validation_results[model].test_rms
