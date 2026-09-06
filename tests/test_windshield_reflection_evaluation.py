from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from calibration.project_io import project_from_dict, project_to_dict
from calibration.types import CalibrationProject, CameraConfig, PatternConfig, PatternType
from calibration.windshield.reflection import (
    REFLECTION_METRIC_VERSION,
    ReflectionDatasetResult,
    ReflectionEvaluationConfig,
    ReflectionImagePair,
    evaluate_reflection,
    evaluate_reflection_dataset,
    evaluate_reflection_reference,
)
from export.reflection import export_reflection_html, export_reflection_yaml


def _base_image(h: int = 120, w: int = 160) -> np.ndarray:
    y, x = np.indices((h, w))
    texture = 80 + 50 * ((x // 8 + y // 8) % 2) + 30 * (x / max(w - 1, 1))
    image = np.clip(texture, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _overlay_rect(image: np.ndarray, value: int = 35, y0: int | None = None, y1: int | None = None) -> np.ndarray:
    out = image.copy().astype(np.int16)
    h, w = image.shape[:2]
    y0 = h * 3 // 4 if y0 is None else y0
    y1 = h if y1 is None else y1
    out[y0:y1, w // 5 : 4 * w // 5, :] += value
    return np.clip(out, 0, 255).astype(np.uint8)


def _gaussian_overlay(image: np.ndarray, center: tuple[int, int], amplitude: float = 50.0, sigma: float = 11.0) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = center
    overlay = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))
    return np.clip(image.astype(np.float32) + overlay[:, :, None], 0, 255).astype(np.uint8)


def test_reference_identical_image_has_near_zero_reflection():
    image = _base_image()

    result = evaluate_reflection_reference(image, image)

    assert result.success
    assert result.metric_version == REFLECTION_METRIC_VERSION
    assert result.mean_strength < 1e-6
    assert result.coverage == pytest.approx(0.0, abs=1e-6)


def test_reference_known_bottom_overlay_reports_bottom_stronger_than_top():
    reference = _base_image()
    normal = _overlay_rect(reference, value=35)

    result = evaluate_reflection_reference(reference_image=reference, normal_image=normal)

    assert result.success
    assert result.mean_strength > 0.0
    assert result.regional_metrics["bottom"].mean_strength > result.regional_metrics["top"].mean_strength
    assert result.bottom_roi_mean_strength > result.regional_metrics["top"].mean_strength
    assert result.positive_mean_strength > 0.0


def test_reference_gaussian_reflection_heatmap_peak_matches_gt_location():
    reference = _base_image()
    normal = _gaussian_overlay(reference, center=(120, 82), amplitude=70, sigma=8)

    result = evaluate_reflection_reference(normal, reference)

    heatmap = np.asarray(result.downsampled_reflection_map)
    peak_row, peak_col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    peak_x = (peak_col + 0.5) * reference.shape[1] / result.heatmap_cols
    peak_y = (peak_row + 0.5) * reference.shape[0] / result.heatmap_rows
    assert abs(peak_x - 120) < 10
    assert abs(peak_y - 82) < 10
    assert result.p95_strength > result.mean_strength


def test_reference_alignment_reduces_false_reflection_from_small_shift():
    normal = _base_image()
    warp = np.float32([[1, 0, 3], [0, 1, -2]])
    shifted_reference = cv2.warpAffine(normal, warp, (normal.shape[1], normal.shape[0]), borderMode=cv2.BORDER_REFLECT)

    no_align = evaluate_reflection_reference(
        normal,
        shifted_reference,
        ReflectionEvaluationConfig(align=False, photometric_normalize=False, allow_unsafe_reference_bypass=True),
    )
    aligned = evaluate_reflection_reference(
        normal,
        shifted_reference,
        ReflectionEvaluationConfig(
            align=True,
            alignment_model="translation",
            photometric_normalize=False,
            allow_unsafe_reference_bypass=True,
        ),
    )

    assert aligned.success
    assert aligned.alignment_status in {"good", "warning"}
    assert aligned.mean_strength < no_align.mean_strength * 0.5


def _blurred_noise_bgr(h: int, w: int, seed: int = 0, sigma: float = 6.0) -> np.ndarray:
    """ECC가 안정적으로 recover할 수 있는, 주기성 없는 non-periodic
    texture(순수 gradient는 aperture problem 때문에 ECC가 shift를 아예
    못 잡아낼 수 있어 이 테스트 목적에 부적합하다 - blur된 random noise를
    쓴다)."""
    rng = np.random.default_rng(seed)
    raw = (rng.random((h, w)) * 255).astype(np.float32)
    blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma)
    return cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_valid_warp_mask_excludes_border_reflect_artifacts():
    """Phase B-1 안정화 - alignment warp가 만드는 BORDER_REFLECT 인공
    영역이 Reflection metric에 섞이면 안 된다. 실제로 겹치는 영역만 보면
    같은 이미지끼리의 비교이므로 reflection이 거의 0이어야 하고, valid
    영역의 크기는 정확히 "이미지 폭 - shift 크기"와 일치해야 한다."""
    h, w = 200, 300
    normal = _blurred_noise_bgr(h, w)
    shift_px = 20
    warp = np.float32([[1, 0, shift_px], [0, 1, 0]])
    reference = cv2.warpAffine(normal, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = evaluate_reflection_reference(
        normal, reference,
        ReflectionEvaluationConfig(align=True, alignment_model="translation", photometric_normalize=True),
    )
    assert result.success
    assert result.alignment_status == "good"
    assert result.alignment_translation_x_px == pytest.approx(shift_px, abs=0.5)
    # Border를 제대로 제외했다면, 실제로 겹치는 영역은 같은 이미지끼리의
    # 비교이므로 reflection이 거의 0에 가까워야 한다.
    assert result.reflection_mean is not None
    assert result.reflection_mean < 0.02
    assert result.reflection_max is not None
    assert result.reflection_max < 0.1


def test_valid_warp_mask_matches_exact_expected_border_width():
    """`align_reference_to_normal()`의 valid_mask 자체를 직접 검증한다 -
    shift_px만큼 border가 생겼다면 valid 비율은 정확히
    (w-shift_px)/w이어야 한다(사용자 스펙 B-1, 근사가 아니라 정확한 값)."""
    from calibration.windshield.reflection.alignment import align_reference_to_normal

    h, w = 200, 300
    normal_bgr = _blurred_noise_bgr(h, w)
    normal_luma = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift_px = 20
    warp = np.float32([[1, 0, shift_px], [0, 1, 0]])
    reference_luma = cv2.warpAffine(normal_luma, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = align_reference_to_normal(normal_luma, reference_luma, method="translation")
    assert result.status == "good"
    assert result.valid_mask is not None
    expected_valid_fraction = (w - shift_px) / w
    assert float(result.valid_mask.mean()) == pytest.approx(expected_valid_fraction, abs=0.01)
    # Border는 항상 전체 행에 걸쳐 있어야 한다(순수 x-translation이므로).
    col_valid = result.valid_mask.all(axis=0)
    assert int(col_valid.sum()) == w - shift_px


def test_valid_warp_mask_is_none_when_alignment_disabled():
    """정렬을 아예 하지 않으면(align=False) border 개념 자체가 없다 -
    valid_mask 제약 없이 전체 이미지가 그대로 평가되어야 한다(기존 동작
    유지)."""
    normal = _base_image()
    result = evaluate_reflection_reference(
        normal, normal,
        ReflectionEvaluationConfig(align=False, photometric_normalize=False, allow_unsafe_reference_bypass=True),
    )
    assert result.success
    assert result.reflection_mean == pytest.approx(0.0, abs=1e-6)


def test_alignment_method_typo_is_rejected_not_silently_treated_as_affine():
    """Phase B-2 안정화 - 지원하지 않는 alignment method(오타 포함)가
    조용히 Affine으로 처리되면 안 된다."""
    from calibration.windshield.reflection.alignment import align_reference_to_normal

    normal = _base_image()
    with pytest.raises(ValueError, match="unsupported reflection alignment method"):
        align_reference_to_normal(
            cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY).astype(np.float32),
            cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY).astype(np.float32),
            method="Affine",  # 대문자 오타 - 예전에는 조용히 AFFINE으로 처리됐다
        )
    with pytest.raises(ValueError, match="unsupported reflection alignment method"):
        align_reference_to_normal(
            cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY).astype(np.float32),
            cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY).astype(np.float32),
            method="afifne",  # 오타
        )


def test_alignment_translation_and_affine_are_still_accepted():
    from calibration.windshield.reflection.alignment import align_reference_to_normal

    normal = _base_image()
    luma = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for method in ("translation", "affine"):
        result = align_reference_to_normal(luma, luma, method=method)
        assert result.method == method
        assert result.status in {"good", "warning", "invalid"}


def test_affine_alignment_reports_readable_diagnostics():
    """Phase B-2 - Affine 사용 시 translation/rotation/scale/shear 진단이
    채워져야 한다."""
    h, w = 160, 220
    gradient = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=3.0)
    normal = cv2.cvtColor(gradient.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    warp = np.float32([[1, 0, 5], [0, 1, 3]])
    reference = cv2.warpAffine(normal, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = evaluate_reflection_reference(
        normal, reference,
        ReflectionEvaluationConfig(align=True, alignment_model="affine", photometric_normalize=True),
    )
    assert result.success
    assert result.alignment_method == "affine"
    assert result.alignment_translation_x_px is not None
    assert result.alignment_translation_y_px is not None
    assert result.alignment_rotation_deg is not None
    assert result.alignment_scale is not None
    assert result.alignment_shear_deg is not None


def test_unusually_large_transform_is_downgraded_to_warning_even_with_high_score():
    """Phase B-2 - score가 높아도 transform 자체가 이미지 대각선 대비
    비정상적으로 크면 자동으로 'good'이 되면 안 된다(다른 scene을 잘못
    짝지었을 가능성)."""
    from calibration.windshield.reflection.alignment import (
        DEFAULT_MAX_REASONABLE_TRANSLATION_DIAGONAL_RATIO,
        align_reference_to_normal,
    )

    h, w = 100, 100
    rng = np.random.default_rng(1)
    raw = (rng.random((h, w)) * 255).astype(np.float32)
    clean = cv2.GaussianBlur(raw, (0, 0), sigmaX=5.0)
    diagonal = float(np.hypot(h, w))
    huge_shift = int(diagonal * DEFAULT_MAX_REASONABLE_TRANSLATION_DIAGONAL_RATIO) + 10
    warp = np.float32([[1, 0, huge_shift], [0, 1, 0]])
    shifted = cv2.warpAffine(clean, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = align_reference_to_normal(clean, shifted, method="translation")
    if result.score is not None and result.score >= 0.90:
        assert result.status == "warning"
        assert result.warning_message is not None


def test_production_reference_path_forces_alignment_and_normalization():
    normal = _base_image()
    warp = np.float32([[1, 0, 3], [0, 1, -2]])
    shifted_reference = cv2.warpAffine(normal, warp, (normal.shape[1], normal.shape[0]), borderMode=cv2.BORDER_REFLECT)
    config = ReflectionEvaluationConfig(align=False, photometric_normalize=False)

    result = evaluate_reflection_reference(normal, shifted_reference, config)

    assert result.success
    assert result.alignment_status != "not_run"
    assert result.photometric_normalized is True


def test_explicit_unsafe_debug_bypass_can_disable_alignment_and_normalization():
    image = _base_image()

    result = evaluate_reflection_reference(
        image,
        image,
        ReflectionEvaluationConfig(
            align=False,
            photometric_normalize=False,
            allow_unsafe_reference_bypass=True,
        ),
    )

    assert result.success
    assert result.alignment_status == "not_run"
    assert result.photometric_normalized is False


def test_reference_exposure_gain_bias_normalization_removes_global_change():
    reference = _base_image()
    normal = np.clip(reference.astype(np.float32) * 1.1 + 5.0, 0, 255).astype(np.uint8)

    result = evaluate_reflection_reference(normal, reference, ReflectionEvaluationConfig(photometric_normalize=True))

    assert result.success
    assert result.mean_strength < 0.03
    assert result.coverage < 0.02
    assert result.photometric_gain == pytest.approx(1.1, rel=0.1)
    assert result.contrast_retention == pytest.approx(1.0, abs=0.05)
    assert result.edge_retention == pytest.approx(1.0, abs=0.05)


def test_reference_reflection_plus_exposure_preserves_local_overlay_signal():
    reference = _base_image()
    exposed = np.clip(reference.astype(np.float32) * 1.1 + 5.0, 0, 255).astype(np.uint8)
    normal = _overlay_rect(exposed, value=35)

    result = evaluate_reflection_reference(normal, reference, ReflectionEvaluationConfig(photometric_normalize=True))

    assert result.success
    assert result.regional_metrics["bottom"].mean_strength > result.regional_metrics["top"].mean_strength * 2.0
    assert result.coverage > 0.05


def test_reference_coverage_matches_known_overlay_area():
    reference = np.full((100, 100, 3), 100, dtype=np.uint8)
    normal = reference.copy()
    normal[:10, :, :] = 130  # exactly 10% of the image, relative diff about 0.3

    result = evaluate_reflection_reference(
        normal,
        reference,
        ReflectionEvaluationConfig(
            coverage_threshold=0.08,
            align=False,
            photometric_normalize=False,
            allow_unsafe_reference_bypass=True,
        ),
    )

    assert result.coverage == pytest.approx(0.10, abs=0.01)


def test_reference_contrast_and_edge_retention_drop_for_translucent_overlay():
    reference = _base_image()
    normal = reference.copy()
    h, w = normal.shape[:2]
    patch = normal[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].astype(np.float32)
    normal[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = np.clip(0.35 * patch + 150.0 * 0.65, 0, 255)

    result = evaluate_reflection_reference(
        normal,
        reference,
        ReflectionEvaluationConfig(align=False, allow_unsafe_reference_bypass=True),
    )

    assert result.contrast_retention < 1.0
    assert result.edge_retention < 1.0


def test_reference_saturation_coverage_matches_known_patch():
    reference = np.full((80, 100, 3), 90, dtype=np.uint8)
    normal = reference.copy()
    normal[:, :20, :] = 255

    result = evaluate_reflection_reference(
        normal,
        reference,
        ReflectionEvaluationConfig(
            saturation_threshold=250,
            align=False,
            photometric_normalize=False,
            allow_unsafe_reference_bypass=True,
        ),
    )

    assert result.saturation_coverage == pytest.approx(0.20, abs=0.01)


def test_reflection_glare_and_saturation_metrics_remain_separate():
    reference = np.full((100, 100, 3), 100, dtype=np.uint8)
    reflection_only = reference.copy()
    reflection_only[:20, :, :] = 135
    glare_only = reference.copy()
    glare_only[20:40, :, :] = 235
    saturation_only = reference.copy()
    saturation_only[40:60, :, :] = 255
    cfg = ReflectionEvaluationConfig(
        align=False,
        photometric_normalize=False,
        coverage_threshold=0.08,
        glare_luminance_threshold=220,
        glare_contrast_threshold=80,
        saturation_threshold=250,
        allow_unsafe_reference_bypass=True,
    )

    reflection = evaluate_reflection_reference(reflection_only, reference, cfg)
    glare = evaluate_reflection_reference(glare_only, reference, cfg)
    saturation = evaluate_reflection_reference(saturation_only, reference, cfg)

    assert reflection.reflection_coverage == pytest.approx(0.20, abs=0.01)
    assert reflection.glare_coverage == pytest.approx(0.0, abs=0.01)
    assert reflection.saturation_coverage == pytest.approx(0.0, abs=0.01)
    assert glare.glare_coverage == pytest.approx(0.20, abs=0.02)
    assert glare.glare_strength > 0.0
    assert glare.saturation_coverage == pytest.approx(0.0, abs=0.01)
    assert saturation.saturation_coverage == pytest.approx(0.20, abs=0.01)


def test_no_reference_likelihood_is_higher_for_bright_low_contrast_overlay():
    clean = _base_image()
    reflected = clean.copy()
    reflected[60:105, 30:130, :] = np.clip(0.15 * reflected[60:105, 30:130, :].astype(np.float32) + 190.0 * 0.85, 0, 255)

    clean_result = evaluate_reflection(clean, config=ReflectionEvaluationConfig(mode="no_reference"))
    reflected_result = evaluate_reflection(reflected, config=ReflectionEvaluationConfig(mode="no_reference"))

    assert reflected_result.mode == "no_reference"
    assert reflected_result.reflection_likelihood is not None
    assert reflected_result.reflection_mean is None
    assert reflected_result.reflection_coverage is None
    assert reflected_result.no_reference_is_likelihood
    assert reflected_result.severity_score is None
    assert "likelihood" in reflected_result.warning_message.lower()
    assert reflected_result.mean_strength > clean_result.mean_strength


def test_reflection_evaluator_does_not_mutate_input_images():
    reference = _base_image()
    normal = _overlay_rect(reference)
    reference_before = reference.copy()
    normal_before = normal.copy()

    evaluate_reflection_reference(normal, reference)

    assert np.array_equal(reference, reference_before)
    assert np.array_equal(normal, normal_before)


def test_reference_input_resolution_mismatch_is_invalid():
    result = evaluate_reflection_reference(np.zeros((20, 30, 3), dtype=np.uint8), np.zeros((21, 30, 3), dtype=np.uint8))

    assert not result.success
    assert result.alignment_status == "invalid"
    assert "same resolution" in result.error_message


def test_reference_alignment_failure_is_reported_without_metrics():
    normal = np.zeros((80, 100, 3), dtype=np.uint8)
    reference = np.full((80, 100, 3), 180, dtype=np.uint8)

    result = evaluate_reflection_reference(normal, reference)

    assert not result.success
    assert result.alignment_status == "invalid"
    assert "alignment" in result.error_message.lower()


def test_multi_pair_dataset_aggregates_and_keeps_day_night_groups(tmp_path):
    reference = _base_image()
    mild = _overlay_rect(reference, value=15)
    strong = _overlay_rect(reference, value=45)
    paths = []
    for name, img in (("ref", reference), ("mild", mild), ("strong", strong)):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)

    result = evaluate_reflection_dataset([
        ReflectionImagePair(str(paths[1]), str(paths[0]), "day-scene", day_night="day"),
        ReflectionImagePair(str(paths[2]), str(paths[0]), "night-scene", day_night="night"),
    ])

    assert result.success
    assert result.worst_pair_id == "night-scene"
    assert result.mean_strength > 0.0
    assert result.reference_mean_strength == pytest.approx(result.mean_strength)
    assert result.mean_reflection_likelihood is None
    assert set(result.by_day_night) == {"day", "night"}


def test_dataset_isolates_one_broken_pair_and_keeps_evaluating_the_rest(tmp_path):
    """Phase A-7 안정화 - Pair 03이 INVALID(존재하지 않는 파일)여도 그 뒤에
    오는 Pair 04가 평가되지 않고 dataset 전체가 예외로 죽으면 안 된다.
    GOOD pair만 aggregate에 쓰이고, invalid pair 개수도 명시적으로
    보고되어야 한다."""
    reference = _base_image()
    mild = _overlay_rect(reference, value=15)
    ref_path = tmp_path / "ref.png"
    mild_path = tmp_path / "mild.png"
    cv2.imwrite(str(ref_path), reference)
    cv2.imwrite(str(mild_path), mild)

    result = evaluate_reflection_dataset([
        ReflectionImagePair(str(mild_path), str(ref_path), "pair_01"),
        ReflectionImagePair(str(mild_path), str(ref_path), "pair_02"),
        ReflectionImagePair(str(tmp_path / "does_not_exist.png"), str(ref_path), "pair_03"),
        ReflectionImagePair(str(mild_path), str(ref_path), "pair_04"),
    ])

    assert result.success
    assert len(result.pair_results) == 4  # 실패한 pair도 결과 목록에는 남는다(원인 추적용)
    assert result.num_valid_pairs == 3
    assert result.num_invalid_pairs == 1
    pair_03 = next(r for r in result.pair_results if r.pair_id == "pair_03")
    assert pair_03.success is False
    assert pair_03.error_message is not None
    # Pair 04는 Pair 03의 실패와 무관하게 정상적으로 평가되어야 한다.
    pair_04 = next(r for r in result.pair_results if r.pair_id == "pair_04")
    assert pair_04.success is True
    assert result.warning_message is not None and "1" in result.warning_message


def test_dataset_with_all_pairs_invalid_reports_failure_not_exception(tmp_path):
    """Phase A-7 안정화 - 모든 pair가 invalid면 dataset 자체는 실패해야
    하지만, 예외를 던지는 대신 `success=False`인 결과를 정상적으로
    반환해야 한다."""
    result = evaluate_reflection_dataset([
        ReflectionImagePair(str(tmp_path / "missing_1.png"), pair_id="bad_1"),
        ReflectionImagePair(str(tmp_path / "missing_2.png"), pair_id="bad_2"),
    ])
    assert result.success is False
    assert result.num_valid_pairs == 0
    assert result.num_invalid_pairs == 2
    assert result.error_message is not None


def test_no_reference_dataset_aggregate_uses_likelihood_not_severity(tmp_path):
    normal = _overlay_rect(_base_image())
    path = tmp_path / "normal.png"
    cv2.imwrite(str(path), normal)

    result = evaluate_reflection_dataset([
        ReflectionImagePair(str(path), pair_id="heuristic-scene"),
    ])

    assert result.success
    assert result.mode == "no_reference"
    assert result.mean_reflection_likelihood is not None
    assert result.reference_mean_strength is None
    assert result.severity_score is None


def test_reflection_project_io_and_yaml_export_round_trip(tmp_path):
    pair_result = evaluate_reflection_reference(_overlay_rect(_base_image()), _base_image())
    dataset_result = ReflectionDatasetResult(
        mode="reference",
        pair_results=[pair_result],
        mean_strength=pair_result.mean_strength,
        median_strength=pair_result.mean_strength,
        p95_strength=pair_result.p95_strength,
        worst_pair_id=pair_result.pair_id,
        coverage=pair_result.coverage,
        severity_score=pair_result.severity_score,
    )
    project = CalibrationProject(
        project_name="reflection-test",
        camera_config=CameraConfig(width=160, height=120),
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=5, squares_y=5, square_size=0.02),
        reflection_results={"latest": dataset_result},
    )

    restored = project_from_dict(project_to_dict(project))

    assert "latest" in restored.reflection_results
    restored_result = restored.reflection_results["latest"]
    assert restored_result.metric_version == REFLECTION_METRIC_VERSION
    assert restored_result.pair_results[0].mean_strength == pytest.approx(pair_result.mean_strength)
    assert restored_result.pair_results[0].reflection_mean == pytest.approx(pair_result.mean_strength)
    assert restored_result.pair_results[0].coverage_threshold == pytest.approx(0.08)
    assert restored_result.pair_results[0].photometric_normalized
    assert restored_result.pair_results[0].spatial_map

    path = tmp_path / "reflection_evaluation.yml"
    export_reflection_yaml(restored_result, str(path))
    assert "reflection_evaluation" in path.read_text(encoding="utf-8")

    report_path = tmp_path / "reflection_evaluation.html"
    export_reflection_html(restored_result, str(report_path))
    report_text = report_path.read_text(encoding="utf-8")
    assert "Reflection Dataset Evaluation" in report_text
    assert "geometry calibration" in report_text


def test_reflection_project_io_legacy_reference_and_no_reference_semantics():
    reference_legacy = project_from_dict({
        "format_version": 2,
        "project": {
            "project_name": "legacy-reference",
            "camera_config": {"width": 160, "height": 120},
            "pattern_config": {"type": "charuco", "squares_x": 5, "squares_y": 5, "square_size": 0.02},
            "reflection_results": {
                "legacy": {
                    "mode": "reference",
                    "mean_strength": 0.12,
                    "p95_strength": 0.35,
                    "coverage": 0.22,
                    "pair_results": [{"mode": "reference", "mean_strength": 0.12, "p95_strength": 0.35, "coverage": 0.22}],
                }
            },
        },
    }).reflection_results["legacy"]
    no_reference_legacy = project_from_dict({
        "format_version": 2,
        "project": {
            "project_name": "legacy-no-reference",
            "camera_config": {"width": 160, "height": 120},
            "pattern_config": {"type": "charuco", "squares_x": 5, "squares_y": 5, "square_size": 0.02},
            "reflection_results": {
                "legacy": {
                    "mode": "no_reference",
                    "mean_strength": 0.44,
                    "p95_strength": 0.70,
                    "coverage": 0.31,
                    "severity_score": 88.0,
                    "pair_results": [
                        {
                            "mode": "no_reference",
                            "mean_strength": 0.44,
                            "p95_strength": 0.70,
                            "coverage": 0.31,
                            "severity_score": 88.0,
                        }
                    ],
                }
            },
        },
    }).reflection_results["legacy"]

    assert reference_legacy.reference_mean_strength == pytest.approx(0.12)
    assert reference_legacy.pair_results[0].reflection_mean == pytest.approx(0.12)
    assert no_reference_legacy.mean_reflection_likelihood == pytest.approx(0.44)
    assert no_reference_legacy.reference_mean_strength is None
    assert no_reference_legacy.severity_score is None
    assert no_reference_legacy.pair_results[0].reflection_likelihood == pytest.approx(0.44)
    assert no_reference_legacy.pair_results[0].reflection_mean is None
    assert no_reference_legacy.pair_results[0].reflection_coverage is None
    assert no_reference_legacy.pair_results[0].no_reference_is_likelihood is True
    assert no_reference_legacy.pair_results[0].severity_score is None


def test_reflection_result_dataclass_round_trips_with_asdict():
    result = evaluate_reflection_reference(_overlay_rect(_base_image()), _base_image())

    raw = dataclasses.asdict(result)

    assert raw["metric_version"] == REFLECTION_METRIC_VERSION
    assert raw["reflection_mean"] == pytest.approx(result.mean_strength)
    assert raw["no_reference_is_likelihood"] is False
    assert raw["coverage_threshold"] == pytest.approx(0.08)
    assert raw["regional_metrics"]["bottom"]["mean_strength"] > raw["regional_metrics"]["top"]["mean_strength"]
