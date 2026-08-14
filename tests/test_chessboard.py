"""
tests/test_chessboard.py
=============================

일반 체스보드(PatternType.CHESSBOARD) 지원 회귀 테스트.

핵심 검증 포인트:
1. 실제로 이미지에서 검출이 되는지 (findChessboardCornersSB/classic 둘 다)
2. SB와 classic이 반환하는 코너 순서가 정확히 반대라는 실측 사실 -
   _normalize_chessboard_corner_order()가 이걸 정규화하는지
3. object_points 순서가 코너 순서와 정확히 짝이 맞는지 (id -> row/col 역산이
   ChArUco와 동일한 공식을 재사용하므로, 순서가 안 맞으면 캘리브레이션
   자체는 "성공"하지만 결과가 미묘하게 틀어지는 조용한 버그가 됨)
4. 전체 파이프라인(3모델+검증+straightness)이 체스보드에서도 기존 코드
   변경 없이 동작하는지
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.detector import (
    build_chessboard_object_points,
    detect_chessboard,
    detect_dataset,
    _normalize_chessboard_corner_order,
)
from calibration.types import CameraConfig, PatternConfig, PatternType

pytestmark = pytest.mark.slow


def _pattern() -> PatternConfig:
    return PatternConfig(type=PatternType.CHESSBOARD, squares_x=7, squares_y=5, square_size=0.04)


def _render_chessboard_image(squares_x=7, squares_y=5, square_px=100) -> np.ndarray:
    base = np.full((squares_y * square_px, squares_x * square_px), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                base[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 0
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def test_normalize_corner_order_reverses_when_first_is_farther_from_origin():
    """findChessboardCornersSB가 classic과 정확히 반대 순서를 준다는 실측
    사실(대화 중 재현)에 대한 회귀 테스트 - 정규화 함수가 항상 원점에
    가까운 쪽을 첫 코너로 맞추는지.
    """
    reversed_order = np.array([[500, 400], [300, 200], [10, 10]], dtype=np.float32).reshape(-1, 1, 2)
    normalized = _normalize_chessboard_corner_order(reversed_order)
    assert np.allclose(normalized[0].reshape(-1), [10, 10])
    assert np.allclose(normalized[-1].reshape(-1), [500, 400])


def test_normalize_corner_order_leaves_already_correct_order_unchanged():
    correct_order = np.array([[10, 10], [300, 200], [500, 400]], dtype=np.float32).reshape(-1, 1, 2)
    normalized = _normalize_chessboard_corner_order(correct_order)
    assert np.array_equal(normalized, correct_order)


def test_detect_chessboard_succeeds_on_real_image():
    img = _render_chessboard_image()
    pattern = _pattern()
    result = detect_chessboard(img, pattern, "test_img")

    assert result.success
    assert result.num_corners == 6 * 4
    assert result.corners.shape == (24, 1, 2)
    assert result.ids is not None
    assert result.ids.reshape(-1).tolist() == list(range(24))


def test_object_points_match_corner_order_row_major():
    """object_points가 build_chessboard_object_points()와 정확히 같은
    row-major 순서인지 - 순서가 어긋나면 캘리브레이션이 "성공은 하지만
    결과가 미묘하게 틀어지는" 조용한 버그가 된다.
    """
    pattern = _pattern()
    objp = build_chessboard_object_points(pattern)

    assert objp.shape == (24, 1, 3)
    assert np.allclose(objp[0].reshape(-1), [0, 0, 0])
    assert np.allclose(objp[1].reshape(-1), [pattern.square_size, 0, 0])
    assert np.allclose(objp[6].reshape(-1), [0, pattern.square_size, 0])


def test_detect_chessboard_fails_gracefully_on_blank_image():
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    pattern = _pattern()
    result = detect_chessboard(blank, pattern, "blank")

    assert not result.success
    assert result.num_corners == 0
    assert result.failure_reason is not None
    assert "체스보드" in result.failure_reason


def test_detect_dataset_dispatches_to_chessboard(tmp_path):
    img = _render_chessboard_image()
    path = str(tmp_path / "board.jpg")
    cv2.imwrite(path, img)

    pattern = _pattern()
    dataset = detect_dataset([path], pattern)

    assert dataset.num_total == 1
    assert dataset.num_detected == 1
    frame = dataset.frames[0]
    assert frame.detection.num_corners == 24


def _build_synthetic_distorted_chessboard_dir(tmp_path_factory, n_images=12, seed=1):
    out_dir = tmp_path_factory.mktemp("synthetic_chessboard")
    base = _render_chessboard_image(squares_x=7, squares_y=5, square_px=100)

    W, H = 1280, 720
    true_K = np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]])
    true_D = np.array([-0.22, 0.06, 0.0, 0.0, 0.0])
    map1, map2 = cv2.initUndistortRectifyMap(true_K, true_D, None, true_K, (W, H), cv2.CV_32FC1)

    rng = np.random.default_rng(seed)
    for i in range(n_images):
        scale = 0.4 + rng.random() * 0.3
        bw, bh = int(700 * scale), int(500 * scale)
        small = cv2.resize(base, (bw, bh))
        canvas = np.full((H, W, 3), 200, dtype=np.uint8)
        x0 = int(rng.integers(0, max(W - bw, 1)))
        y0 = int(rng.integers(0, max(H - bh, 1)))
        src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        jitter = 0.05 * bw
        dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
        dst[:, 0] += x0
        dst[:, 1] += y0
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(small, M, (W, H), borderValue=(200, 200, 200))
        mask = cv2.warpPerspective(np.full((bh, bw), 255, dtype=np.uint8), M, (W, H))
        canvas[mask > 0] = warped[mask > 0]
        distorted = cv2.remap(canvas, map1, map2, interpolation=cv2.INTER_LINEAR, borderValue=(200, 200, 200))
        cv2.imwrite(str(out_dir / f"img_{i:02d}.jpg"), distorted)

    return str(out_dir)


def test_full_pipeline_with_chessboard_pattern(tmp_path_factory):
    """검출 -> 3모델 -> Hold-out -> 추천 -> straightness까지 체스보드로
    전부 이어서 돈다 - straightness.py 등 다른 모듈을 하나도 안 고쳤는데도
    (id -> row/col 공식이 동일해서) 그대로 재사용되는지가 핵심.
    """
    from calibration.compare import run_all_models
    from calibration.frame_quality import compute_frame_quality_scores
    from calibration.models.common import infer_image_size
    from calibration.quality import analyze_dataset_quality
    from calibration.recommender import compute_model_scores
    from calibration.straightness import compute_straightness_residual
    from calibration.validation import validate_all_models

    image_dir = _build_synthetic_distorted_chessboard_dir(tmp_path_factory)
    pattern = _pattern()
    camera_config = CameraConfig(width=1280, height=720)

    import glob
    paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
    dataset = detect_dataset(paths, pattern)
    assert dataset.num_detected >= 8, f"검출 성공률이 너무 낮음: {dataset.num_detected}/{dataset.num_total}"

    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern, image_size, use_reprojection=False)

    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    assert any(r.success for r in results), "3개 모델이 전부 실패함"

    compute_frame_quality_scores(dataset, pattern, image_size, use_reprojection=True)
    validation_results = validate_all_models(dataset, camera_config, pattern, test_ratio=0.3)
    scores = compute_model_scores(calibration_results, validation_results)
    assert sum(1 for s in scores if s.is_recommended) == 1

    frame = dataset.enabled_frames[0]
    successful_model = next(m for m, r in calibration_results.items() if r.success)
    cal = calibration_results[successful_model]
    residual, n_lines = compute_straightness_residual(
        [frame], pattern, cal.camera_matrix, cal.distortion, successful_model
    )
    assert residual is not None
    assert n_lines > 0


# ---------------------------------------------------------------------------
# UI 레이어 - 패턴 타입 콤보, 행 숨김/보임, PatternConfig 생성
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_ui_default_pattern_type_is_charuco_with_visible_rows(qapp):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        assert win.pattern_type_combo.currentData() == PatternType.CHARUCO
        assert win._pattern_form.isRowVisible(4)  # Marker size
        assert win._pattern_form.isRowVisible(5)  # Dictionary
    finally:
        win.close()


def test_ui_switching_to_chessboard_hides_marker_and_dictionary_rows(qapp):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        idx = win.pattern_type_combo.findData(PatternType.CHESSBOARD)
        win.pattern_type_combo.setCurrentIndex(idx)

        assert not win._pattern_form.isRowVisible(4)
        assert not win._pattern_form.isRowVisible(5)

        pattern_config = win._current_pattern_config()
        assert pattern_config.type == PatternType.CHESSBOARD
        assert pattern_config.marker_size is None
        assert pattern_config.dictionary is None
    finally:
        win.close()


def test_ui_full_pipeline_with_chessboard_selected(qapp, tmp_path_factory):
    """UI에서 Chessboard를 고르고 실제 이미지로 검출~3모델까지 돌아가는지."""
    import glob
    from ui.main_window import MainWindow
    from calibration.compare import run_all_models

    win = MainWindow()
    try:
        idx = win.pattern_type_combo.findData(PatternType.CHESSBOARD)
        win.pattern_type_combo.setCurrentIndex(idx)
        win.squares_x_spin.setValue(7)
        win.squares_y_spin.setValue(5)
        win.square_size_spin.setValue(40.0)
        win.width_spin.setValue(1280)
        win.height_spin.setValue(720)

        image_dir = _build_synthetic_distorted_chessboard_dir(tmp_path_factory, n_images=10, seed=2)
        paths = sorted(glob.glob(f"{image_dir}/*.jpg"))

        camera_config = win._current_camera_config()
        pattern_config = win._current_pattern_config()
        assert pattern_config.type == PatternType.CHESSBOARD

        dataset = detect_dataset(paths, pattern_config)
        assert dataset.num_detected >= 6

        results = run_all_models(dataset, camera_config)
        assert any(r.success for r in results)

        win.dataset_view.set_dataset(dataset)
        assert win.dataset_view.table.rowCount() == dataset.num_total
    finally:
        win.close()
