"""
tests/test_external_compare_view.py
========================================

ui/external_compare_view.py - "⑦ 외부 결과 비교" 탭이 실제로 계산 결과를
화면(표, verdict, 이미지 콤보)에 반영하는지 확인하는 배선 테스트.
계산 정확성 자체(승/패 판정, verdict 문구)는 tests/test_external_compare.py가
이미 담당하므로, 여기서는 "UI가 그 결과를 놓치지 않고 그대로 그리는지"만 본다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication  # noqa: E402

from calibration.compare import run_all_models  # noqa: E402
from calibration.types import (  # noqa: E402
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from calibration.validation import validate_all_models  # noqa: E402
from export.opencv import export_opencv_yaml  # noqa: E402
from ui.external_compare_view import ExternalCompareView  # noqa: E402

W, H = 640, 480
TRUE_K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.22, 0.06, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5,
        square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
    )


def _synthetic_dataset(pattern: PatternConfig, n_frames: int = 24, seed: int = 3) -> Dataset:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y), pattern.square_size, pattern.marker_size, aruco_dict
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 20:
        attempts += 1
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.25, (rng.random() - 0.5) * 0.25, 0.3 + rng.random() * 0.25])
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(-1, 2)
        if np.any(proj < 0) or np.any(proj[:, 0] > W) or np.any(proj[:, 1] > H):
            continue
        image_id = f"uiext_{len(frames):02d}"
        # path="-"이면 imread가 None을 돌려줘 시각화 갱신이 조용히 스킵되므로,
        # 이미지 비교 패널까지 확인하려면 실제 흰 캔버스를 파일로 저장해둔다.
        frames.append((image_id, proj, pts3d, ids))

    assert len(frames) >= n_frames * 0.8
    return frames


@pytest.fixture(scope="module")
def dataset_and_config(tmp_path_factory):
    pattern = _pattern_config()
    raw_frames = _synthetic_dataset(pattern)
    out_dir = tmp_path_factory.mktemp("uiext_images")

    frames: list[Frame] = []
    blank = np.full((H, W, 3), 255, dtype=np.uint8)
    for image_id, proj, pts3d, ids in raw_frames:
        path = str(out_dir / f"{image_id}.jpg")
        cv2.imwrite(path, blank)
        info = ImageInfo(image_id=image_id, path=path, width=W, height=H)
        det = DetectionResult(
            image_id=image_id, success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=pts3d.shape[0],
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    dataset = Dataset(frames=frames)
    camera_config = CameraConfig(width=W, height=H)
    return dataset, camera_config, pattern


@pytest.fixture(scope="module")
def validation_results(dataset_and_config):
    dataset, camera_config, pattern = dataset_and_config
    return validate_all_models(dataset, camera_config, pattern, test_ratio=0.3)


@pytest.fixture(scope="module")
def external_yaml_path(dataset_and_config, tmp_path_factory):
    dataset, camera_config, pattern = dataset_and_config
    results = run_all_models(dataset, camera_config, estimate_fisheye_uncertainty=False)
    pinhole = {r.model_name: r for r in results}[CameraModelType.PINHOLE]
    path = str(tmp_path_factory.mktemp("uiext_yaml") / "camera.yaml")
    export_opencv_yaml(pinhole, camera_config, pattern, path)
    return path


def test_manual_input_comparison_populates_table_and_verdict(qapp, dataset_and_config, validation_results):
    dataset, camera_config, pattern = dataset_and_config
    view = ExternalCompareView()
    view.set_context(dataset, camera_config, pattern, validation_results)

    view.my_model_combo.setCurrentIndex(view.my_model_combo.findData(CameraModelType.PINHOLE))
    view.external_model_combo.setCurrentIndex(view.external_model_combo.findData(CameraModelType.PINHOLE))
    view.fx_spin.setValue(float(TRUE_K[0, 0]))
    view.fy_spin.setValue(float(TRUE_K[1, 1]))
    view.cx_spin.setValue(float(TRUE_K[0, 2]))
    view.cy_spin.setValue(float(TRUE_K[1, 2]))
    view.distortion_edit.setText("0, 0, 0, 0, 0")  # 왜곡을 무시한, 명백히 더 나쁜 파라미터

    view._on_run_comparison()

    assert view._last_result is not None
    assert view._last_result.mine.success and view._last_result.external.success
    # 표가 실제로 채워졌는지 (N/A나 빈 셀이 아니라 숫자 텍스트) 확인.
    item = view.table.item(0, 0)
    assert item is not None and item.text() not in ("", "-", "N/A")
    assert "내 결과" in view.verdict_label.text() or "Pinhole" in view.verdict_label.text()
    assert view.image_combo.count() > 0


def test_load_yaml_button_flow_prefills_model_and_runs(qapp, dataset_and_config, validation_results, external_yaml_path):
    dataset, camera_config, pattern = dataset_and_config
    view = ExternalCompareView()
    view.set_context(dataset, camera_config, pattern, validation_results)

    # 실제 QFileDialog를 띄우지 않고, 그 다음 단계(파일 검증 + 모델 힌트 반영)만 재현.
    from export.opencv import detect_model_hint_from_opencv_yaml, load_camera_matrix_and_distortion_from_opencv_yaml
    load_camera_matrix_and_distortion_from_opencv_yaml(external_yaml_path)
    view._loaded_yaml_path = external_yaml_path
    hint = detect_model_hint_from_opencv_yaml(external_yaml_path)
    assert hint == CameraModelType.PINHOLE
    idx = view.external_model_combo.findData(hint)
    view.external_model_combo.setCurrentIndex(idx)

    view.my_model_combo.setCurrentIndex(view.my_model_combo.findData(CameraModelType.PINHOLE))
    view._on_run_comparison()

    assert view._last_result is not None
    assert view._last_result.external.label  # 라벨이 비어있지 않음


def test_run_without_context_shows_warning_and_does_not_crash(qapp, monkeypatch):
    view = ExternalCompareView()
    calls = []
    monkeypatch.setattr(
        "ui.external_compare_view.QMessageBox.warning",
        lambda *a, **k: calls.append(a),
    )
    # set_context를 아예 안 부른 상태 - 크래시 없이 경고만 뜨고 조용히 리턴해야 함.
    view.distortion_edit.setText("0,0,0,0,0")
    view._on_run_comparison()
    assert view._last_result is None
    assert calls, "데이터 없음 경고가 떠야 함"


def test_invalid_distortion_text_is_rejected_gracefully(qapp, monkeypatch, dataset_and_config, validation_results):
    dataset, camera_config, pattern = dataset_and_config
    view = ExternalCompareView()
    view.set_context(dataset, camera_config, pattern, validation_results)
    view.distortion_edit.setText("not,a,number")

    calls = []
    monkeypatch.setattr(
        "ui.external_compare_view.QMessageBox.warning",
        lambda *a, **k: calls.append(a),
    )
    view._on_run_comparison()  # 내부에서 ValueError -> QMessageBox.warning, 크래시 없어야 함
    assert view._last_result is None
    assert calls, "잘못된 입력 경고가 떠야 함"
