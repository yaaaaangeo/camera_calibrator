"""
tests/test_ui_feedback_fixes.py
====================================

사용자 피드백으로 고친 UI 항목들의 회귀 테스트:
  1. 검출 실패 이유가 Dataset 테이블에 표시되는지
  2. Dataset Diversity 막대그래프들이 동일한 위치/크기로 정렬되는지
  3. Model Comparison 표에서 Complexity 행이 빠졌는지
  4. Square/Marker size가 UI에서는 mm로 보이고 내부적으로는 m로 정확히 변환되는지
  5. 실패한 모델을 선택하면 버튼을 누르기 전에 미리 상태가 보이는지
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication

from calibration.types import CalibrationMethod, CalibrationResult, CameraModelType


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_detection_failure_reason_populated_no_markers():
    """마커가 아예 없는 이미지는 명확한 한국어 이유가 담겨야 한다."""
    import numpy as np
    from calibration.detector import detect_charuco
    import cv2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict)

    blank = np.full((480, 640), 255, dtype=np.uint8)
    result = detect_charuco(blank, board, "blank")

    assert not result.success
    assert result.failure_reason is not None
    assert "마커" in result.failure_reason
    assert "no charuco" not in result.failure_reason.lower()


def test_dataset_view_shows_failure_reason_in_status_cell(qapp, tmp_path):
    import numpy as np
    import cv2
    from ui.dataset_view import DatasetView
    from calibration.types import Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, np.full((480, 640, 3), 255, dtype=np.uint8))

    info = ImageInfo(image_id="blank", path=path, width=640, height=480)
    det = DetectionResult(
        image_id="blank", success=False, num_corners=0,
        failure_reason="ArUco 마커가 하나도 검출되지 않음 (조명/초점/각도 확인 필요)",
    )
    frame = Frame(image_info=info, detection=det, status=FrameStatus.DETECTION_FAILED)
    dataset = Dataset(frames=[frame])

    view = DatasetView()
    view.set_dataset(dataset)
    status_item = view.table.item(0, 1)

    assert "검출 실패" in status_item.text()
    assert "마커" in status_item.text()
    assert status_item.toolTip() == det.failure_reason


def test_diversity_bars_are_aligned(qapp):
    """라벨 길이가 달라도 모든 막대의 시작 x좌표(=크기)가 동일해야 한다."""
    from ui.dataset_view import DiversityBarsWidget

    widget = DiversityBarsWidget()
    widget.show()
    QApplication.processEvents()

    x_positions = {key: bar.pos().x() for key, bar in widget._bars.items()}
    overall_x = widget._overall_bar.pos().x()

    assert len(set(x_positions.values())) == 1, f"막대 시작 위치가 서로 다름: {x_positions}"
    assert overall_x == list(x_positions.values())[0]
    widget.close()


def test_dataset_view_is_scrollable_on_short_screens(qapp):
    from ui.dataset_view import DatasetView

    view = DatasetView()
    try:
        assert view.scroll_area.widgetResizable()
        view.resize(800, 320)
        view.show()
        QApplication.processEvents()
        assert view.scroll_area.verticalScrollBar() is not None
    finally:
        view.close()


def test_mouse_wheel_does_not_change_numbers_combos_or_tabs(qapp, monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from ui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_autosave_recovery", lambda self: None)
    win = MainWindow()
    try:
        win._show_intrinsic_workspace()
        win.show()
        QApplication.processEvents()
        targets = [
            (win.width_spin, win.width_spin.value),
            (win.dictionary_combo, win.dictionary_combo.currentIndex),
            (win.tabs.tabBar(), win.tabs.currentIndex),
        ]
        for widget, getter in targets:
            before = getter()
            event = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, 120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
            )
            QApplication.sendEvent(widget, event)
            assert getter() == before
    finally:
        win.close()


def test_result_view_table_has_no_complexity_row(qapp):
    from ui.result_view import ResultView

    view = ResultView()
    assert view.table.rowCount() == 14
    labels = [view.table.verticalHeaderItem(i).text() for i in range(view.table.rowCount())]
    assert "Complexity" not in labels
    assert labels == [
        "Train RMS", "Test RMS", "Test P95", "Edge RMS", "Straightness",
        "Radial Edge", "AIC", "BIC", "Stability", "Observability",
        "Undistortion", "Model Score", "Selection Conf.", "Recommend",
    ]
    view.close()


def test_pattern_size_widgets_use_mm_and_convert_to_meters(qapp):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        assert win.square_size_spin.suffix().strip() == "mm"
        assert win.marker_size_spin.suffix().strip() == "mm"

        win.square_size_spin.setValue(40.0)
        win.marker_size_spin.setValue(30.0)
        pattern_config = win._current_pattern_config()

        assert abs(pattern_config.square_size - 0.04) < 1e-9
        assert abs(pattern_config.marker_size - 0.03) < 1e-9
    finally:
        win.close()


def test_camera_setup_panel_can_collapse_and_expand(qapp, monkeypatch):
    from ui.main_window import MainWindow
    from ui.theme import APP_STYLESHEET

    monkeypatch.setattr(MainWindow, "_offer_autosave_recovery", lambda self: None)
    win = MainWindow()
    try:
        win._show_intrinsic_workspace()
        win.show()
        QApplication.processEvents()
        expanded_height = win.settings_group.height()
        assert win.settings_group.isCheckable()  # 기존 토글 동작은 그대로 유지
        assert 'QGroupBox#settingsPanel::indicator' in APP_STYLESHEET
        assert 'width: 0px; height: 0px' in APP_STYLESHEET  # 별도 checkbox는 보이지 않음

        win.settings_group.setChecked(False)
        QApplication.processEvents()
        assert not win.settings_content.isVisible()
        assert "▶" in win.settings_group.title()
        assert win.settings_group.height() < expanded_height

        win.settings_group.setChecked(True)
        QApplication.processEvents()
        assert win.settings_content.isVisible()
        assert "▼" in win.settings_group.title()
    finally:
        win.close()


def test_advanced_calibration_only_expands_for_object_releasing(qapp, monkeypatch):
    from ui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_autosave_recovery", lambda self: None)
    win = MainWindow()
    try:
        win._show_intrinsic_workspace()
        win.show()
        QApplication.processEvents()
        group = win.result_view.advanced_group
        content = win.result_view.advanced_content

        assert not group.isEnabled()
        assert not group.isChecked()
        assert content.isHidden()

        # Standard에서는 프로그램으로 checked를 요청해도 내용이 펼쳐지지 않는다.
        group.setChecked(True)
        assert not group.isChecked()
        assert content.isHidden()

        win.calibration_method_combo.setCurrentIndex(
            win.calibration_method_combo.findData(CalibrationMethod.OBJECT_RELEASING)
        )
        assert group.isEnabled()
        group.setChecked(True)
        assert group.isChecked()
        assert not content.isHidden()
        assert "▼" in group.title()

        win.calibration_method_combo.setCurrentIndex(
            win.calibration_method_combo.findData(CalibrationMethod.STANDARD)
        )
        assert not group.isEnabled()
        assert not group.isChecked()
        assert content.isHidden()
        assert "▶" in group.title()
    finally:
        win.close()


def test_subset_recalibration_falls_back_to_ranking_model_when_combo_data_is_none(qapp):
    """Qt combo가 일시적으로 빈 상태여도 None model을 worker로 보내지 않아야 한다."""
    from calibration.types import (
        Dataset, SceneQualityAnalysis, SceneQualityEntry,
    )
    from ui.scene_quality_view import SceneQualityView

    view = SceneQualityView()
    try:
        model = CameraModelType.BROWN_CONRADY
        result = CalibrationResult(model_name=model, success=True)
        analysis = SceneQualityAnalysis(
            model_name=model,
            scenes=[
                SceneQualityEntry(f"scene-{index}", rank=index, quality_score=90 - index)
                for index in range(1, 4)
            ],
        )
        view.set_context(Dataset(), None, {model: result}, analysis, None)
        view._set_all_checked(True)
        emitted = []
        view.recalibrate_requested.connect(
            lambda frame_ids, selected_model: emitted.append((frame_ids, selected_model))
        )

        view.model_combo.clear()  # currentData() == None 재현
        view._request_recalibration()

        assert emitted
        assert emitted[0][1] == model
        assert len(emitted[0][0]) == 3
    finally:
        view.close()


def test_scene_ranking_compares_full_and_best_subset_undistortion(qapp, tmp_path):
    import cv2
    import numpy as np
    from calibration.types import (
        CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo,
        SceneQualityAnalysis, SceneQualityEntry, SubsetCalibrationResult,
    )
    from ui.scene_quality_view import SceneQualityView

    image_path = tmp_path / "scene.png"
    yy, xx = np.indices((240, 320))
    source = np.dstack((xx % 256, yy % 256, (xx + yy) % 256)).astype(np.uint8)
    cv2.imwrite(str(image_path), source)
    corners = np.array([[80, 60], [240, 60], [80, 180], [240, 180]], dtype=np.float32).reshape(-1, 1, 2)
    frame = Frame(
        image_info=ImageInfo("scene-1", str(image_path), 320, 240, sharpness=500),
        detection=DetectionResult(
            "scene-1", True, corners=corners, num_corners=len(corners),
        ),
        status=FrameStatus.DETECTED,
    )
    model = CameraModelType.BROWN_CONRADY
    K = np.array([[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]])
    original_result = CalibrationResult(
        model_name=model,
        camera_matrix=K.copy(),
        distortion=np.array([[-0.25], [0.08], [0.0], [0.0], [0.0]]),
        per_frame_error={"scene-1": 0.8},
        success=True,
    )
    subset_result = SubsetCalibrationResult(
        model_name=model,
        selected_frame_ids=["scene-1"],
        calibration_result=CalibrationResult(
            model_name=model,
            camera_matrix=K.copy(),
            distortion=np.array([[-0.08], [0.01], [0.0], [0.0], [0.0]]),
            success=True,
        ),
    )
    analysis = SceneQualityAnalysis(
        model_name=model,
        scenes=[SceneQualityEntry("scene-1", rank=1, quality_score=90)],
    )
    view = SceneQualityView()
    try:
        view.resize(1000, 700)
        view.show()
        view.set_context(
            Dataset(frames=[frame]), CameraConfig(320, 240),
            {model: original_result}, analysis, None,
        )
        QApplication.processEvents()
        assert view.original_preview_label.pixmap() is not None
        assert view.best_subset_preview_label.pixmap().isNull()
        assert "Re-Calibrate Selected Scenes" in view.best_subset_preview_label.text()

        view.set_context(
            Dataset(frames=[frame]), CameraConfig(320, 240),
            {model: original_result}, analysis, subset_result,
        )
        QApplication.processEvents()

        assert view.original_preview_label.pixmap() is not None
        assert view.best_subset_preview_label.pixmap() is not None
        assert (
            view.original_preview_label.pixmap().cacheKey()
            != view.best_subset_preview_label.pixmap().cacheKey()
        )
        assert view.original_calibration_group.title() == "Original Calibration (1 scenes)"
        assert view.best_subset_calibration_group.title() == "Best Subset Calibration (1 scenes)"
        assert "Same source image undistorted with each calibration" in view.preview_status_label.text()
    finally:
        view.close()


def test_main_window_uses_documented_top_level_tabs(qapp):
    from PySide6.QtCore import Qt
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        # Lite 버전: Coverage Map/Dataset Diversity는 ① Dataset 안으로
        # 통합됐고, 사용자가 직접 프레임을 검토/제외하는 Outlier 탭은
        # 제거됐다. Diagnosis 탭은 빠졌고, Export 탭은 Camera Setup 오른쪽
        # 열의 Export 버튼(다이얼로그)으로 옮겨갔다. Validation/Stability/
        # External Compare 탭도 제거됐고, 옛 Error Analysis(Undistort
        # Preview/Straightness Map을 합치고 Edge Error Map은 뺀 것)는
        # "② Preview"로 이름이 바뀌었다 - 그만큼 뒤 탭들의 번호가 앞당겨진다.
        assert labels == [
            "① Dataset",
            "② Preview",
            "③ Scene Ranking",
            "④ Model Comparison",
        ]
        assert win.result_view.model_comparison_scroll_area.widgetResizable()
        assert (
            win.result_view.model_comparison_scroll_area.widget()
            is win.result_view.model_comparison_content
        )
        assert win.result_view.table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        expected_rows_height = sum(
            win.result_view.table.rowHeight(row)
            for row in range(win.result_view.table.rowCount())
        )
        assert win.result_view.table.height() >= expected_rows_height
    finally:
        win.close()


def test_main_window_has_help_menu_next_to_file_menu(qapp):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        labels = [action.text() for action in win.menuBar().actions()]
        assert labels[:3] == ["파일", "설명", "도구"]
    finally:
        win.close()


def test_main_window_starts_on_calibration_home_and_keeps_intrinsic_tabs(qapp):
    from ui.main_window import MainWindow

    win = MainWindow()
    try:
        assert win.workspace_stack.currentWidget() is win.home_view
        win._show_intrinsic_workspace()
        assert win.workspace_stack.currentWidget() is win.intrinsic_workspace
        assert win.tabs.tabText(0) == "① Dataset"
    finally:
        win.close()


def test_model_status_label_warns_on_failed_model(qapp):
    from ui.result_view import ResultView

    view = ResultView()
    try:
        fail_result = CalibrationResult(
            model_name=CameraModelType.FISHEYE, success=False, error_message="테스트 실패 사유"
        )
        ok_result = CalibrationResult(model_name=CameraModelType.PINHOLE, success=True, rms_error=0.42)
        view.set_comparison(
            {CameraModelType.PINHOLE: ok_result, CameraModelType.FISHEYE: fail_result}, {}, []
        )

        view.select_model(CameraModelType.FISHEYE)
        assert "실패" in view.model_status_label.text()
        assert "테스트 실패 사유" in view.model_status_label.text()
        assert not view.export_opencv_button.isEnabled()
        assert not view.cross_dataset_button.isEnabled()

        view.select_model(CameraModelType.PINHOLE)
        assert "사용 가능" in view.model_status_label.text()
        assert view.export_opencv_button.isEnabled()
        assert view.cross_dataset_button.isEnabled()
    finally:
        view.close()


def test_model_status_label_warns_on_unrun_model(qapp):
    """아예 계산이 안 된 모델(dict에 키 자체가 없음)도 명확히 구분해서 알려줘야 한다."""
    from ui.result_view import ResultView

    view = ResultView()
    try:
        ok_result = CalibrationResult(model_name=CameraModelType.PINHOLE, success=True, rms_error=0.3)
        view.set_comparison({CameraModelType.PINHOLE: ok_result}, {}, [])

        view.select_model(CameraModelType.FISHEYE)
        assert "계산되지 않았습니다" in view.model_status_label.text()
        assert not view.export_opencv_button.isEnabled()
    finally:
        view.close()


def test_dataset_table_status_column_stretches_without_wrapping(qapp):
    """"상태" 컬럼이 남는 공간을 가져가야 하고(Stretch), 실패 이유가 길어도
    줄바꿈하지 않고 한 줄로 말줄임 처리해 검출 성공 행과 높이가 같아야 한다
    (긴 실패 이유 때문에 그 행만 유독 높아지는 문제 수정).
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QHeaderView
    from calibration.types import Dataset, DetectionResult, Frame, FrameStatus, ImageInfo
    from ui.dataset_view import DatasetView

    detected_info = ImageInfo(image_id="frame_0000", path="/fake/0.jpg", width=640, height=480)
    detected_det = DetectionResult(image_id="frame_0000", success=True, num_corners=24)
    failed_info = ImageInfo(image_id="frame_0001", path="/fake/1.jpg", width=640, height=480)
    failed_det = DetectionResult(
        image_id="frame_0001", success=False, num_corners=0,
        failure_reason="마커는 검출됐지만 체스보드 코너 보간 실패 (보드 일부만 보이거나 각도가 너무 큼)",
    )
    dataset = Dataset(frames=[
        Frame(image_info=detected_info, detection=detected_det, status=FrameStatus.DETECTED),
        Frame(image_info=failed_info, detection=failed_det, status=FrameStatus.DETECTION_FAILED),
    ])

    view = DatasetView()
    try:
        header = view.table.horizontalHeader()
        assert header.sectionResizeMode(1) == QHeaderView.Stretch, "상태(컬럼1)가 Stretch여야 함"
        assert not view.table.wordWrap(), "긴 실패 이유로 행 높이가 들쭉날쭉해지지 않도록 줄바꿈은 꺼져 있어야 함"

        # 좁은 창이어도 줄바꿈이 없으니 두 행 높이가 같아야 한다.
        view.resize(500, 400)
        view.show()
        QCoreApplication.processEvents()
        view.set_dataset(dataset)
        QCoreApplication.processEvents()
        detected_row_height = view.table.rowHeight(0)
        failed_row_height = view.table.rowHeight(1)
        assert failed_row_height == detected_row_height, (
            "검출 실패 행이 검출 성공 행과 높이가 같아야 함 (긴 실패 이유는 tooltip으로만 표시)"
        )
        # 전체 문구는 여전히 tooltip으로 확인 가능해야 한다.
        assert view.table.item(1, 1).toolTip() == failed_det.failure_reason
    finally:
        view.close()
