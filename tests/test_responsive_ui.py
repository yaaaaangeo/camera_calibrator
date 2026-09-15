"""Offscreen resize regressions for layout-owned responsive GUI behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is required for UI tests")

from PySide6.QtWidgets import QApplication, QBoxLayout, QScrollArea, QTextBrowser

from ui.help_view import WindshieldGuideDialog
from ui.main_window import MainWindow
from ui.windshield_common import ResponsiveRow
from ui.windshield_workspace import WindshieldWorkspace


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


SIZES = [(1920, 1080), (1366, 768), (1280, 720), (1024, 768), (900, 650), (800, 600)]


def _process_layout(app: QApplication) -> None:
    for _ in range(3):
        app.processEvents()


def _assert_children_do_not_overlap(row: ResponsiveRow) -> None:
    widgets = [
        row.box_layout.itemAt(i).widget()
        for i in range(row.box_layout.count())
        if row.box_layout.itemAt(i).widget() is not None
        and row.box_layout.itemAt(i).widget().isVisible()
    ]
    for index, first in enumerate(widgets):
        for second in widgets[index + 1 :]:
            assert not first.geometry().intersects(second.geometry())


def test_windshield_all_requested_window_sizes_are_stable(qapp):
    workspace = WindshieldWorkspace()
    workspace.show()

    for width, height in SIZES:
        workspace.resize(width, height)
        for tab_index in range(workspace.tabs.count()):
            workspace.tabs.setCurrentIndex(tab_index)
            _process_layout(qapp)
            assert workspace.tabs.currentWidget().isVisible()

    assert workspace.tabs.count() == 6
    assert workspace.tabs.tabBar().usesScrollButtons()
    assert not workspace.tabs.tabBar().expanding()
    workspace.close()


def test_long_windshield_pages_scroll_and_controls_remain_reachable(qapp):
    workspace = WindshieldWorkspace()
    workspace.resize(800, 600)
    workspace.show()
    workspace.tabs.setCurrentIndex(2)
    _process_layout(qapp)

    for index in range(4):
        assert isinstance(workspace.tabs.widget(index), QScrollArea)

    model_scroll = workspace.tabs.widget(2)
    assert model_scroll.verticalScrollBar().maximum() > 0
    model_scroll.ensureWidgetVisible(workspace.export_button)
    _process_layout(qapp)
    export_center = workspace.export_button.mapTo(
        model_scroll.viewport(), workspace.export_button.rect().center()
    )
    assert model_scroll.viewport().rect().contains(export_center)
    assert workspace.run_button.isEnabled()
    assert not workspace.export_button.isEnabled()  # enabled only after a successful result
    workspace.close()


def test_rows_and_charts_stack_at_narrow_width_without_overlap(qapp):
    workspace = WindshieldWorkspace()
    workspace.resize(800, 600)
    workspace.show()
    workspace.tabs.setCurrentIndex(0)
    _process_layout(qapp)

    base_row = workspace.base_load_buttons[0].parentWidget()
    assert isinstance(base_row, ResponsiveRow)
    assert base_row.box_layout.direction() == QBoxLayout.TopToBottom
    _assert_children_do_not_overlap(base_row)

    workspace.tabs.setCurrentIndex(2)
    _process_layout(qapp)
    chart_row = workspace.radial_chart.parentWidget()
    assert isinstance(chart_row, ResponsiveRow)
    assert chart_row.box_layout.direction() == QBoxLayout.TopToBottom
    _assert_children_do_not_overlap(chart_row)

    workspace.resize(1366, 768)
    _process_layout(qapp)
    assert chart_row.box_layout.direction() == QBoxLayout.LeftToRight
    _assert_children_do_not_overlap(chart_row)
    workspace.close()


def test_reflection_and_ghost_subtabs_are_scrollable(qapp):
    workspace = WindshieldWorkspace()
    for outer_index in (4, 5):
        outer = workspace.tabs.widget(outer_index)
        assert outer.tabBar().usesScrollButtons()
        for subtab_index in range(outer.count()):
            assert isinstance(outer.widget(subtab_index), QScrollArea)


def test_windshield_guide_uses_canonical_markdown(qapp):
    dialog = WindshieldGuideDialog()
    browser = dialog.findChild(QTextBrowser, "windshieldGuideBrowser")
    assert browser is not None
    text = browser.toPlainText()
    assert "30초 요약" in text
    assert "Base K,D는 Windshield Calibration 중 다시 최적화하지 않습니다" in text


def test_guide_and_primary_workflow_buttons_keep_their_connections(qapp, monkeypatch):
    import ui.help_view as help_module
    from PySide6.QtWidgets import QMessageBox

    calls = []

    class FakeGuideDialog:
        def __init__(self, parent):
            calls.append(("guide_init", parent))

        def exec(self):
            calls.append(("guide_exec", None))

    monkeypatch.setattr(help_module, "WindshieldGuideDialog", FakeGuideDialog)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, _message: calls.append(("warning", title)),
    )
    workspace = WindshieldWorkspace()

    workspace.windshield_guide_button.click()
    workspace.base_load_buttons[0].click()
    workspace.load_dataset_button.click()
    workspace.run_button.click()

    assert ("guide_exec", None) in calls
    assert ("warning", "Base Camera") in calls
    assert ("warning", "Dataset") in calls
    assert ("warning", "Windshield Calibration") in calls


def test_main_camera_settings_stack_and_scroll_at_800x600(qapp, monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_autosave_recovery", lambda self: None)
    window = MainWindow()
    window.workspace_stack.setCurrentWidget(window.intrinsic_workspace)
    window.resize(800, 600)
    window.show()
    _process_layout(qapp)

    row = window.settings_content.findChild(ResponsiveRow)
    assert row is not None
    assert row.box_layout.direction() == QBoxLayout.TopToBottom
    assert window.settings_scroll_area.verticalScrollBar().maximum() > 0
    window.settings_scroll_area.ensureWidgetVisible(window.cancel_button)
    _process_layout(qapp)
    assert window.run_button.parentWidget() is not None
    window.close()


def test_main_camera_settings_put_pattern_left_and_camera_center(qapp, monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_autosave_recovery", lambda self: None)
    window = MainWindow()
    try:
        row = window.settings_content.findChild(ResponsiveRow)
        assert row is not None
        assert row.box_layout.itemAt(0).widget().objectName() == "calibrationPatternColumn"
        assert row.box_layout.itemAt(1).widget().objectName() == "cameraSetupColumn"
    finally:
        window.close()
