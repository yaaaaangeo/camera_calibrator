"""Offscreen resize regressions for layout-owned responsive GUI behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is required for UI tests")

from PySide6.QtWidgets import QApplication, QBoxLayout

from ui.main_window import MainWindow
from ui.responsive_layout import ResponsiveRow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _process_layout(app: QApplication) -> None:
    for _ in range(3):
        app.processEvents()


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
