"""
tests/test_theme_mode.py
========================

화면 테마 (다크 / 라이트): 두 팔레트가 같은 토큰을 갖는지, 고른 테마가 저장·적용되는지.
"""

from __future__ import annotations

import json

import pytest

from ui import theme


@pytest.fixture
def ui_config(tmp_path, monkeypatch):
    path = tmp_path / "ui.json"
    monkeypatch.setattr(theme, "UI_CONFIG_PATH", path)
    monkeypatch.delenv("CALIB_THEME", raising=False)
    yield path
    theme._use("dark")
    theme.APP_STYLESHEET = theme.build_stylesheet()


def test_palettes_define_the_same_tokens():
    assert set(theme.PALETTES["dark"]) == set(theme.PALETTES["light"])


def test_light_is_white_background_black_text(ui_config):
    theme._use("light")
    assert theme.Theme.BG_PRIMARY.upper() == "#FFFFFF"
    assert theme.Theme.TEXT_PRIMARY == "#111111"
    sheet = theme.build_stylesheet()
    assert "{Theme" not in sheet and "background-color: #FFFFFF" in sheet
    assert "#111111" in sheet  # 글자색


def test_mode_is_saved_and_env_overrides(ui_config, monkeypatch):
    assert theme.current_mode() == "dark"
    theme.save_mode("light")
    assert json.loads(ui_config.read_text())["theme"] == "light"
    assert theme.current_mode() == "light"
    monkeypatch.setenv("CALIB_THEME", "dark")
    assert theme.current_mode() == "dark"
    with pytest.raises(ValueError):
        theme.save_mode("blue")


def test_menu_saves_choice_and_can_defer_restart(ui_config, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication([])
    from ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    win = MainWindow()
    try:
        assert win.theme_actions["dark"].isChecked() and not win.theme_actions["light"].isChecked()
        win.theme_actions["light"].trigger()
        assert theme.saved_mode() == "light"
        assert "다음 실행부터 라이트" in win.status_label.text()
        assert win.theme_actions["dark"].isChecked()  # 지금 화면은 아직 다크
    finally:
        win.close()
        win.deleteLater()

