import json
import pytest
from PySide6.QtWidgets import QApplication,QDialog,QFileDialog,QPlainTextEdit


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_factory_report_view_is_readonly_and_preserves_controls(qapp,tmp_path,monkeypatch):
    from ui.external_compare_view import ExternalCompareView
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'schema_version':1,'conclusion':'UNCLEAR','candidates':{},'source':'<script>bad()</script>'}))
    view=ExternalCompareView();before=view.fx_spin.value();dialogs=[]
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(report),''))
    def capture(dialog):
        editor=dialog.findChild(QPlainTextEdit)
        assert editor.isReadOnly() and '<script>' in editor.toPlainText()
        dialogs.append(dialog)
        return 0
    monkeypatch.setattr(QDialog,'exec',capture)
    view.factory_report_button.click()
    assert len(dialogs)==1 and view.fx_spin.value()==before
    view.close()
