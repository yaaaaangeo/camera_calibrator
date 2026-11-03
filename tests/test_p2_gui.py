import json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from ui.automatic_calibration_dialog import AutomaticCalibrationDialog


def test_automatic_panel_uses_current_board_and_renders_status(tmp_path,pattern_config,camera_config):
    app=QApplication.instance() or QApplication([])
    dialog=AutomaticCalibrationDialog(tmp_path,pattern_config,camera_config,'/camera')
    dialog.serial.setText('SN123')
    args=dialog.command()
    import yaml
    raw=yaml.safe_load(Path(args[args.index('--profile')+1]).read_text(encoding='utf-8'))
    assert raw['p1']['pattern']['squares_x']==pattern_config.squares_x
    assert args[:3]==['-m','automation','live']
    dialog.display({'state':'CAPTURING','saved':12,'guidance':['move board']})
    assert '12' in dialog.status.text() and 'move board' in dialog.details.toPlainText()
    dialog.close()


def test_manual_live_dialog_exposes_separate_automatic_entry(tmp_path,pattern_config):
    from ui.live_capture_dialog import LiveCaptureDialog
    app=QApplication.instance() or QApplication([])
    dialog=LiveCaptureDialog(str(tmp_path),pattern_config)
    assert dialog.automatic_calibration_button.isEnabled()
    assert dialog.capture_button is not None and dialog.auto_capture_check.isChecked()
    dialog.close()


def test_old_cancel_timer_cannot_kill_a_new_run(tmp_path,pattern_config):
    from PySide6.QtCore import QProcess
    app=QApplication.instance() or QApplication([])
    dialog=AutomaticCalibrationDialog(tmp_path,pattern_config)
    original=dialog.process; killed=[]
    class Process:
        def state(self): return QProcess.Running
        def kill(self): killed.append(True)
    dialog.process=Process(); dialog.generation=2
    dialog._force_stop(1); assert not killed
    dialog._force_stop(2); assert killed
    dialog.process=original; dialog.close()


def test_actual_gui_process_reports_failed_replay_without_touching_source(tmp_path,pattern_config):
    import cv2
    import numpy as np
    import time
    from PySide6.QtCore import QProcess
    app=QApplication.instance() or QApplication([])
    images=tmp_path/'images'; images.mkdir()
    source=images/'blank.png'; cv2.imwrite(str(source),np.zeros((480,640,3),np.uint8))
    before=source.read_bytes()
    dialog=AutomaticCalibrationDialog(tmp_path/'outputs',pattern_config)
    dialog.serial.setText('GUI-TEST'); dialog.source.setCurrentText('Image directory'); dialog.path.setText(str(images))
    dialog.start_run()
    deadline=time.monotonic()+20
    while dialog.process.state()!=QProcess.NotRunning and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.01)
    app.processEvents()
    try:
        assert dialog.process.state()==QProcess.NotRunning
        assert dialog.session is not None and (dialog.session/'p2_manifest.json').is_file()
        checkpoint=json.loads((dialog.session/'p2_manifest.json').read_text(encoding='utf-8'))
        assert checkpoint['state']=='ERROR' and checkpoint['rejected']==1
        assert dialog.start.isEnabled() and source.read_bytes()==before
    finally:
        if dialog.process.state()!=QProcess.NotRunning:
            dialog.process.kill(); dialog.process.waitForFinished(3000)
        dialog.close()
