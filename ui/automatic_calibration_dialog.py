"""Thin QProcess adapter for the Qt-independent live controller."""
from __future__ import annotations
import dataclasses
import codecs
import json
import os
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QPlainTextEdit, QVBoxLayout)
import yaml


class AutomaticCalibrationDialog(QDialog):
    def __init__(self, output_dir, pattern_config, camera_config=None, topic='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('자동 Intrinsic 캘리브레이션 (P2)')
        self.resize(760, 650)
        self.pattern, self.camera = pattern_config, camera_config
        self.session = None
        self.buffer = ''
        self.closing = False
        self.generation = 0
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.profile = QLineEdit('default')
        self.model = QLineEdit('CAMERA')
        self.serial = QLineEdit()
        self.serial.setPlaceholderText('이 영상 토픽에 연결된 실제 카메라 Serial Number')
        self.source = QComboBox()
        self.source.addItems(['ROS topic', 'Image directory', 'Rosbag'])
        self.path = QLineEdit()
        self.path.setPlaceholderText('재생 이미지 폴더 또는 rosbag 경로')
        self.topic = QLineEdit(topic)
        self.msg_type = QComboBox()
        self.msg_type.addItems(['sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'])
        self.root = QLineEdit(str(Path(output_dir).resolve() / 'automatic'))
        self.resume = QLineEdit()
        self.resume.setPlaceholderText('비워두면 새 세션; 복구할 P2 세션 폴더 지정 가능')
        self.inputs = [self.profile, self.model, self.serial, self.source, self.path,
                       self.topic, self.msg_type, self.root, self.resume]
        for title, field in [('P2 프로파일 YAML', self.profile), ('카메라 모델', self.model),
                             ('Serial Number', self.serial), ('입력 종류', self.source),
                             ('재생 경로', self.path), ('영상 토픽', self.topic), ('메시지 타입', self.msg_type),
                             ('출력 루트', self.root), ('복구 세션', self.resume)]:
            form.addRow(title, field)
        layout.addLayout(form)
        self.status = QLabel('대기 중 · 기본 준비도 기준은 실험용입니다. 보드 사양과 프로파일을 확인하세요.')
        self.status.setWordWrap(True)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.document().setMaximumBlockCount(300)
        layout.addWidget(self.status)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        self.start = QPushButton('자동 캘리브레이션 시작')
        self.cancel = QPushButton('중지 / 취소')
        self.close_button = QPushButton('닫기')
        self.start.clicked.connect(self.start_run)
        self.cancel.clicked.connect(self.cancel_run)
        self.close_button.clicked.connect(self.close)
        self.cancel.setEnabled(False)
        for button in (self.start, self.cancel, self.close_button):
            row.addWidget(button)
        layout.addLayout(row)

    def command(self):
        from automation.identity import build_identity
        from automation.live.config import load_live_profile, resolve_live_profile
        build_identity(self.model.text(), self.serial.text())
        spec = self.profile.text().strip() or 'default'
        if spec == 'default':
            raw = load_live_profile().raw
            if self.resume.text().strip():
                raw = json.loads((Path(self.resume.text().strip()) / 'p2_manifest.json').read_text(encoding='utf-8'))['profile']
            else:
                keys = ('type', 'squares_x', 'squares_y', 'square_size', 'marker_size', 'dictionary',
                        'circle_grid_type', 'aprilgrid_variant')
                raw['p1'] = {'pattern': {k: getattr(getattr(self.pattern, k), 'value', getattr(self.pattern, k))
                                          for k in keys if getattr(self.pattern, k, None) is not None}}
                if self.camera is not None:
                    raw['p1']['camera'] = {'width': self.camera.width, 'height': self.camera.height}
            resolve_live_profile(raw)
            folder = Path(self.root.text()).expanduser().resolve() / '_profiles'
            folder.mkdir(parents=True, exist_ok=True)
            spec = str(folder / f'p2_{uuid.uuid4().hex}.yaml')
            with open(spec, 'x', encoding='utf-8') as stream:
                yaml.safe_dump(raw, stream, allow_unicode=True)
        else:
            load_live_profile(spec)
            spec = str(Path(spec).expanduser().resolve())
        args = ['-m', 'automation', 'live', '--profile', spec, '--camera-model', self.model.text(),
                '--serial', self.serial.text(), '--output-root', self.root.text()]
        if self.resume.text().strip():
            args += ['--resume', self.resume.text().strip()]
        kind = self.source.currentText()
        if kind == 'ROS topic':
            args += ['--ros', '--topic', self.topic.text(), '--msg-type', self.msg_type.currentText()]
        elif kind == 'Image directory':
            args += ['--replay', self.path.text()]
        else:
            args += ['--bag', self.path.text(), '--topic', self.topic.text()]
        return args

    def start_run(self):
        if self.process.state() != QProcess.NotRunning:
            return
        try:
            args = self.command()
        except Exception as exc:
            self.status.setText(f'설정 오류: {exc}')
            return
        env = QProcessEnvironment.systemEnvironment()
        import automation
        root = str(Path(automation.__file__).resolve().parent.parent)
        env.insert('PYTHONPATH', os.pathsep.join(filter(None, [root, env.value('PYTHONPATH')])))
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('PYTHONUTF8', '1')
        self.process.setProcessEnvironment(env)
        self.buffer = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.generation += 1
        self.session = None
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        for field in self.inputs:
            field.setEnabled(False)
        self.process.start(sys.executable, args)

    def _read(self):
        self.buffer += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            try:
                self.display(json.loads(line))
            except ValueError:
                self.details.appendPlainText(line)

    def _read_error(self):
        self.details.appendPlainText(bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace'))

    def display(self, data):
        if data.get('session'):
            self.session = Path(data['session'])
        state = data.get('state', '?')
        self.status.setText(f"{state} · 연결 {data.get('connection', '?')} · 저장 {data.get('saved', 0)}장 · "
                            f"계산 {data.get('calculations', 0)}회 · 추가 촬영 {data.get('recaptures', 0)}회")
        parts = []
        if data.get('buffer'):
            stats = data['buffer']
            parts.append(f"영상 수신 {stats['received']} · 처리 {stats['delivered']} · 오래된 프레임 교체 {stats['replaced']}")
        detection = data.get('detection')
        if detection:
            parts.append(f"검출: {detection.get('success')} / 코너 {detection.get('num_corners')} / "
                         f"중심 {detection.get('board_center_px')} / 크기 {detection.get('board_area_ratio')} / "
                         f"기울기 {detection.get('board_tilt_deg')} / 경계 여유 {detection.get('min_edge_margin_px')}")
            parts.append(f"최근 검출 처리 {detection.get('processing_seconds', 0)*1000:.1f} ms")
        ready = data.get('readiness')
        if ready:
            if ready.get('coverage_basis') == 'observable':
                parts.append(f"Full-frame Coverage {ready['full_frame_coverage_pct']:.1f}% / "
                             f"Observable Coverage {ready['coverage_pct']:.1f}% (공간 준비도; 정확도 검증 아님)")
            parts.append(f"Coverage {ready['coverage_pct']:.1f}% / 위치 구역 {ready['center_cells']}/9 / "
                         f"다양성 {ready['diversity']}")
            if ready.get('observability'):
                obs = ready['observability']
                parts.append(f"Observability: {obs.get('report', {}).get('observability_score')} / "
                             f"사전 계산 {obs.get('seconds', 0):.2f}초")
        parts.extend(data.get('guidance', []))
        if data.get('calibration_progress'):
            parts.append('계산 단계: ' + ', '.join(f"{s['name']}={s['status']}" for s in data['calibration_progress']))
        for attempt in data.get('attempts', []):
            parts.append(f"P2 판정 {attempt['outcome'].get('verdict')} / P1 원판정 {attempt.get('p1_verdict')} / 모델 {attempt.get('recommendation')} / "
                         f"결과 {attempt['outcome'].get('session_dir')}")
            if attempt.get('p2_acceptance_path'):
                parts.append(f"P2 검토 근거: {attempt['p2_acceptance_path']}")
        if self.session:
            parts.append(f'세션: {self.session}')
        self.details.setPlainText('\n'.join(parts))

    def cancel_run(self):
        if self.session is not None and self.process.state() != QProcess.NotRunning:
            try:
                (self.session / 'cancel.request').touch()
            except OSError as exc:
                self.status.setText(f'취소 요청 저장 오류: {exc}')
        elif self.process.state() != QProcess.NotRunning:
            self.process.terminate()
        generation = self.generation
        QTimer.singleShot(10000, lambda: self._force_stop(generation))

    def _force_stop(self, generation):
        if generation == self.generation and self.process.state() != QProcess.NotRunning:
            self.process.kill()  # task owner-watchdog terminates only its own descendants

    def _process_error(self, _error):
        self.status.setText(self.process.errorString())
        if self.process.state() == QProcess.NotRunning:
            self.start.setEnabled(True)
            self.cancel.setEnabled(False)
            for field in self.inputs:
                field.setEnabled(True)

    def _finished(self, code, _status):
        self._read()
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        for field in self.inputs:
            field.setEnabled(True)
        self.status.setText(self.status.text() + f' · 프로세스 종료 {code}')
        if self.closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.process.state() != QProcess.NotRunning:
            self.closing = True
            self.cancel_run()
            event.ignore()
        else:
            super().closeEvent(event)
