"""
Live dual ROS topic capture for stereo calibration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from calibration.ros_live import (
    ROS_LIVE_BACKEND,
    DualLiveTopicSubscriber,
    LiveTopic,
    StereoLivePair,
    build_live_dual_capture_qa_report,
)
from calibration.detector import build_detect_fn
from calibration.stereo import match_common_charuco_corners


def _pixmap(img_bgr: np.ndarray, max_width: int = 420) -> QPixmap:
    h, w = img_bgr.shape[:2]
    if w > max_width:
        img_bgr = cv2.resize(img_bgr, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())


class StereoLiveCaptureDialog(QDialog):
    pair_ready = Signal(object)
    error_ready = Signal(str)

    def __init__(self, output_dir: str, pattern_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Dual Camera Capture")
        self.setMinimumWidth(900)

        self.output_dir = Path(output_dir)
        self.pattern_config = pattern_config
        self._detect_fn = build_detect_fn(pattern_config) if pattern_config is not None else None
        self.cam1_dir = self.output_dir / "camera1"
        self.cam2_dir = self.output_dir / "camera2"
        self.cam1_dir.mkdir(parents=True, exist_ok=True)
        self.cam2_dir.mkdir(parents=True, exist_ok=True)
        self.captured_paths_cam1: list[str] = []
        self.captured_paths_cam2: list[str] = []

        self._subscriber: DualLiveTopicSubscriber | None = None
        self._latest_pair: StereoLivePair | None = None
        self._topics: list[LiveTopic] = []
        self._subscribe_started_at: float | None = None
        self._last_sync_delta_ms: float | None = None

        self.pair_ready.connect(self._on_pair_ready)
        self.error_ready.connect(self._on_error)

        layout = QVBoxLayout(self)
        self.status_label = QLabel(self._backend_text())
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        topic_row = QHBoxLayout()
        self.topic1_combo = QComboBox()
        self.topic2_combo = QComboBox()
        self.refresh_button = QPushButton("토픽 새로고침")
        self.refresh_button.clicked.connect(self._refresh_topics)
        self.qa_button = QPushButton("QA 체크")
        self.qa_button.clicked.connect(self._show_qa_report)
        self.qa_save_button = QPushButton("QA 저장")
        self.qa_save_button.clicked.connect(self._save_qa_report)
        self.subscribe_button = QPushButton("구독 시작")
        self.subscribe_button.clicked.connect(self._toggle_subscribe)
        topic_row.addWidget(QLabel("Camera 1"))
        topic_row.addWidget(self.topic1_combo, stretch=1)
        topic_row.addWidget(QLabel("Camera 2"))
        topic_row.addWidget(self.topic2_combo, stretch=1)
        topic_row.addWidget(self.refresh_button)
        topic_row.addWidget(self.qa_button)
        topic_row.addWidget(self.qa_save_button)
        topic_row.addWidget(self.subscribe_button)
        layout.addLayout(topic_row)

        preview_grid = QGridLayout()
        self.preview1 = QLabel("Camera 1 preview")
        self.preview2 = QLabel("Camera 2 preview")
        for label in (self.preview1, self.preview2):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(260)
            label.setProperty("surface", "image")
        preview_grid.addWidget(self.preview1, 0, 0)
        preview_grid.addWidget(self.preview2, 0, 1)
        layout.addLayout(preview_grid)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("동기 Pair 캡처")
        self.capture_button.setProperty("role", "primary")
        self.capture_button.clicked.connect(self._capture_pair)
        self.capture_button.setEnabled(False)
        self.count_label = QLabel("캡처된 stereo pairs: 0쌍")
        capture_row.addWidget(self.capture_button)
        capture_row.addWidget(self.count_label)
        capture_row.addStretch(1)
        layout.addLayout(capture_row)

        self.capture_coach_label = QLabel("동기 pair가 들어오면 실시간 Capture Coach가 표시됩니다.")
        self.capture_coach_label.setWordWrap(True)
        self.capture_coach_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.capture_coach_label)

        done_row = QHBoxLayout()
        self.done_button = QPushButton("완료 (Detection으로 보내기)")
        self.done_button.clicked.connect(self._on_done)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        done_row.addStretch(1)
        done_row.addWidget(cancel)
        done_row.addWidget(self.done_button)
        layout.addLayout(done_row)

        enabled = ROS_LIVE_BACKEND is not None
        self.refresh_button.setEnabled(enabled)
        self.qa_button.setEnabled(True)
        self.subscribe_button.setEnabled(enabled)
        if enabled:
            self._refresh_topics()

    def _backend_text(self) -> str:
        if ROS_LIVE_BACKEND is None:
            return "ROS1/ROS2 실시간 backend를 찾지 못했습니다. ROS 환경을 source한 뒤 다시 실행하세요."
        return f"ROS backend: {ROS_LIVE_BACKEND}. 두 이미지 토픽의 timestamp 차이가 30 ms 이하일 때 pair로 묶습니다."

    def _refresh_topics(self) -> None:
        self.topic1_combo.clear()
        self.topic2_combo.clear()
        try:
            sub = DualLiveTopicSubscriber()
            self._topics = sub.list_image_topics()
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(str(exc))
            return
        for topic in self._topics:
            label = f"{topic.name} ({topic.msg_type})"
            self.topic1_combo.addItem(label, topic)
            self.topic2_combo.addItem(label, topic)
        if self.topic2_combo.count() > 1:
            self.topic2_combo.setCurrentIndex(1)
        self.status_label.setText(f"이미지 토픽 {len(self._topics)}개 발견")

    def _show_qa_report(self) -> None:
        report = self._build_current_qa_report()
        QMessageBox.information(self, "Live Dual Capture QA", report.format())

    def _build_current_qa_report(self):
        report = build_live_dual_capture_qa_report(
            topics=self._topics,
            selected_topic1=self.topic1_combo.currentData(),
            selected_topic2=self.topic2_combo.currentData(),
            output_dir=str(self.output_dir),
            max_sync_delta_ms=30.0,
            subscribed=self._subscriber is not None,
            captured_pair_count=len(self.captured_paths_cam1),
            last_sync_delta_ms=self._last_sync_delta_ms,
            subscribe_elapsed_sec=(
                time.monotonic() - self._subscribe_started_at
                if self._subscribe_started_at is not None else None
            ),
        )
        return report

    def _save_qa_report(self) -> None:
        report = self._build_current_qa_report()
        default = self.output_dir / "live_dual_capture_qa.txt"
        path, _ = QFileDialog.getSaveFileName(self, "QA 리포트 저장", str(default), "Text (*.txt)")
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(report.format(), encoding="utf-8")
        self.status_label.setText(f"QA 리포트 저장 완료: {path}")

    def _toggle_subscribe(self) -> None:
        if self._subscriber is not None:
            self._subscriber.stop()
            self._subscriber = None
            self._subscribe_started_at = None
            self.subscribe_button.setText("구독 시작")
            self.capture_button.setEnabled(False)
            return
        topic1 = self.topic1_combo.currentData()
        topic2 = self.topic2_combo.currentData()
        if topic1 is None or topic2 is None:
            QMessageBox.warning(self, "토픽 없음", "Camera 1/2 이미지 토픽을 선택하세요.")
            return
        if topic1.name == topic2.name:
            QMessageBox.warning(self, "같은 토픽", "Camera 1과 Camera 2는 서로 다른 이미지 토픽이어야 합니다.")
            return
        try:
            self._subscriber = DualLiveTopicSubscriber(max_sync_delta_ms=30.0)
            self._subscriber.start(
                topic1.name,
                topic1.msg_type,
                topic2.name,
                topic2.msg_type,
                lambda pair: self.pair_ready.emit(pair),
                lambda message: self.error_ready.emit(message),
            )
        except Exception as exc:  # noqa: BLE001
            self._subscriber = None
            self._subscribe_started_at = None
            QMessageBox.critical(self, "구독 실패", str(exc))
            return
        self._subscribe_started_at = time.monotonic()
        self.subscribe_button.setText("구독 중지")
        self.status_label.setText("두 토픽을 구독 중입니다. 동기 pair가 들어오면 미리보기가 갱신됩니다.")

    def _on_pair_ready(self, pair: StereoLivePair) -> None:
        self._latest_pair = pair
        self._last_sync_delta_ms = pair.sync_delta_ms
        self.preview1.setPixmap(_pixmap(pair.image_cam1))
        self.preview2.setPixmap(_pixmap(pair.image_cam2))
        self.capture_button.setEnabled(True)
        self.status_label.setText(f"Latest sync delta: {pair.sync_delta_ms:.1f} ms")
        self._render_live_capture_coach(pair)

    def _render_live_capture_coach(self, pair: StereoLivePair) -> None:
        if self._detect_fn is None:
            self.capture_coach_label.setText(
                f"Sync Δt: {pair.sync_delta_ms:.1f} ms · Pattern 설정이 없어 detection coach는 비활성화됨"
            )
            return
        try:
            det1 = self._detect_fn(pair.image_cam1, "live_cam1")
            det2 = self._detect_fn(pair.image_cam2, "live_cam2")
            if not det1.success or not det2.success:
                self.capture_coach_label.setText(
                    "LIVE CAPTURE COACH\n"
                    f"Camera1: {'Detected' if det1.success else det1.failure_reason or 'Detection failed'}\n"
                    f"Camera2: {'Detected' if det2.success else det2.failure_reason or 'Detection failed'}\n"
                    f"Sync Δt: {pair.sync_delta_ms:.1f} ms"
                )
                return
            obs = match_common_charuco_corners(
                det1,
                det2,
                pair_id="live",
                timestamp_cam1=pair.timestamp_cam1,
                timestamp_cam2=pair.timestamp_cam2,
                image_size_cam1=(pair.image_cam1.shape[1], pair.image_cam1.shape[0]),
                image_size_cam2=(pair.image_cam2.shape[1], pair.image_cam2.shape[0]),
            )
            self.preview1.setPixmap(_pixmap(self._overlay_detection(pair.image_cam1, det1, obs.common_ids, obs.quality_status)))
            self.preview2.setPixmap(_pixmap(self._overlay_detection(pair.image_cam2, det2, obs.common_ids, obs.quality_status)))
            warnings = " / ".join(obs.quality_warnings[:3]) if obs.quality_warnings else "캡처하기 좋은 pair입니다."
            self.capture_coach_label.setText(
                "LIVE CAPTURE COACH\n"
                f"Detected corners: cam1={det1.num_corners}, cam2={det2.num_corners}, "
                f"common={obs.common_count}\n"
                f"Quality: {obs.quality_score:.1f} ({obs.quality_status}), "
                f"Sync Δt: {pair.sync_delta_ms:.1f} ms\n"
                f"Hint: {warnings}"
            )
        except Exception as exc:  # noqa: BLE001
            self.capture_coach_label.setText(f"Live coach 계산 실패: {exc}")

    @staticmethod
    def _overlay_detection(img_bgr: np.ndarray, detection, common_ids: np.ndarray, status: str) -> np.ndarray:
        out = img_bgr.copy()
        pts = np.asarray(detection.corners, dtype=float).reshape(-1, 2)
        ids = np.asarray(detection.ids, dtype=int).reshape(-1)
        common = {int(v) for v in np.asarray(common_ids).reshape(-1).tolist()}
        for (x, y), corner_id in zip(pts, ids):
            is_common = int(corner_id) in common
            color = (0, 220, 0) if is_common else (0, 165, 255)
            center = (int(round(x)), int(round(y)))
            cv2.circle(out, center, 4, color, -1)
            cv2.putText(out, str(int(corner_id)), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        status_color = (0, 220, 0) if status == "Good" else (0, 180, 255) if status == "Warning" else (40, 40, 230)
        cv2.rectangle(out, (8, 8), (220, 38), (0, 0, 0), -1)
        cv2.putText(out, f"{status}: common {len(common)}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
        cv2.circle(out, (14, 56), 5, (0, 220, 0), -1)
        cv2.putText(out, "common", (26, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (100, 56), 5, (0, 165, 255), -1)
        cv2.putText(out, "non-common", (112, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        return out

    def _on_error(self, message: str) -> None:
        self.status_label.setText(message)

    def _capture_pair(self) -> None:
        if self._latest_pair is None:
            return
        seq = len(self.captured_paths_cam1) + 1
        stamp_ms = int(round(time.time() * 1000.0))
        stem = f"stereo_pair_{seq:04d}_{stamp_ms}"
        path1 = self.cam1_dir / f"{stem}.png"
        path2 = self.cam2_dir / f"{stem}.png"
        cv2.imwrite(str(path1), self._latest_pair.image_cam1)
        cv2.imwrite(str(path2), self._latest_pair.image_cam2)
        self._write_timestamp_sidecar(path1, self._latest_pair.timestamp_cam1, self._latest_pair.sync_delta_ms)
        self._write_timestamp_sidecar(path2, self._latest_pair.timestamp_cam2, self._latest_pair.sync_delta_ms)
        self.captured_paths_cam1.append(str(path1))
        self.captured_paths_cam2.append(str(path2))
        count = len(self.captured_paths_cam1)
        hint = " · 50쌍 이상 권장" if count < 50 else " · 권장 수량 충족"
        self.count_label.setText(f"캡처된 stereo pairs: {count}쌍{hint}")

    @staticmethod
    def _write_timestamp_sidecar(path: Path, timestamp_sec: float, sync_delta_ms: float) -> None:
        payload = {
            "timestamp_sec": float(timestamp_sec),
            "sync_delta_ms": float(sync_delta_ms),
            "source": "live_dual_capture",
        }
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_done(self) -> None:
        count = len(self.captured_paths_cam1)
        if count < 2:
            QMessageBox.warning(self, "Pair 부족", "Stereo calibration에는 최소 2쌍 이상의 캡처가 필요합니다.")
            return
        if count < 50:
            reply = QMessageBox.question(
                self,
                "50쌍 미만",
                f"현재 {count}쌍입니다. 실사용 품질은 50쌍 이상을 권장합니다.\n그래도 진행할까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._subscriber is not None:
            self._subscriber.stop()
            self._subscriber = None
        super().closeEvent(event)
