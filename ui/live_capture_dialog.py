"""
camera_calibrator.ui.live_capture_dialog
=============================================

실시간 ROS 토픽 구독 + 캡처 다이얼로그.

설계: 자동으로 계속 찍는 게 아니라 - 라이브 프리뷰를 보면서 사용자가
직접 [캡처] 버튼을 눌러 원하는 자세에서 저장하는 방식을 기본으로 한다.
캘리브레이션 데이터셋 품질은 장수보다 자세 다양성이 중요하므로(설계 문서 7번),
사용자가 보드를 이리저리 움직이며 좋은 자세에서 직접 캡처하는 게 무작정
자동 캡처보다 낫다. 다만 편의를 위해 "N초마다 자동 캡처" 옵션도 둔다
(rosbag 추출의 시간 기반 샘플링과 같은 이유).

스레드 안전성: ROS 콜백은 백그라운드 스레드에서 온다. 여기서는 절대
위젯을 직접 건드리지 않고, Qt Signal.emit()으로만 넘긴다 - PySide6는
스레드가 다른 emit을 자동으로 큐잉된 연결(Queued Connection)로 처리해서
GUI 스레드에서 안전하게 슬롯이 실행되게 해준다.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from calibration.detector import build_detect_fn
from calibration.quality import compute_live_coverage_bars
from calibration.ros_live import ROS_LIVE_BACKEND, LiveTopicSubscriber
from calibration.types import CameraConfig, Frame, ImageInfo, PatternConfig
from ui.live_coverage_bars import LiveCoverageBarsWidget

_PREVIEW_MAX_WIDTH = 480


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


class LiveCaptureDialog(QDialog):
    """반환값: self.captured_paths (list[str]) - 다이얼로그가 accept()된 뒤 읽는다."""

    _frame_ready = Signal(object, float)  # (img_bgr, timestamp_sec) - ROS 콜백 스레드 -> GUI 스레드
    _decode_error = Signal(str)  # 프레임은 왔지만 디코딩 실패 - ROS 콜백 스레드 -> GUI 스레드

    def __init__(
        self,
        output_dir: str,
        pattern_config: PatternConfig | None = None,
        camera_config: CameraConfig | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("실시간 ROS 토픽 구독")
        self.setMinimumWidth(540)

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.captured_paths: list[str] = []

        self._subscriber: LiveTopicSubscriber | None = None
        self._latest_frame: np.ndarray | None = None
        self._last_auto_capture_t: float | None = None

        # 캡처될 때마다 여기 채워지는 (프레임별) 검출 결과 - X/Y/Size/Skew
        # 바를 실시간으로 갱신하는 데 쓰인다. 패턴/카메라 설정이 없으면
        # (예: 다이얼로그를 독립적으로 띄우는 다른 진입점) 바 갱신은 그냥 건너뛴다.
        self._pattern_config = pattern_config
        self._camera_config = camera_config
        self._detect_fn = None
        if pattern_config is not None:
            try:
                self._detect_fn = build_detect_fn(pattern_config)
            except ValueError:
                self._detect_fn = None
        self._detected_frames: list[Frame] = []

        self._frame_ready.connect(self._on_frame_ready)
        self._decode_error.connect(self._on_decode_error)

        layout = QVBoxLayout(self)

        backend_label = QLabel(self._backend_status_text())
        backend_label.setWordWrap(True)
        layout.addWidget(backend_label)

        topic_row = QHBoxLayout()
        self.topic_combo = QComboBox()
        self.refresh_button = QPushButton("토픽 새로고침")
        self.refresh_button.clicked.connect(self._refresh_topics)
        self.subscribe_button = QPushButton("구독 시작")
        self.subscribe_button.clicked.connect(self._toggle_subscribe)
        topic_row.addWidget(self.topic_combo, stretch=1)
        topic_row.addWidget(self.refresh_button)
        topic_row.addWidget(self.subscribe_button)
        layout.addLayout(topic_row)

        self.preview_label = QLabel("구독을 시작하면 여기에 실시간 영상이 표시됩니다.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background:#222; color:#aaa;")
        layout.addWidget(self.preview_label)

        # X/Y/Size/Skew 커버리지 바 - 캡처할 때마다 갱신된다 (아래 _save_frame 참고).
        # pattern_config가 없어 detect_fn을 못 만든 경우엔 위젯을 숨긴다 -
        # 계산할 수 없는 바를 0%로 보여주면 "부족하다"는 잘못된 신호가 된다.
        self.coverage_bars = LiveCoverageBarsWidget()
        self.coverage_bars.setVisible(self._detect_fn is not None)
        if self._detect_fn is None:
            hint = QLabel(
                "⚠ 패턴 설정이 없어 X/Y/Size/Skew 실시간 커버리지 바를 표시할 수 없습니다."
            )
            hint.setStyleSheet("color:#c0392b;")
            layout.addWidget(hint)
        layout.addWidget(self.coverage_bars)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("📸 캡처")
        self.capture_button.clicked.connect(self._manual_capture)
        self.capture_button.setEnabled(False)
        self.auto_capture_check = QCheckBox("자동 캡처, 간격(초):")
        self.auto_interval_spin = QDoubleSpinBox()
        self.auto_interval_spin.setRange(0.2, 30.0)
        self.auto_interval_spin.setValue(1.0)
        self.auto_interval_spin.setSingleStep(0.5)
        capture_row.addWidget(self.capture_button)
        capture_row.addWidget(self.auto_capture_check)
        capture_row.addWidget(self.auto_interval_spin)
        layout.addLayout(capture_row)

        self.count_label = QLabel("캡처된 이미지: 0장")
        layout.addWidget(self.count_label)

        button_row = QHBoxLayout()
        self.done_button = QPushButton("완료 (불러오기)")
        self.done_button.clicked.connect(self._on_done)
        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)

        if ROS_LIVE_BACKEND is not None:
            self._refresh_topics()

    # ------------------------------------------------------------------

    def _backend_status_text(self) -> str:
        if ROS_LIVE_BACKEND == "ros1":
            return "ROS1(rospy) 환경이 감지됐습니다."
        if ROS_LIVE_BACKEND == "ros2":
            return "ROS2(rclpy) 환경이 감지됐습니다."
        return (
            "⚠ ROS를 찾을 수 없습니다. rospy/rclpy는 pip로 설치되지 않으며, "
            "이 컴퓨터에 ROS1 또는 ROS2가 설치되고 환경이 source 되어 있어야 합니다.\n"
            "이미 녹화된 bag 파일만 있다면 [rosbag에서 불러오기]를 대신 사용하세요."
        )

    def _refresh_topics(self) -> None:
        if ROS_LIVE_BACKEND is None:
            return
        try:
            if self._subscriber is None:
                self._subscriber = LiveTopicSubscriber()
            topics = self._subscriber.list_image_topics()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "토픽 조회 실패", str(e))
            return

        self.topic_combo.clear()
        if not topics:
            self.topic_combo.addItem("(이미지 토픽을 찾지 못했습니다)")
            return
        for t in topics:
            self.topic_combo.addItem(f"{t.name}  ({t.msg_type})", userData=t)

    def _toggle_subscribe(self) -> None:
        if self._subscriber is not None and self.subscribe_button.text() == "구독 중지":
            self._stop_subscription()
            return

        topic_obj = self.topic_combo.currentData()
        if topic_obj is None:
            QMessageBox.warning(self, "토픽 없음", "먼저 토픽을 새로고침하고 선택하세요.")
            return

        try:
            if self._subscriber is None:
                self._subscriber = LiveTopicSubscriber()
            self._subscriber.start(
                topic_obj.name, topic_obj.msg_type,
                on_frame=lambda img, t: self._frame_ready.emit(img, t),
                on_error=lambda detail: self._decode_error.emit(detail),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "구독 실패", str(e))
            return

        self.subscribe_button.setText("구독 중지")
        self.capture_button.setEnabled(True)
        self.preview_label.setText("프레임 수신 대기 중...")

    def _stop_subscription(self) -> None:
        if self._subscriber is not None:
            self._subscriber.stop()
        self.subscribe_button.setText("구독 시작")
        self.capture_button.setEnabled(False)

    # ------------------------------------------------------------------
    # 프레임 수신 (GUI 스레드에서 실행됨 - Signal을 통해 안전하게 넘어옴)
    # ------------------------------------------------------------------

    def _on_frame_ready(self, img_bgr: np.ndarray, t_sec: float) -> None:
        self._latest_frame = img_bgr
        self.preview_label.setPixmap(_cv_to_qpixmap(img_bgr))

        if self.auto_capture_check.isChecked():
            now = time.monotonic()
            interval = self.auto_interval_spin.value()
            if self._last_auto_capture_t is None or (now - self._last_auto_capture_t) >= interval:
                self._save_frame(img_bgr)
                self._last_auto_capture_t = now

    def _on_decode_error(self, detail: str) -> None:
        """프레임은 도착했는데 디코딩에 실패하는 상황 - "환경 감지되고 토픽도
        맞는데 계속 대기 중"처럼 보이는 문제의 실제 원인(대부분 지원 안 하는
        인코딩)을 화면에 바로 보여준다. 캡처된 이미지가 하나도 없으면
        프리뷰 영역에 표시하고, 이미 캡처된 게 있으면 상태만 갱신한다.
        """
        self.preview_label.setText(f"⚠ {detail}")
        self.count_label.setText(f"캡처된 이미지: {len(self.captured_paths)}장  |  ⚠ {detail}")

    def _manual_capture(self) -> None:
        if self._latest_frame is not None:
            self._save_frame(self._latest_frame)

    def _save_frame(self, img_bgr: np.ndarray) -> None:
        idx = len(self.captured_paths)
        filename = self._output_dir / f"live_{idx:04d}_{time.time():.3f}.jpg"
        cv2.imwrite(str(filename), img_bgr)
        self.captured_paths.append(str(filename))
        self.count_label.setText(f"캡처된 이미지: {len(self.captured_paths)}장")
        self._update_coverage_bars(img_bgr, image_id=filename.stem)

    def _update_coverage_bars(self, img_bgr: np.ndarray, image_id: str) -> None:
        """방금 캡처한 프레임에 검출을 돌려서 X/Y/Size/Skew 바를 갱신.

        캡처 직후에만 돌리는 이유: 매 라이브 프레임(수십 fps)마다 검출을
        돌리면 GUI 스레드가 버벅일 수 있고, 어차피 이 프로젝트는 "자동으로
        계속 찍지 않고 사용자가 캡처한 프레임만" 데이터셋에 반영한다는
        설계 원칙(파일 상단 docstring)을 따른다 - 실시간 피드백도 같은
        원칙을 지켜야 사용자가 보는 바 = 실제로 저장되는 데이터셋과 일치한다.
        """
        if self._detect_fn is None or self._camera_config is None:
            return
        h, w = img_bgr.shape[:2]
        info = ImageInfo(image_id=image_id, path="", width=w, height=h)
        detection = self._detect_fn(img_bgr, image_id)
        self._detected_frames.append(Frame(image_info=info, detection=detection))

        bars = compute_live_coverage_bars(
            self._detected_frames,
            image_size=(self._camera_config.width, self._camera_config.height),
        )
        self.coverage_bars.set_bars(bars)

    # ------------------------------------------------------------------

    def _on_done(self) -> None:
        self._stop_subscription()
        if not self.captured_paths:
            reply = QMessageBox.question(
                self, "캡처된 이미지 없음",
                "캡처한 이미지가 없습니다. 그래도 닫을까요?",
            )
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def reject(self) -> None:
        self._stop_subscription()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override naming)
        self._stop_subscription()
        super().closeEvent(event)
