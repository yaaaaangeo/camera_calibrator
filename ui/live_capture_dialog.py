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

스레드 안전성/부하 제한: ROS 콜백은 백그라운드 스레드에서 온다. 프레임은
Qt 이벤트 큐로 emit하지 않고 단일 슬롯 LatestFrameBuffer에 최신 한 장만
남긴다. GUI 스레드의 10 FPS QTimer가 슬롯을 소비하므로 30/60 FPS나 4K 토픽도
오래된 프레임이 무한히 쌓이지 않는다. LiveDetectionWorker도 별도의 1슬롯만
사용해 detector가 바쁘면 중간 프레임을 버린다. 저빈도 결과/오류만 Signal로
GUI에 넘긴다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from calibration.detector import build_detect_fn, maximum_pattern_corners
from calibration.latest_frame import LatestFrameBuffer
from calibration.quality import compute_coverage_grid, compute_live_coverage_bars, coverage_warnings
from calibration.ros_live import ROS_LIVE_BACKEND, LiveTopicSubscriber
from calibration.types import (
    CameraConfig,
    Dataset,
    DetectionResult,
    Frame,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from ui.live_coverage_bars import LiveCoverageBarsWidget
from ui.theme import set_tone
from ui.worker import LiveDetectionWorker

logger = logging.getLogger(__name__)

_PREVIEW_MAX_WIDTH = 430
_PREVIEW_FPS = 10

# 이 정도 프레임이 검출되기 전엔 4x4 그리드 자체가 통계적으로 의미가 없다
# (예: 1장만 찍었는데 "이 구역이 부족하다"고 말해봐야 당연한 얘기라 노이즈에
# 가깝다) - 사후 Coverage 탭과 같은 low_threshold를 쓰되, 최소 프레임 수만
# 실시간 코칭 전용으로 추가한다.
_MIN_FRAMES_FOR_LIVE_COACHING = 3
_MAX_WARNINGS_SHOWN = 2
_CORNER_COLOR_BGR = (0, 185, 118)  # Theme.ACCENT (#76B900), OpenCV BGR order
_HULL_COLOR_BGR = (55, 110, 85)


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> QPixmap:
    # 4K 원본을 RGB로 복사한 뒤 축소하면 프레임마다 수십 MB를 불필요하게
    # 복사한다. BGR 상태에서 먼저 축소해 Qt로 넘기는 데이터 자체를 작게 유지한다.
    h, w = img_bgr.shape[:2]
    if w > max_width:
        preview_h = max(1, round(h * max_width / w))
        img_bgr = cv2.resize(img_bgr, (max_width, preview_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    return pixmap


def render_detection_overlay(
    raw_frame: np.ndarray,
    detection: DetectionResult | None,
    maximum_corners: int,
    *,
    max_width: int | None = None,
    show_ids: bool = False,
) -> np.ndarray:
    """Render live corner feedback on a copy, never on the capture frame.

    The UI passes ``max_width`` so a 1080p/4K source is reduced before drawing
    and Qt conversion. Tests and other callers may omit it to retain the input
    dimensions. A failed detection returns an unchanged copy; its explanation
    is shown in the separate status label so the raw preview remains visible.
    """
    if raw_frame.ndim == 2:
        preview = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
    else:
        preview = raw_frame.copy()

    source_h, source_w = raw_frame.shape[:2]
    if max_width is not None and source_w > max_width:
        preview_h = max(1, round(source_h * max_width / source_w))
        preview = cv2.resize(preview, (max_width, preview_h), interpolation=cv2.INTER_AREA)

    if (
        detection is None
        or not detection.success
        or detection.corners is None
        or detection.num_corners <= 0
    ):
        return preview

    preview_h, preview_w = preview.shape[:2]
    scale_x = preview_w / max(source_w, 1)
    scale_y = preview_h / max(source_h, 1)
    source_points = detection.corners.reshape(-1, 2)
    points = np.column_stack(
        (source_points[:, 0] * scale_x, source_points[:, 1] * scale_y)
    ).astype(np.int32)

    if len(points) >= 3:
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        cv2.polylines(preview, [hull], True, _HULL_COLOR_BGR, 1, cv2.LINE_AA)

    ids = detection.ids.reshape(-1) if detection.ids is not None else np.arange(len(points))
    for index, (x, y) in enumerate(points):
        cv2.circle(preview, (int(x), int(y)), 3, _CORNER_COLOR_BGR, -1, cv2.LINE_AA)
        if show_ids and index < len(ids):
            cv2.putText(
                preview,
                str(int(ids[index])),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                _CORNER_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )

    count_text = f"{detection.num_corners} / {maximum_corners} corners"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(count_text, font, font_scale, thickness)
    x = max(8, preview_w - text_w - 14)
    y = max(text_h + 8, preview_h - 12)
    cv2.rectangle(
        preview,
        (x - 6, y - text_h - 6),
        (min(preview_w - 1, x + text_w + 6), min(preview_h - 1, y + baseline + 5)),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        preview, count_text, (x, y), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA
    )
    return preview


class LiveCaptureDialog(QDialog):
    """반환값: self.captured_paths (list[str]) - 다이얼로그가 accept()된 뒤 읽는다."""

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
        self._frame_buffer = LatestFrameBuffer()
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(round(1000 / _PREVIEW_FPS))
        self._preview_timer.timeout.connect(self._consume_latest_frame)
        self._detection_thread: QThread | None = None
        self._detection_worker: LiveDetectionWorker | None = None
        self._live_detection_active = False
        self._latest_live_detection: DetectionResult | None = None

        # 캡처될 때마다 여기 채워지는 (프레임별) 검출 결과 - X/Y/Size/Skew
        # 바를 실시간으로 갱신하는 데 쓰인다. 패턴/카메라 설정이 없으면
        # (예: 다이얼로그를 독립적으로 띄우는 다른 진입점) 바 갱신은 그냥 건너뛴다.
        self._pattern_config = pattern_config
        self._camera_config = camera_config
        self._maximum_corners = maximum_pattern_corners(pattern_config) if pattern_config else 0
        self._detect_fn = None
        if pattern_config is not None:
            try:
                self._detect_fn = build_detect_fn(pattern_config)
            except ValueError:
                self._detect_fn = None
        self._detected_frames: list[Frame] = []

        self._decode_error.connect(self._on_decode_error)

        layout = QVBoxLayout(self)

        backend_label = QLabel(self._backend_status_text())
        backend_label.setWordWrap(True)
        layout.addWidget(backend_label)

        layout.addWidget(self._build_pattern_summary())

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
        # Keep enough room for a useful preview without forcing the status and
        # coverage rows to overlap on 800px-high developer displays.
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setProperty("surface", "image")
        self.preview_label.setProperty("tone", "muted")
        layout.addWidget(self.preview_label)

        self.detection_status_label = QLabel(self._initial_detection_status())
        self.detection_status_label.setWordWrap(True)
        self.detection_status_label.setProperty("tone", "muted")
        layout.addWidget(self.detection_status_label)

        self.stream_status_label = QLabel(
            f"프리뷰 최대 {_PREVIEW_FPS} FPS · 오래된 프레임은 자동 폐기"
        )
        self.stream_status_label.setProperty("tone", "muted")
        layout.addWidget(self.stream_status_label)

        # X/Y/Size/Skew 커버리지 바 - 캡처할 때마다 갱신된다 (아래 _save_frame 참고).
        # pattern_config가 없어 detect_fn을 못 만든 경우엔 위젯을 숨긴다 -
        # 계산할 수 없는 바를 0%로 보여주면 "부족하다"는 잘못된 신호가 된다.
        self.coverage_bars = LiveCoverageBarsWidget()
        self.coverage_bars.setVisible(self._detect_fn is not None)
        if self._detect_fn is None:
            hint = QLabel(
                "⚠ 패턴 설정이 없어 X/Y/Size/Skew 실시간 커버리지 바를 표시할 수 없습니다."
            )
            hint.setProperty("tone", "bad")
            layout.addWidget(hint)
        layout.addWidget(self.coverage_bars)

        # 4x4 구역별("좌측 상단", "우측 하단" 등) 실시간 다양성 코칭 문구.
        # 캘리브레이션을 다 끝내야 사후 Coverage 탭(ui/coverage_view.py)에서나
        # 보이던 걸, 촬영 도중에 바로 보여줘서 애초에 좋은 데이터셋을 찍게
        # 유도한다 - quality.py의 동일한 compute_coverage_grid()/coverage_warnings()를
        # 그대로 재사용하므로 "촬영 중에 본 안내"와 "캘리브레이션 후 리포트"가
        # 같은 기준으로 계산된다 (기준이 서로 다르면 사용자가 혼란스러워짐).
        self.coverage_warning_label = QLabel(
            "이미지를 3장 이상 캡처하면 구역별 다양성 안내가 시작됩니다."
        )
        self.coverage_warning_label.setWordWrap(True)
        self.coverage_warning_label.setProperty("tone", "muted")
        self.coverage_warning_label.setVisible(self._detect_fn is not None)
        layout.addWidget(self.coverage_warning_label)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("📸 캡처")
        self.capture_button.setProperty("role", "primary")
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

    def _build_pattern_summary(self) -> QGroupBox:
        group = QGroupBox("Calibration Pattern")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(4)

        if self._pattern_config is None:
            grid.addWidget(QLabel("Pattern"), 0, 0)
            grid.addWidget(QLabel("설정 없음"), 0, 1)
            return group

        pattern_labels = {
            PatternType.CHARUCO: "ChArUco",
            PatternType.CHESSBOARD: "Chessboard",
            PatternType.APRILGRID: "AprilGrid",
        }
        pattern = self._pattern_config
        # 구버전 프로젝트/외부 호출에서는 dataclass 타입 힌트와 달리
        # PatternType enum 대신 "charuco" 같은 문자열이 들어올 수 있다.
        # dict.get()의 default 인자는 매번 먼저 평가되므로
        # pattern.type.value를 default에 직접 쓰면 문자열 입력에서 즉시 죽는다.
        pattern_type_value = getattr(pattern.type, "value", str(pattern.type))
        rows = [
            ("Pattern", pattern_labels.get(pattern.type, pattern_type_value)),
            ("Squares X", str(pattern.squares_x)),
            ("Squares Y", str(pattern.squares_y)),
            ("Square Size", f"{pattern.square_size * 1000.0:.2f} mm"),
        ]
        if pattern.marker_size is not None:
            rows.append(("Marker Size", f"{pattern.marker_size * 1000.0:.2f} mm"))
        if pattern.dictionary:
            rows.append(("Dictionary", pattern.dictionary))
        for index, (label, value) in enumerate(rows):
            row = index // 3
            pair_column = (index % 3) * 2
            key_label = QLabel(label)
            key_label.setProperty("tone", "muted")
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(key_label, row, pair_column)
            grid.addWidget(value_label, row, pair_column + 1)
            grid.setColumnStretch(pair_column + 1, 1)
        return group

    def _initial_detection_status(self) -> str:
        if self._pattern_config is None:
            return "Live Detection: 패턴 설정 없음"
        return f"Live Detection 대기 중 · 0 / {self._maximum_corners} corners"

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
            self._frame_buffer.clear(reset_stats=True)
            self._latest_live_detection = None
            self._start_detection_worker()
            self._subscriber.start(
                topic_obj.name, topic_obj.msg_type,
                # ROS 스레드에서는 Qt 이벤트를 프레임마다 쌓지 않는다. 단일 슬롯을
                # 최신 프레임으로 교체하고 GUI의 고정 주기 타이머가 가져간다.
                on_frame=self._frame_buffer.put,
                on_error=lambda detail: self._decode_error.emit(detail),
            )
        except Exception as e:  # noqa: BLE001
            self._stop_detection_worker()
            QMessageBox.critical(self, "구독 실패", str(e))
            return

        self.subscribe_button.setText("구독 중지")
        self.capture_button.setEnabled(True)
        self.preview_label.setText("프레임 수신 대기 중...")
        self.detection_status_label.setText(self._initial_detection_status())
        set_tone(self.detection_status_label, "muted")
        self._last_auto_capture_t = None
        self._preview_timer.start()

    def _stop_subscription(self) -> None:
        self._preview_timer.stop()
        if self._subscriber is not None:
            self._subscriber.stop()
        self._frame_buffer.clear()
        self._latest_frame = None
        self._latest_live_detection = None
        self._stop_detection_worker()
        self.subscribe_button.setText("구독 시작")
        self.capture_button.setEnabled(False)

    def _start_detection_worker(self) -> None:
        if self._pattern_config is None or self._detect_fn is None:
            return
        self._stop_detection_worker()

        worker = LiveDetectionWorker(self._pattern_config)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.initialize)
        worker.result_ready.connect(self._on_live_detection_ready)
        worker.error.connect(self._on_live_detection_error)
        thread.finished.connect(worker.deleteLater)

        self._detection_worker = worker
        self._detection_thread = thread
        self._live_detection_active = True
        thread.start()

    def _stop_detection_worker(self) -> None:
        worker = self._detection_worker
        thread = self._detection_thread
        self._live_detection_active = False
        if worker is not None:
            worker.request_stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            # Normal detection is well below this timeout. Waiting only during
            # explicit stop/close prevents QObject/QThread cross-thread teardown
            # crashes while never blocking the live preview path itself.
            if not thread.wait(3000):
                logger.warning("Live detection thread 종료를 기다리는 중입니다.")
                thread.wait()
        self._detection_worker = None
        self._detection_thread = None

    # ------------------------------------------------------------------
    # 프레임 표시 (GUI 타이머에서 실행 - ROS 콜백은 단일 슬롯만 갱신)
    # ------------------------------------------------------------------

    def _consume_latest_frame(self) -> None:
        pending = self._frame_buffer.take()
        if pending is None:
            return
        img_bgr, t_sec = pending
        # This is always the undecorated ROS frame. Manual/automatic capture
        # reads this reference; overlay rendering below only receives a copy.
        self._latest_frame = img_bgr
        if self._detection_worker is not None:
            self._detection_worker.submit_frame(img_bgr, t_sec)
        self._render_live_preview()

        stats = self._frame_buffer.stats()
        status = (
            f"수신 {stats.received} · 표시 {stats.delivered} · "
            f"오래된 프레임 폐기 {stats.replaced} · 프리뷰 최대 {_PREVIEW_FPS} FPS"
        )
        if self._detection_worker is not None:
            detection_stats = self._detection_worker.buffer_stats()
            status += (
                f" · 검출 처리 {detection_stats.delivered} · "
                f"검출 대기 교체 {detection_stats.replaced}"
            )
        self.stream_status_label.setText(status)

        if self.auto_capture_check.isChecked():
            now = time.monotonic()
            interval = self.auto_interval_spin.value()
            if self._last_auto_capture_t is None or (now - self._last_auto_capture_t) >= interval:
                self._save_frame(img_bgr)
                self._last_auto_capture_t = now

    def _render_live_preview(self) -> None:
        if self._latest_frame is None:
            return
        preview = render_detection_overlay(
            self._latest_frame,
            self._latest_live_detection,
            self._maximum_corners,
            max_width=_PREVIEW_MAX_WIDTH,
        )
        self.preview_label.setPixmap(_cv_to_qpixmap(preview))

    def _on_live_detection_ready(self, detection: DetectionResult) -> None:
        if not self._live_detection_active:
            return
        self._latest_live_detection = detection
        if detection.success:
            details = [
                "Detected",
                f"{detection.num_corners} / {self._maximum_corners} corners",
            ]
            if detection.board_area_ratio is not None:
                details.append(f"Coverage: {detection.board_area_ratio * 100.0:.1f}%")
            if detection.min_edge_margin_px is not None:
                details.append(f"Edge margin: {detection.min_edge_margin_px:.0f} px")
            set_tone(self.detection_status_label, "good")
            self.detection_status_label.setText("  ·  ".join(details))
        else:
            reason = detection.failure_reason or "pattern not detected"
            set_tone(self.detection_status_label, "bad")
            self.detection_status_label.setText(
                f"⚠ Not detected · 0 / {self._maximum_corners} corners · {reason}"
            )
        # Show a completed detection immediately; subsequent 10 FPS raw frames
        # continue using the most recent bounded-latency result.
        self._render_live_preview()

    def _on_live_detection_error(self, detail: str) -> None:
        if not self._live_detection_active:
            return
        self._latest_live_detection = None
        set_tone(self.detection_status_label, "bad")
        self.detection_status_label.setText(f"⚠ {detail}")
        self._render_live_preview()

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
        """방금 캡처한 프레임에 검출을 돌려서 X/Y/Size/Skew 바 + 구역별 다양성
        코칭 문구를 함께 갱신.

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
        self._update_coverage_coaching()

    def _update_coverage_coaching(self) -> None:
        """4x4 구역 기준 "이 구역이 아직 부족합니다" 코칭 문구 갱신.

        사후 Coverage 탭(ui/coverage_view.py)과 완전히 동일한 계산 함수
        (quality.compute_coverage_grid / coverage_warnings)를 재사용한다 -
        새 로직을 따로 만들면 실시간 안내와 사후 리포트의 기준이 미묘하게
        어긋날 위험이 있다.
        """
        num_detected = sum(
            1 for f in self._detected_frames if f.detection and f.detection.success
        )
        if num_detected < _MIN_FRAMES_FOR_LIVE_COACHING:
            remaining = _MIN_FRAMES_FOR_LIVE_COACHING - num_detected
            set_tone(self.coverage_warning_label, "muted")
            self.coverage_warning_label.setText(
                f"이미지를 {remaining}장 더 캡처하면 구역별 다양성 안내가 시작됩니다."
            )
            return

        temp_dataset = Dataset(frames=self._detected_frames)
        cells = compute_coverage_grid(temp_dataset, self._camera_config)
        warnings = coverage_warnings(cells)

        if not warnings:
            set_tone(self.coverage_warning_label, "good")
            self.coverage_warning_label.setText("✓ 지금까지 촬영한 자세가 구역별로 고르게 분포되어 있습니다.")
            return

        shown = warnings[:_MAX_WARNINGS_SHOWN]
        text = "⚠ " + "  ".join(shown)
        if len(warnings) > _MAX_WARNINGS_SHOWN:
            text += f"  (+{len(warnings) - _MAX_WARNINGS_SHOWN}개 구역 더)"
        set_tone(self.coverage_warning_label, "bad")
        self.coverage_warning_label.setText(text)

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
