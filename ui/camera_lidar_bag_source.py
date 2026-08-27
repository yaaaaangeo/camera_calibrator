"""
Bag topic discovery + selection + timeline preview for the Camera-LiDAR
FAST-Calib workspace.

Opens a bag, discovers ALL image topics and ALL PointCloud2 topics
separately (never auto-picks "the first" of either -- the user always
chooses), and lets the user scrub a timeline to load a Camera+LiDAR
preview pair near an arbitrary timestamp. Emits `scene_loaded` with the
loaded (ImageFrame, PointCloudFrame, camera_topic, lidar_topic) so the
workspace can build a CalibrationScene the same way it already does for
file-based input -- the calibration pipeline downstream doesn't know or
care that this data came from a bag.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from calibration.rosbag_reader import list_pointcloud_topics, read_bag_duration
from camera_lidar.types import ImageFrame, PointCloudFrame
from ui.worker import BagPreviewWorker, BagTopicDiscoveryWorker, run_worker_in_thread

_SLIDER_STEPS = 1000
_SCATTER_SIZE = 220
_SCATTER_POINT_COLOR = (118, 185, 0)  # RGB, matches Theme.ACCENT


def _pointcloud_xy_scatter_pixmap(points: np.ndarray, size: int = _SCATTER_SIZE) -> QPixmap:
    """Lightweight top-down XY scatter preview of a point cloud, rendered
    directly as pixels -- no plotting library dependency."""
    canvas = np.full((size, size, 3), 20, dtype=np.uint8)
    if points.shape[0] > 0:
        xy = points[:, :2]
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        margin = size * 0.08
        scale = (size - 2 * margin) / float(span.max())
        px = ((xy - min_xy) * scale + margin).astype(int)
        px[:, 1] = size - 1 - px[:, 1]  # flip Y so it reads like a top-down map
        px = np.clip(px, 0, size - 1)
        canvas[px[:, 1], px[:, 0]] = _SCATTER_POINT_COLOR
    h, w, ch = canvas.shape
    qimage = QImage(canvas.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class CameraLidarBagSource(QGroupBox):
    scene_loaded = Signal(object, object, str, str)  # ImageFrame, PointCloudFrame, camera_topic, lidar_topic

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Bag Source (open a .bag / .db3 / .mcap and pick topics)", parent)
        self.bag_path: str | None = None
        self.duration_sec: float = 0.0
        self._pending_topic_workers = 0
        self._camera_thread = None
        self._camera_worker = None
        self._lidar_thread = None
        self._lidar_worker = None
        self._preview_thread = None
        self._preview_worker = None

        layout = QVBoxLayout(self)

        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open Bag...")
        self.open_button.clicked.connect(self._on_open_bag)
        open_row.addWidget(self.open_button)
        self.bag_info_label = QLabel("(no bag loaded)")
        open_row.addWidget(self.bag_info_label, stretch=1)
        layout.addLayout(open_row)

        topic_row = QHBoxLayout()
        topic_row.addWidget(QLabel("Camera Topic"))
        self.camera_topic_combo = QComboBox()
        self.camera_topic_combo.currentIndexChanged.connect(self._on_topic_changed)
        topic_row.addWidget(self.camera_topic_combo, stretch=1)
        topic_row.addWidget(QLabel("LiDAR Topic"))
        self.lidar_topic_combo = QComboBox()
        self.lidar_topic_combo.currentIndexChanged.connect(self._on_topic_changed)
        topic_row.addWidget(self.lidar_topic_combo, stretch=1)
        layout.addLayout(topic_row)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, _SLIDER_STEPS)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.timeline_slider)

        preview_row = QHBoxLayout()
        self.image_preview_label = QLabel("(camera preview)")
        self.image_preview_label.setProperty("surface", "image")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(160)
        preview_row.addWidget(self.image_preview_label, stretch=1)
        self.cloud_preview_label = QLabel("(point cloud preview, top-down XY)")
        self.cloud_preview_label.setProperty("surface", "image")
        self.cloud_preview_label.setAlignment(Qt.AlignCenter)
        self.cloud_preview_label.setMinimumHeight(160)
        preview_row.addWidget(self.cloud_preview_label, stretch=1)
        layout.addLayout(preview_row)

        load_row = QHBoxLayout()
        self.timeline_label = QLabel("t = 0.00 s / 0.00 s")
        load_row.addWidget(self.timeline_label)
        self.load_preview_button = QPushButton("Load Preview at t")
        self.load_preview_button.setEnabled(False)
        self.load_preview_button.clicked.connect(self._on_load_preview)
        load_row.addWidget(self.load_preview_button)
        load_row.addStretch(1)
        layout.addLayout(load_row)

    # ------------------------------------------------------------------
    # Open bag -> topic discovery (camera + lidar, independently)
    # ------------------------------------------------------------------

    def _on_open_bag(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Bag 파일 선택", "", "ROS bag (*.bag *.db3 *.mcap);;All files (*)"
        )
        if not path:
            return
        self.bag_path = path
        self.bag_info_label.setText(f"{os.path.basename(path)}  ·  토픽 검색 중...")
        self.camera_topic_combo.clear()
        self.lidar_topic_combo.clear()
        self.timeline_slider.setEnabled(False)
        self.load_preview_button.setEnabled(False)

        try:
            self.duration_sec = read_bag_duration(path)
        except Exception:  # noqa: BLE001 -- surfaced via the topic-discovery workers' own errors below
            self.duration_sec = 0.0

        self._pending_topic_workers = 2

        cam_worker = BagTopicDiscoveryWorker(path, label="카메라 이미지")
        cam_thread = run_worker_in_thread(cam_worker, self)
        cam_worker.topics_ready.connect(self._on_camera_topics_ready)
        cam_worker.error.connect(self._on_error)
        cam_worker.finished.connect(self._on_topic_worker_finished)
        self._camera_thread, self._camera_worker = cam_thread, cam_worker
        cam_thread.start()

        lidar_worker = BagTopicDiscoveryWorker(path, list_fn=list_pointcloud_topics, label="LiDAR PointCloud2")
        lidar_thread = run_worker_in_thread(lidar_worker, self)
        lidar_worker.topics_ready.connect(self._on_lidar_topics_ready)
        lidar_worker.error.connect(self._on_error)
        lidar_worker.finished.connect(self._on_topic_worker_finished)
        self._lidar_thread, self._lidar_worker = lidar_thread, lidar_worker
        lidar_thread.start()

    def _on_camera_topics_ready(self, topics: list, _bag_path: str) -> None:
        for t in topics:
            self.camera_topic_combo.addItem(f"{t.name}  ({t.msg_type.split('/')[-1]}, {t.count})", t.name)

    def _on_lidar_topics_ready(self, topics: list, _bag_path: str) -> None:
        for t in topics:
            self.lidar_topic_combo.addItem(f"{t.name}  ({t.msg_type.split('/')[-1]}, {t.count})", t.name)

    def _on_topic_worker_finished(self) -> None:
        self._pending_topic_workers -= 1
        if self._pending_topic_workers > 0:
            return
        if self.camera_topic_combo.count() == 0:
            QMessageBox.warning(self, "이미지 토픽 없음", "이 bag에서 카메라 이미지 토픽을 찾지 못했습니다.")
        if self.lidar_topic_combo.count() == 0:
            QMessageBox.warning(self, "PointCloud 토픽 없음", "이 bag에서 LiDAR PointCloud2 토픽을 찾지 못했습니다.")
        duration_text = f"{self.duration_sec:.1f} sec" if self.duration_sec else "알 수 없음"
        self.bag_info_label.setText(f"{os.path.basename(self.bag_path)}  ·  {duration_text}")
        ready = self.camera_topic_combo.count() > 0 and self.lidar_topic_combo.count() > 0
        self.timeline_slider.setEnabled(ready)
        self.load_preview_button.setEnabled(ready)

    # ------------------------------------------------------------------
    # Timeline + preview
    # ------------------------------------------------------------------

    def _current_t_sec(self) -> float:
        return (self.timeline_slider.value() / float(_SLIDER_STEPS)) * self.duration_sec

    def _on_slider_moved(self, _value: int) -> None:
        self.timeline_label.setText(f"t = {self._current_t_sec():.2f} s / {self.duration_sec:.2f} s")

    def _on_topic_changed(self) -> None:
        # 요구사항: Topic을 바꾸면 Preview도 즉시 갱신 (사용자가 다시 버튼을
        # 눌러야만 바뀐 토픽이 반영되면, 실수로 옛 카메라 미리보기를 새
        # 토픽 것으로 착각하기 쉽다). 아직 한 번도 Preview를 로드하지
        # 않았다면(버튼 비활성 상태) 아무것도 하지 않는다.
        if self.load_preview_button.isEnabled():
            self._on_load_preview()

    def _on_load_preview(self) -> None:
        if self.bag_path is None:
            return
        camera_topic = self.camera_topic_combo.currentData()
        lidar_topic = self.lidar_topic_combo.currentData()
        if not camera_topic or not lidar_topic:
            return
        worker = BagPreviewWorker(self.bag_path, camera_topic, lidar_topic, self._current_t_sec())
        thread = run_worker_in_thread(worker, self)
        worker.preview_ready.connect(self._on_preview_ready)
        worker.error.connect(self._on_error)
        self._preview_thread, self._preview_worker = thread, worker
        self.load_preview_button.setEnabled(False)
        thread.finished.connect(lambda: self.load_preview_button.setEnabled(True))
        thread.start()

    def _on_preview_ready(self, image_frame: ImageFrame, cloud_frame: PointCloudFrame) -> None:
        rgb = cv2.cvtColor(image_frame.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_preview_label.setPixmap(QPixmap.fromImage(qimage).scaledToHeight(160, Qt.SmoothTransformation))
        pixmap = _pointcloud_xy_scatter_pixmap(cloud_frame.points)
        self.cloud_preview_label.setPixmap(pixmap.scaledToHeight(160, Qt.SmoothTransformation))

        camera_topic = self.camera_topic_combo.currentData()
        lidar_topic = self.lidar_topic_combo.currentData()
        self.scene_loaded.emit(image_frame, cloud_frame, camera_topic, lidar_topic)

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Bag 오류", message)
