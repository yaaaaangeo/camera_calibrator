"""
camera_calibrator.ui.library_view
=====================================

Library 탭 - 카메라(센서)별로 지금까지 계산한 캘리브레이션 결과를 목록으로
보여준다. 계산은 하지 않는다 - calibration/library.py가 이미 정리해 둔
summary.json/project.ccproj를 읽어 표시만 한다 ("UI는 계산하지 않는다" 원칙,
ui/live_coverage_bars.py와 같은 이유).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.library import (
    LibraryRunSummary,
    delete_camera,
    delete_run,
    list_cameras,
    list_runs,
    load_run_project,
    update_run_note,
)
from calibration.models.common import undistort_image
from calibration.types import CameraModelType

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [
    CameraModelType.PINHOLE,
    CameraModelType.BROWN_CONRADY,
    CameraModelType.EXTENDED_PINHOLE,
    CameraModelType.FISHEYE,
]
_PREVIEW_MAX_WIDTH = 420


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


class LibraryView(QWidget):
    """반환값 없음 - 순수 조회 화면. back_requested만 밖으로 알린다."""

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dirty = True
        self._current_runs: list[LibraryRunSummary] = []
        self._current_run: LibraryRunSummary | None = None

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        back = QPushButton("← Calibration Home")
        back.clicked.connect(self.back_requested.emit)
        top_row.addWidget(back)
        top_row.addStretch(1)
        refresh = QPushButton("새로고침")
        refresh.clicked.connect(self.refresh)
        top_row.addWidget(refresh)
        layout.addLayout(top_row)

        title = QLabel("LIBRARY")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        camera_group = QGroupBox("카메라")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_list = QListWidget()
        self.camera_list.currentTextChanged.connect(self._on_camera_selected)
        self.camera_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.camera_list.customContextMenuRequested.connect(self._show_camera_context_menu)
        camera_layout.addWidget(self.camera_list)
        splitter.addWidget(camera_group)

        run_group = QGroupBox("계산 기록 (최신순, 우클릭으로 삭제)")
        run_layout = QVBoxLayout(run_group)
        self.run_list = QListWidget()
        self.run_list.currentRowChanged.connect(self._on_run_selected)
        self.run_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.run_list.customContextMenuRequested.connect(self._show_run_context_menu)
        run_layout.addWidget(self.run_list)
        splitter.addWidget(run_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        summary_row = QHBoxLayout()

        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Model", "Success", "RMS (px)", "Hold-out RMS (px)", "Dist. coeffs"]
        )
        # Model/RMS/Hold-out RMS는 남는 폭을 나눠 가지며 넓게(Stretch),
        # Success/Distortion coeffs는 내용(숫자 하나, "성공")만큼만 차지하게
        # 한다 - 그냥 마지막 컬럼만 늘리면 값이 작은 Distortion coeffs가
        # 쓸데없이 커지고 정작 자주 보는 Model/RMS 칸은 좁게 남는다.
        header = self.summary_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)              # Model
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)     # Success
        header.setSectionResizeMode(2, QHeaderView.Stretch)              # RMS
        header.setSectionResizeMode(3, QHeaderView.Stretch)              # Hold-out RMS
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)     # Distortion coeffs
        header.setStretchLastSection(False)
        self.summary_table.verticalHeader().setVisible(False)  # 1,2,3... 행 번호는 필요 없음
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        summary_row.addWidget(self.summary_table, stretch=2)

        note_group = QGroupBox("메모")
        note_layout = QVBoxLayout(note_group)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("이 기록이 어떤 것에 대한 것인지 짧게 남겨두세요")
        self.note_edit.setEnabled(False)
        self.note_edit.returnPressed.connect(self._save_note)
        note_layout.addWidget(self.note_edit)
        self.note_save_button = QPushButton("저장")
        self.note_save_button.setEnabled(False)
        self.note_save_button.clicked.connect(self._save_note)
        note_layout.addWidget(self.note_save_button)
        note_layout.addStretch(1)
        summary_row.addWidget(note_group, stretch=1)

        layout.addLayout(summary_row)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        preview_row.addWidget(self.model_combo)
        preview_button = QPushButton("왜곡 보정 전/후 미리보기")
        preview_button.clicked.connect(self._update_preview)
        preview_row.addWidget(preview_button)
        preview_row.addStretch(1)
        layout.addLayout(preview_row)

        images_row = QHBoxLayout()
        original_group = QGroupBox("원본")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel("계산 기록을 선택하고 [미리보기]를 누르세요.")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setProperty("surface", "image")
        original_layout.addWidget(self.original_label)
        images_row.addWidget(original_group)

        undistorted_group = QGroupBox("왜곡 보정 후")
        undistorted_layout = QVBoxLayout(undistorted_group)
        self.undistorted_label = QLabel("-")
        self.undistorted_label.setAlignment(Qt.AlignCenter)
        self.undistorted_label.setProperty("surface", "image")
        undistorted_layout.addWidget(self.undistorted_label)
        images_row.addWidget(undistorted_group)
        layout.addLayout(images_row)

        self.status_label = QLabel(
            "아직 저장된 계산 결과가 없습니다. 캘리브레이션을 한 번 실행하면 여기 쌓입니다."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        """새 실행이 Library에 저장됐음을 표시 - 다음에 이 탭이 보일 때 다시 스캔한다."""
        self._dirty = True

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override naming)
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        self._dirty = False
        current = self.camera_list.currentItem().text() if self.camera_list.currentItem() else None
        self.camera_list.clear()
        cameras = list_cameras()
        if not cameras:
            self.status_label.setText(
                "아직 저장된 계산 결과가 없습니다. 캘리브레이션을 한 번 실행하면 여기 쌓입니다."
            )
            self.run_list.clear()
            self.summary_table.setRowCount(0)
            return
        self.camera_list.addItems(cameras)
        if current and current in cameras:
            self.camera_list.setCurrentRow(cameras.index(current))
        else:
            self.camera_list.setCurrentRow(0)

    def _show_camera_context_menu(self, pos) -> None:
        item = self.camera_list.itemAt(pos)
        if item is None:
            return
        sensor_name = item.text()
        menu = QMenu(self)
        delete_action = menu.addAction("이 카메라의 모든 기록 삭제")
        chosen = menu.exec(self.camera_list.mapToGlobal(pos))
        if chosen != delete_action:
            return
        reply = QMessageBox.question(
            self, "카메라 삭제",
            f"'{sensor_name}' 카메라의 모든 계산 기록을 삭제할까요? 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if delete_camera(sensor_name):
            self.status_label.setText(f"'{sensor_name}' 카메라를 삭제했습니다.")
            self.refresh()
        else:
            self.status_label.setText(f"'{sensor_name}' 카메라를 삭제하지 못했습니다.")

    def _show_run_context_menu(self, pos) -> None:
        item = self.run_list.itemAt(pos)
        if item is None:
            return
        row = self.run_list.row(item)
        if row < 0 or row >= len(self._current_runs):
            return
        run = self._current_runs[row]
        menu = QMenu(self)
        delete_action = menu.addAction("삭제")
        chosen = menu.exec(self.run_list.mapToGlobal(pos))
        if chosen != delete_action:
            return
        reply = QMessageBox.question(
            self, "계산 기록 삭제",
            f"이 계산 기록을 삭제할까요? 되돌릴 수 없습니다.\n\n{run.created_at}  ·  {run.num_images}장",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if delete_run(run.run_dir):
            self.status_label.setText("계산 기록을 삭제했습니다.")
            current_camera = self.camera_list.currentItem()
            if current_camera is not None:
                self._on_camera_selected(current_camera.text())
        else:
            self.status_label.setText("계산 기록을 삭제하지 못했습니다.")

    def _on_camera_selected(self, sensor_name: str) -> None:
        self.run_list.clear()
        self.summary_table.setRowCount(0)
        self._current_runs = []
        self._current_run = None
        if not sensor_name:
            return
        self._current_runs = list_runs(sensor_name)
        for run in self._current_runs:
            success_models = [
                _MODEL_LABELS.get(CameraModelType(m), m) for m, s in run.models.items() if s.success
            ]
            note_suffix = f"  ·  📝 {run.note}" if run.note else ""
            self.run_list.addItem(
                f"{run.created_at}  ·  {run.num_images}장  ·  "
                f"성공: {', '.join(success_models) or '없음'}{note_suffix}"
            )
        if self._current_runs:
            self.run_list.setCurrentRow(0)

    def _on_run_selected(self, row: int) -> None:
        self.summary_table.setRowCount(0)
        self.original_label.setText("계산 기록을 선택하고 [미리보기]를 누르세요.")
        self.original_label.setPixmap(QPixmap())
        self.undistorted_label.setText("-")
        self.undistorted_label.setPixmap(QPixmap())
        self.note_edit.clear()
        self.note_edit.setEnabled(row >= 0)
        self.note_save_button.setEnabled(row >= 0)
        if row < 0 or row >= len(self._current_runs):
            self._current_run = None
            return
        run = self._current_runs[row]
        self._current_run = run
        self.note_edit.setText(run.note)
        self.summary_table.setRowCount(len(_MODEL_ORDER))
        for r, model in enumerate(_MODEL_ORDER):
            s = run.models.get(model.value)
            self.summary_table.setItem(r, 0, QTableWidgetItem(_MODEL_LABELS[model]))
            if s is None:
                for c in range(1, 5):
                    self.summary_table.setItem(r, c, QTableWidgetItem("N/A"))
                continue
            self.summary_table.setItem(
                r, 1, QTableWidgetItem("성공" if s.success else f"실패: {s.error_message or ''}")
            )
            self.summary_table.setItem(
                r, 2, QTableWidgetItem(f"{s.rms_error:.4f}" if s.rms_error is not None else "N/A")
            )
            self.summary_table.setItem(
                r, 3, QTableWidgetItem(f"{s.test_rms:.4f}" if s.test_rms is not None else "N/A")
            )
            self.summary_table.setItem(
                r, 4,
                QTableWidgetItem(str(s.distortion_count) if s.distortion_count is not None else "N/A"),
            )

    def _save_note(self) -> None:
        if self._current_run is None:
            return
        note = self.note_edit.text().strip()
        updated = update_run_note(self._current_run.run_dir, note)
        if updated is None:
            self.status_label.setText("메모를 저장하지 못했습니다 (이 기록의 파일을 찾을 수 없습니다).")
            return
        self._current_run.note = note
        row = self.run_list.currentRow()
        if 0 <= row < self.run_list.count():
            success_models = [
                _MODEL_LABELS.get(CameraModelType(m), m)
                for m, s in self._current_run.models.items() if s.success
            ]
            note_suffix = f"  ·  📝 {note}" if note else ""
            self.run_list.item(row).setText(
                f"{self._current_run.created_at}  ·  {self._current_run.num_images}장  ·  "
                f"성공: {', '.join(success_models) or '없음'}{note_suffix}"
            )
        self.status_label.setText("메모를 저장했습니다.")

    def _update_preview(self) -> None:
        if self._current_run is None:
            self.status_label.setText("먼저 목록에서 계산 기록을 선택하세요.")
            return
        # PySide6는 QComboBox userData로 넣은 str-Enum(CameraModelType)을 콤보
        # 내부적으로 평범한 str로 되돌려줄 때가 있다 - .value 접근 전에 항상
        # 다시 enum으로 정규화한다.
        model = CameraModelType(self.model_combo.currentData())
        model_summary = self._current_run.models.get(model.value)
        if model_summary is None or not model_summary.success:
            self.status_label.setText(f"{_MODEL_LABELS.get(model, model)} 모델은 이 기록에서 성공하지 않았습니다.")
            return
        if not self._current_run.sample_image:
            self.status_label.setText("이 기록에는 미리보기용 샘플 이미지가 없습니다.")
            return

        try:
            project, missing = load_run_project(self._current_run.run_dir)
        except Exception as e:  # noqa: BLE001
            self.status_label.setText(f"기록을 불러오지 못했습니다: {e}")
            return

        result = project.calibration_results.get(model)
        if result is None or not result.success:
            self.status_label.setText(f"{_MODEL_LABELS.get(model, model)} 모델 결과를 이 기록에서 찾지 못했습니다.")
            return

        image_path = Path(self._current_run.run_dir) / self._current_run.sample_image
        img = cv2.imread(str(image_path))
        if img is None:
            self.status_label.setText(f"샘플 이미지를 읽을 수 없습니다: {image_path}")
            return

        try:
            undistorted = undistort_image(img, result, project.camera_config)
        except ValueError as e:
            self.status_label.setText(str(e))
            return

        self.original_label.setPixmap(_cv_to_qpixmap(img))
        self.undistorted_label.setPixmap(_cv_to_qpixmap(undistorted))
        note = f" (참고: 원본 이미지 중 {len(missing)}장을 찾을 수 없습니다)" if missing else ""
        self.status_label.setText(
            f"{_MODEL_LABELS.get(model, model)} 모델 기준 왜곡 보정 결과입니다 "
            f"(기록: {self._current_run.created_at}){note}"
        )
