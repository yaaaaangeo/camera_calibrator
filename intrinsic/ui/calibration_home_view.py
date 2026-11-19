"""
Calibration workflow selector.

차량부터 고른다 - 두 툴이 같은 차량 프로필(shared.vehicle)을 쓴다:
  * Camera Intrinsic: Camera Name 이 그 차량의 카메라 목록이 되고, 렌즈 등록이 그 차량 카메라에 걸린다.
  * CAMERA–LiDAR: 차량 선택 창 없이 바로 그 차량으로 열린다.
차량에 달지 않은 카메라는 "차량 없음 (카메라 단품)" - Camera Intrinsic 만 쓸 수 있다.
[GitHub에서 가져오기] 는 결과 저장소의 차량 정보(다른 PC 가 올린 것)를 받아온다.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shared import vehicle as shared_vehicle
from ui.theme import Theme


class CalibrationHomeView(QWidget):
    intrinsic_requested = Signal()
    room_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)
        root.addStretch(1)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(18)

        title = QLabel("CALIBRATION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        content_layout.addWidget(title)

        subtitle = QLabel("What do you want to calibrate?")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 16px; color: {Theme.TEXT_SECONDARY};")
        content_layout.addWidget(subtitle)
        content_layout.addWidget(self._vehicle_box())

        cards = QGridLayout()
        cards.setHorizontalSpacing(16)
        cards.setVerticalSpacing(16)
        self.intrinsic_start = None
        cards.addWidget(
            self._card(
                "CAMERA INTRINSIC",
                "Calibrate\n- fx\n- fy\n- cx\n- cy\n- distortion coefficients\n\nOutput\nK, D",
                "START",
                self.intrinsic_requested.emit,
            ),
            0,
            0,
        )
        self.intrinsic_start = self._last_button
        cards.addWidget(
            self._card(
                "CAMERA–LiDAR",
                "카메라·LiDAR 캘리브레이션\n보드를 이용해 LiDAR 와 여러 카메라의\n"
                "외부파라미터를 구합니다\n\n"
                "- 실시간 토픽 또는 bag 캡처\n- 차량 / 세션별 결과 저장\n\nOutput\nR, t",
                "START",
                self.room_requested.emit,
            ),
            0,
            1,
        )
        self.room_start = self._last_button
        content_layout.addLayout(cards)
        root.addWidget(content, alignment=Qt.AlignCenter)
        root.addStretch(1)
        self.reload_vehicles()

    # ------------------------------------------------------------------ 차량

    def _vehicle_box(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 1px solid {Theme.BORDER}; background: {Theme.BG_SECONDARY}; }}")
        lay = QVBoxLayout(frame)
        row = QHBoxLayout()
        label = QLabel("차량")
        label.setStyleSheet("font-size: 15px; font-weight: 700; border: none;")
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.setEditable(True)
        self.vehicle_combo.setInsertPolicy(QComboBox.NoInsert)
        self.vehicle_combo.setMinimumWidth(320)
        self.vehicle_combo.lineEdit().setPlaceholderText("차량을 고르거나 새 이름을 입력하세요 (예: torres03)")
        self.vehicle_combo.setToolTip("결과 폴더·Git 경로·카메라 이름 목록·렌즈가 차량별로 정해집니다.\n"
                                      "차량에 달지 않은 카메라는 '차량 없음 (카메라 단품)'.")
        self.vehicle_combo.currentTextChanged.connect(self._vehicle_text_changed)
        self.import_button = QPushButton("GitHub에서 가져오기")
        self.import_button.setToolTip("결과 저장소(⚙ Git 설정)의 vehicles/ 에서 차량·카메라 이름·렌즈를 받아옵니다.")
        self.import_button.clicked.connect(self._import_from_git)
        row.addWidget(label)
        row.addWidget(self.vehicle_combo, 1)
        row.addWidget(self.import_button)
        lay.addLayout(row)
        self.vehicle_info = QLabel()
        self.vehicle_info.setWordWrap(True)
        self.vehicle_info.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; border: none;")
        lay.addWidget(self.vehicle_info)
        frame.setMinimumWidth(616)
        return frame

    def reload_vehicles(self, select: str | None = None) -> None:
        current = self.vehicle_combo.currentText() if select is None else select
        self.vehicle_combo.blockSignals(True)
        self.vehicle_combo.clear()
        self.vehicle_combo.addItem(shared_vehicle.NO_VEHICLE_LABEL)
        self.vehicle_combo.addItems(shared_vehicle.list_vehicles())
        self.vehicle_combo.setCurrentIndex(-1)
        self.vehicle_combo.setEditText(current)
        self.vehicle_combo.blockSignals(False)
        self._vehicle_text_changed(current)

    def is_vehicle_chosen(self) -> bool:
        return bool(self.vehicle_combo.currentText().strip())

    def vehicle(self) -> str:
        """고른 차량 이름. 차량 없음이면 ""."""
        text = self.vehicle_combo.currentText().strip()
        if not text or text == shared_vehicle.NO_VEHICLE_LABEL:
            return shared_vehicle.NO_VEHICLE
        return shared_vehicle.vehicle_name(text)

    def set_vehicle(self, name: str) -> None:
        """다른 화면(CAMERA–LiDAR 의 차량 바꾸기)에서 바뀐 차량을 홈에도 반영한다."""
        self.reload_vehicles(name or shared_vehicle.NO_VEHICLE_LABEL)

    def _vehicle_text_changed(self, text: str) -> None:
        chosen = bool(text.strip())
        none = text.strip() == shared_vehicle.NO_VEHICLE_LABEL
        if not chosen:
            self.vehicle_info.setText("먼저 차량을 고르세요. 차량에 달지 않은 카메라는 '차량 없음 (카메라 단품)'.")
        else:
            name = self.vehicle()
            shown = "" if none or name == text.strip() else f"폴더 이름으로 '{name}' 을(를) 씁니다. "
            self.vehicle_info.setText(shown + shared_vehicle.summary(name))
        if self.intrinsic_start is not None:
            self.intrinsic_start.setEnabled(chosen)
            self.room_start.setEnabled(chosen and not none)
            self.room_start.setToolTip("CAMERA–LiDAR 는 차량을 골라야 합니다" if none else "")

    def confirm_vehicle(self) -> str:
        """START 직전: 새 차량이면 프로필을 만든다. 고른 차량 이름 ("" = 차량 없음)."""
        name = self.vehicle()
        if name:
            shared_vehicle.ensure(name)
            self.reload_vehicles(name)
        return name

    def _import_from_git(self) -> None:
        from ui.vehicle_import_dialog import VehicleImportDialog

        dialog = VehicleImportDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.imported:
            current = self.vehicle_combo.currentText().strip()
            self.reload_vehicles(current or dialog.imported[0])

    def _card(self, title: str, body: str, button_text: str, callback) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setMinimumWidth(300)
        frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {Theme.BORDER}; background: {Theme.BG_SECONDARY}; }}"
        )
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 18px; font-weight: 700; border: none;")
        layout.addWidget(heading)
        text = QLabel(body)
        text.setAlignment(Qt.AlignLeft)
        text.setWordWrap(True)
        text.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; border: none;")
        layout.addWidget(text, stretch=1)
        button = QPushButton(button_text)
        button.setProperty("role", "primary")
        button.clicked.connect(callback)
        layout.addWidget(button)
        self._last_button = button
        return frame
