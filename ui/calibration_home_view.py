"""
Calibration workflow selector.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.theme import Theme


class CalibrationHomeView(QWidget):
    intrinsic_requested = Signal()
    stereo_requested = Signal()

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

        title = QLabel("CAMERA CALIBRATION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        content_layout.addWidget(title)

        subtitle = QLabel("What do you want to calibrate?")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 16px; color: {Theme.TEXT_SECONDARY};")
        content_layout.addWidget(subtitle)

        cards = QGridLayout()
        cards.setHorizontalSpacing(16)
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
        cards.addWidget(
            self._card(
                "CAMERA ↔ CAMERA",
                "Calibrate\n- Relative Rotation\n- Relative Translation\n\nOutput\nR, T\n4x4 Transform\nRectification",
                "START",
                self.stereo_requested.emit,
            ),
            0,
            1,
        )
        content_layout.addLayout(cards)
        root.addWidget(content, alignment=Qt.AlignCenter)
        root.addStretch(1)

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
        return frame
