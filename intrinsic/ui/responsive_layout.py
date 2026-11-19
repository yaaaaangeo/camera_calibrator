"""
camera_calibrator.ui.responsive_layout
=========================================

Small layout helpers shared by the main window: a row that stacks its children
when space is scarce, and a QFormLayout configuration that wraps long rows.
(Moved out of the removed Windshield UI package unchanged.)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QBoxLayout, QFormLayout, QSizePolicy, QWidget


class ResponsiveRow(QWidget):
    """A layout-owned row which stacks its children when space is scarce.

    The widget never computes child geometry.  It only changes the direction of
    a ``QBoxLayout`` and leaves sizing/positioning entirely to Qt's layout
    engine, which also makes it safe under OS-level DPI scaling.
    """

    def __init__(self, breakpoint: int = 760, parent: QWidget | None = None):
        super().__init__(parent)
        self.breakpoint = breakpoint
        self.box_layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.box_layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
        self.box_layout.addWidget(widget, stretch)

    def addStretch(self, stretch: int = 1) -> None:
        self.box_layout.addStretch(stretch)

    def resizeEvent(self, event) -> None:
        direction = (
            QBoxLayout.TopToBottom
            if event.size().width() < self.breakpoint
            else QBoxLayout.LeftToRight
        )
        if self.box_layout.direction() != direction:
            self.box_layout.setDirection(direction)
            self.updateGeometry()
        super().resizeEvent(event)


def configure_form_layout(form: QFormLayout) -> None:
    """Allow labels/fields to wrap instead of competing for horizontal space."""
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
    for index in range(form.count()):
        widget = form.itemAt(index).widget()
        if widget is not None and hasattr(widget, "setWordWrap"):
            widget.setWordWrap(True)
