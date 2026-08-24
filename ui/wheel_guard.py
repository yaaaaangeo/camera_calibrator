"""마우스 휠로 입력값/탭이 우발적으로 바뀌는 것을 앱 전체에서 차단."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QTabBar, QWidget


class WheelChangeGuard(QObject):
    """숫자 입력, 콤보박스, 탭 바 위의 wheel event를 소비한다.

    값 변경은 클릭/키보드/드롭다운으로만 가능하다. spinbox 내부 line edit처럼
    실제 event target이 자식 위젯인 경우도 부모를 따라 올라가 차단한다.
    """

    _GUARDED_TYPES = (QAbstractSpinBox, QComboBox, QTabBar)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Wheel:
            return super().eventFilter(watched, event)

        widget = watched if isinstance(watched, QWidget) else None
        while widget is not None:
            if isinstance(widget, self._GUARDED_TYPES):
                return True
            widget = widget.parentWidget()
        return super().eventFilter(watched, event)
