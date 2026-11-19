"""Qt 공용 부품: 백그라운드 작업 실행, BGR 이미지 -> QPixmap.

두 기능이 각자 쓰던 것(인트린직 ui/worker.run_worker_in_thread, 화면마다 복사된
_cv_to_qpixmap, CAMERA–LiDAR 의 RoomStep.run_task, Git 업로드 대화상자의 작업 스레드)을
하나로 모았다.
"""

from __future__ import annotations

import traceback
from typing import Callable

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def bgr_to_qpixmap(img_bgr: np.ndarray, max_width: int | None = None) -> QtGui.QPixmap:
    """OpenCV BGR(또는 회색) 이미지 -> QPixmap. `max_width` 보다 넓으면 먼저 줄인다.

    4K 원본을 RGB 로 복사한 뒤 줄이면 프레임마다 수십 MB 를 쓸데없이 복사하므로,
    BGR 상태에서 먼저 줄여 Qt 로 넘기는 데이터를 작게 유지한다.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    if max_width and w > max_width:
        img_bgr = cv2.resize(img_bgr, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


def run_worker_in_thread(worker: QtCore.QObject, parent: QtCore.QObject) -> QtCore.QThread:
    """Signal 을 가진 QObject 작업자(`run` 슬롯, `finished` 시그널)를 새 스레드에 올린다.

    호출부는 `thread = run_worker_in_thread(worker, self)` 뒤 필요한 signal 을 연결하고
    `thread.start()` 만 하면 된다.
    """
    thread = QtCore.QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread


class _TaskWorker(QtCore.QObject):
    finished = QtCore.Signal(object, bool, object)
    progress = QtCore.Signal(str)

    def __init__(self, fn: Callable, with_progress: bool):
        super().__init__()
        self.fn, self.with_progress = fn, with_progress
        self.callbacks = None

    @QtCore.Slot()
    def run(self):
        try:
            result = self.fn(self.progress.emit) if self.with_progress else self.fn()
            self.finished.emit(self, True, result)
        except Exception as e:  # noqa: BLE001 - 실패는 on_fail 로 넘긴다
            self.finished.emit(self, False, e)


class TaskRunner(QtCore.QObject):
    """함수 하나를 UI 스레드 밖에서 돌리고, 결과 콜백은 UI 스레드에서 부른다.

        runner = TaskRunner(widget)
        runner.run(lambda: heavy(), on_done=self.show, busy=self.button)
        runner.run(lambda log: upload(log=log), on_done=..., on_progress=label.setText)  # 진행 메시지

    완료 처리는 이 객체(UI 스레드에 있음)의 슬롯이 받으므로 Qt 가 UI 스레드로 넘겨준다 -
    작업 스레드에서 위젯을 건드리는 일이 없다. 실행 중에는 `busy` 위젯을 잠근다.
    `on_fail` 이 없으면 오류 대화상자를 띄운다.
    """

    def __init__(self, owner: QtWidgets.QWidget):
        super().__init__(owner)
        self._owner = owner
        self._tasks: list[tuple[QtCore.QThread, _TaskWorker]] = []

    def run(self, fn: Callable, on_done: Callable, on_fail: Callable | None = None,
            busy: QtWidgets.QWidget | None = None, on_progress: Callable[[str], None] | None = None) -> None:
        th = QtCore.QThread(self)
        worker = _TaskWorker(fn, with_progress=on_progress is not None)
        worker.moveToThread(th)
        th.started.connect(worker.run)
        if on_progress is not None:
            worker.progress.connect(on_progress)
        if busy is not None:
            busy.setEnabled(False)
        worker.callbacks = (on_done, on_fail, busy, th)
        worker.finished.connect(self._finished)
        self._tasks.append((th, worker))
        th.start()

    def running(self) -> bool:
        return bool(self._tasks)

    def shutdown(self, wait_ms: int = 2000) -> None:
        for th, _ in self._tasks:
            th.quit()
            th.wait(wait_ms)

    @QtCore.Slot(object, bool, object)
    def _finished(self, worker, ok, payload):
        on_done, on_fail, busy, th = worker.callbacks
        if busy is not None:
            busy.setEnabled(True)
        th.quit()
        th.wait()
        self._tasks = [t for t in self._tasks if t[0] is not th]
        if ok:
            on_done(payload)
        elif on_fail is not None:
            on_fail(payload)
        else:
            text = "".join(traceback.format_exception(type(payload), payload, payload.__traceback__))
            QtWidgets.QMessageBox.critical(self._owner, "오류", f"{type(payload).__name__}: {payload}\n\n{text}"[:3000])
