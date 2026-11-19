"""
ui.vehicle_import_dialog
========================

[GitHub에서 가져오기] - 결과 저장소의 ``vehicles/<차량>/`` (차량 프로필 + 렌즈 프리셋)을
읽어 이 PC 에 반영한다. 다른 PC 에서 Git 업로드한 차량·카메라 이름·렌즈를 그대로 쓰기 위함.

저장소 주소는 Git 업로드와 같은 설정(⚙ Git 설정)을 쓴다. 받아오기는 백그라운드에서 돈다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from export import git_upload as gu
from shared.qt import TaskRunner
from ui.git_upload_dialog import GitSettingsDialog, _tone


class VehicleImportDialog(QDialog):
    """가져온 차량 이름은 `imported` 에 남는다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GitHub에서 차량 정보 가져오기")
        self.resize(780, 460)
        self.config = gu.load_config()
        self.vehicles: list[gu.RemoteVehicle] = []
        self.imported: list[str] = []
        self._runner = TaskRunner(self)

        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        self.settings_button = QPushButton("⚙ Git 설정…")
        self.settings_button.clicked.connect(self._open_settings)
        self.refresh_button = QPushButton("다시 받기")
        self.refresh_button.clicked.connect(self._fetch)
        top = QHBoxLayout()
        top.addWidget(self.target_label, 1)
        top.addWidget(self.refresh_button)
        top.addWidget(self.settings_button)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["차량", "이 PC 와 비교", "카메라", "렌즈"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        note = QLabel("체크한 차량의 카메라 이름·토픽·렌즈 프리셋을 이 PC 에 반영합니다. 저장소 값이 우선이고, "
                      "저장소에 없는 이 PC 의 카메라는 남깁니다 (바꾸기 전 파일은 vehicle.yaml.bak).")
        note.setWordWrap(True)
        self.import_button = QPushButton("가져오기")
        self.import_button.setProperty("role", "primary")
        self.import_button.clicked.connect(self._import)
        self.close_button = QPushButton("닫기")
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(note)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)
        self._show_target()
        self._fetch()

    def _show_target(self) -> None:
        c = self.config
        self.target_label.setText(f"<b>저장소</b> {c.remote or '<i>미설정</i>'} · <b>브랜치</b> {c.branch}")

    def _open_settings(self) -> None:
        dialog = GitSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.config = dialog.config
            self._show_target()
            self._fetch()

    def _lock(self, locked: bool) -> None:
        for w in (self.refresh_button, self.settings_button, self.import_button, self.close_button):
            w.setEnabled(not locked)

    def _fetch(self) -> None:
        if not self.config.remote:
            _tone(self.status_label, "결과 저장소 주소가 없습니다. ⚙ Git 설정에서 넣으세요.", "bad")
            self.import_button.setEnabled(False)
            return
        if self._runner.running():
            return
        self._lock(True)
        config = self.config
        self._runner.run(lambda log: gu.fetch_vehicles(config, log), on_done=self._fetched,
                         on_fail=self._failed, on_progress=lambda m: _tone(self.status_label, m))

    def _fetched(self, vehicles: list) -> None:
        self._lock(False)
        self.vehicles = vehicles
        self.table.setRowCount(0)
        for v in vehicles:
            r = self.table.rowCount()
            self.table.insertRow(r)
            cams = [str((e or {}).get("name")) for e in (v.profile.get("cameras") or {}).values()]
            cams += [str(n) for n in (v.profile.get("lenses") or {}) if str(n) not in cams]
            first = QTableWidgetItem(v.name)
            first.setFlags(first.flags() | Qt.ItemIsUserCheckable)
            first.setCheckState(Qt.Unchecked if v.status == "같음" else Qt.Checked)
            self.table.setItem(r, 0, first)
            for k, text in enumerate([v.status, ", ".join(cams) or "—", ", ".join(str(e.get("name")) for e in v.lenses) or "—"], 1):
                self.table.setItem(r, k, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.import_button.setEnabled(bool(vehicles))
        _tone(self.status_label, f"저장소에 차량 {len(vehicles)}대" if vehicles else
              "저장소에 차량 정보가 아직 없습니다 (vehicles/ 폴더). 차량을 고르고 Git 업로드하면 생깁니다.",
              "good" if vehicles else "")

    def _failed(self, exc: Exception) -> None:
        self._lock(False)
        self.import_button.setEnabled(False)
        _tone(self.status_label, str(exc) if isinstance(exc, gu.GitUploadError) else f"{type(exc).__name__}: {exc}", "bad")

    def _import(self) -> None:
        chosen = [v for r, v in enumerate(self.vehicles) if self.table.item(r, 0).checkState() == Qt.Checked]
        if not chosen:
            _tone(self.status_label, "가져올 차량을 체크하세요.", "bad")
            return
        done = []
        try:
            for v in chosen:
                done += gu.import_vehicle(v)
                self.imported.append(v.name)
        except Exception as exc:  # noqa: BLE001 - 사용자에게 그대로 보여준다
            _tone(self.status_label, f"가져오기 실패: {exc}", "bad")
            return
        _tone(self.status_label, "반영했습니다: " + ", ".join(done), "good")
        self.accept()

    def reject(self) -> None:
        if self._runner.running():
            return
        super().reject()
