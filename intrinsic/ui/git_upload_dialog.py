"""
ui.git_upload_dialog
====================

Git 업로드 화면 두 개 - Camera Intrinsic 과 CAMERA–LiDAR 가 같이 쓴다.

* ``GitSettingsDialog`` (⚙ Git 설정): 저장소 주소·브랜치·작성자, 저장소 안 경로 규칙,
  올릴 파일 종류. ``~/.config/calibration-tool/git_upload.json`` 에 저장한다.
* ``GitUploadDialog`` (⬆ Git 업로드): 지금 설정으로 올라갈 파일 목록과 커밋 메시지를
  보여주고, 사람이 [업로드] 를 눌러야 올린다. [설정…] 으로 바꾸면 목록을 다시 만든다.

git 작업은 백그라운드 스레드에서 돌려 화면이 멈추지 않게 한다.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from export import git_upload as gu
from shared.qt import TaskRunner
from ui.theme import Theme

PlanBuilder = Callable[[gu.GitUploadConfig], "tuple[gu.UploadPlan, str]"]


def _size(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    n /= 1024
    if n < 1024:
        return f"{n:.1f} KB"
    return f"{n / 1024:.1f} MB"


def _tone(label: QLabel, text: str, tone: str = "") -> None:
    color = {"bad": Theme.BAD, "good": Theme.GOOD}.get(tone)
    text = text.replace("\n", "<br>")
    label.setText(f"<span style='color:{color}'>{text}</span>" if color else text)


# ---------------------------------------------------------------------- 설정


class GitSettingsDialog(QDialog):
    """⚙ Git 설정. [저장] 하면 설정 파일에 쓰고 `config` 에 새 값을 남긴다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Git 업로드 설정")
        self.resize(760, 640)
        config = gu.load_config()
        self.config = config
        self._runner = TaskRunner(self)

        self.remote_edit = QLineEdit(config.remote)
        self.remote_edit.setPlaceholderText("결과 전용 저장소 - 예: git@github.com:team/calibration-results.git")
        self.check_button = QPushButton("연결 확인")
        self.check_button.clicked.connect(self._check)
        remote_row = QHBoxLayout()
        remote_row.addWidget(self.remote_edit, 1)
        remote_row.addWidget(self.check_button)
        self.branch_edit = QLineEdit(config.branch)
        git_name, git_email = gu.git_identity()
        self.author_name_edit = QLineEdit(config.author_name or git_name)
        self.author_name_edit.setPlaceholderText("커밋에 남길 이름")
        self.author_email_edit = QLineEdit(config.author_email or git_email)
        self.author_email_edit.setPlaceholderText("커밋에 남길 이메일")
        repo_box = QGroupBox("저장소")
        repo_form = QFormLayout(repo_box)
        repo_form.addRow("주소", remote_row)
        repo_form.addRow("브랜치", self.branch_edit)
        repo_form.addRow("작성자 이름", self.author_name_edit)
        repo_form.addRow("작성자 이메일", self.author_email_edit)

        self.intrinsic_path_edit = QLineEdit(config.intrinsic_path)
        self.intrinsic_checks = {}
        intr_box = QGroupBox("Camera Intrinsic")
        intr_form = QFormLayout(intr_box)
        intr_form.addRow("저장소 안 경로", self.intrinsic_path_edit)
        intr_form.addRow("", QLabel("쓸 수 있는 항목: {vehicle} 차량 (차량 없음 → no_vehicle) · {camera} 카메라 이름 · {session} 세션 · {date} 날짜"))
        for key, label in gu.INTRINSIC_FILES.items():
            box = QCheckBox(label)
            box.setChecked(key in config.intrinsic_files)
            self.intrinsic_checks[key] = box
            intr_form.addRow("" if self.intrinsic_checks.keys() - {key} else "올릴 파일", box)
        self.recommended_only = QCheckBox("모델 YAML 은 추천 모델 하나만")
        self.recommended_only.setChecked(config.intrinsic_recommended_only)
        self.intrinsic_checks["models"].toggled.connect(self.recommended_only.setEnabled)
        self.recommended_only.setEnabled(self.intrinsic_checks["models"].isChecked())
        intr_form.addRow("", self.recommended_only)

        self.extrinsic_path_edit = QLineEdit(config.extrinsic_path)
        self.extrinsic_checks = {}
        ext_box = QGroupBox("CAMERA–LiDAR")
        ext_form = QFormLayout(ext_box)
        ext_form.addRow("저장소 안 경로", self.extrinsic_path_edit)
        ext_form.addRow("", QLabel("쓸 수 있는 항목: {vehicle} 차량 · {session} 세션 · {date} 날짜 · {camera} 카메라"))
        for key, label in gu.EXTRINSIC_FILES.items():
            box = QCheckBox(label)
            box.setChecked(key in config.extrinsic_files)
            self.extrinsic_checks[key] = box
            ext_form.addRow("" if self.extrinsic_checks.keys() - {key} else "올릴 파일", box)

        self.upload_vehicle = QCheckBox("차량 정보도 같이 올리기 (vehicles/<차량>/vehicle.yaml · lenses.yaml)")
        self.upload_vehicle.setToolTip("카메라 이름·토픽·렌즈. 다른 PC 의 홈 화면 [GitHub에서 가져오기] 가 이것을 읽습니다.")
        self.upload_vehicle.setChecked(config.upload_vehicle)

        defaults = QPushButton("기본값 (결과값만)")
        defaults.clicked.connect(self._defaults)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("저장")
        buttons.button(QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(defaults)
        bottom.addStretch(1)
        bottom.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(repo_box)
        layout.addWidget(intr_box)
        layout.addWidget(ext_box)
        layout.addWidget(self.upload_vehicle)
        layout.addWidget(self.status_label)
        layout.addLayout(bottom)

    def _defaults(self) -> None:
        d = gu.GitUploadConfig()
        self.intrinsic_path_edit.setText(d.intrinsic_path)
        self.extrinsic_path_edit.setText(d.extrinsic_path)
        for key, box in self.intrinsic_checks.items():
            box.setChecked(key in d.intrinsic_files)
        for key, box in self.extrinsic_checks.items():
            box.setChecked(key in d.extrinsic_files)
        self.recommended_only.setChecked(d.intrinsic_recommended_only)
        self.upload_vehicle.setChecked(d.upload_vehicle)

    def current(self) -> gu.GitUploadConfig:
        return gu.GitUploadConfig(
            remote=self.remote_edit.text().strip(),
            branch=self.branch_edit.text().strip() or "main",
            author_name=self.author_name_edit.text().strip(),
            author_email=self.author_email_edit.text().strip(),
            intrinsic_path=self.intrinsic_path_edit.text().strip(),
            intrinsic_files=[k for k, b in self.intrinsic_checks.items() if b.isChecked()],
            intrinsic_recommended_only=self.recommended_only.isChecked(),
            extrinsic_path=self.extrinsic_path_edit.text().strip(),
            extrinsic_files=[k for k, b in self.extrinsic_checks.items() if b.isChecked()],
            upload_vehicle=self.upload_vehicle.isChecked(),
        )

    def _problems(self, config: gu.GitUploadConfig) -> list[str]:
        out = []
        if config.author_email and "@" not in config.author_email:
            out.append("작성자 이메일 형식이 아닙니다")
        for title, template, fields in (("Camera Intrinsic", config.intrinsic_path, gu.INTRINSIC_PATH_FIELDS),
                                        ("CAMERA–LiDAR", config.extrinsic_path, gu.EXTRINSIC_PATH_FIELDS)):
            problem = gu.check_path_template(template, fields)
            if problem:
                out.append(f"{title} 경로: {problem}")
        if "{camera}" not in config.extrinsic_path and len(set(config.extrinsic_files) - {"session.yaml"}) > 0:
            out.append("CAMERA–LiDAR 경로에 {camera} 가 없으면 카메라끼리 같은 파일명으로 덮어씁니다")
        if not config.intrinsic_files:
            out.append("Camera Intrinsic 에서 올릴 파일을 하나 이상 고르세요")
        if not config.extrinsic_files:
            out.append("CAMERA–LiDAR 에서 올릴 파일을 하나 이상 고르세요")
        return out

    def _save(self) -> None:
        config = self.current()
        problems = self._problems(config)
        if problems:
            _tone(self.status_label, "\n".join(problems), "bad")
            return
        gu.save_config(config)
        self.config = config
        self.accept()

    def _check(self) -> None:
        remote = self.remote_edit.text().strip()
        if not remote:
            _tone(self.status_label, "저장소 주소를 입력하세요", "bad")
            return
        _tone(self.status_label, "연결 확인 중…")
        self._runner.run(
            lambda: gu.connection_check(remote),
            on_done=lambda problem: _tone(self.status_label, problem or "연결됨 - 이 PC 에서 읽을 수 있습니다",
                                          "bad" if problem else "good"),
            on_fail=lambda exc: _tone(self.status_label, str(exc), "bad"),
            busy=self.check_button)

    def reject(self) -> None:
        if self._runner.running():
            return
        super().reject()


# ---------------------------------------------------------------------- 업로드


class GitUploadDialog(QDialog):
    """`build_plan(config)` 이 만든 파일을 올린다. 성공하면 `result` 에 UploadResult 가 남는다."""

    def __init__(self, build_plan: PlanBuilder, title: str = "Git 업로드", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 560)
        self.build_plan = build_plan
        self.config = gu.load_config()
        self.plan = gu.UploadPlan()
        self.result: gu.UploadResult | None = None
        self._runner = TaskRunner(self)

        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        self.settings_button = QPushButton("⚙ 설정…")
        self.settings_button.clicked.connect(self._open_settings)
        top = QHBoxLayout()
        top.addWidget(self.target_label, 1)
        top.addWidget(self.settings_button)
        self.message_edit = QLineEdit()
        self.message_edit.textEdited.connect(lambda _t: self.message_edit.setProperty("auto", False))
        form = QFormLayout()
        form.addRow("커밋 메시지", self.message_edit)
        self.files_label = QLabel()
        self.files_view = QPlainTextEdit()
        self.files_view.setReadOnly(True)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.upload_button = QPushButton("⬆  업로드")
        self.upload_button.setProperty("role", "primary")
        self.upload_button.clicked.connect(self._start)
        self.close_button = QPushButton("닫기")
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.upload_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(form)
        layout.addWidget(self.files_label)
        layout.addWidget(self.files_view, 1)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)
        self._rebuild()

    def _rebuild(self) -> None:
        c = self.config
        who = f"{c.author_name} <{c.author_email}>" if c.author_name and c.author_email else "<i>작성자 미설정</i>"
        self.target_label.setText(f"<b>저장소</b> {c.remote or '<i>미설정</i>'}<br><b>브랜치</b> {c.branch} · <b>작성자</b> {who}")
        try:
            self.plan, message = self.build_plan(c)
        except gu.GitUploadError as exc:
            self.plan, message = gu.UploadPlan(), ""
            _tone(self.status_label, str(exc), "bad")
        else:
            _tone(self.status_label, "")
        if not self.message_edit.text().strip() or self.message_edit.property("auto"):
            self.message_edit.setText(message)
            self.message_edit.setProperty("auto", True)
        listing = [f"{f.dest}    ({_size(f.size)})" for f in self.plan.files]
        if self.plan.skipped:
            listing += ["", f"제외 (파일당 {gu.MAX_FILE_MB} MB 초과):"] + [f"  {d}" for d in self.plan.skipped]
        self.files_view.setPlainText("\n".join(listing) if listing else "(올릴 파일이 없습니다 - ⚙ 설정에서 파일 종류를 확인하세요)")
        self.files_label.setText(f"올라갈 파일 {len(self.plan.files)}개 · 합계 {_size(self.plan.total_bytes)}")
        self.upload_button.setEnabled(bool(self.plan.files))

    def _open_settings(self) -> None:
        dialog = GitSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.config = dialog.config
            self._rebuild()

    def _lock(self, locked: bool) -> None:
        for w in (self.upload_button, self.close_button, self.settings_button, self.message_edit):
            w.setEnabled(not locked)

    def _start(self) -> None:
        c = self.config
        if not c.remote or not c.author_name or "@" not in c.author_email:
            _tone(self.status_label, "저장소 주소와 작성자를 먼저 설정하세요.", "bad")
            self._open_settings()
            c = self.config
            if not c.remote or not c.author_name or "@" not in c.author_email:
                return
        message = self.message_edit.text().strip() or "calibration result"
        plan = self.plan
        self._lock(True)
        _tone(self.status_label, "시작…")
        self._runner.run(lambda log: gu.upload(c, plan, message, log=log), on_done=self._done,
                         on_fail=lambda exc: self._failed(str(exc) if isinstance(exc, gu.GitUploadError)
                                                          else f"{type(exc).__name__}: {exc}"),
                         on_progress=lambda m: _tone(self.status_label, m))

    def _done(self, result: gu.UploadResult) -> None:
        self.result = result
        self._lock(False)
        self.upload_button.setEnabled(False)
        if result.nothing_to_commit:
            _tone(self.status_label, f"이미 올라가 있습니다 (변경 없음) · {result.branch} @ {result.commit}", "good")
        else:
            _tone(self.status_label, f"업로드 완료 · {result.files}개 파일 · {result.branch} @ {result.commit}", "good")

    def _failed(self, message: str) -> None:
        self._lock(False)
        _tone(self.status_label, message, "bad")

    def reject(self) -> None:
        if self._runner.running():
            return  # 업로드 중에는 닫지 않는다
        super().reject()
