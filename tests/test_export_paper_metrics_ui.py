"""
tests/test_export_paper_metrics_ui.py
====================================

"Export Paper Metrics" UI 배선 회귀 테스트.

계산 로직(calibration/paper_evidence.py) 자체는 tests/test_paper_evidence.py가
이미 검증했으므로, 여기서는 UI 배선만 다룬다:
  - 버튼/signal이 실제 존재하는지
  - 필요한 결과(calibration/validation/repeated K-Fold)가 없을 때 export를
    실행하지 않고 안내만 보여주는지
  - Repeated K-Fold 결과가 MainWindow state(self.repeated_kfold_results)에
    저장되고, Export가 그 저장된 값을 재사용할 뿐 다시 계산하지 않는지
  - 폴더 선택 취소 시 아무 파일도 안 만드는지
  - 실제로 폴더에 파일이 생성되고 성공 메시지에 그 목록이 표시되는지
  - backend 예외가 나도 GUI가 죽지 않고 에러 메시지를 보여주는지
  - 새 calibration 실행이 이전 Repeated K-Fold 결과를 stale로 남기지 않는지
  - Paper Intrinsic Stability / All-Parameter Stability가 export에서
    분리되어 있는지
  - 기존 OpenCV YAML Export가 영향받지 않았는지

무거운 실제 calibration 파이프라인은 돌리지 않는다 - 가벼운 fake
CalibrationResult/ValidationResult/RepeatedKFoldResult로 MainWindow state를
직접 채운다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets", reason="PySide6.QtWidgets is not importable in this environment")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    ParameterUncertainty,
    PatternConfig,
    PatternType,
    RepeatedKFoldResult,
    ValidationResult,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _fake_calibration_result(model: CameraModelType, paper_stability=90.0, overall_stability=62.0) -> CalibrationResult:
    return CalibrationResult(
        model_name=model, success=True,
        camera_matrix=np.eye(3), distortion=np.zeros((4, 1)),
        rms_error=1.0,
        param_uncertainty_bootstrap=ParameterUncertainty(
            method="bootstrap", n_bootstrap_success=18, n_bootstrap_total=20,
            fx_stability=paper_stability, fy_stability=paper_stability,
            cx_stability=paper_stability, cy_stability=paper_stability,
            paper_intrinsic_stability=paper_stability,
            overall_stability=overall_stability,
        ),
    )


def _fake_validation_result() -> ValidationResult:
    return ValidationResult(
        test_rms=1.0, success=True,
        train_frame_ids=["train_a", "train_b"], test_frame_ids=["test_a"],
    )


def _fake_repeated_result(k=3, n_repeats=2, base_seed=1) -> RepeatedKFoldResult:
    return RepeatedKFoldResult(
        k=k, n_repeats=n_repeats, base_seed=base_seed,
        n_successful_runs=k * n_repeats, total_folds=k * n_repeats,
        fully_successful_folds=k * n_repeats,
    )


def _minimal_state(win, *, with_calibration=True, with_validation=True, with_kfold=True) -> None:
    win.camera_config = CameraConfig(width=640, height=480, sensor_name="paper-metrics-test")
    win.pattern_config = PatternConfig(type=PatternType.CHESSBOARD, squares_x=7, squares_y=5, square_size=0.03)
    win.dataset = Dataset(frames=[])
    if with_calibration:
        win.calibration_results = {
            CameraModelType.BROWN_CONRADY: _fake_calibration_result(CameraModelType.BROWN_CONRADY),
        }
    if with_validation:
        win.validation_results = {CameraModelType.BROWN_CONRADY: _fake_validation_result()}
    if with_kfold:
        win.repeated_kfold_results = {CameraModelType.BROWN_CONRADY: _fake_repeated_result()}


class _Warnings:
    """QMessageBox.warning/critical/information 호출을 가로채 텍스트만 모은다."""
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, *args, **kwargs):
        # QMessageBox.warning(parent, title, text, ...) - text가 args[2].
        text = args[2] if len(args) > 2 else kwargs.get("text", "")
        self.calls.append(str(text))


# ---------------------------------------------------------------------------
# 1) 버튼/signal 존재
# ---------------------------------------------------------------------------

class TestButtonExists:
    def test_result_view_has_export_button_and_signal(self, qapp):
        from ui.result_view import ResultView

        view = ResultView()
        try:
            assert hasattr(view, "kfold_export_button")
            assert hasattr(view, "export_paper_metrics_requested")
            assert view.kfold_export_button.text() == "Export Paper Metrics"
        finally:
            view.close()

    def test_main_window_has_handler(self, qapp):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert hasattr(win, "_on_export_paper_metrics_requested")
        finally:
            win.close()

    def test_clicking_button_reaches_main_window_handler(self, qapp, monkeypatch):
        """IntrinsicWorkspace.connect_owner_handlers()가 __init__ 시점에 이미
        result_view.export_paper_metrics_requested -> MainWindow._on_export_paper_metrics_requested
        를 연결해 둔다 - 여기서 다시 connect()하면 중복 연결이 되므로, 기존
        연결이 monkeypatch된 핸들러까지 실제로 타는지만 확인한다."""
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            called = {"n": 0}
            monkeypatch.setattr(win, "_on_export_paper_metrics_requested", lambda: called.__setitem__("n", called["n"] + 1))
            win.result_view.kfold_export_button.click()
            assert called["n"] == 1
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 2) 필요한 데이터가 없을 때 export를 실행하지 않고 안내만 표시
# ---------------------------------------------------------------------------

class TestDependencyChecks:
    def test_missing_calibration_blocks_export(self, qapp, monkeypatch):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win, with_calibration=False)
            warnings = _Warnings()
            monkeypatch.setattr(QMessageBox, "warning", staticmethod(warnings))
            dialog_calls = {"n": 0}
            monkeypatch.setattr(
                QFileDialog, "getExistingDirectory",
                staticmethod(lambda *a, **k: dialog_calls.__setitem__("n", dialog_calls["n"] + 1) or ""),
            )
            win._on_export_paper_metrics_requested()
            assert warnings.calls and "Calibration" in warnings.calls[0]
            assert dialog_calls["n"] == 0, "필요한 데이터가 없으면 폴더 선택창도 열면 안 된다"
        finally:
            win.close()

    def test_missing_validation_blocks_export(self, qapp, monkeypatch):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win, with_validation=False)
            warnings = _Warnings()
            monkeypatch.setattr(QMessageBox, "warning", staticmethod(warnings))
            win._on_export_paper_metrics_requested()
            assert warnings.calls and "Validation" in warnings.calls[0]
        finally:
            win.close()

    def test_missing_repeated_kfold_blocks_export(self, qapp, monkeypatch):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win, with_kfold=False)
            warnings = _Warnings()
            monkeypatch.setattr(QMessageBox, "warning", staticmethod(warnings))
            win._on_export_paper_metrics_requested()
            assert warnings.calls and "Repeated K-Fold" in warnings.calls[0]
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 3) Repeated K-Fold 결과가 state에 저장되고, Export가 재사용(재계산 없음)
# ---------------------------------------------------------------------------

class TestRepeatedKFoldStateReuse:
    def test_results_ready_stores_into_state(self, qapp):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert win.repeated_kfold_results == {}
            fake_results = {CameraModelType.BROWN_CONRADY: _fake_repeated_result()}
            win._on_repeated_kfold_results_ready(fake_results)
            assert win.repeated_kfold_results is fake_results
        finally:
            win.close()

    def test_export_uses_stored_result_without_recomputing(self, qapp, monkeypatch, tmp_path):
        import ui.main_window as main_window_module
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            stored_repeated = win.repeated_kfold_results

            recompute_calls = {"n": 0}

            def _fail_if_recomputed(*a, **k):
                recompute_calls["n"] += 1
                raise AssertionError("Export가 Repeated K-Fold를 다시 계산하면 안 된다")

            # calibration.kfold.run_repeated_kfold_all_models가 다시 호출되면
            # 바로 실패하도록 - export 핸들러 경로에서는 절대 불려서는 안 된다.
            import calibration.kfold as kfold_module
            monkeypatch.setattr(kfold_module, "run_repeated_kfold_all_models", _fail_if_recomputed)

            captured = {}

            def _fake_export(output_dir, single_holdout, repeated, stability_by_model=None, metadata=None, spatial_rows_by_model=None):
                captured["output_dir"] = output_dir
                captured["single_holdout"] = single_holdout
                captured["repeated"] = repeated
                return {"paper_summary.csv": str(output_dir) + "/paper_summary.csv"}

            monkeypatch.setattr(main_window_module.paper_evidence, "export_paper_metrics", _fake_export)
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
            monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

            win._on_export_paper_metrics_requested()

            assert recompute_calls["n"] == 0
            assert captured["repeated"] is stored_repeated
            assert captured["single_holdout"] is win.validation_results
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 4) 폴더 선택 취소 -> 아무 파일도 생성하지 않음
# ---------------------------------------------------------------------------

class TestCancelDoesNothing:
    def test_default_export_does_not_open_a_directory_dialog(self, qapp, monkeypatch, tmp_path):
        import ui.main_window as main_window_module
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            dialog_calls = {"n": 0}
            monkeypatch.setattr(
                QFileDialog, "getExistingDirectory",
                staticmethod(lambda *a, **k: dialog_calls.__setitem__("n", dialog_calls["n"] + 1)),
            )
            export_calls = {"n": 0}
            monkeypatch.setattr(
                main_window_module.paper_evidence, "export_paper_metrics",
                lambda output_dir, *a, **k: (
                    export_calls.__setitem__("n", export_calls["n"] + 1)
                    or {"paper_summary.csv": str(output_dir) + "/paper_summary.csv"}
                ),
            )
            monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
            win._on_export_paper_metrics_requested()
            assert export_calls["n"] == 1
            assert dialog_calls["n"] == 0
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 5) 실제 폴더에 파일 생성 + 성공 메시지에 실제 파일 목록 표시
# ---------------------------------------------------------------------------

class TestSuccessfulExport:
    def test_files_are_actually_written_and_message_lists_them(self, qapp, monkeypatch, tmp_path):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            messages = []
            monkeypatch.setattr(
                QMessageBox, "information",
                staticmethod(lambda *a, **k: messages.append(a[2] if len(a) > 2 else "")),
            )

            win._on_export_paper_metrics_requested()

            assert messages, "성공 시 QMessageBox.information이 호출돼야 한다"
            message = messages[0]
            expected_files = [
                "paper_summary.csv", "repeated_kfold_folds.csv", "repeated_kfold_repeats.csv",
                "pairwise_win_counts.csv", "stability_parameters.csv",
                "split_manifest.json", "paper_metrics.json",
            ]
            for name in expected_files:
                assert name in message, f"{name}이 성공 메시지에 없음"
                path = win.output_manager.paper_directory() / name
                assert path.exists(), f"{name} 파일이 실제로 생성되지 않음"
                assert path.stat().st_size > 0
        finally:
            win.close()

    def test_export_accepts_string_model_and_pattern_values(self, qapp, monkeypatch, tmp_path):
        """PySide Signal/QVariant가 str-Enum을 plain str로 바꿔도 export한다."""
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            model = CameraModelType.BROWN_CONRADY.value
            win.pattern_config.type = PatternType.CHESSBOARD.value
            win.calibration_results = {model: next(iter(win.calibration_results.values()))}
            win.validation_results = {model: next(iter(win.validation_results.values()))}
            win.repeated_kfold_results = {model: next(iter(win.repeated_kfold_results.values()))}

            monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
            monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

            win._on_export_paper_metrics_requested()

            metrics = (win.output_manager.paper_directory() / "paper_metrics.json").read_text(encoding="utf-8")
            assert '"target_type": "chessboard"' in metrics
            assert '"brown_conrady"' in metrics
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 6) backend 예외 시 crash 없이 에러 표시
# ---------------------------------------------------------------------------

class TestExportFailure:
    def test_backend_exception_shows_error_without_crashing(self, qapp, monkeypatch, tmp_path):
        import ui.main_window as main_window_module
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))

            def _raise(*a, **k):
                raise RuntimeError("synthetic export failure")

            monkeypatch.setattr(main_window_module.paper_evidence, "export_paper_metrics", _raise)
            errors = []
            monkeypatch.setattr(
                QMessageBox, "critical",
                staticmethod(lambda *a, **k: errors.append(a[2] if len(a) > 2 else "")),
            )

            win._on_export_paper_metrics_requested()  # 예외가 여기서 새어나오면 테스트 자체가 실패한다

            assert errors and "synthetic export failure" in errors[0]
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 7) 새 calibration 실행이 stale Repeated K-Fold 결과를 남기지 않음
# ---------------------------------------------------------------------------

class TestStaleResultPrevention:
    def test_new_pipeline_run_clears_repeated_kfold_results(self, qapp, monkeypatch):
        """PipelineWorker/QThread는 fake로 대체해서 실제 이미지 처리는 전혀
        하지 않고, _on_run_pipeline() 맨 앞의 상태 초기화 로직만 검증한다."""
        import ui.main_window as main_window_module
        from ui.main_window import MainWindow

        class _FakeSignal:
            def connect(self, *_a, **_k):
                pass

        class _FakeThread:
            finished = _FakeSignal()

            def start(self):
                pass

        class _FakeWorker:
            progress = _FakeSignal()
            progress_value = _FakeSignal()
            dataset_ready = _FakeSignal()
            quality_ready = _FakeSignal()
            models_ready = _FakeSignal()
            validation_ready = _FakeSignal()
            recommendation_ready = _FakeSignal()
            error = _FakeSignal()
            cancelled = _FakeSignal()

            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr(main_window_module, "run_worker_in_thread", lambda worker, parent: _FakeThread())
        monkeypatch.setattr(main_window_module, "PipelineWorker", _FakeWorker)

        win = MainWindow()
        try:
            _minimal_state(win)
            assert win.repeated_kfold_results, "전제 조건: 이전 실행의 K-Fold 결과가 남아있어야 함"

            win.image_paths = ["dataset_a_fake.jpg"]
            win._on_run_pipeline()

            assert win.repeated_kfold_results == {}, "새 calibration 실행 후에도 이전 dataset의 K-Fold 결과가 남아있음(stale)"
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 8) Paper Intrinsic Stability / All-Parameter Stability 분리
# ---------------------------------------------------------------------------

class TestStabilitySeparationInExport:
    def test_stability_csv_keeps_paper_and_all_parameter_columns_distinct(self, qapp, monkeypatch, tmp_path):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            _minimal_state(win)
            from export.output_manager import OutputManager
            win.output_manager = OutputManager(tmp_path)
            win.windshield_workspace.set_output_manager(win.output_manager)
            # _fake_calibration_result가 paper=90.0, overall=62.0으로 일부러
            # 다르게 만들어 뒀다 - 두 값이 export에서 섞이면 이 테스트가 잡는다.
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
            monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

            win._on_export_paper_metrics_requested()

            content = (win.output_manager.paper_directory() / "stability_parameters.csv").read_text(encoding="utf-8")
            assert "paper_intrinsic_stability" in content
            assert "all_parameter_stability" in content
            assert "90.0" in content
            assert "62.0" in content
        finally:
            win.close()

    def test_stability_view_detail_labels_are_distinct(self, qapp):
        from ui.stability_view import format_stability_detail

        pu = ParameterUncertainty(
            method="bootstrap", paper_intrinsic_stability=90.0, overall_stability=62.0,
        )
        text = format_stability_detail("Fisheye", pu)
        assert "Paper Intrinsic Stability" in text
        assert "All-Parameter Stability" in text
        assert "90.0" in text
        assert "62.0" in text


# ---------------------------------------------------------------------------
# 9) 기존 OpenCV YAML Export는 영향받지 않음
# ---------------------------------------------------------------------------

class TestOpenCvExportUnaffected:
    def test_opencv_export_handler_and_button_still_present(self, qapp):
        from ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert hasattr(win, "_on_export_opencv")
            assert hasattr(win.result_view, "export_opencv_button")
            assert win.result_view.export_opencv_button.text() == "Export OpenCV YAML"
        finally:
            win.close()
