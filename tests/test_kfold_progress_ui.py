"""
tests/test_kfold_progress_ui.py
====================================

Repeated K-Fold 실시간 진행률(progress bar / 모델별 WAITING-RUNNING-COMPLETE-
FAILED 상태 / 모델 완료 즉시 결과 행 채우기) 회귀 테스트.

75 fold(K=5 x Repeats=5 x 3모델) 계산이 오래 걸려도 사용자가 "멈춘 건지
계산 중인지" 알 수 있어야 한다는 게 목적이라, 여기서는 실제 cv2 calibration을
돌리지 않고 calibration.kfold.KFoldProgressEvent를 직접 만들어
ResultView/RepeatedKFoldWorker에 주입하는 방식으로 UI 반응만 검증한다(fold
split/seed/metric 계산 자체는 tests/test_kfold.py가 이미 검증한다).
"""

from __future__ import annotations

import pytest

# ui/result_view.py, ui/kfold_worker.py 모두 PySide6가 있어야 import 가능.
pytest.importorskip("PySide6.QtWidgets", reason="PySide6.QtWidgets is not importable in this environment")

from PySide6.QtWidgets import QApplication

from calibration.kfold import KFoldProgressEvent
from calibration.types import CameraModelType, RepeatedKFoldResult


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _fake_repeated_result(model: CameraModelType, k: int = 5, n_repeats: int = 5) -> RepeatedKFoldResult:
    total = k * n_repeats
    return RepeatedKFoldResult(
        k=k, n_repeats=n_repeats,
        mean_test_rms=1.23, std_test_rms=0.05,
        mean_test_p95=2.34, std_test_p95=0.08,
        mean_edge_rms=1.5, std_edge_rms=0.1,
        mean_test_straightness=0.04, std_test_straightness=0.01, n_straightness_folds=total,
        n_successful_runs=total, total_folds=total,
        fully_successful_folds=total, partial_success_folds=0, failed_folds=0,
    )


class TestResultViewKFoldProgressDisplay:
    def test_start_progress_clears_stale_previous_result(self, qapp):
        """모델 완료 전(=새 실행을 막 시작한 시점)에는 이전 실행의 결과 숫자가
        표에 그대로 남아있으면 안 된다 - stale data와 헷갈리는 문제 방지."""
        from ui.result_view import ResultView

        view = ResultView()
        old_result = _fake_repeated_result(CameraModelType.BROWN_CONRADY)
        view.set_repeated_kfold_results({CameraModelType.BROWN_CONRADY: old_result})
        # 이전 실행 결과가 실제로 표에 반영됐는지 먼저 확인(전제 조건).
        assert "1.23" in view.kfold_table.item(0, 1).text()

        view.start_repeated_kfold_progress(k=5, n_repeats=5)

        for row in range(3):
            for col in range(1, 6):
                text = view.kfold_table.item(row, col).text()
                assert "1.23" not in text, "새 실행 시작 후에도 이전 결과 숫자가 표에 남아있다"
                assert text in ("Waiting...", "Running..."), f"row={row} col={col}: {text!r}"
        view.close()

    def test_start_progress_sets_first_model_running_others_waiting(self, qapp):
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)

        assert view.kfold_status_table.item(0, 1).text() == "RUNNING"
        assert view.kfold_status_table.item(0, 2).text() == "0/25"
        assert view.kfold_status_table.item(1, 1).text() == "WAITING"
        assert view.kfold_status_table.item(2, 1).text() == "WAITING"
        assert view.kfold_progress_bar.maximum() == 75
        assert view.kfold_progress_bar.value() == 0
        view.close()

    def test_fold_completed_events_update_progress_bar_and_model_row(self, qapp):
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)

        event = KFoldProgressEvent(
            completed_folds=13, total_folds=75,
            model=CameraModelType.BROWN_CONRADY, model_completed_folds=13, model_total_folds=25,
            stage="fold_completed",
        )
        view.update_repeated_kfold_progress(event)

        assert view.kfold_progress_bar.value() == 13
        assert view.kfold_status_table.item(0, 1).text() == "RUNNING"
        assert view.kfold_status_table.item(0, 2).text() == "13/25"
        view.close()

    def test_model_completed_fills_row_immediately_without_waiting_for_others(self, qapp):
        """Brown 25 folds가 끝나는 즉시(Rational/Fisheye가 시작도 하기 전에)
        Brown 행에 metric이 채워져야 한다."""
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)

        result = _fake_repeated_result(CameraModelType.BROWN_CONRADY)
        event = KFoldProgressEvent(
            completed_folds=25, total_folds=75,
            model=CameraModelType.BROWN_CONRADY, model_completed_folds=25, model_total_folds=25,
            stage="model_completed", partial_result=result,
        )
        view.update_repeated_kfold_progress(event)

        assert view.kfold_status_table.item(0, 1).text() == "COMPLETE"
        assert view.kfold_status_table.item(0, 2).text() == "25/25"
        assert "1.23" in view.kfold_table.item(0, 1).text()  # Test RMS 채워짐

        # Rational/Fisheye는 아직 시작 이벤트가 안 왔으므로 그대로 WAITING/Waiting...
        assert view.kfold_status_table.item(1, 1).text() == "WAITING"
        assert view.kfold_table.item(1, 1).text() == "Waiting..."
        assert view.kfold_status_table.item(2, 1).text() == "WAITING"
        view.close()

    def test_all_completed_sets_progress_bar_to_full(self, qapp):
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)
        event = KFoldProgressEvent(
            completed_folds=75, total_folds=75,
            model=None, model_completed_folds=0, model_total_folds=25,
            stage="all_completed",
        )
        view.update_repeated_kfold_progress(event)

        assert view.kfold_progress_bar.value() == view.kfold_progress_bar.maximum() == 75
        view.close()

    def test_error_marks_running_models_failed_and_keeps_progress_value(self, qapp):
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)
        view.update_repeated_kfold_progress(
            KFoldProgressEvent(
                completed_folds=13, total_folds=75,
                model=CameraModelType.BROWN_CONRADY, model_completed_folds=13, model_total_folds=25,
                stage="fold_completed",
            )
        )
        assert view.kfold_progress_bar.value() == 13

        view.set_repeated_kfold_error("boom")

        assert view.kfold_status_table.item(0, 1).text() == "FAILED"
        # progress bar 값은 실패 시점의 진행 상황을 그대로 보존해야 한다(뒤로 되돌리지 않음).
        assert view.kfold_progress_bar.value() == 13
        assert "boom" in view.kfold_summary_label.text()
        view.close()

    def test_final_results_ready_marks_all_models_complete(self, qapp):
        """results_ready 최종 dict로도(progress_event 연결이 없었더라도) 3개
        모델 모두 COMPLETE + 채워진 결과 행으로 정리돼야 한다."""
        from ui.result_view import ResultView

        view = ResultView()
        view.start_repeated_kfold_progress(k=5, n_repeats=5)
        results = {
            CameraModelType.BROWN_CONRADY: _fake_repeated_result(CameraModelType.BROWN_CONRADY),
            CameraModelType.EXTENDED_PINHOLE: _fake_repeated_result(CameraModelType.EXTENDED_PINHOLE),
            CameraModelType.FISHEYE: _fake_repeated_result(CameraModelType.FISHEYE),
        }
        view.set_repeated_kfold_results(results)

        for row in range(3):
            assert view.kfold_status_table.item(row, 1).text() == "COMPLETE"
            assert "1.23" in view.kfold_table.item(row, 1).text()
        view.close()


class TestRepeatedKFoldWorkerSignals:
    """ui/kfold_worker.py의 signal 배선 - 실제 QThread 없이 run()을 직접
    호출해서(동기 실행) finished/error/progress_event가 올바르게 나오는지만
    확인한다(main_window.py의 thread.finished.connect(...)가 Run 버튼을
    재활성화하는 실제 배선은 이미 존재하는 cross_dataset_button과 동일한
    검증된 패턴이라 여기서는 worker가 그 신호들을 정확히 내는지에 집중)."""

    def test_progress_event_and_results_ready_emitted_on_success(self, qapp, monkeypatch):
        from ui.kfold_worker import RepeatedKFoldWorker
        import ui.kfold_worker as kfold_worker_module

        def fake_run_all_models(dataset, camera_config, pattern_config, k, n_repeats, base_seed, n_jobs, progress_callback=None):
            if progress_callback is not None:
                progress_callback(KFoldProgressEvent(
                    completed_folds=1, total_folds=k * n_repeats * 3,
                    model=CameraModelType.BROWN_CONRADY, model_completed_folds=1, model_total_folds=k * n_repeats,
                    stage="fold_completed",
                ))
            return {CameraModelType.BROWN_CONRADY: _fake_repeated_result(CameraModelType.BROWN_CONRADY, k, n_repeats)}

        monkeypatch.setattr(kfold_worker_module, "run_repeated_kfold_all_models", fake_run_all_models)

        worker = RepeatedKFoldWorker(dataset=None, camera_config=None, pattern_config=None, k=5, n_repeats=5)
        events = []
        results_holder = {}
        finished_flag = {"done": False}
        worker.progress_event.connect(events.append)
        worker.results_ready.connect(lambda r: results_holder.update(r))
        worker.finished.connect(lambda: finished_flag.__setitem__("done", True))

        worker.run()

        assert finished_flag["done"], "finished signal이 나와야 Run 버튼이 재활성화된다"
        assert len(events) == 1 and events[0].stage == "fold_completed"
        assert CameraModelType.BROWN_CONRADY in results_holder

    def test_error_and_finished_emitted_on_failure(self, qapp, monkeypatch):
        """계산 중 예외가 나도 error + finished가 모두 나와야 한다 - finished가
        나와야(성공/실패 무관) Run 버튼이 다시 활성화된다."""
        from ui.kfold_worker import RepeatedKFoldWorker
        import ui.kfold_worker as kfold_worker_module

        def fake_run_all_models(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(kfold_worker_module, "run_repeated_kfold_all_models", fake_run_all_models)

        worker = RepeatedKFoldWorker(dataset=None, camera_config=None, pattern_config=None, k=5, n_repeats=5)
        errors = []
        finished_flag = {"done": False}
        worker.error.connect(errors.append)
        worker.finished.connect(lambda: finished_flag.__setitem__("done", True))

        worker.run()

        assert finished_flag["done"]
        assert len(errors) == 1
        assert "synthetic failure" in errors[0]
