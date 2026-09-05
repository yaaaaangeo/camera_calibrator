"""
camera_calibrator.ui.windshield_worker
===========================================

Windshield Refraction Calibration을 QThread 워커로 분리한다.

ui/windshield_workspace.py::_on_run_windshield_calibration이 예전에는
run_windshield_calibration(...)/run_residual_ray_calibration_with_
diagnostics(...)를 GUI 스레드에서 직접(동기) 호출했다 - Residual Ray의
STAGE A/B + Repeated Hold-out(기본 5개 seed)까지 합치면 수 초~수십 초가
걸릴 수 있어, 그동안 Qt 이벤트 루프가 막혀 창 이동/크기 조절/다른 탭
렌더링이 전부 멈추고 OS가 "응답 없음"으로 표시할 수 있었다.

ui/worker.py가 이미 이 문제(Standard 4모델 계산, Self-check 등)를 QObject
worker + run_worker_in_thread() 패턴으로 풀어뒀으므로 그 패턴을 그대로
재사용한다. Windshield calibration(calibrate_spherical/calibrate_residual_ray)은
ui/worker.py의 ProcessPoolExecutor 서브메커니즘이 필요했던 이유
(cv2.calibrateCamera/cv2.fisheye.calibrate가 GIL을 놓아준다는 보장이 없어서
발생하는 문제)에 해당하는 cv2 호출을 전혀 쓰지 않는다 - scipy.optimize.
least_squares 기반 순수 Python/NumPy 최적화라 QThread만으로 충분하다.

이 워커는 Qt 위젯을 전혀 참조하지 않는다 - 생성자는 순수 data object
(Dataset/WindshieldConfig/CameraConfig)만 받는다. UI 위젯 접근은 전부
main thread(WindshieldWorkspace)의 signal 핸들러 안에서만 일어난다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from calibration.types import CameraConfig, Dataset
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModelType
from calibration.windshield.residual_ray import run_residual_ray_calibration_with_diagnostics
from calibration.windshield.residual_rbf import run_residual_rbf_calibration_with_diagnostics
from calibration.windshield.spline import run_spline_calibration_with_diagnostics
from calibration.windshield.validation import run_windshield_calibration, split_windshield_train_test


class WindshieldCalibrationWorker(QObject):
    """Residual Ray는 run_residual_ray_calibration_with_diagnostics()(진단
    포함) 경로로, 그 외(Baseline/Spherical)는 기존 run_windshield_calibration()
    dispatcher로 그대로 보낸다 - 계산 로직 자체는 여기서 재구현하지 않는다."""

    progress = Signal(str)
    result_ready = Signal(object)     # WindshieldCalibrationResult
    not_implemented = Signal(str)     # Spline 등 "Coming soon" 스텁 - 일반 오류와 구분해서 안내
    error = Signal(str)
    finished = Signal()

    def __init__(self, dataset: Dataset, config: WindshieldConfig, camera_config: CameraConfig):
        super().__init__()
        self._dataset = dataset
        self._config = config
        self._camera_config = camera_config

    def run(self) -> None:
        try:
            self.progress.emit("Windshield calibration 실행 중...")
            if self._config.windshield_model == WindshieldModelType.RESIDUAL_RAY:
                # run_windshield_calibration() dispatcher와 동일하게 여기서도
                # config.test_ratio/split_seed로 직접 split한다 - Residual Ray만
                # 진단(Repeated Hold-out 등)이 추가된 별도 진입점을 타므로, split
                # 방식 자체는 다른 모델과 완전히 동일하게 맞춘다(Split Logic 재사용
                # 원칙, calibration/windshield/validation.py 모듈 docstring 참고).
                train_ids, test_ids = split_windshield_train_test(
                    self._dataset, self._camera_config, self._config.test_ratio, self._config.split_seed,
                )
                hint = self._config.residual_ray_hint or {}
                method = str(hint.get("method", "grid")).lower()
                if method == "rbf":
                    result = run_residual_rbf_calibration_with_diagnostics(
                        self._dataset, self._config, self._camera_config, train_ids, test_ids,
                    )
                elif method == "neural":
                    # torch는 선택적 의존성이라 여기서만 lazy import한다 - Neural을
                    # 실제로 선택했을 때만 이 import를 거치므로, PyTorch가 없는
                    # 환경에서도 Grid/RBF/다른 모델은 정상 동작한다(ImportError는
                    # 아래 광범위 except가 잡아 명확한 메시지로 UI에 보여준다).
                    from calibration.windshield.neural_residual import run_neural_residual_calibration_with_diagnostics
                    result = run_neural_residual_calibration_with_diagnostics(
                        self._dataset, self._config, self._camera_config, train_ids, test_ids,
                    )
                else:
                    result = run_residual_ray_calibration_with_diagnostics(
                        self._dataset, self._config, self._camera_config, train_ids, test_ids,
                    )
            elif self._config.windshield_model == WindshieldModelType.SPLINE:
                # Spline도 Repeated Hold-out/Ray/Surface Stability 진단이 필요한
                # "advanced" 모델이라 Residual Ray와 동일하게 전용 오케스트레이터를
                # 거친다(run_windshield_calibration의 얇은 dispatch가 아니라).
                train_ids, test_ids = split_windshield_train_test(
                    self._dataset, self._camera_config, self._config.test_ratio, self._config.split_seed,
                )
                result = run_spline_calibration_with_diagnostics(
                    self._dataset, self._config, self._camera_config, train_ids, test_ids,
                )
            else:
                result = run_windshield_calibration(self._dataset, self._config, self._camera_config)
            self.result_ready.emit(result)
        except NotImplementedError as e:
            self.not_implemented.emit(str(e))
        except Exception as e:  # noqa: BLE001 - UI에 원인을 그대로 보여주기 위해 광범위하게 캐치
            self.error.emit(f"계산 실패: {e}")
        finally:
            self.finished.emit()
