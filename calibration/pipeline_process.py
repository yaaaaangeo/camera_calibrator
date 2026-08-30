"""
camera_calibrator.calibration.pipeline_process
====================================================

ui/worker.py의 PipelineWorker와 app/cli.py의 --outlier 플로우가 "GUI/CLI
스레드가 아니라 완전히 별도의 OS 프로세스"에서 무거운 계산(3모델 캘리브레이션
+ Hold-out 검증, 이상치 반복 재계산)을 돌리기 위한 진입점 함수들을 모아둔다.

--- 왜 QThread만으로는 부족한가 (실제 사용자 버그) ---

detect_dataset()의 병렬 검출은 이미 ProcessPoolExecutor를 쓰고 있어서 문제가
없었다. 그런데 그 다음 단계인 run_all_models()/validate_all_models()
(cv2.calibrateCamera, cv2.fisheye.calibrate 등)는 PipelineWorker.run() 안에서
그냥 순차 호출됐다 - QThread(백그라운드 파이썬 스레드)에서 돈다는 이유로
"UI를 안 막는다"고 여겨졌지만, 실제로는 파이썬 GIL은 프로세스 전체에
하나뿐이라 문제였다: cv2의 C++ 확장 함수들은 오래 걸리는 계산 도중 GIL을
놓아준다는 보장이 없고, 실제로 놓아주지 않는 경우가 많다. 그러면 계산
스레드가 GIL을 계속 붙들고 있는 동안 GUI 스레드는 Qt 이벤트(창 이동,
다시 그리기, 버튼 클릭 처리 등)를 실행할 파이썬 코드 자체를 돌릴 GIL을
얻지 못해 완전히 멈춘다 - 이게 OS가 "python3 is not responding"을 띄우는
실제 원인이다. 이미지가 몇 장 안 되면 계산이 순식간이라 안 보이다가,
수백 장(사용자 사례: rosbag에서 뽑은 307장)에서 Rational model(14개
왜곡 계수)까지 켜면 3모델 x (전체 학습 + Hold-out 재학습)로 번들 조정이
여러 번 돌아 몇 초~몇십 초씩 걸릴 수 있어 뚜렷하게 나타난다.

완전히 별도의 OS 프로세스로 계산을 돌리면, 그 프로세스는 자기만의 파이썬
인터프리터/GIL을 가지므로 부모 프로세스(GUI가 있는 쪽)의 GIL은 계산
시간과 무관하게 항상 비어 있다 - 근본적인 해결책이다.

--- 이 모듈의 함수들이 지켜야 할 규칙 ---

concurrent.futures.ProcessPoolExecutor.submit()의 인자로 그대로 pickle되어
자식 프로세스에 전달되고, 반환값도 다시 pickle되어 돌아와야 하므로:
    1. 반드시 모듈 최상위 함수로 정의한다 (클래스 메서드/클로저는 pickle 불가 -
       detector.py의 ProcessPoolExecutor 워커 함수들과 같은 이유).
    2. 인자/반환값은 calibration/types.py의 순수 dataclass(+numpy 배열)만
       사용한다 - 전부 이미 project_io.py에서 JSON 직렬화까지 되는 타입들이라
       pickle은 항상 더 관대하게 통과한다.
"""

from __future__ import annotations

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CalibrationMethod,
    Dataset,
    OutlierResult,
    PatternConfig,
    ValidationResult,
)


def run_models_and_validation(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    test_ratio: float,
    use_rational_model: bool,
    calibration_method: CalibrationMethod = CalibrationMethod.STANDARD,
    model_jobs: int = 2,
) -> tuple[dict[CameraModelType, CalibrationResult], dict[CameraModelType, ValidationResult], CalibrationResult | None]:
    """3모델 계산(run_all_models) + Hold-out 검증(validate_all_models)을
    자식 프로세스 하나에서 이어서 실행한다.

    두 단계를 한 번의 프로세스 제출로 묶은 이유: dataset을 매번 따로
    pickle해서 자식 프로세스로 보내는 오버헤드(이미지 수백 장 기준 무시 못할
    크기)를 반으로 줄이기 위해서다. 두 함수 모두 dataset을 읽기만 하고
    바꾸지 않으므로(기존 설계 그대로) 이렇게 묶어도 결과는 완전히 동일하다.
    """
    from calibration.compare import run_all_models
    from calibration.validation import validate_all_models

    if not isinstance(calibration_method, CalibrationMethod):
        calibration_method = CalibrationMethod(str(calibration_method))

    results_list = run_all_models(
        dataset,
        camera_config,
        use_rational_model=use_rational_model,
        model_jobs=model_jobs,
    )
    calibration_results = {r.model_name: r for r in results_list}
    object_releasing_result = None
    if calibration_method == CalibrationMethod.OBJECT_RELEASING:
        from calibration.models.object_releasing import calibrate_object_releasing_brown_conrady

        object_releasing_result = calibrate_object_releasing_brown_conrady(
            dataset,
            camera_config,
            pattern_config,
        )

    validation_results = validate_all_models(
        dataset, camera_config, pattern_config,
        test_ratio=test_ratio, use_rational_model=use_rational_model,
    )
    return calibration_results, validation_results, object_releasing_result


def run_outlier_pruning_and_validation(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    reference_model: CameraModelType,
    max_iterations: int,
    test_ratio: float,
    use_rational_model: bool,
) -> tuple[
    Dataset,
    CalibrationResult,
    OutlierResult,
    list[str],
    dict[CameraModelType, CalibrationResult],
    dict[CameraModelType, ValidationResult],
]:
    """이상치 반복 재계산 + Coverage 재분석 + 3모델 재계산 + Hold-out
    재검증까지 전부 자식 프로세스 하나에서 실행한다.

    주의 - dataset을 반드시 반환값에 포함시켜야 하는 이유:
    recalibrate_with_outlier_pruning()은 dataset.frames[i].status를
    DISABLED_OUTLIER로 바꾸는 in-place 부수효과가 있다. 같은 프로세스/스레드
    안에서라면 그 부수효과가 호출부가 들고 있는 같은 객체에 그대로
    반영되지만, 여기서는 자식 프로세스가 pickle로 전달받은 "복사본"에만
    반영된다. 그래서 이 함수가 끝난 뒤의 dataset(상태가 바뀐 바로 그 객체)을
    명시적으로 반환하고, 호출부가 자기 쪽 dataset 참조를 통째로 이걸로
    교체해야 한다 - 안 그러면 "이상치 제외"가 반영되지 않는 조용한 버그가
    생긴다.
    """
    from calibration.outlier import recalibrate_with_outlier_pruning
    from calibration.quality import analyze_dataset_quality
    from calibration.frame_quality import compute_frame_quality_scores
    from calibration.models.common import infer_image_size

    ref_result, outlier_result = recalibrate_with_outlier_pruning(
        dataset, camera_config, reference_model,
        max_iterations=max_iterations, use_rational_model=use_rational_model,
    )

    warnings = analyze_dataset_quality(dataset, camera_config)

    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    calibration_results, validation_results, _object_releasing_result = run_models_and_validation(
        dataset, camera_config, pattern_config, test_ratio, use_rational_model, model_jobs=2,
    )

    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    return dataset, ref_result, outlier_result, warnings, calibration_results, validation_results
