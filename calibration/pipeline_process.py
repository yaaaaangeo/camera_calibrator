"""
camera_calibrator.calibration.pipeline_process
====================================================

ui/worker.py의 PipelineWorker와 app/cli.py의 --outlier 플로우가 "GUI/CLI
스레드가 아니라 완전히 별도의 OS 프로세스"에서 무거운 계산(Standard 4모델
캘리브레이션 + Hold-out 검증, 이상치 반복 재계산, Object-Releasing을
썼다면 그 결과/Hold-out/Standard와의 비교까지)을 돌리기 위한 진입점 함수들을
모아둔다.

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
왜곡 계수)까지 켜면 Standard 4모델 x (전체 학습 + Hold-out 재학습)로 번들
조정이 여러 번 돌아 몇 초~몇십 초씩 걸릴 수 있어 뚜렷하게 나타난다.

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
    ObjectReleasingValidationResult,
    OutlierResult,
    PatternConfig,
    StandardVsObjectReleasingComparison,
    ValidationResult,
)


def run_models_and_validation(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    test_ratio: float,
    calibration_method: CalibrationMethod = CalibrationMethod.STANDARD,
    model_jobs: int = 2,
) -> tuple[
    dict[CameraModelType, CalibrationResult],
    dict[CameraModelType, ValidationResult],
    CalibrationResult | None,
    ObjectReleasingValidationResult | None,
    StandardVsObjectReleasingComparison | None,
]:
    """Standard 4-model 계산(run_all_models) + Hold-out 검증(validate_all_models)을
    자식 프로세스 하나에서 이어서 실행한다.

    calibration_method가 OBJECT_RELEASING이면 같은 프로세스 안에서 이어서:
      1. calibrate_object_releasing_brown_conrady()로 Advanced 결과를 계산하고
      2. validate_object_releasing_holdout()으로 RO 전용 Hold-out을 계산하고
      3. compare_standard_vs_object_releasing_brown()으로 Standard Brown-Conrady와의
         공정 비교(같은 full-board eligible 데이터셋 + 같은 train/test 분할)를 계산한다.
    이 셋은 Standard 4모델/AIC-BIC와 완전히 분리된 결과이며, 반환 튜플 뒤쪽
    세 자리에 항상 담는다(Standard method면 전부 None).

    여러 단계를 한 번의 프로세스 제출로 묶은 이유: dataset을 매번 따로
    pickle해서 자식 프로세스로 보내는 오버헤드(이미지 수백 장 기준 무시 못할
    크기)를 줄이기 위해서다. 각 함수 모두 dataset을 읽기만 하고 바꾸지
    않으므로(기존 설계 그대로) 이렇게 묶어도 결과는 완전히 동일하다.
    """
    from calibration.compare import run_all_models
    from calibration.validation import validate_all_models

    if not isinstance(calibration_method, CalibrationMethod):
        calibration_method = CalibrationMethod(str(calibration_method))

    results_list = run_all_models(
        dataset,
        camera_config,
        model_jobs=model_jobs,
    )
    calibration_results = {r.model_name: r for r in results_list}
    object_releasing_result = None
    object_releasing_validation_result = None
    standard_vs_object_releasing_comparison = None
    if calibration_method == CalibrationMethod.OBJECT_RELEASING:
        from calibration.models.object_releasing import calibrate_object_releasing_brown_conrady
        from calibration.object_releasing_validation import (
            compare_standard_vs_object_releasing_brown,
            validate_object_releasing_holdout,
        )

        object_releasing_result = calibrate_object_releasing_brown_conrady(
            dataset,
            camera_config,
            pattern_config,
        )
        object_releasing_validation_result = validate_object_releasing_holdout(
            dataset, camera_config, pattern_config, test_ratio=test_ratio,
        )
        standard_vs_object_releasing_comparison = compare_standard_vs_object_releasing_brown(
            dataset, camera_config, pattern_config, test_ratio=test_ratio,
        )

    validation_results = validate_all_models(
        dataset, camera_config, pattern_config,
        test_ratio=test_ratio,
    )
    return (
        calibration_results,
        validation_results,
        object_releasing_result,
        object_releasing_validation_result,
        standard_vs_object_releasing_comparison,
    )


def run_calibration_optimizer(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    train_ids: list[str],
    holdout_ids: list[str],
    settings,
):
    """Pickle-safe process entry point for the SciPy refinement backend."""
    from calibration.optimizer import ScipyCalibrationOptimizer

    return ScipyCalibrationOptimizer().optimize(
        dataset, camera_config, pattern_config, model, train_ids, holdout_ids, settings
    )


def run_scene_subset_calibration(
    dataset: Dataset,
    selected_frame_ids: list[str],
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    original_diversity,
    original_coverage_pct: float,
    frozen_holdout_ids: list[str] | None = None,
):
    """ProcessPool에 전달할 수 있는 scene subset 재계산 진입점."""
    from calibration.scene_quality import run_subset_calibration

    return run_subset_calibration(
        dataset, selected_frame_ids, camera_config, pattern_config, model,
        original_diversity=original_diversity,
        original_coverage_pct=original_coverage_pct,
        frozen_holdout_ids=frozen_holdout_ids,
    )


def run_outlier_pruning_and_validation(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    reference_model: CameraModelType,
    max_iterations: int,
    test_ratio: float,
) -> tuple[
    Dataset,
    CalibrationResult,
    OutlierResult,
    list[str],
    dict[CameraModelType, CalibrationResult],
    dict[CameraModelType, ValidationResult],
]:
    """이상치 반복 재계산 + Coverage 재분석 + Standard 4모델 재계산 + Hold-out
    재검증까지 전부 자식 프로세스 하나에서 실행한다.

    설계 문서 9번 - Train/Test Leakage 완전 제거. 이 함수는 두 갈래로
    나뉜다(app/cli.py의 `--outlier` 흐름과 동일한 원칙, validation.py 상단
    주석 참고):

    1. **최종 배포용 계산** - `dataset`(원본) 전체에 outlier pruning을 적용해
       `ref_result`/`calibration_results`를 만든다. train/test 구분 없이
       "좋은 데이터를 전부 쓴다"는 의도된 설계다(배포용 K/D는 애초에 평가
       대상이 아니므로 leakage 개념 자체가 적용되지 않는다).
    2. **Leak-safe Hold-out Validation** - outlier 판단 이전 상태를
       deepcopy해둔 `validation_dataset`에서 **먼저 train/test로 분할**하고,
       그 다음 train 서브셋에만 outlier pruning을 적용한다
       (`recalibrate_train_with_outlier_pruning`) - test 프레임은 outlier
       판단에 전혀 관여하지 않는다.

    이전 버전은 (1)에서 만든 outlier-pruned `dataset`을 그대로
    `run_models_and_validation`(내부에서 `validate_all_models`가 split을
    수행)에 넘겨 validation까지 만들었다 - outlier 판단이 "분할 전" 전체
    데이터셋에 적용됐으므로 test가 될 프레임의 정보가 outlier 판단에 이미
    새어 들어간 뒤였다(전형적인 leakage). 이 함수는 현재 `app/cli.py`/
    `ui/worker.py` 어디서도 호출되지 않는 dead code이지만(테스트에서만 참조),
    나중에 UI가 연결하면 그대로 재발할 수 있는 지뢰라 leak-safe 경로로
    다시 작성한다.

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
    import copy

    from calibration.outlier import recalibrate_with_outlier_pruning
    from calibration.quality import analyze_dataset_quality
    from calibration.frame_quality import compute_frame_quality_scores
    from calibration.models.common import infer_image_size
    from calibration.models.pinhole import calibrate_pinhole
    from calibration.validation import (
        _subset_dataset,
        recalibrate_train_with_outlier_pruning,
        split_train_test,
        validate_holdout,
    )

    # Leak-safe validation이 outlier 판단 이전 상태에서 출발하도록, 아래
    # 전체 데이터 기준 outlier 제거를 적용하기 전에 복사해둔다.
    validation_dataset = copy.deepcopy(dataset)

    # --- 1. 최종 배포용 계산: 전체 데이터 기준 outlier 제거 ---
    ref_result, outlier_result = recalibrate_with_outlier_pruning(
        dataset, camera_config, reference_model,
        max_iterations=max_iterations,
    )

    warnings = analyze_dataset_quality(dataset, camera_config)

    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    calibration_results, _validation_results_leaky, _object_releasing_result, _object_releasing_validation, _ro_comparison = (
        run_models_and_validation(
            dataset, camera_config, pattern_config, test_ratio, model_jobs=2,
        )
    )
    calibration_results[reference_model] = ref_result

    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    # --- 2. Leak-safe Hold-out Validation: 복사본에서 먼저 분할, 그 다음
    #        train 서브셋에만 outlier pruning ---
    train_ids, test_ids = split_train_test(validation_dataset, camera_config, test_ratio)
    _, _val_outlier_result, ref_validation = recalibrate_train_with_outlier_pruning(
        validation_dataset, camera_config, pattern_config, reference_model, train_ids, test_ids,
        max_iterations=max_iterations,
    )
    validation_results = {reference_model: ref_validation}

    # 나머지 모델도 leak-safe하게(같은 train/test 분할, K/D는 각 모델
    # 자신의 train 서브셋에서만 확정) 평가한다 - app/cli.py의 --outlier
    # 흐름과 동일한 패턴(fisheye는 pinhole 초기값으로 발산 방지).
    train_subset = _subset_dataset(validation_dataset, train_ids)
    pinhole_init = calibrate_pinhole(train_subset, camera_config)
    for m in calibration_results:
        if m == reference_model:
            continue
        fisheye_guess = pinhole_init if m == CameraModelType.FISHEYE else None
        validation_results[m] = validate_holdout(
            validation_dataset, camera_config, pattern_config, m, train_ids, test_ids,
            fisheye_initial_guess=fisheye_guess,
        )

    return dataset, ref_result, outlier_result, warnings, calibration_results, validation_results
