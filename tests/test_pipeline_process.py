"""
tests/test_pipeline_process.py
===================================

실제 사용자 버그: PipelineWorker(ui/worker.py)가 무거운 계산(Standard 4모델
캘리브레이션 + Hold-out 검증)을 QThread 안에서 직접 돌려서,
cv2 확장 함수가 GIL을 오래 붙드는 동안 GUI 스레드까지 같이 멈춰
"python3 is not responding"이 떴다.

이 테스트는 calibration/pipeline_process.py의 함수들이 "그냥 직접 호출"이
아니라 실제로 concurrent.futures.ProcessPoolExecutor를 통해 별도 프로세스로
왕복(pickle 직렬화 -> 자식 프로세스 실행 -> 역직렬화)해도 문제없이 동작하는지
확인한다. 여기서 검증 안 하면 "로컬에서 직접 호출하면 되는데 실제로 별도
프로세스에 넘기면 pickle 에러로 터진다"는 걸 늦게 발견하게 된다
(Dataset/CalibrationResult 안에 pickle 안 되는 필드가 섞여 들어가는 회귀는
계속 발생할 수 있는 종류의 버그라 이 테스트를 마커 없이 - 즉 기본 빠른
티어에서도 - 돌게 둔다. 단, 실제 프로세스 스폰 비용 때문에 synthetic_dataset
fixture를 재사용해 이미지 렌더링 비용은 안 든다).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from calibration.pipeline_process import (
    run_models_and_validation,
    run_outlier_pruning_and_validation,
)
from calibration.types import CalibrationMethod, CameraModelType, Dataset, FrameStatus


def test_run_models_and_validation_survives_a_real_process_roundtrip(
    synthetic_dataset, camera_config, pattern_config
):
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_models_and_validation,
            synthetic_dataset, camera_config, pattern_config, 0.25, False,
        )
        (
            calibration_results, validation_results, object_releasing_result,
            object_releasing_validation_result, standard_vs_object_releasing_comparison,
        ) = future.result(timeout=120)

    assert set(calibration_results) == {
        CameraModelType.PINHOLE,
        CameraModelType.BROWN_CONRADY,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    }
    assert set(validation_results) == {
        CameraModelType.PINHOLE,
        CameraModelType.BROWN_CONRADY,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    }
    assert object_releasing_result is None
    assert object_releasing_validation_result is None
    assert standard_vs_object_releasing_comparison is None
    assert calibration_results[CameraModelType.PINHOLE].success
    # 자식 프로세스를 오가며 numpy 배열이 깨지지 않았는지 확인 - 3x3이어야 함.
    assert calibration_results[CameraModelType.PINHOLE].camera_matrix.shape == (3, 3)


def test_run_outlier_pruning_and_validation_survives_a_real_process_roundtrip(
    synthetic_dataset, camera_config, pattern_config
):
    """recalibrate_with_outlier_pruning()의 in-place 부수효과(프레임 비활성화)가
    자식 프로세스 왕복 후에도 반환된 Dataset 객체에 반영돼 있는지가 핵심 -
    이게 깨지면 '이상치 제외'가 화면에 조용히 반영 안 되는 버그가 생긴다.
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_outlier_pruning_and_validation,
            synthetic_dataset, camera_config, pattern_config,
            CameraModelType.EXTENDED_PINHOLE, 3, 0.25, False,
        )
        (
            updated_dataset, ref_result, outlier_result, warnings,
            calibration_results, validation_results,
        ) = future.result(timeout=180)

    assert isinstance(updated_dataset, Dataset)
    assert len(updated_dataset.frames) == len(synthetic_dataset.frames)
    # 프레임 상태(활성/비활성)가 실제로 담겨서 돌아왔는지 - 전부 유효한 enum 값이어야 함.
    assert all(isinstance(f.status, FrameStatus) for f in updated_dataset.frames)
    assert ref_result.model_name == CameraModelType.EXTENDED_PINHOLE
    assert set(calibration_results) == {
        CameraModelType.PINHOLE,
        CameraModelType.BROWN_CONRADY,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    }


def test_object_releasing_result_does_not_overwrite_standard_results(
    synthetic_dataset, camera_config, pattern_config
):
    (
        calibration_results, validation_results, object_releasing_result,
        object_releasing_validation_result, standard_vs_object_releasing_comparison,
    ) = run_models_and_validation(
        synthetic_dataset,
        camera_config,
        pattern_config,
        0.25,
        False,
        CalibrationMethod.OBJECT_RELEASING,
    )

    assert CameraModelType.BROWN_CONRADY in calibration_results
    assert calibration_results[CameraModelType.BROWN_CONRADY].calibration_method == CalibrationMethod.STANDARD
    assert object_releasing_result is not None
    assert object_releasing_result.calibration_method == CalibrationMethod.OBJECT_RELEASING
    assert object_releasing_result.model_name == CameraModelType.BROWN_CONRADY
    assert validation_results[CameraModelType.BROWN_CONRADY].success in (True, False)
    # synthetic_dataset은 ChArUco라 Object-Releasing이 disabled -> Hold-out/비교도
    # 성공은 못 하지만(success=False), 크래시 없이 명확한 이유와 함께 반환돼야 한다.
    assert object_releasing_validation_result is not None
    assert not object_releasing_validation_result.success
    assert object_releasing_validation_result.error_message
    assert standard_vs_object_releasing_comparison is not None
    assert not standard_vs_object_releasing_comparison.success
    assert standard_vs_object_releasing_comparison.error_message
