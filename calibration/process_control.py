"""
camera_calibrator.calibration.process_control
==================================================

사용자가 "취소"를 눌렀을 때 이미 실행 중인 ProcessPoolExecutor 워커
프로세스를 즉시 강제 종료하기 위한 공용 헬퍼.

concurrent.futures.ProcessPoolExecutor에는 "지금 실행 중인 작업"을 멈추는
공개 API가 없다 (3.9+의 cancel_futures=True도 아직 시작 안 한 작업만
취소한다 - cv2.calibrateCamera처럼 이미 시작된 C++ 호출은 그대로 끝까지
돈다). 내부 프로세스 목록(_processes)에 직접 접근해 terminate()하는 건
비공개 API에 기대는 것이지만, 커뮤니티에서 널리 쓰이는 방식이고 이 프로젝트의
워커 프로세스들은 공유 상태 없이 순수 계산만 하므로(detector.py의 코너 검출,
pipeline_process.py의 모델 계산) 중간에 죽여도 부작용이 없다.
"""

from __future__ import annotations


class PipelineCancelled(Exception):
    """사용자가 검출/캘리브레이션 실행 중 취소를 요청했을 때 발생시킨다."""


def terminate_executor_processes(executor) -> None:
    """executor가 띄운 자식 프로세스를 전부 즉시 강제 종료한다."""
    for process in list(getattr(executor, "_processes", {}).values()):
        if process.is_alive():
            process.terminate()
