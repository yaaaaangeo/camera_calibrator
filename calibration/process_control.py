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

import multiprocessing as mp
import time


# Linux의 ProcessPoolExecutor 기본 시작 방식은 fork다. Qt GUI가 이미 QThread,
# OpenCV 내부 thread, concurrent.futures 관리 thread를 만든 뒤 fork하면 자식이
# 부모의 잠긴 mutex만 복사해 영원히 멈출 수 있다. 특히 자동 저장 프로젝트를
# 복구하면 Preview/Scene Quality 등 여러 Qt/OpenCV 객체가 먼저 초기화되므로
# 이 문제가 훨씬 쉽게 재현된다. spawn은 깨끗한 interpreter에서 worker를
# 시작하므로 GUI에서 실행하는 모든 process pool이 이 context를 공유해야 한다.
_PROCESS_POOL_CONTEXT = mp.get_context("spawn")


class PipelineCancelled(Exception):
    """사용자가 검출/캘리브레이션 실행 중 취소를 요청했을 때 발생시킨다."""


def safe_process_pool_context():
    """Qt/QThread 안에서도 안전한 ProcessPoolExecutor용 multiprocessing context."""
    return _PROCESS_POOL_CONTEXT


def terminate_executor_processes(executor) -> None:
    """executor가 띄운 자식 프로세스를 전부 즉시 강제 종료한다."""
    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    # terminate()는 signal만 보낸다. 여기서 짧게 회수하지 않으면 interpreter
    # 종료 시 multiprocessing atexit가 같은 process.join()에서 멈추고,
    # 그 사이 Qt가 살아 있는 QThread를 파괴해 abort할 수 있다.
    terminate_deadline = time.monotonic() + 1.0
    for process in processes:
        process.join(timeout=max(0.0, terminate_deadline - time.monotonic()))

    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        if hasattr(process, "kill"):
            process.kill()
    kill_deadline = time.monotonic() + 1.0
    for process in survivors:
        process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
