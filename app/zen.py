"""
camera_calibrator.app.zen
=============================

이스터에그: "The Zen of Python"(`import this`)을 패러디한
"The Zen of Camera Calibration".

여기 적힌 문구들은 전부 지어낸 게 아니라, 이 프로젝트가 실제로 겪은 설계
결정/버그/원칙에서 그대로 가져왔다 (README, docstring, 커밋 히스토리 참고).
`app/main.py`가 시작할 때마다 이 중 하나를 무작위로 골라 터미널에 출력한다 -
캘리브레이션 실행과는 완전히 무관한, 순수한 재미용 장치다.
"""

from __future__ import annotations

import random

ZEN_LINES: list[str] = [
    "RMS가 가장 낮다고 정답은 아니다 — 근거 없는 확신보다 근거 있는 의심이 낫다.",
    "Fisheye는 초기값을 주지 않으면 발산한다 — 좋은 시작점 없이 좋은 결과를 바라지 마라.",
    "opencv-python과 opencv-contrib-python을 같이 깔지 마라 — 같은 이름을 쓰는 둘은 결국 하나만 산다.",
    "체스보드는 방향을 기억하지 못한다 — 대칭은 아름답지만 모호함을 낳는다.",
    "파일을 지우지 마라, 상태만 바꿔라 — 지운 것은 되돌릴 수 없지만 상태는 언제든 되돌릴 수 있다.",
    "장수보다 자세 다양성이 중요하다 — 같은 각도로 백 번 찍느니 다른 각도로 열 번 찍어라.",
    "cv2.aruco.CharucoDetector는 pickle이 안 된다 — 모든 것이 나눠지진 않는다.",
    "Edge RMS가 N/A라면, 보드가 중앙에만 머물렀다는 뜻이다 — 가장자리를 보지 않으면 가장자리를 알 수 없다.",
    "OpenCV 5.0은 fisheye 플래그의 자리를 옮겼다 — 이름이 같아도 주소는 바뀔 수 있다.",
    "3장으로도 캘리브레이션은 되지만, 그것이 좋다는 뜻은 아니다.",
    "붓스트랩은, 모른다는 사실조차 모르는 것보다는 낫다.",
    "좌측 상단이 비어있다면 좌측 상단을 채워라 — 평균은 거짓말을 한다.",
    "커버리지는 균등해야 하고, 자세는 다양해야 하고, 판단은 근거가 있어야 한다.",
    "이미지를 삭제한 적은 한 번도 없다 — 그래서 원본이 사라져도 프로젝트는 살아남는다.",
    "검출 실패는 상태일 뿐, 죄가 아니다.",
    "calibrateCameraExtended는 공짜로 주지만, cv2.fisheye는 아무것도 공짜로 주지 않는다.",
    "병렬로 돌리되, 순서는 지켜라.",
    "조용히 실패하는 것보다 시끄럽게 로그를 남기는 것이 낫다.",
]


def random_zen_line() -> str:
    return random.choice(ZEN_LINES)


def print_zen_greeting() -> None:
    """터미널에 '🐑: <문구>' 한 줄을 출력한다. GUI/캘리브레이션 로직과
    완전히 무관 - 실패해도(예: 콘솔이 없는 환경) 앱 실행에 영향이 없도록
    조용히 무시한다.
    """
    try:
        print(f"🐑: {random_zen_line()}")
    except Exception:  # noqa: BLE001 - 이스터에그 때문에 앱이 죽으면 안 됨
        pass
