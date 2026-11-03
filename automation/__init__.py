"""Headless intrinsic calibration automation layer.

이 패키지는 기존 계산 코드(calibration/, export/, app/cli.py)를 바꾸지 않고,
그 위에서 "이미지 폴더 → 검출 → 이미지 선별 → 4모델 계산 → Hold-out →
추천 → 품질 판정 → 결과 저장"을 한 번의 명령으로 실행하는 제어 계층이다.

계산은 항상 기존 ``app.cli.main()``이 자식 프로세스에서 수행하고, 이 패키지는
입력 검증·실행 기록(run_manifest)·합격 판정(acceptance)만 담당한다.
"""

AUTOMATION_SCHEMA_VERSION = 1
