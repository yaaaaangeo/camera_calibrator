# 기여 가이드 (Contributing)

## 준비

```bash
git clone https://github.com/yaaaaangeo/camera_calibrator.git
cd camera_calibrator
pip install -e ".[dev]"
```

## 테스트 실행

```bash
./run_tests.sh          # 전체 테스트
pytest tests/test_cli.py # 파일 하나만
```

`tests/test_cli.py` 등 rosbag 관련 테스트는 optional dependency인
`rosbags`가 설치되어 있을 때만 실행되고, 없으면 조용히 스킵됩니다
(`pip install -e ".[ros]"`로 설치).

## Lint

```bash
pip install -e ".[dev]"
ruff check .
```

CI에도 동일한 명령으로 도는 별도 `lint` job이 있습니다. `pyproject.toml`의
`[tool.ruff]` 설정은 의도적으로 pyflakes 동급 규칙(`F`: 미사용 import,
미정의 이름 등)만 켜뒀습니다 — import 정렬이나 타입힌트 스타일 같은
opinionated 규칙까지 켜면 이 코드베이스의 기존 컨벤션과 계속 충돌해서
노이즈가 되기 때문입니다. PR을 올리기 전에 로컬에서 한 번 돌려보는
것을 권장합니다.

## 코드 구조

디렉토리 구조 전체는 README의 "아키텍처" 섹션(§4)을 참고하세요. 기여하기
전에 알아두면 좋은 몇 가지 컨벤션입니다:

- **계산 로직(`calibration/`, `export/`)과 UI(`ui/`)를 분리합니다.**
  `ui/*.py`는 계산을 직접 구현하지 않고 `calibration/*.py`의 함수를
  호출만 합니다 (`ui/worker.py`가 그 호출을 QThread로 감쌉니다).
- Import는 `from calibration.detector import ...`처럼 절대 경로로 쓰고,
  relative import는 쓰지 않습니다.

## 릴리스 체크리스트

1. `CHANGELOG.md`의 `[Unreleased]` 섹션에 쌓인 항목들을
   `## [X.Y.Z] — YYYY-MM-DD`로 이름을 바꾸고, 그 위에 새 빈
   `## [Unreleased]` 섹션을 추가하세요. 버전 번호는
   [Semantic Versioning](https://semver.org/)을 따릅니다 — 하위 호환
   기능 추가는 MINOR(`0.X.0`), 버그 수정만 있다면 PATCH(`0.0.X`), 기존
   CLI 플래그/config 스키마를 깨는 변경이면 MAJOR입니다.
2. `pyproject.toml`의 `version`을 같은 값으로 올리세요.
3. `./run_tests.sh`와 `ruff check .`가 통과하는지 확인한 뒤 태그를
   찍으세요.
