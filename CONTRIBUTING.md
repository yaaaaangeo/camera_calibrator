# 기여 가이드 (Contributing)

## 준비

```bash
git clone https://github.com/yaaaaangeo/cam_lidar_eval.git
cd cam_lidar_eval
pip install -e .
```

## 테스트 실행

```bash
./run_tests.sh                       # 전체 테스트
python3 tests/test_edge_alignment.py # 파일 하나만
```

모든 테스트 파일은 pytest 없이도 단독으로 동작하지만, 선호한다면
`pytest tests/`도 그대로 사용할 수 있습니다.

`tests/test_lidar.py`/`tests/test_camera.py`/`tests/test_cli.py`의
rosbag 관련 테스트는 optional dependency인 `rosbags`가 설치되어 있을
때만 실행되고, 없으면 조용히 스킵됩니다 (`pip install -e ".[rosbag]"`로
설치).

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

- **모듈은 단일 네임스페이스 아래로 중첩되지 않은 flat top-level
  패키지**(`input`, `geometry`, `evaluation`, `quality`, `visualization`,
  `report`, `app`)입니다. 새 모듈도 이 구조를 따르세요 — import는
  `from evaluation.edge_alignment import ...`처럼 쓰고, relative import는
  쓰지 않습니다.
- **Metric은 plain dict가 아니라 dataclass를 반환합니다.** JSON/HTML로의
  직렬화는 오직 `report/builder.py`에서만 일어납니다 — 무엇이 리포트에
  실릴 가치가 있는지 판단하고 NaN/Inf를 정리하는 유일한 지점입니다. 다른
  곳에서 metric 결과를 직접 직렬화하지 마세요.
- **모든 threshold는 센서 상대적**이며 `quality.noise_floor`의 배수에서
  유도됩니다 — 새로운 하드코딩된 절대 픽셀 threshold를 도입하지 마세요.
- **새 metric은 known ground truth를 가진 합성 장면으로 검증해야
  합니다.** M2/M3/M4가 하는 방식과 동일하게(`tests/test_edge_alignment.py`의
  `_make_synthetic_scene` 참고): 그냥 크래시 없이 도는지만 확인하지 말고,
  calibration을 흔들었을 때 metric이 *올바른 방향*으로 움직이는지
  assert하세요.
- **Advanced(Phase-5) metric은 절대 `quality_score`에 영향을 주면 안
  됩니다.** 이들은 opt-in 진단용(`--advanced`)이며 MVP 점수 집합에
  포함되지 않습니다.

## 새 metric 추가하기

1. `evaluation/your_metric.py`에 구현하고, 최소한 `classification`
   필드(`"GOOD" | "WARNING" | "BAD" | "FAIL"`)를 가진 dataclass를
   반환하세요.
2. `tests/test_your_metric.py`에 합성 장면 테스트를 추가하세요.
3. MVP 점수에 포함된다면 `quality/quality_score.py`에 연결하세요. 부가
   진단용이라면 `report/builder.py`에 `*_summary()` 함수를 추가하고
   `report/html.py`의 advanced 섹션에 연결하세요 (`plane_consistency_summary`
   패턴 참고).
4. CLI에서 실행 가능해야 한다면 `app/cli.py`에도 연결하세요.

## 릴리스 체크리스트

버전 문자열이 두 곳(`pyproject.toml`의 `[project].version`,
`report/builder.py`의 `TOOL_VERSION`)에 손으로 동기화되어 있어야 하는
값이라 자동으로 안 맞으면 조용히 벌어집니다. 릴리스할 때마다:

1. `CHANGELOG.md`의 `[Unreleased]` 섹션에 쌓인 항목들을
   `## [X.Y.Z] — YYYY-MM-DD`로 이름을 바꾸고, 그 위에 새 빈
   `## [Unreleased]` 섹션을 추가하세요. 버전 번호는
   [Semantic Versioning](https://semver.org/)을 따릅니다 — 하위 호환
   기능 추가는 MINOR(`0.X.0`), 버그 수정만 있다면 PATCH(`0.0.X`), 기존
   CLI 플래그/config 스키마/report 구조를 깨는 변경이면 MAJOR입니다.
2. `pyproject.toml`의 `version`을 같은 값으로 올리세요.
3. `report/builder.py`의 `TOOL_VERSION`을 같은 값으로 올리세요 (이
   값은 `report.json`/`report.html`의 `metadata.tool_version`에 그대로
   실립니다).
4. `./run_tests.sh`와 `ruff check .`가 통과하는지 확인한 뒤 태그를
   찍으세요.
