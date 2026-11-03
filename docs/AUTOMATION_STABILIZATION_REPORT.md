# P1-0/P1-1 자동화 안정화 검증 — 2026-11-02

최종 결과 확인: **2026-11-02 17:33 KST**. 이번 최종 정리 단계에서는 애플리케이션 및
테스트 코드를 변경하지 않고 기존 로그·JUnit XML·프로세스 상태를 확인했다.

| 고유 테스트 기준 | 수 |
|---|---:|
| PASS | **1,358** |
| FAIL | **4** |
| SKIP | **4** |
| XFAIL | **4** |
| 미완료 / 결과 미확인 | **6** |
| 합계 | **1,376** |

상태를 확인한 항목은 1,370개다. **전체 테스트 통과 또는 전체 검증 완료로 판정하지
않는다.** 미완료 6개는 성공에 포함하지 않았다. 단일 프로세스 중단 후 분리 실행과
후속 검증을 합친 고유 node ID 집계이며, 실행 횟수의 단순 합계가 아니다.
기계 판독용 집계는 `.verification/test-summary.json`, 각 테스트의 상태·근거 원장은
`.verification/test-result-ledger.json`에 있다.

## 작업 범위와 변경 보존

`windshield` 브랜치에서 시작했다. 시작 시 이미 수정되어 있던 파일은
`app/cli.py`, `calibration/image_quality.py`, `calibration/image_selection.py`,
`calibration/validation.py`, `docs/OUTPUT_SESSIONS.md`, `export/output_manager.py`,
`pyproject.toml`이다. `automation/`, `docs/AUTOMATION.md`, 자동화 관련 테스트
8개 파일도 미추적 상태였다. 이 구현을 보존하고 검토·검증했다.
커밋, 푸시, 배포는 하지 않았다. 검증 결과는 기존 출력 폴더와 분리된
`.verification/`에 저장했다.

기존 미커밋 구현은 정식 `--session-dir`, 자동화 패키징 및 내장 프로파일,
설정 가능한 패턴 기준 노출 검사, 판정 사유 분류, OpenCV 스레드 설정,
일부 모델만 계산할 때의 검증 표 출력을 포함하고 있었다. 동적 OutputManager
교체용 `automation/cli_child.py`는 현재 작업 트리에 없으며, runner는 기존 CLI를
자식 프로세스로 실행한다.

최종 작업 트리의 기존 추적 파일 수정 목록은 위 7개다. 미추적 소스/문서 목록은
아래와 같다(이 보고서 이외에는 작업 시작 시 이미 존재했다).

- `automation/__init__.py`, `automation/__main__.py`, `automation/cli.py`,
  `automation/runner.py`, `automation/profile.py`, `automation/acceptance.py`,
  `automation/identity.py`, `automation/run_manifest.py`, `automation/profiles/default.yaml`
- `tests/_automation_test_utils.py`, `tests/test_automation_acceptance.py`,
  `tests/test_automation_e2e.py`, `tests/test_automation_profile.py`,
  `tests/test_automation_runner.py`, `tests/test_cli_session_dir.py`,
  `tests/test_image_selection_exposure.py`, `tests/test_validation_format_subset.py`
- `docs/AUTOMATION.md`, `docs/AUTOMATION_STABILIZATION_REPORT.md`

검증 전용 신규 파일은 `.verification/audit.py`, `.verification/package_probe.py`,
`.verification/finish_suite.py`, `.verification/summarize_tests.py`,
로그·JUnit XML·설치본·분리된 출력 Session들이다. 배포 소스에 포함하지 않는다.

이번 검토에서 추가로 수정한 부분:

| 파일 | 발견한 문제와 보완 |
|---|---|
| `app/cli.py` | 기존 결과 검사 중 권한 오류도 `--session-dir` 오류로 처리한다. 기본 Session 생성의 OSError도 traceback 대신 오류 메시지와 종료 코드 1로 반환한다. 정상 실행 경로·이름 규칙은 유지한다. |
| `export/output_manager.py` | 예약된 출력 디렉터리 자리에 파일·심볼릭 링크·Windows junction이 있으면 결과 충돌로 거부한다. 링크가 다른 결과나 원본 폴더로 출력을 우회시키는 것을 막는다. |
| `automation/acceptance.py` | NaN/무한대 지표를 누락으로 취급하고, 필수 지표 누락이면 `criteria_complete=false`로 기록한다. 추천 점수와 운영 임계값은 변경하지 않았다. |
| `automation/profile.py` | 노출/선별 설정 및 timeout의 비유한 숫자와 1을 초과한 포화 비율 설정을 사전 거부한다. |
| `tests/test_cli_session_dir.py`, `tests/test_automation_acceptance.py`, `tests/test_automation_profile.py` | 위 실패 경로와 원본 보존 검증을 추가했다. |
| 이 보고서 | 새 문서. 실행 증거와 기존 실패 분류를 기록한다. |

`--output-dir`의 flat 출력과 기본 시간 기반 Session 생성은 유지한다.
`--session-dir`는 명시한 폴더에 기존 Session 구조를 생성하며 기존 결과나 claim이
있으면 종료 코드 1로 거부한다. 자동화는 원자적 mkdir로 새 세션 이름을 예약하고
해당 경로를 CLI에 전달한다. 보호 claim은 실행 후에도 남으므로 실패했던 Session을
재사용하지 말고 새 경로를 사용한다. 기존 CLI의 명시적 `--save-project`,
`--json-summary`, `--log-file` 경로 계약은 유지되므로 수동 재실행 시 이 경로들도
새 위치로 지정해야 한다.

## 설치 및 실행 검증

Windows / Python 3.11 / OpenCV 5.0.0에서 실행했다. 일반 설치와 editable 설치를
각각 `.verification/installed`, `.verification/editable`로 격리했다.
`pip install . --no-deps --no-build-isolation --target ...` 및 대응하는 `-e .`를
사용했다. 기존 환경의 의존성을 사용했으며, clean 환경에서 전체 의존성을
다운로드하는 검증은 하지 않았다. 빌드 도구 wheel만 별도 검증 폴더에 설치했다.

두 설치 모두 저장소 밖 임시 cwd에서 import, 내장 `default` 프로파일 로딩,
`automation check`, 합성 720p 24장 `automation run`을 통과했다.
일반 설치는 `python -m automation`, editable target은 `.pth`를
`site.addsitedir`로 처리한 뒤 같은 모듈을 `runpy`로 실행했다.
두 run은 완료 상태이며 기본 운영 임계값 미정으로 WARN(종료 코드 3)이다.
`app.cli`와 `ui` import도 확인했다. 근거: `.verification/package-results.json`.

`automation*` 패키지 검색 및 `profiles/*.yaml` package-data 설정이 일반 wheel에도
포함된다. `importlib.resources`와 `pathlib`, `os.pathsep`를 사용하므로 저장소 cwd에
의존하지 않는다. Ubuntu/Jetson 장비에서의 설치·실행은 미실행이며 Windows 결과를
다른 OS의 실측 결과로 간주하지 않는다.

## 실제 이미지 및 재현성

저장소 `image/`의 실제 촬영 JPEG 55장을 사용했다. 보드 사양 기록이 없어 기존
문서와 검출 결과에 따라 10×7 ChArUco / DICT_5X5_100, 칸 0.04 m·마커 0.03 m를
가정했다. 물리적 치수와 마커/칸 비율 확인이 필요하며 이 결과는 그 가정에 조건부다.
촬영 장비 연결과 새로운 실촬영, GUI 조작을 통한 비교는 미실행이다.

| 보드 항목 | 검증에 사용한 값 | 확인 수준 |
|---|---|---|
| 패턴 | ChArUco, **10×7 칸(squares)** | 기존 문서·마커 ID 분포(0~34) 및 검출 결과로 추정. 실물 사양 확인 전. |
| Dictionary | **DICT_5X5_100** | 해당 설정에서 55/55장 검출. 실물 제조 사양서 확인 전. |
| 칸 크기(square_size) | **0.04 m (40 mm)** | 실제 치수 기록이 없어 가정. |
| 마커 크기(marker_size) | **0.03 m (30 mm)** | 실제 치수 기록이 없어 가정. 마커/칸 비율 **0.75**. |

칸과 마커를 같은 비율로 확대·축소하는 이상적인 경우 K/D는 균일한 보드 크기
배율의 영향을 받지 않지만, 이동 벡터·깊이 등의 물리 단위는 달라진다. 실제 마커/칸
비율이 가정과 다르면 검출·보간 좌표와 보정 결과에도 영향이 있을 수 있다.
따라서 이번 수치는 **가정한 보드 설정에서의 검출 성공, 실행 재현성 및 CLI 일치성**의
증거이며, 실제 K/D의 정답 정확도나 운영 품질 합격의 증거로 확대 해석하지 않는다.
카메라 모델명·serial도 실제 장비 식별 정보가 없어 검증용 `AUDIT` / `REAL`을 썼다.

검출 55/55, 선별 35/55, 중복 장면 제외 20장이다. 검출 실패·blur 제외·노출 제외는
각각 0장이다. `global`과 `pattern` 모드의 선택 이미지 집합이 정확히 같다.
실제 제외 경로는 검출 실패, Laplacian 분산 기반 blur, 선택 모드의 노출 검사,
pHash/장면 유사도 기반 중복 제거, 목표 수·위치/거리/기울기 분포 선별이다.
`image_quality.py`의 motion blur 방향성 지표와 전역 대비 표준편차는 WARNING으로
제공되며 `image_selection.py`의 독립적인 강제 제외 조건은 아니다.
pattern 모드도 코너 주변 대비/선명도나 가장자리 코너 위치 오차를 직접 평가하지는
않는다. 기존 검출 성공과 선명도 검사를 유지하고 코너 주변 포화·보드 ROI 밝기를
추가로 사용한다. 향후 개선 시 코너별 국소 지표와 촬영 정답 데이터가 필요하다.
밝은 배경 합성 이미지에서 기존 global의 정상 패턴 제외와 pattern의 유지,
gain을 가한 실제 이미지의 과노출 제외는 이미지 노출 테스트로 확인했다.
원본 55장은 검증 전후 SHA-256이 모두 동일하다.

단일 스레드(`opencv_threads=1`), seed=42, test_ratio=0.25로 자동화 2회와 직접 CLI
1회를 실행했다. 자동화 반복 간 모든 모델의 K/D 및 Train/Test/Edge RMS, 실제 분할,
추천 결과가 정확히 같았다. 직접 CLI와 자동화의 모델 요약도 정확히 같다.
Train 27장 / Test 8장, 누수 없음, 모든 모델이 동일한 분할을 사용한다.

| 모델 | fx / fy | 전체 학습 RMS | Hold-out 학습 RMS | Test RMS | Edge RMS |
|---|---|---:|---:|---:|---:|
| Pinhole | 2998.989 / 2912.649 | 33.852197 | 31.629171 | 44.229990 | 50.180924 |
| Brown-Conrady | 1899.851 / 1908.084 | 2.600584 | 2.308475 | 4.271152 | 3.662548 |
| Rational | 2303.070 / 2309.792 | 7.863552 | 4.284946 | 6.221400 | 6.397316 |
| Fisheye | 1893.700 / 1899.533 | 1.885608 | 1.292665 | 4.161213 | 3.334465 |

RMS 단위는 px이다. 반올림하지 않은 K/D 전체와 각 실행 산출물은
`.verification/real/audit.json` 및 그 안에 기록된 Session 경로에 있다.
추천은 모두 Brown-Conrady이며 판정은 WARN이다. 사유는 Train RMS Sanity WARNING,
기존 POOR 등급·REJECT 신뢰도, 필수 운영 기준 미설정이다. 계산 성공과 품질 합격은
별개로 유지된다. 운영 RMS 기준을 새로 정하지 않았다.

기존 Sanity ERROR는 물리적으로 무효인 결과로 FAIL, Sanity WARNING은 WARN,
Diagnosis ERROR는 촬영/데이터 보완 권고를 포함하므로 기본 WARN, Diagnosis WARNING은
기록, Hold-out 근거 부족 및 LOW/REJECT 신뢰도는 WARN으로 유지한다.
필수 지표 누락은 FAIL이며 `missing_metric` 사유를 기록한다. 계산 자체 실패는
`calculation_failure`, 임계값 미달은 `quality`로 구별한다.

기본 스레드 수에서의 미세 변동은 고정 seed만으로 제거되지 않는다. 이 검증에서는
계산 알고리즘을 바꾸지 않고 기존 단일 스레드 옵션을 확인했다. 합성 E2E의 기존
허용오차(K/D rtol=1e-5, atol=1e-6; RMS rel=1e-4)는 기존 문서의 기본 스레드 실측에
근거한 값으로 유지했다. 실제 데이터 비교는 허용오차를 늘리지 않고 정확히 비교했다.

## 성능과 캐시 판단

이번 안전성 보완은 계산/검출 최적화를 포함하지 않으므로 개선 전후 속도 향상을
주장하지 않는다. 전체 테스트 및 설치 검증과 동시 실행한 측정이라 독점 실행 성능과
직접 비교할 수 없다.
동일 프로세스의 기존 이미지 전처리 캐시가 warm 상태일 수도 있으므로 각 구간 시간을
독립된 cold-start 측정이나 개선 전후 A/B 결과로 해석하지 않는다.

실제 이미지 6장을 해상도별로 복사·축소한 순차 검출 시간:

| 해상도 | 6장 검출 | 장당 |
|---|---:|---:|
| 1280×720 | 0.482초 | 80 ms |
| 1920×1080 | 0.813초 | 136 ms |
| 3840×2160 | 2.662초 | 444 ms |

실제 원본 55장 단독 검출 구간은 24.451초였다. 검출된 동일 dataset을 사용한 선별
구간은 global 0.006초, pattern 2.901초였다. pattern은 코너 주변 지표 계산을 위해
이미지를 다시 읽는 비용이 있다. 자동화 첫 실행 전체는 89.913초이며 CLI 단계가
74.30초로 가장 크다. 이 단계에는 35장 재검출, 모델 계산, Hold-out, bootstrap 및
export가 포함되므로 시간 전체를 중복 검출 비용으로 해석하면 안 된다.

중복 검출 경로는 `automation.runner._run_image_selection → detect_dataset`과
`app.cli`의 이미지 입력 검출이다. 모델 결과 캐시는 검출 캐시와 다르다.
프로세스 간 검출 캐시는 추가하지 않았다. 이미지 해시·전체 Pattern 설정·검출 코드와
OpenCV 버전·전처리 옵션을 검증하고 무효 캐시를 재검출하는 공개 입력 계약이 필요하다.
현 단계에서 검출 품질/결과 일치성을 유지할 만큼 이 경로를 확장하는 것보다
기존 CLI 경로를 그대로 유지하는 편이 변경 범위에 적합하다.

## 테스트 결과

### Spline 최종 확인

| 모듈 | 결과 | 완료 여부와 근거 |
|---|---|---|
| `tests/test_windshield_spline.py` | **9 PASS / 9개** | 정상 종료 요약과 `spline-extra.xml` 확인. `spline-extra.log` 기준 2,056.26초(34분 16초). |
| `tests/test_windshield_spline_stabilization.py` | **7 PASS / 6 미완료 / 13개** | `spline-stabilization-extra.log`의 명시적 PASSED 7개만 반영. 최종 XML·종료 요약 없음. 최종 확인 시 실행 중인 Python 프로세스가 없고 기존 실행 세션도 조회 불가. 종료 원인·종료 코드는 확인할 수 없음. **모듈 전체 성공으로 처리하지 않음.** |

위 근거 파일은 모두 `.verification/`에 있다. 안정화에서 확인한 통과 7개는
geometry/normal 3개, Stage B K/D 보존, 최종 stage 기록, pose 범위,
Test 데이터 오염 시 학습 결과 불변 검사다. XML이 없더라도 개별 PASSED 로그가
명확한 이 7개만 통과에 포함했다.

다음 6개는 모두 `tests/test_windshield_spline_stabilization.py`의 **미완료/미검증**
항목이다. 시작 여부와 관계없이 완료 결과가 확인되지 않았다는 뜻이며, 실패나 통과로
추정하지 않는다.

1. `test_auto_grid_selection_does_not_leak_outer_test_set`
2. `test_select_best_spline_grid_resolution_keeps_base_intrinsics_immutable`
3. `test_diagnostics_orchestrator_populates_all_diag_fields`
4. `test_repeated_holdout_reports_stability_fields_when_successful`
5. `test_run_repeated_holdout_spline_keeps_base_intrinsics_immutable`
6. `test_diagnostics_orchestrator_repeated_holdout_never_sees_outer_test_ids`

현재 실행 중인 테스트가 없으므로 이번 최종 정리에서는 장기 테스트를 새로 시작하지
않았다. 완료된 기본 9개와 안정화 통과 7개를 불필요하게 재실행하지 않았다.

### 단일 프로세스 실행과 분리 실행

| 실행 구분 | 결과 | 해석 |
|---|---|---|
| 최초 전체 실행(slow 제외 없음, 당시 1,358개 수집) | 출력 확인 790개: **PASS 774 / FAIL 10 / SKIP 3 / XFAIL 3** | 최종 JUnit XML 생성 전 프로세스 종료. 나머지 568개 결과 미확인. 단일 프로세스 전체 완료가 아님. |
| 나머지 568개의 파일별 분리·Spline 별도 실행 | **PASS 560 / FAIL 0 / SKIP 1 / XFAIL 1 / 미완료 6** | `remaining/*.xml`, `remaining/results.json`, Spline 로그/XML 사용. Spline 통과 7개는 부분 로그 근거. |
| 후속 최종 집중 검증 | **111 PASS** | `final-cli-permissions.xml`. 재검사와 신규 항목을 포함하므로 111개를 전체 수에 다시 더하지 않음. |
| 고유 테스트 최종 집계 | **PASS 1,358 / FAIL 4 / SKIP 4 / XFAIL 4 / 미완료 6 = 1,376** | 최초 1,358개 + 신규 테스트 18개. 동일 node ID는 유효한 후속 결과로 갱신하며 중복 합산하지 않음. ERROR/XPASS는 각각 0. |

최초 실패 10개 중 기본 CLI Session 생성 권한 오류 4개는 오류 처리 보완 후
**원래 제한 환경에서도 통과**했다(`final-cli-permissions.xml`). 사용자 홈 쓰기
권한을 확대한 결과가 아니다. Jetson bash 문법 검사 2개는 접근 불가능한
WindowsApps bash 별칭이 원인이었고, 설치된 Git Bash를 PATH 앞에 둔 재검사에서
통과했다(`environment-recheck.xml`). 이는 문법 검사이며 실제 Jetson 설치·실행
검증을 의미하지 않는다. 남은 실패 4개는 아래와 같다.

최초 프로세스의 정확한 종료 원인은 확인하지 못했다. 이어질 예정이던 GUI 스레드
전달 테스트는 파일별 실행에서 통과했다. 변경 전 `HEAD`의 responsive UI →
ROS/rosbag 묶음 연속 실행에서도 **Windows fatal exception 0xc0000374 (heap
corruption)**로 종료됐다(`baseline-gui-sequence.log`). 당시 stack은
`ui/main_window.py::_build_settings_panel` 및
`test_main_camera_settings_stack_and_scroll_at_800x600`을 가리켰다.
최초 실행과 종료 지점이 달라 같은 원인이라고 단정하지 않는다. 기존 GUI 연속 실행의
네이티브 불안정성이 재현된 것이며, 파일별 통과가 단일 프로세스 안정성을 보증하지 않는다.

### 기존 실패 4개: 원인·재현·영향

아래 원인은 로그로 확인한 **직접 실패 조건**이다. 최적화·모델 식별성 등 더 깊은
근본 원인을 모두 규명했다는 뜻은 아니다. 가중치·수학 알고리즘·테스트 기대값은
변경하지 않았다.

| 테스트명 | 확인된 직접 원인 | 변경 전 `HEAD` 재현 | 기능 영향도 |
|---|---|---|---|
| `tests/test_cli.py::test_cli_chessboard_does_not_require_marker_size` | 합성 체스보드 4장에서 모든 모델 계산/추천 실패. 기대한 종료 코드 0 대신 2. 상세 최적화 실패 원인은 미확정. | **동일 실패 재현**. `baseline-failures.xml/.log`, `environment-recheck.log`. | **계산 경로 영향**: 이 적은 프레임의 체스보드 입력에서 CLI 완료를 보장하지 못함. ChArUco 55장 자동화 실패나 체스보드 전체 미지원을 뜻하지 않음. |
| `tests/test_recommender_accuracy.py::test_recommender_picks_extended_pinhole_when_distortion_is_clean` | 노이즈 없는 Extended/Rational 합성 시나리오에서 테스트 helper의 최저 점수 모델이 기대한 `extended_pinhole` 대신 `brown_conrady`. 순위 차이의 근본 원인 분리는 미완료. | **동일 선택 불일치 재현**. `baseline-accuracy-complete.xml/.log`. | **모델 식별 정확도 영향**: 해당 합성 시나리오의 기대 모델을 고르지 못함. 계산·저장 성공과 자동화/CLI 수치 일치와는 별개의 품질 문제. |
| `tests/test_recommender_accuracy.py::test_recommender_majority_correct_under_noise[true_fisheye_noisy]` | 시드 1/2/3에서 fisheye 정답 0/3(기대 ≥2/3). 실제 선택 Rational, Brown, Brown. 노이즈·모델 적합·점수의 근본 원인 분리는 미완료. | **동일 0/3 실패 재현**. `baseline-accuracy-complete.xml/.log`(해당 2개 총 2 failed). | **노이즈 조건 추천 신뢰성 영향**: 해당 시나리오의 모델 추천 정확도 한계. 실제 55장에는 정답 모델이 없어 그 추천의 정확성을 보증하지 못함. |
| `tests/test_responsive_ui.py::test_rows_and_charts_stack_at_narrow_width_without_overlap` | 800px 창에서 `ResponsiveRow`가 기대한 TopToBottom 대신 LeftToRight 유지. | **동일 assertion 재현**. `baseline-ui.xml/.log`, 현재 `ui.xml/.log`. | **좁은 GUI 화면 배치 영향**: 컨트롤 stacking·겹침 방지 기대 미충족. 헤드리스 CLI 계산 및 별도로 관찰한 네이티브 종료와 구분. |

위 4개는 이번 변경으로 새로 생긴 실패로 분류하지 않는다. 문제가 없거나 해결됐다는
뜻도 아니다. 현재 환경의 기존 결함·정확도 한계로 남긴다.

### SKIP / XFAIL

SKIP 4개:

- `tests/test_bootstrap.py::TestComputeParameterBootstrap::test_fisheye_dispatch_uses_fisheye_calibrate`: 합성 데이터의 fisheye 초기화 실패 시 Skip하는 기존 환경 의존 조건.
- `tests/test_neural_torch_lazy_import.py::test_calling_require_torch_loads_torch_lazily`: PyTorch 미설치.
- `tests/test_reflection_suppression_torch_lazy_import.py::test_calling_require_torch_loads_torch_lazily`: PyTorch 미설치.
- `tests/test_windshield_project_io.py::test_real_trained_neural_result_state_dict_round_trips_through_project`: PyTorch 미설치.

XFAIL 4개(기존 표시 유지):

- `tests/test_cli.py::test_cli_unsupported_pattern_type_returns_exit_code_1`
- `tests/test_recommender_accuracy.py::test_recommender_majority_correct_under_noise[true_extended_noisy]`
- `tests/test_recommender_accuracy.py::test_recommender_struggles_with_mild_distortion_and_low_data`
- `tests/test_windshield_ghost.py::test_regular_led_grid_large_radius_with_smooth_brightness_gradient_is_a_known_limitation`

실제 neural 학습 검증은 완료한 것으로 보지 않는다. Ubuntu/Jetson 장비 실행,
카메라 연결 및 GUI 수동 비교도 미검증이다. 기존 실패를 숨기기 위해 테스트를 삭제하거나
완화하거나 새 XFAIL/Skip으로 바꾸지 않았다.

### 장기 테스트 로그와 중복 실행 방지

이번 실행에는 먼저 시작한 별도 Spline 실행과 파일별 큐가 중복된 구간이 있었다.
뒤늦게 시작한 큐의 중복 실행은 정리했고 먼저 시작한 실행의 결과를 사용했다.
중복 결과를 고유 테스트 수에 더하지 않았다. 이 낭비가 없었다고 주장하지 않으며,
다음 실행부터 아래 원칙을 적용한다.

1. 시작 전 `test-result-ledger.json`에서 node ID별 상태와 근거를 확인한다.
2. 동일 테스트/모듈이 실행 중이면 새 실행을 만들지 않고 기존 실행 결과를 수집한다.
3. 장기 실행마다 고유 실행 ID, 개별 stdout/stderr 로그, JUnit XML, 선택 node ID, 시작/종료 시각·종료 코드를 남긴다. XML이 없어도 부분 로그를 보존한다.
4. 새 변경·실패 조사 등 명시적인 재검증 사유 없이 PASS/SKIP/XFAIL을 다시 실행하지 않는다. 현재 재개 대상은 위 **미완료 6개**뿐이다. 공유 fixture 재계산과 완료된 테스트의 중복 실행은 구분한다.
5. 기존 실행 스크립트를 그대로 다시 돌리면 별도 실행 결과까지 자동 중복 제거한다고 보장할 수 없다. 이번 최종 정리에서는 실행기를 수정하지 않았으며 결과 원장을 기준으로 대상을 선별해야 한다.


## 남은 검증

실제 보드 사양 확인, 센서 포화·블루밍·HDR을 포함한 실촬영 과노출/모션 블러 검증,
가장자리 코너 정밀도 검증, Ubuntu 및 Jetson 실행, 카메라 연결 및 GUI 수동 비교가
필요하다. pattern 노출 지표는 밝은 배경의 오제외를 줄이는 휴리스틱이며 실제 센서의
코너 위치 정확도를 보증하지 않는다. 중앙값 기반이므로 일부 코너의 국소 손상을
놓칠 수 있다. 선택 모드가 다른 모델 결과를 직접 비교하면 공정하지 않으므로
동일 선택 이미지 집합과 분할을 사용해야 한다.

2차 자동화 확장 전에는 기존 GUI 연속 실행의 heap corruption과 반응형 UI 실패,
체스보드/추천 정확도 실패를 별도 재현·수정하고, 실제 보드 사양과 운영 품질 기준을
확정해야 한다. 실제 데이터에서 Rational의 학습 RMS가 Brown보다 높은 결과는
좋은 해에 수렴했음을 보증하지 못하므로 단일 스레드 재현성과 계산 품질을 혼동하면
안 된다. 검출 캐시의 공개 입력 계약·무효화 검증은 별도 후속 과제로 남긴다.
