# P3 구현 및 로컬 검증 보고서

검증일: 2026-11-02. 작업 브랜치는 `windshield`이며 기존 미커밋 P1/P2 변경을 보존했다. 기존 calibration 알고리즘·추천 가중치·P1/P2 Acceptance 규칙과 Output Session은 변경하지 않았다. 코드 저장소 commit/push/deploy와 외부 GitHub/GitLab 게시를 수행하지 않았다. commit/push는 허용된 임시 로컬 Bare 테스트 저장소에서만 실행했다.

## 구현 결과

완료된 Session → 허용 결과 수집/정리 → 불변 Bundle → 로컬 Outbox → Git 게시 → Commit SHA/receipt → 카메라별 이력 조회를 구현했다. 기존 P1 run/P2 live의 **post-export CLI 어댑터**로 연결했다. Publisher 실패는 계산 결과/품질판정/기존 CLI 종료 코드를 변경하지 않으며 재시도는 저장된 Bundle만 사용한다.

신규 모듈은 `automation/publish/`의 `base.py`, `storage.py`, `security.py`, `config.py`, `bundle.py`, `git_publisher.py`, `outbox.py`, `history.py`, `service.py`, `cli.py`, `__init__.py`다. `ResultPublisher.publish(bundle) -> PublishReceipt` 인터페이스와 Git 구현을 분리했다. GitHub/GitLab 공통 Git transport이며 서비스별 API에 의존하지 않는다.

수정한 기존 제품 파일은 `automation/cli.py`, `automation/live/cli.py`의 명시적 게시 옵션/후처리와 `.gitignore`뿐이다. GUI/Runner/수학 모듈은 수정하지 않았다. 신규 테스트는 `_publish_test_utils.py`, `test_publish_bundle.py`, `test_publish_git.py`, `test_publish_outbox_history.py`, `test_publish_cli.py`, `test_publish_e2e.py`다. 사용 안내/설정 예시는 [AUTOMATION_P3.md](AUTOMATION_P3.md), [publisher.example.json](publisher.example.json)에 제공했다.

## 기존 코드 분석과 재사용

- `automation/runner.py`는 완료 Session, run_manifest, JSON/YAML/HTML/Acceptance를 이미 생성한다. 계산 본체를 바꾸지 않고 CLI 반환 직후 게시한다.
- `identity.py`의 식별자 검증을 재사용하고 미확인 S/N을 자동 생성하지 않았다.
- `run_manifest.py`의 provenance/입력 파일 hash/model metric/split/recommendation을 안전한 구조로 추출한다. 출판용 atomic writer는 concurrent temporary filename/fsync와 lock이 필요해 별도로 구현했다.
- `OutputManager` 경로/기존 export serializer는 유지한다. 이미지를 복사하는 `calibration.library`를 원격 업로드에 사용하면 원본 이미지가 따라갈 수 있어 사용하지 않았다.
- `calibration.calibration_io.load_standard_calibration`으로 실제 YAML을 읽어 K/D/해상도를 검증한다. 기존 external_compare의 원칙처럼 조건 호환성을 먼저 검사하고, 재학습 없이 동일 광선의 투영 차이를 보고한다.
- `automation/orchestrator.py`는 없으며 신규 `publish/service.py`를 후처리 경계로 사용했다. P2는 완료된 최종 attempt의 P2 Acceptance를 보존한다.

## Bundle/Content Hash/보안

필수 결과가 누락되면 안전한 **상대 파일명 목록**으로 보고한다. 실패/미완료 실행을 정상 Bundle로 승격하지 않고 로컬 errors 기록만 남긴다. 기존 Session은 읽기 전용이며 staging에서만 복사/정리한다. P2 WARN/FAIL을 P1 PASS로 바꾸지 않는다.

공개 패키지는 sanitized run_manifest/Calibration JSON/Acceptance, 실제 YAML, structured summary와 재생성 HTML이다. 원래 HTML의 상세 그래프/이미지/스크립트는 전달하지 않는다. 원문 HTML을 그대로 게시했다고 주장하지 않으며 원본 해시와 정리 방식을 기록한다. 원본 이미지/bag/project/cache/log/환경변수/개인 경로는 전송하지 않는다.

Content Hash v1은 카메라 식별자, 정렬된 입력 이미지 내용 hash, 실제 수치/품질판정/추천/split/코드버전, YAML K/D/해상도, 좌표계 선언을 canonical JSON으로 SHA256한다. Run ID/시각/실행 시간/로컬 경로는 제외한다. 파일 자체의 SHA256과 의미 기반 Content Hash를 분리했다. 상세 규칙과 제외 키 범주는 사용 문서에 기록했다.

검증 중 Windows checkout이 JSON의 LF를 CRLF로 바꿔 byte hash가 달라지는 문제를 확인했다. staging 텍스트를 LF로 통일하고 검증 시 CRLF/LF 변환만 허용했다. 의미/내용 편집은 계속 거부한다. Git checkout에서의 history 검증 회귀도 포함한다.

파일당10MiB/전체25MiB/32파일, allowlist, 민감 문자열/계정/절대경로 검사, traversal/link/reparse guard, YAML-JSON 수치 일관성 검사를 적용했다. 토큰 설정 필드/URL credential은 거부하며 Git Credential Helper/SSH agent를 사용한다. Git stderr/traceback은 원격 패키지나 오류 기록에 복사하지 않는다. 자동 프라이버시 필터가 임의 자유문장의 개인정보를 완벽하게 식별하는 것은 아니므로 실제 외부 활성화 전 staging 검토가 필요하다.

## Git와 Outbox 검증

매 게시마다 새 임시 저장소만 초기화하고, 설정한 branch 최신 상태를 읽어 불변 Run과 카메라 index를 추가한다. source/user checkout은 바꾸지 않는다. force push 없이 일반 push하며 다른 사용자가 먼저 갱신하면 non-fast-forward로 안전하게 실패하고 새 상태에서 재시도한다. 중복 Content Hash는 기존 Commit SHA를 반환한다. 같은 Run ID의 다른 내용은 덮어쓰지 않는다.

Outbox PENDING/PUBLISHING/PUBLISHED/FAILED/RETRY_WAIT, 지수 backoff/횟수 제한, 인증 실패의 수동 reset, 상태 JSON 원자 교체와 backup/journal 복구를 구현했다. 모든 상태 증거가 사라지면 자동 재시도 대신 수동 검토를 요구한다. Push 직후 프로세스를 중단시키는 시험 후 재시도에서 기존 remote 결과를 찾았고 commit 수가1로 유지됐다.

Index 손상 후에도 immutable Run 스캔으로 이력을 읽고 다음 게시 때 index를 재구축했다. index에는 코드 SHA/content hash/판정/모델/날짜를 저장한다. 실제 게시 SHA는 receipt와 Git path history로 조회한다. 자기 commit SHA를 동일 파일에 넣기 위한 두 번째 commit은 만들지 않는다.

## P1/P2 통합

알려진 K의 합성 이미지24장으로 실제 P1/P2 subprocess CLI를 실행했다. 각 실행은 계산/선별/기존 export 후 새 Publisher로 로컬 Bare 저장소에 게시됐다. 저장된 Git summary의 K/D는 해당 계산 프로젝트와 배열 단위로 일치했다. 두 카메라 S/N(P3-p1/P3-p2)의 별도 이력과 두 개의 실제 게시 Commit SHA를 확인했다. 원본 합성 이미지도 전후 hash가 일치했다.

이 통합 실행은 기본 미설정 운영 기준에 따른 **WARN**이었다. PASS/WARN/FAIL Bundle 보존은 별도 fixture 시험으로 검증했다. 품질FAIL을 실제 수학 실패로 위조하지 않았으며 정상 계산 완료/Acceptance FAIL과 실행 오류는 구분한다. Publisher 오류·disabled 동작은 별도의 CLI 후처리/Outbox 시험에서 계산 종료 코드와 원본을 유지했다.

P2 integration은 기존 연결 시험용 replay profile을 사용하며 일부 수량/공간 기준이 완화돼 있다. P2의 실데이터 준비도/실ROS 검증을 대체하지 않는다. 실제 ROS/물리 카메라, 실제 GitHub/GitLab 인증·push, protected branch/조직 보안정책은 **미검증**이다.

기존 run_manifest가 없는 일반 app.cli/수동 GUI 결과를 카메라/SN으로 임의 추정하여 가져오는 기능은 포함하지 않았다. 기존 수동 GUI는 변경하지 않았고 기존 P1 automation CLI/완료 P2 세션으로 게시한다.

## 이력 비교

모델과 S/N별 경로를 분리하고 대소문자가 다른 식별자도 별도 hash 경로로 저장한다. 이력은 index에 의존하지 않으며 로컬 Outbox 또는 결과 checkout에서 JSON/CLI로 읽는다. 기본은 offline 조회다.

동일 Camera Model/SN, 해상도, distortion model, 명시적으로 확인된 좌표계/ROI일 때만 K/D delta와 동일 광선63개의 projection RMS/P95/max를 제공한다. 조건이 다르거나 ROI가 미확인일 때 수치 비교를 차단한다. 안정성/추천/판정/P95/edge 등 기존 metric도 보존한다. K/D 변화만으로 불량이나 우열을 결정하지 않는다. 기존 출력에 물리 ROI가 없으므로 자동 게시의 기본 좌표계는 unknown이며, 사용자 확인 후 `--publish-full-frame`/`--publish-roi`를 사용해야 비교할 수 있다.

## 실행과 테스트

기록은 `.verification/p3/`에 모았다. 장시간 계산의 stdout/stderr/JUnit을 개별 보존했으며 완료된 calibration은 게시 재검증을 위해 다시 계산하지 않았다. 수정 후 검사에서는 필요한 Bundle/상태/파일 누락 경로만 재확인했다.

```powershell
venv/Scripts/python.exe -m pytest -q tests/test_publish_bundle.py tests/test_publish_git.py tests/test_publish_outbox_history.py tests/test_publish_cli.py --basetemp=.verification/p3/verified --junitxml=.verification/p3/verified.xml
venv/Scripts/python.exe -m pytest -q tests/test_publish_bundle.py::test_required_file_missing_is_not_success --basetemp=.verification/p3/missing-final --junitxml=.verification/p3/missing-final.xml
venv/Scripts/python.exe -m pytest -q tests/test_publish_e2e.py --basetemp=.verification/p3/e2e --junitxml=.verification/p3/e2e.xml
venv/Scripts/python.exe -m pytest -q tests/test_automation_acceptance.py tests/test_automation_profile.py tests/test_automation_runner.py tests/test_automation_e2e.py tests/test_cli_session_dir.py --basetemp=.verification/p3/p1-regression --junitxml=.verification/p3/p1-regression.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_live.py tests/test_p2_readiness.py tests/test_p2_runtime.py tests/test_p2_gui.py tests/test_p2_bag.py tests/test_p2_observability_diagnostics.py tests/test_p2_gate_policy.py tests/test_p2_gate_synthetic.py tests/test_p2_e2e.py --basetemp=.verification/p3/p2-regression --junitxml=.verification/p3/p2-regression.xml
venv/Scripts/python.exe -m pytest -q tests/test_cli.py --basetemp=.verification/p3/cli-regression --junitxml=.verification/p3/cli-regression.xml
venv/Scripts/python.exe -m pytest -q tests/test_responsive_ui.py::test_rows_and_charts_stack_at_narrow_width_without_overlap --junitxml=.verification/p3/gui-known.xml
```

GUI 관련은 `QT_QPA_PLATFORM=offscreen`, `PYTHONUTF8=1`로 실행했다. 기존CLI의 unsupported-pattern strict XFAIL은 최초에 작업 영역 밖 기본 출력 폴더가 sandbox에서 거부되어 exit1을 반환했고 **환경 원인의 XPASS(strict)**가 됐다. 필요한 파일 쓰기 권한을 확보해 해당1건을 재실행하여 원래XFAIL을 확인했다. 최신ID별 결과로 대체하고 환경 문제를 제품 회귀로 잘못 집계하지 않았다. 재실행 증거는 `cli-permissions.xml`이다.

최종 집계는 아래와 같으며 동일ID의 재실행은 중복 합산하지 않는다.

| 구분 | PASS | FAIL | SKIP | XFAIL | 고유 수 |
|---|---:|---:|---:|---:|---:|
| P3 단위/보안/Git/Outbox/history/CLI | 59 | 0 | 0 | 0 | 59 |
| P3 실제 P1/P2계산→로컬Git | 2 | 0 | 0 | 0 | 2 |
| 기존 P1 회귀 | 138 | 0 | 0 | 0 | 138 |
| 기존 P2/GUI/복구/E2E | 89 | 0 | 0 | 0 | 89 |
| 기존 CLI | 28 | 1 | 0 | 1 | 30 |
| 기존 좁은 GUI의 알려진 문제 | 0 | 1 | 0 | 0 | 1 |
| **합계** | **316** | **2** | **0** | **1** | **319** |

P3 신규61개는 모두PASS이고 이번 실행 범위에서 신규 회귀 실패는 확인하지 않았다. 위 표는 이번 부분집합이며 기존 전체1376개 재실행이 아니다. 미실행한 기존 추천 테스트2개와 Spline미완료6개를 통과로 바꾸지 않았다.

P3 전체 흐름 시험2개는204.89초, P1회귀138개는325.26초, P2회귀89개는175.96초에 완료됐다. 여러 그룹을 동시에 실행한 환경의 소요시간이며 처리 성능 보장값은 아니다. 설치본은 저장소 밖 임시 폴더에서 publish/history/run 도움말과 비활성 게시→PENDING→이력 조회를 확인했고 신규 Publisher 모듈이 작업 소스와 일치했다. 증거는 `install-final.log`, `package-check.json`, `final-integrity.json`이다.

재확인한 기존FAIL은 `test_cli_chessboard_does_not_require_marker_size`(합성4장 전모델 실패)와 `test_rows_and_charts_stack_at_narrow_width_without_overlap`(800px 세로배치 실패)다. 이전 P1 안정화에서 HEAD 재현이 확인된 동일 문제다. 기존XFAIL은 `test_cli_unsupported_pattern_type_returns_exit_code_1`이다. 세부 원장/집계는 `test-result-ledger.json`, `test-summary.json`에 기록한다.

## 실제 외부 게시 전 설정 및 한계

외부 결과 저장소를 운영자가 별도로 준비하고, 코드 저장소와 다른 URL/명시적 branch, 최소 권한의 Credential Helper/SSH agent, 공개 가능한 Camera Model/SN/결과, ROI 조건과 보존 정책을 확인해야 한다. 설정 기본값은 disabled이며 이번 작업은 외부 업로드 승인 요청이나 실제 업로드를 포함하지 않는다.

GitLab 전환에는 GitLab 결과 저장소 remote/branch, 해당 credential/SSH 설정 및 **새 Outbox 경로**를 지정한다. Publisher/Bundle/계산 코드를 바꿀 필요가 없다. 기존 queue는 처음 정한 target에 바인딩되므로 대기 항목을 다른 저장소로 암묵 전송하지 않는다.

자동 민감정보 검사가 모든 자유문장/기업 정보/개인정보를 보증하지는 않는다. 자체 token 환경변수 API, 원격 이력 자동동기화, GUI 업로드 패널, 승인/배포 기능은 구현하지 않았다. 임의 코드SHA/운영 S/N을 만들어 채우지 않으며 installed/dirty 실행에서 코드 provenance가 불완전할 수 있다. 대량 이력의 index/scan 성능, 실제 서버 정책/인증/네트워크, 여러 OS의 장기 outbox 부하 시험은 후속 검증 대상이다.

현재 한 Outbox는 단일 writer lock으로 게시/상태 변경을 직렬화한다. 동시 호출의 BUSY는 안전하게 실패할 수 있으며 원본 Session을 유지한 채 기존 결과 게시를 재호출해야 한다. 자동화된 다중 카메라의 고빈도 enqueue/비동기 worker 확장은 후속 과제다. 이미 PUBLISHED인 receipt는 역사적 성공 기록이며, 이후 서버에서 파일/branch를 삭제한 상황까지 자동 동기화해 감사하지 않는다.
