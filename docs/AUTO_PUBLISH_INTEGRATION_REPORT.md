# 계산 완료 후 자동 게시 통합 검증

## 결과

기존 `automation experiment run`에 ROSbag → P2 → P1 → Export → Bundle → Publisher 연결이 이미 있었다. 새 계산/게시 파이프라인을 만들지 않았다. 합성 ROS2 bag으로 명령 한 번에 계산부터 **로컬 Bare Git 게시까지** 완료했다. 신규 결과의 GitHub Push는 별도 승인 대기이며 아직 수행하지 않았다.

고유 테스트 최종 결과: **315 PASS / 0 FAIL / 0 SKIP / 0 XFAIL**. 신규 테스트 8개 포함. 별도 장시간 MCAP 계산 테스트 1개는 중복 계산을 피하려고 선택 제외했다. 전체 프로젝트 테스트 수는 아니다. JUnit을 합쳐 중복을 제거한 근거는 `.verification/ap/test-ledger.json`이다.

초기 자동 재시도 재현 테스트 3개는 변경 전 실패했다. 추가한 게시 순서 검증에서는 P1 상태가 소문자 `completed`인데 대문자를 기대한 테스트 오류 1개를 수정했고, 완료된 합성 결과를 재사용하여 검증했다. 합성 bag의 Calibration은 한 번만 실행했다. 이전 결과/로그를 덮어쓰지 않도록 이번 실행 로그 경로를 분리했다.

## 기존 실행 흐름과 안전 경계

1. `automation/experiment/config.py`: YAML 설정과 identity/input/publisher 로드.
2. `automation/experiment/runner.py`: preflight → 입력 fingerprint → `capture_readiness_calibration_export`.
3. `automation/live/task.py`: P1 실행 완료 후 결과 재검토. `automation/runner.py`는 필수 Export 확인, 결과 수집, Acceptance, finalize를 끝낸 후에만 run manifest를 `completed`로 기록한다.
4. 실험 Runner는 완료된 P2 outcome 및 P1 manifest hash를 확인한 뒤 선택적 Factory 비교를 수행한다.
5. publication 단계에서 `Outbox.enqueue` → `build_bundle`/`verify_bundle` → `Outbox.process` → `GitPublisher.publish` 순서로 실행한다. 불완전한 Export/실패한 계산은 게시하지 않는다.
6. GitPublisher는 임시 독립 checkout에서 결과 Commit 후 일반 Push와 원격 확인을 수행한다. 코드 checkout에 Commit하지 않는다.

실험 상태 `RUNNING` → 계산/Export 완료 → Outbox `PENDING` → `PUBLISHING` → `PUBLISHED` → 실험 `COMPLETED`. Publisher가 꺼져 있으면 실험은 `COMPLETED`여도 게시 상태는 **PENDING**이다. WARN의 CLI 종료 코드는 **3**이며 실행 실패 코드 2와 구분해야 한다.

품질 WARN/Confidence REJECT는 진단 이력으로 게시할 수 있다. 계산 실패와 품질 실패는 다른 상태다. 완료된 계산의 품질 FAIL도 Bundle에서 허용되지만, 이는 차량 적용 승인과 무관하다. 이번 실제 합성 결과는 WARN이며 운영 Acceptance 기준 미설정 때문에 PASS가 아니다. 게시 상태와 `UNREVIEWED / NOT_DEPLOYED`는 별개로 유지된다.

## 최소 변경

- `automation/experiment/config.py`: `publish.retry_in_process` 엄격한 boolean 설정 추가. 기본 false.
- `automation/experiment/runner.py`: 위 옵션이 true일 때 기존 Outbox의 RETRY_WAIT/백오프/최대 시도 횟수를 사용해 같은 명령 안에서 재시도. 영구 인증 오류를 자동 초기화하지 않는다. 대기 전 이벤트와 매 시도 결과를 저장한다. 취소하면 INTERRUPTED와 Outbox를 보존한다.
- `docs/experiment.example.yaml`: 선택형 설정 설명.
- `tests/test_experiment_autopublish.py`: 일시 오류 후 자동 복구, 횟수 제한, 영구 오류, Export 누락 2종, 엄격한 설정, 대기 중 취소/재개, 계산 실패 차단 등 8개 테스트.
- `tests/test_experiment_e2e.py`: 계산/Export 완료가 publication보다 앞서는지 검증. 로그 목적지를 환경변수로 분리 가능하게 하여 기존 증거 보존.

기존 Calibration 알고리즘/K/D/추천 가중치/Confidence/등급 임계값은 변경하지 않았다. 기존 기본 경로는 첫 실패 시 종료되는 동작을 유지한다.

## 프로파일과 명령

기존 ROSbag 프로파일의 입력·보드·Mask·선정 설정을 유지하고 아래 게시 섹션만 설정한다. 새로운 결과의 외부 Push는 해당 Run 승인 후에만 enabled를 true로 바꾼다.

```yaml
publish:
  results_repository: true
  full_frame: true  # 실제 원본 좌표계 확인된 경우만 선언
  retry_in_process: true
  config:
    enabled: false # 승인 후 true
    remote: https://github.com/yaaaaangeo/calibration-results-test.git
    branch: calibration-results
    outbox: ./separate-experiment-outbox
    max_attempts: 3
    base_backoff_sec: 2
    max_backoff_sec: 10
    timeout_sec: 90
```

```powershell
# 읽기 전용 실행 계획 / 사전 점검
venv\Scripts\python.exe -m automation experiment run --profile <experiment.yaml> --dry-run
venv\Scripts\python.exe -m automation experiment preflight --profile <experiment.yaml>

# 승인되고 enabled:true인 프로파일이면 Export 후 별도 Push 명령 없이 게시
venv\Scripts\python.exe -m automation experiment run --profile <experiment.yaml>

# 종료/영구 실패 후 필요할 때 계산 없이 게시만 재시도
venv\Scripts\python.exe -m automation experiment retry-publish --profile <experiment.yaml>
# 원인을 해결한 영구 실패만 --reset-failed 명시
```

백오프 중 대기는 동기 방식이며 최대 시도 횟수로 제한된다. 프로세스 종료 후 자동 실행되는 상주 Worker/스케줄러는 아니다. 재시도 횟수가 소진되면 결과와 Outbox를 보존하고 종료하므로 운영 감시가 필요하다. 설정 경로는 YAML 파일 위치 기준이며 Outbox는 대상 remote/branch에 결합된다.

## A. 로컬 합성 ROSbag 결과

기존 합성 테스트 fixture: 24장의 known-K ChArUco 이미지를 ROS2 bag에 기록하고 P2가 16장을 채택했다. 기존 테스트 전용 Readiness 설정을 사용했다. 이는 실제 카메라의 운영 Readiness 기준을 완화한 검증이 아니며, 실데이터 품질 통과 근거로 쓰면 안 된다. 원본 bag/이미지 해시는 보존됐다.

- 4개 모델 계산 및 Hold-out, 추천, YAML/JSON/HTML Export, Factory 비교, 로컬 Git 게시 완료.
- 추천 Brown-Conrady; Acceptance WARN; Calibration Confidence 85.0/HIGH; 품질 excellent. **합성 데이터의 수치이며 실차 결과가 아니다.**
- 새 Run ID: `20261103T091807Z-73f40066`.
- 로컬 결과 Commit: `bb469d03d530f89027300d2b2189eec443d80779`.
- 동일 Run 재실행에서 계산 재사용, Commit 1개 유지.
- 모의 일시 Push 실패는 같은 실험 호출 안에서 2회 시도로 자동 복구. 영구 오류는 1회 후 중단, 최대 2회 테스트는 2회 후 중단, Bundle/결과 보존 확인.
- 계산 실패 또는 ROS YAML/HTML 누락 시 Publisher 호출 및 Commit 생성이 없음을 확인.

실제 CLI 로그 `.verification/ap/bag-pipeline-stdout.log`의 UTC 발췌:

```text
09:18:02 stage_started capture_readiness_calibration_export
09:18:25 stage_finished capture_readiness_calibration_export COMPLETED (22.640s)
09:18:25 stage_started factory_comparison
09:18:25 stage_finished factory_comparison COMPLETED (0.281s)
09:18:25 stage_started publication
09:18:28 stage_finished publication COMPLETED (3.047s)
09:18:28 completed verdict=WARN publication=PUBLISHED
```

P1 내부 result_collection/acceptance/finalize도 completed임을 확인했다. 테스트 로그: `.verification/ap-suite.log`, `.verification/ap-regression.log`, `.verification/ap-order-fixed.xml`. 실행 상태/타이밍은 `.verification/ap/bag-pipeline-result.json`에 있다.

## B. 실제 GitHub 테스트: 승인 대기

대상 저장소가 현재 private이고 연결 계정의 push 권한이 있음을 다시 확인했다. **이번 요청에서는 외부 Push를 수행하지 않았다.**

이번 합성 결과만 재사용해 별도 실험 job `github-auto-synthetic-20261103` 및 전용 Outbox를 준비했다. 원본 E2E Rational/P5와 identity/Run ID가 다르다. `--reuse-session` 경로에서 입력 해시·camera·보드 일치를 검사한 후, Publisher disabled 상태로 Bundle을 생성해 PENDING까지 실행했다. 이는 새 Calibration을 한 번 더 실행한 검증이 아니다. 승인 후 아래 같은 `experiment run` 명령이 결과 재사용 후 자동 게시한다. 별도 `publish` 명령은 필요 없다.

```powershell
# 현재는 외부 게시 disabled이므로 PENDING을 유지한다.
# 승인 후 같은 프로파일의 enabled만 true로 바꾸고 실행한다.
venv\Scripts\python.exe -m automation experiment run --profile .verification/ap/github-pending.yaml
```

승인 대상:

- Remote: `https://github.com/yaaaaangeo/calibration-results-test.git`
- Branch: `calibration-results`
- Run ID: `20261103T091807Z-73f40066`
- 모델 Brown-Conrady; WARN / Calibration Confidence 85.0 HIGH / UNREVIEWED / NOT_DEPLOYED.
- 실제 카메라 성능 또는 차량 적용 승인과 무관한 **합성 ROS2 bag 테스트 결과**.

게시 예정 경로:

```text
cameras/TEST__94ee059335e5/SYNTH-BAG__176dadd1a268/
  index.json
  runs/2026-11-03_20261103T091807Z-73f40066/
    acceptance.json
    bundle.json
    calibration.json
    intrinsic/camera_brown_conrady.yaml
    intrinsic/camera_info.yaml
    p1_acceptance.json
    p2_acceptance.json
    report.html
    run_manifest.json
    semantic.json
    summary.json
```

11ファイルのBundleとindex計12ファイル。正確な各パスとSHA-256は `.verification/ap/github-approval.json` に記録。既存のRational履歴は別cameraディレクトリにあり変更対象外。

Bundle `verify_bundle` は全ハッシュ・許可ファイル・機密情報スキャンに合格。HTMLは安全な構造化情報から再生成し、画像・bag・project・認証情報・機密絶対パスは含まない。合成Factoryファイル/比較結果は今回のGitHub候補には含めず、ローカル統合検証でのみ確認した。

承認後の未実施チェック：実験Runnerからの実GitHub Push、remote SHA/全ファイルハッシュ、既存Rational履歴不変、同一Runの重複防止。実ネットワーク障害、実機接続、独立撮影精度、運用認証更新/ブランチ保護は別途確認が必要。


## ?? ?? ?? ?? ??

- Remote: `https://github.com/yaaaaangeo/calibration-results-test.git`, branch `calibration-results`.
- Run ID: `20261103T091807Z-73f40066`, Brown-Conrady, ?? ROS2 bag ??.
- Local result Commit SHA = Remote branch SHA: **`c18418e2b7e6bdfdfb0392228bade45761a11f82`**.
- GitHub Commit: https://github.com/yaaaaangeo/calibration-results-test/commit/c18418e2b7e6bdfdfb0392228bade45761a11f82
- ?? Rational Commit `65026cc79c0124e76d691d8fea69558a3371fd77`? ???? ????. ?? Commit ? 2??? ?? Rational? ?? ?? ???? ????.
- ??? 12? ??? ????. Bundle 11? ?? ?? ?? blob SHA-256? ?? ?? ???? ????, index? ??? ? Run? ???? content hash? ????. 12? ?? ??? ????.
- ?? ?? ?? ???? ?? ??. ?? ???/bag/project/P5 ??? ???? ???.
- ?? `experiment run` ??? ?? ??? ??? ?? SHA? ?? ?? ???, Commit ??? ????.
- ?? ?? Session ? Export/Bundle, P5 ?? ??? ?? ??? ?? ????. ?? ??? HEAD? ???? ???.
- WARN / Calibration Confidence 85.0 HIGH / UNREVIEWED / NOT_DEPLOYED ??. ?? ???? HIGH? ?? ?? ??? ???? ???? ? ??.

?? ?? ?? ?? (UTC):

```text
09:26:51 calibration_reused verdict=WARN
09:26:51 stage_started publication
09:26:59 stage_finished publication COMPLETED (8.453s)
09:26:59 completed verdict=WARN publication=PUBLISHED
```

? ??? ???? WARN ?? ?? 3?? ?? ????. ??? CLI `automation experiment run`? subprocess? ????? ?? Publisher ???? ?? Push ???? ???? ???. ?? Session? ?? ????? capture/calibration ??? ????? ???? ?? stage ??? ?? ??? ????.

?? ?? ??: `.verification/ap/remote-check/result.json` (?? ?? SHA-256), `before.json` (?? ??/?? ???), `first-auto-push-stdout.log`, `same-run-repeat-stdout.log`. ?? Manifest: `.verification/ap/AUTO_PUBLISH_VERIFICATION_MANIFEST.json`. ?? ?? ? ????? Calibration/?? ???? ?? ???? ???.
