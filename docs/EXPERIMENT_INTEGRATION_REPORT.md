# 실험 전 통합 자동화 검증 보고서

2026-11-03 KST. P1~P4 연결과 실험 실행 준비를 점검했다.
소프트웨어 연결은 합성 bag과 기존 실제 이미지 Session에서 검증됐으며,
**실제 실험 bag·장비와 외부 GitHub 게시 권한은 미검증**이다.

## 구현과 사용 인터페이스

새 명령은 `python -m automation experiment`이며 `preflight`, `run`, `retry-publish`를
제공한다. 하나의 YAML에 입력·camera·P2/P1·출력·Publisher·선택 P4 설정을 넣는다.
기존 `run/live/factory/publish/history`는 유지한다. 명령/프로파일/실험 절차/게시 활성화는
[EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md), 설정 예제는
[experiment.example.yaml](experiment.example.yaml)에 정리했다.

신규 파일은 `automation/experiment/{__init__,config,preflight,runner,cli}.py`,
`tests/test_experiment.py`, `tests/test_experiment_e2e.py`, 위 사용 문서·예제·이 보고서다.
기존 수정 파일은 다음 3개뿐이다.

- `automation/cli.py`: experiment dispatch/help
- `automation/publish/config.py`: inline 설정 검증 재사용 및 결과 트리 검증
- `automation/publish/git_publisher.py`: URL rewrite와 코드 저장소 기본/대상 브랜치 방어

ROSbag/P2/P1/P4의 기존 알고리즘은 수정하지 않았다. P2 BagReplay와 LiveController,
기존 P1 Runner, P4 후처리와 P3 Outbox/GitPublisher를 연결했다.

## Preflight와 Dry-run 결과

| 항목 | 이번 확인 결과 |
|---|---|
| Python | PASS: 가상환경 3.11.9 |
| 필수 라이브러리 | PASS: NumPy 2.4.6, SciPy 1.17.1, YAML 6.0.3, OpenCV 5.0.0, PySide6 6.11.2 |
| Bag 라이브러리 | PASS: rosbags 0.11.5 |
| ROS1/ROS2 bag 읽기 | PASS: 합성 fixture, Image/CompressedImage decode·토픽 검사 |
| 다음 실험의 실제 bag/토픽 | 미확인: 파일이 아직 제공되지 않음 |
| 패턴 구성 | PASS: parser와 보드 생성 검사 |
| 물리 보드/실제 SN | WARN: 사용자 선언이며 실측·장치 조회하지 않음 |
| 출력/Outbox | PASS: 테스트 경로 write/fsync, 공간/충돌 검사; 기본 read-only 검사는 WARN |
| 저장 공간 | 테스트 당시 약 27 GiB 여유. 예제 요구 5 GiB. 실제 bag/capture 용량은 현장 재확인 |
| Git CLI/로컬 Bare | PASS: 읽기·게시·중복 방지 검증 |
| 실제 GitHub remote/인증 | 미확인: 결과 저장소/계정이 지정되지 않음 |
| Git Push 권한 | 항상 별도 UNVERIFIED: read probe만으로 입증하지 않음 |
| Publisher 기본값 | disabled, 외부 업로드 없음 |
| Dry-run | PASS: 출력/lock/outbox·계산·Push를 만들지 않는 snapshot 검사 |

실제 쓰기 검사는 명시적 `--verify-write` 또는 실행 `run`에서만 자신의 임시 파일로 한다.
Dry-run은 write/network probe 옵션이 있어도 실행하지 않는다. 첫 이미지 decode만으로
bag 전체 건전성을 보장하지 않는다. 무인 Git read probe는 오류 원문/자격증명을 출력하지
않으며 public anonymous read와 Push 권한을 구분한다.

예제 그대로의 Preflight/Dry-run은 **FAIL이 정상**이다. 실제 camera/SN과 입력 파일을
채우기 전 실행을 막는다. 이는 내일 실험이 이미 준비 완료됐다는 의미가 아니다.

## 통합 실행 검증

**합성 ROS2 bag 전체 연결:** 알려진 K/D로 만든 ChArUco 이미지 24장을 PNG
CompressedImage bag에 기록했다. 기존 테스트용 P2 profile에서 16장 저장/선정,
Train 14 / Test 2로 계산했고 4개 모델이 모두 성공했다. 선택 모델 Brown,
기존 운영 기준 미설정 등에 따라 최종 품질은 WARN이다.

이 profile은 기존 P2 E2E의 작은 합성 fixture용 설정이며, production Readiness 기준의
검증이나 임계값 완화를 의미하지 않는다. 배포 예제와 기본 정책은 기존 값 그대로다.
Factory는 알려진 합성 정답을 사용했고 실제 제조사 값으로 주장하지 않는다.
P4 Fixed/Local/Seeded가 같은 Test에서 평가됐고 결론은 UNCLEAR였다.
YAML·정제 Factory 보고서가 로컬 결과 전용 Bare 저장소로 게시됐으며 WARN/UNREVIEWED/
NOT_DEPLOYED가 유지됐다. 이미지·bag·ccproj는 게시 tree에 없음을 확인했다.

단계 시간은 input fingerprint 약 0.00초, P2/P1 capture·readiness·계산·export 약
21.70초, P4 약 0.25초, 로컬 게시 약 2.59초였다. 테스트 전체는 이미지 생성 등도 포함한다.
다시 실행한 작업은 동일 계산 Session/manifest SHA와 동일 게시 SHA를 재사용했고,
remote 커밋 수가 1개로 유지됐다.

최초 E2E 검증 코드가 Windows 기본 cp949로 UTF-8 결과를 읽는 오류를 냈다.
실제 pipeline은 이미 WARN/PUBLISHED로 완료돼 있었다. 테스트의 읽기 encoding을
수정하고 원래 bag 해시를 확인한 후 완료 결과를 재사용해 검증했다. 전체 Calibration을
불필요하게 재실행하지 않았다. 초기 로그와 수정 후 XML을 모두 보존했다.

증거: `.verification/integration/bag-pipeline-stdout.log`, `bag-e2e-verified.xml`,
`bag-pipeline-result.json`, `bag-reuse.log`, `bag-evidence/`.

**기존 실제 이미지 55장:** 기존 `2026-11-02_163000_AUDIT_REAL` 완료 Session을
`--reuse-session` 경로로 편입했다. Camera/SN 문자열, 입력 55장 내용 해시와 보드 설정을
확인하고 기존 계산을 재사용했다. P4 시험용 참조 비교와 로컬 Bare 게시까지 완료했다.
품질 WARN, 비교 UNCLEAR, 로컬 결과 커밋 1개이며 원본 이미지/Session 해시가 일치했다.

여기서 Factory는 Train-derived proxy로 실제 제조사 데이터가 아니다. 보드는 추정
ChArUco 10×7, square 0.04m / marker 0.03m / DICT_5X5_100, 이미지 3840×2160이다.
실물 보드·정확한 hardware/SN·ROI/binning/orientation은 미확인이다. 따라서 실제 Factory
우열이나 절대 Calibration 정확도를 입증하지 않는다. 과거 판정을 현재 프로파일 기준으로
재평가하거나 P2 촬영을 다시 수행한 시험도 아니다.

증거: `.verification/integration/existing-real55-result.json`, `existing-real55.log`,
`real55-preflight.json`, `real55-results.git`.

## 실패 복구와 게시 안전성

고정 job_id+설정 fingerprint, 입력 내용 해시, OS lock, atomic manifest, stage/event log를
사용한다. 충돌 경로를 덮어쓰지 않는다. 완료된 계산은 재사용하며 변경된 입력은 거부한다.
단계 시간, 실패 단계와 원인, P1 기존 단계 시간 및 P2 task 시간 근거를 남긴다.

취소·중단과 source 실패 후 같은 P2 Session 복구, 계산 완료 후 게시 중단/실패 및
입력이 없어도 재계산 없이 Outbox 재게시를 검증했다. 기존 P3 backoff·영구 실패 reset·
Push 후 crash 복구·동시 갱신 충돌·내용 중복 방지를 유지했다. Factory 설정이 변경되면
옛 bundle을 조용히 재사용하지 않고 새 비교를 패키징한다.

알려진 코드 저장소 URL, URL rewrite로 우회한 주소, 로컬 non-bare, 기본/대상 브랜치에
소스가 있는 remote를 막는다. source tree가 기본 브랜치에 있고 깨끗한 결과 브랜치만
별도로 있는 경우도 거부한다. remote 검사는 실제 게시 시 임시 checkout에서 수행한다.
결과 저장소 선언만으로 소유권/용도를 인증하는 것은 아니다.

원본·실험 log는 P3 whitelist에 넣지 않고 기존 정제·토큰·개인 경로 검사만 통과한
파생 bundle을 게시한다. force push나 코드 checkout의 commit을 수행하지 않는다.
실제 GitHub Push는 이번 작업에서 0회다. 테스트를 위한 새 임시 저장소와 로컬 Bare의
commit/push만 사용자 요청 범위 안에서 수행했다.

## 테스트 집계

**최종 고유 테스트 321개: PASS 321 / FAIL 0 / SKIP 0 / XFAIL 0. 신규 27개 모두 PASS.**
범위별 고유 수는 아래와 같다. 재실행된 동일 node는 최신 결과 한 번만 집계했다.
기존 단일 테스트를 2개 parameterized case로 확장한 경우 옛 node를 추가로 세지 않았다.

| 범위 | 고유 PASS |
|---|---:|
| 신규 experiment unit / 실제 합성 bag E2E | 25 / 2 |
| P1 profile·acceptance·runner | 111 |
| P2 gate·readiness·live·runtime | 74 |
| 기존 ROS1/ROS2 BagReplay | 2 |
| P3 Git·Outbox/history·bundle·CLI | 59 |
| P4 Factory core | 48 |

주요 실행 명령:

```text
venv/Scripts/python.exe -m pytest tests/test_experiment.py -q --junitxml=...
venv/Scripts/python.exe -m pytest tests/test_experiment_e2e.py -q --junitxml=...
venv/Scripts/python.exe -m pytest tests/test_publish_git.py tests/test_publish_outbox_history.py tests/test_publish_bundle.py tests/test_publish_cli.py -q --junitxml=...
venv/Scripts/python.exe -m pytest tests/test_automation_profile.py tests/test_automation_acceptance.py tests/test_automation_runner.py -q --junitxml=...
venv/Scripts/python.exe -m pytest tests/test_p2_gate_policy.py tests/test_p2_readiness.py tests/test_p2_live.py tests/test_p2_runtime.py -q --junitxml=...
venv/Scripts/python.exe -m pytest tests/test_factory.py tests/test_p2_bag.py -q --junitxml=...
```

정확한 node/결과/XML은 `.verification/integration/test-result-ledger.json`,
`test-summary.json`에 있다. 단위·회귀는 그룹별 프로세스와 별도 로그로 실행했다.
새 장시간 bag pipeline Calibration은 1회이며 harness 수정 후에는 캐시를 검증했다.
`compileall`과 `git diff --check`도 통과했다.

전체 저장소 모든 테스트를 재실행하지는 않았다. 기존 알려진 실패 4개와 장시간 Spline
미완료 6개는 이번 수정으로 해결했다고 주장하지 않으며, 이전 보고서 상태를 유지한다.
과거 전체 집계에 이번 결과를 합산하지 않는다.

## 실험 전 남은 확인

실제 bag의 토픽·encoding·전체 decode, 해상도/ROI, 실물 보드 사양과 노출·다양성,
기존 P2 기본 Readiness를 만족하는지 현장에서 확인해야 한다. 실제 ROS 장비·카메라
연결과 실물 촬영 특성은 이번 합성 bag으로 검증되지 않았다.

실제 GitHub 결과 저장소/자격증명/쓰기 권한/branch protection/SSO/네트워크·대용량
history fetch도 미검증이다. 로컬 Bare 성공은 이를 대신하지 않는다. 게시 실패 시 계산은
보존되고 재게시만 수행한다. bag resume은 앞에서 다시 decode할 수 있어 오래 걸릴 수
있고, checkpoint가 모호한 초기화 중단은 수동 확인이 필요하다. 입력 해시도 bag 크기에
따라 I/O 시간이 든다. 디스크 최소 공간은 실제 capture/export/bundle 용량의 보증이 아니다.

기존 파일 455개의 시작 해시와 비교한 결과 기존 수정은 위 3개, 삭제 0개였다.
원본 이미지·기존 결과·미커밋 변경과 Calibration/추천/acceptance 알고리즘을 보존했다.
HEAD `acb22f147b277a600649e9ece84729c66dee1fb1`은 동일하다.
증거는 `.verification/integration/final-integrity.json`이다. 코드 저장소 commit/push,
실제 외부 Push, 배포는 수행하지 않았다.
