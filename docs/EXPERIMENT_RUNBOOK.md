# ROSbag 통합 실험 실행 절차

하나의 YAML로 기존 P2 입력·Readiness·선정 → P1 계산·모델 비교·YAML Export →
선택 P4 비교 → P3 bundle/outbox/게시를 실행한다. 새 Calibration 알고리즘이나 품질
판정을 적용하지 않는다. 예제는 `experiment.example.yaml`이다.

## 프로파일 준비

예제를 작업용 파일로 복사하고 Camera Model/SN, bag 경로, 정확한 이미지 토픽,
실물 보드 사양과 출력 경로를 수정한다. 경로는 **프로파일 파일 위치 기준**이다.
ROS1은 `.bag`, ROS2는 `metadata.yaml`과 `.db3`/`.mcap`이 포함된 녹화 디렉터리를
우선 사용한다. 지원하지 않는 실제 형식은 Preflight가 실패로 표시한다.

통합 `experiment` 입력에 `.db3` 파일을 지정하면 같은 폴더의 원래 `metadata.yaml`에서
소속을 검증하고 **전체 녹화 디렉터리**로 정규화한다. 따라서 모든 split segment가
입력 해시에 포함된다. metadata 없는 SQLite 파일을 추정하여 읽지 않는다.
표준 ROS2 profile의 단일 `.mcap`도 지원한다. 확장자가 MCAP인 것만으로 ROS2 이미지를
보장하지 않으며, loggerd/PYUV 등의 사용자 codec은 별도 decoder가 필요하다.

`live.p1.pattern`의 square/marker 크기는 m 단위다. 예제 7×5는 실제 보드의 사양이
아니다. 기존 실제 55장 진단은 추정 10×7 보드를 사용했으므로 그대로 혼용하면 안 된다.
해상도를 알면 `live.p1.camera.width/height`도 지정한다. `live.capture/readiness/recovery`의
생략값은 기존 P2 기본값이다. 준비도 미달 시 임계값을 내려 성공을 강제하지 않는다.

`input.replay_interval`은 bag timestamp 배속이 아닌 프레임 전달의 wall-clock 간격이다.
기존 P2 기본 capture interval 1초에 맞춰 기본값은 1.1초다. 더 빠르게 읽으면 capture
간격 제한으로 많은 프레임이 저장되지 않을 수 있다. P2가 준비되면 계산을 시작하므로
bag 전체 메시지를 반드시 소비하는 모드는 아니다. 필요한 추가 촬영이 있으면 남은
입력을 계속 사용하며, 입력 소진·횟수 제한은 명시적 실패로 끝난다.

`input.extract_fps`는 별도 설정이다. bag record timestamp 기준 최대 샘플 FPS이며,
생략하면 모든 메시지를 후보로 전달한다. 예제의 1.0은 1초 이상 간격의 프레임만
디코딩한다. 이후 P2 wall-clock capture interval·노출·중복·다양성 검사가 적용되므로
최종 저장 FPS가 이보다 낮을 수 있다. 해상도는 `live.p1.camera.width/height`로 예상
원본 크기를 지정한다. 임의 resize/crop은 하지 않고 불일치나 중간 해상도 변경을 거부한다.

P4 제조사 파일이 없으면 `factory`를 생략한다. Local 계산은 계속된다. P4를 쓸 때만
검증된 reference/좌표 JSON 경로와 seeded 옵션을 추가한다. 제조사 파일의 정확한
projection convention과 좌표 호환성은 P4 사용 문서의 조건을 따른다.

## 실험 전 점검과 Dry-run

저장소 루트에서 프로젝트의 가상환경 인터프리터를 사용한다.

```powershell
venv\Scripts\python.exe -m automation experiment preflight --profile my-experiment.yaml
venv\Scripts\python.exe -m automation experiment run --profile my-experiment.yaml --dry-run
venv\Scripts\python.exe -m automation experiment preflight --profile my-experiment.yaml --verify-write
```

Preflight는 Python/필수 버전·API, bag 형식과 토픽 목록, 첫 이미지 decode, 패턴 구성,
출력·Outbox 경로/공간/충돌, Git CLI/remote/Publisher를 검사한다. 토픽이 틀리면
`image_topics` 목록에서 정확한 토픽을 확인해 수정한다. 첫 메시지 검사만으로 bag의 모든
메시지·encoding이 정상이라고 판단하지 않는다. 실물 보드/하드웨어 식별은 WARN이다.

기본 Preflight는 읽기 전용이다. 쓰기 권한은 메타데이터 검사만 하므로 WARN으로 남는다.
`--verify-write`는 가장 가까운 기존 출력/Outbox 부모에서 자신의 임시 파일만
쓰기·fsync·닫기 하여 실제 쓰기를 확인한다. 실행 `run`은 이 검사를 자동 수행한다.
계산 중 디스크 고갈이나 이후 권한 변경까지 보장하지 않는다.

Dry-run은 출력 폴더·lock·outbox·계산·삭제·Push를 만들지 않는다. `--network`,
`--verify-write`를 같이 넣어도 이 두 probe는 비활성화한다. bag metadata/첫 이미지
읽기와 계획 출력은 수행한다. shell에서 stdout을 파일로 redirect한 경우 그 로그는
호출자가 만든 파일이다. Python 일반 모듈 캐시와 검사 대상 결과 쓰기는 구분한다.

PASS는 해당 검사 범위의 확인, WARN은 미확인/검토 필요, FAIL은 진행 불가다.
예제의 `REPLACE_WITH...` 식별자와 없는 bag은 의도적으로 FAIL이다.
Preflight/Dry-run 종료 코드 0은 품질 PASS가 아니며, FAIL 검사가 있으면 1이다.

## 실행과 결과 확인

외부 업로드를 끈 상태로 먼저 실행한다.

```powershell
venv\Scripts\python.exe -m automation experiment run --profile my-experiment.yaml
```

`output.root/experiments/<job_id>/`에 다음을 남긴다.

- `experiment.json`: 전체 상태, 품질 verdict, 실패 단계/이유, 단계별 시간, P1/P2 시간 근거,
  P4 결과 경로, 게시 job/receipt
- `events.jsonl`: 단계 시작·완료·실패·재사용 이력
- `capture.log`: P2 상태·준비도·선정·계산 진행 JSON
- `capture_runs/`: 기존 P2 checkpoint/captures와 P1 시도별 원본 Session

P1 Session의 `reports/`, `intrinsic/`, `automation/acceptance.json`, 선택된 모델과
K/D·RMS·Test 지표를 확인한다. WARN은 계산 완료이지만 검토가 필요한 결과다.
게시 성공도 승인·배포를 의미하지 않는다.

실행 종료 코드는 품질 PASS=0 / WARN=3 / FAIL=4, Preflight/설정=1,
실행·게시 실패=2, 취소·중단=130이다. `calibration_verdict`와 `publication.status`를
별도로 확인한다. Publisher disabled이면 PENDING 로컬 bundle이 정상 결과다.

## 중단·재시작·재게시

같은 프로파일과 job_id로 `run`을 다시 실행하면 완료 계산을 재사용한다. 입력 해시와
설정을 확인하고, 바뀐 입력·패턴·camera 설정은 같은 작업으로 섞지 않는다. 다른 실험은
새 job_id를 사용한다. 같은 job_id의 동시 실행은 OS lock으로 거부한다.

Ctrl+C 후 다시 실행하면 기존 P2 Session을 복구한다. bag reader는 앞에서 다시 읽을 수
있으며 기존 capture hash/선정 상태를 재사용한다. frame 위치를 빠르게 seek하는 기능은
추가하지 않았다. 복구할 Session이 여러 개이거나 초기화 checkpoint가 없으면 추측하지
않고 수동 검토로 중단한다. 기존 P2 횟수·timeout·동일 입력 재계산 방지를 유지한다.

게시만 실패했다면 다음 명령을 사용한다. 이 경로는 bag, 이미지 검출, Calibration,
Factory 최적화를 다시 실행하지 않는다.

```powershell
venv\Scripts\python.exe -m automation experiment retry-publish --profile my-experiment.yaml
# 인증/영구 오류 원인을 수정한 뒤 명시적으로 retry counter를 초기화할 경우
venv\Scripts\python.exe -m automation experiment retry-publish --profile my-experiment.yaml --reset-failed
```

재게시에는 기존 Outbox와 Git branch/remote 바인딩을 유지한다. retry wait는 backoff를
존중한다. 원래 입력 파일이 없어도 완료된 결과와 Outbox가 있으면 재게시할 수 있다.
`retry-publish --dry-run`도 입력을 읽지 않고 재게시 계획만 확인한다.
Outbox/Session 자체가 손상되거나 유실된 경우 성공으로 처리하지 않는다.

기존 이미지 기반 P1 Session을 명시적으로 편입할 수도 있다.

```powershell
venv\Scripts\python.exe -m automation experiment run --profile image-experiment.yaml --reuse-session <completed-session>
```

모델/SN, 원래 입력 이미지 내용 해시, 물리 패턴과 완료 상태를 확인한다. 과거 품질 판정을
그대로 보존하며 현재 프로파일의 계산/acceptance 설정을 소급 적용하지 않는다.
bag provenance를 추정하여 과거 Session을 편입하지는 않는다.

## GitHub 게시 활성화

현재 개발 작업에서는 외부 Push를 실행하지 않았다. 현장 운영자가 별도 **결과 전용**
저장소를 준비한 뒤 프로파일의 다음 부분을 수정한다.

```yaml
publish:
  results_repository: true
  full_frame: false  # 실제 좌표 확인 후에만 true
  config:
    enabled: true
    remote: https://github.com/OWNER/camera-calibration-results.git
    branch: calibration-results
    outbox: ./experiment-outbox
```

계산 후 활성화할 때 기존 outbox 경로를 바꾸지 않는다. 위 예시 outbox는 파일 위치
기준이므로 실제 설정은 사용 중인 경로를 유지한다. 원격/branch가 이미 바인딩된 큐의
대상을 조용히 바꾸지 않는다. 대상 변경은 별도 작업·Outbox로 구분한다.

인증은 URL/YAML에 토큰을 넣지 않고 Git의 credential manager 또는 준비된 SSH key를
사용한다. [GitHub의 인증 캐싱 안내](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)를
따른다. 무인 Publisher는 대화형 로그인을 수행하지 않으므로 자격증명과 SSH host trust를
미리 준비해야 한다.

```powershell
venv\Scripts\python.exe -m automation experiment preflight --profile my-experiment.yaml --network
```

`--network`는 `ls-remote` 읽기 probe만 수행한다. [Git 문서](https://git-scm.com/docs/git-ls-remote)상
이는 remote refs 조회이며, 공개 저장소에서는 익명으로도 성공할 수 있다. 성공해도 Push
권한, 실제 인증 계정, SSO/branch protection·조직 정책은 미검증이다. 실제 외부 Push는
운영자가 허용한 실험 실행에서만 검증할 수 있다.

원본 이미지/bag/프로젝트는 bundle에 넣지 않는다. P3 allowlist와 토큰·개인 경로 정제를
유지하며, 코드 저장소 URL/URL rewrite/기본·대상 브랜치의 소스 트리를 거부한다.
일반 결과 메타데이터와 plain-text 메모 외의 root 파일·디렉터리는 제한한다.
Push는 임시 checkout에서만 수행하고 force를 사용하지 않는다.
