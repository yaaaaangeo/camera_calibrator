# P3 결과 게시와 카메라별 이력

P3는 완료된 Output Session을 읽어 별도 안전 패키지와 로컬 Outbox를 만들고, 명시적으로 활성화된 경우 결과 전용 Git 저장소에 게시한다. 계산·추천·P1/P2 판정 규칙을 변경하지 않는다. 게시 성공은 품질 승인이나 차량 적용이 아니다.

## 설정과 명령

[publisher.example.json](publisher.example.json)을 로컬 설정 파일로 복사한다. 기본은 `enabled=false`이며 remote도 비어 있다. 실제 결과 저장소를 별도로 준비하고 URL과 branch를 지정해야 한다. P3는 외부 저장소를 생성하지 않는다. 코드 저장소 URL을 결과 저장소로 사용하지 않는다.

```json
{
  "enabled": false,
  "remote": "",
  "branch": "calibration-results",
  "outbox": "publish_outbox",
  "max_attempts": 5,
  "base_backoff_sec": 5,
  "max_backoff_sec": 3600,
  "timeout_sec": 90
}
```

`--publish`가 없으면 기존 run/live는 Publisher를 호출하거나 네트워크에 접근하지 않는다. `--publish`가 있어도 설정이 비활성화면 안전 패키지를 **PENDING으로 로컬 저장**한다. 자동 게시하려면 `--publish`와 설정의 `enabled=true`가 모두 필요하다. `publish --session`도 enabled 설정을 따른다.

```powershell
# 기존 P1 계산 후 선택적으로 게시
venv/Scripts/python.exe -m automation run --profile default --images ./captures --serial SN123 --camera-model CAMERA --publish --publish-config publisher.local.json

# 기존 P2 촬영/계산 완료 후 같은 Publisher 사용
venv/Scripts/python.exe -m automation live --profile p2.yaml --ros --topic /camera/image_raw --serial SN123 --camera-model CAMERA --publish --publish-config publisher.local.json

# 이미 완료된 P1 세션 또는 COMPLETED P2 캡처 세션을 패키징/게시
venv/Scripts/python.exe -m automation publish --session ./completed-session --publish-config publisher.local.json

# 재시도 시간이 된 항목만 처리. 계산은 다시 실행하지 않음
venv/Scripts/python.exe -m automation publish retry --publish-config publisher.local.json
venv/Scripts/python.exe -m automation publish retry --job <job-id> --publish-config publisher.local.json

# 인증/설정 수정 후 영구 실패를 명시적으로 초기화할 때만 사용
venv/Scripts/python.exe -m automation publish retry --reset-failed --publish-config publisher.local.json

# 네트워크 없이 로컬 Outbox 이력 조회
venv/Scripts/python.exe -m automation history --serial SN123 --publish-config publisher.local.json
venv/Scripts/python.exe -m automation history --serial SN123 --camera-model CAMERA --compare RUN-A RUN-B --publish-config publisher.local.json

# 결과 저장소의 로컬 체크아웃을 읽기 전용으로 조회
venv/Scripts/python.exe -m automation history --serial SN123 --repository ./results-checkout
```

P1 `run`의 기존 종료 코드는 유지한다. P2 `live`의 제어 종료0도 기존처럼 COMPLETED를 뜻하며 품질PASS가 아니다. 두 경로 모두 게시 실패 때문에 계산 종료 코드를 바꾸지 않고 별도 `event=publication` JSON을 출력한다. 독립 `publish` 명령은 실패/재시도 대기에2, 게시 완료/비활성 PENDING에0을 반환한다.

GUI는 이번에 변경하지 않았다. 완료된 수동/자동 결과에 대한 게시 인터페이스는 CLI다. `run_manifest.json`과 명시적인 Camera Model/S/N이 없는 일반 `app.cli`/수동 GUI 세션을 운영 식별자로 추정해서 게시하지 않는다. 이 경우 기존 `automation run`으로 provenance가 있는 결과를 만들거나 별도 가져오기 설계가 필요하다.

## 파일과 저장 구조

필수 입력은 `run_manifest.json`, `reports/calibration.json`, `reports/report.html`, `automation/acceptance.json`, `intrinsic/camera_info.yaml`, 하나 이상의 실제 `intrinsic/camera_<model>.yaml`이다. 없으면 상대 파일명을 담은 오류를 기록하고 게시하지 않는다. P2의 `automation/p2_acceptance.json`이 있으면 그 유효 판정을 사용하고 P1 원판정도 별도로 보존한다. `automation/run_summary.json`이 있으면 P95/안정성 등 기존 metric을 수집한다.

```text
<outbox>/
  items/<content-hash>/
    bundle/                       # 검증된 불변 패키지
    state.json / state.backup.json
    events.jsonl / receipt.json
  errors/                         # 실행 오류/불완전 패키지의 로컬 기록

<결과 저장소>/
  cameras/<model>__<hash12>/<serial>__<hash12>/
    index.json
    runs/<date>_<run-id>/
      bundle.json                 # 업로드 파일 SHA256/원본 파일 SHA256/요약
      run_manifest.json           # 경로/환경/개인정보 제거한 derivative
      calibration.json
      acceptance.json             # P2가 있으면 P2 최종 판정
      p1_acceptance.json
      p2_acceptance.json          # 실제 파일이 있는 경우만
      summary.json / semantic.json
      report.html                 # 구조화된 결과로 재생성한 안전 보고서
      intrinsic/camera_<model>.yaml
      intrinsic/camera_info.yaml
```

모델/SN의 정확한 대소문자를 유지하면서 각 식별자의 SHA256 앞12자리를 붙여 Windows의 대소문자 비구분 경로에서도 다른 식별자를 합치지 않는다. 알려지지 않은 S/N을 생성하지 않고 UNKNOWN/UNSET 등은 거부한다. 수동 입력은 `identity_verified=false`로 기록하며 제조사/EEPROM 검증으로 간주하지 않는다.

원본 Session/이미지/bag/project/cache/log는 수정하거나 업로드하지 않는다. HTML 원문은 경로·스크립트·이미지 등을 포함할 수 있어 업로드용 HTML을 안전한 structured summary로 다시 만든다. 따라서 기존 상세 그래프/이미지는 업로드 보고서에 포함하지 않는다. 원문과 derivative를 혼동하지 않도록 원본 파일 해시와 정리 방식을 Bundle에 기록한다.

## Content Hash v1

SHA256(canonical JSON)을 사용한다. canonical JSON은 키 정렬, 고정 구분자, UTF-8, 비유한 수 금지다. 입력 구성은 다음과 같다.

1. 카메라 모델/SN/source를 포함한 정확한 식별자.
2. 원본 입력 이미지의 **내용 SHA256 정렬 목록**. 목록이 없는 이전 기록이면 aggregate fingerprint를 사용한다. 원래 aggregate도 provenance에 보존한다.
3. 안전하게 정리한 모델별K/D·metric·추천·split·품질판정·코드SHA/dirty 정보와 Calibration JSON/P1 Acceptance.
4. 실제 intrinsic/ROS YAML에서 읽은 K/D·해상도·모델의 수치 표현.
5. 좌표계/ROI 선언.

Run ID, 실행/생성 시각, elapsed/seconds, 로컬 경로/환경/명령행 등 비결과 메타데이터는 제외한다. 이미지/frame ID는 해당 이미지의 내용 해시 기반 별칭으로 바꾼다. 원래 파일명·절대경로는 전달하지 않는다. 실제 K/D/metric/판정/split/입력 또는 코드 버전이 바뀌면 다른 내용으로 취급한다. 원문 HTML의 시간/레이아웃 변화만으로 새 결과를 만들지는 않는다.

`content_hash`와 `bundle.json.files`의 개별 업로드 파일 SHA256은 별도다. 업로드 텍스트는 LF로 저장하고 Git 체크아웃의 CRLF 변환만 정규화해 검사한다. 그 외 파일 편집은 해시 불일치로 거부한다. 같은 카메라/같은 Content Hash면 기존 게시 정보와 Commit SHA를 반환한다. Run ID가 같고 내용이 다르면 덮어쓰지 않고 충돌로 실패한다. 내용이 같고 Run ID만 다르면 기존 원격 Run을 재사용하고 로컬 신규 실행의 outbox 기록은 남긴다.

## 재시도와 Git 안전성

상태는 PENDING → PUBLISHING → PUBLISHED 또는 RETRY_WAIT/FAILED다. 로컬 패키지와 최초 상태를 모두 만든 뒤 원자적 rename으로 등록한다. 상태 JSON은 fsync 후 원자 교체하며 backup과 revision이 있는 journal로 복구한다. 모든 상태가 사라지면 임의로 처음부터 재시도하지 않고 수동 검토 FAILED로 둔다.

기본 재시도는5/10/20/40초 간격이며 최대5회, 대기 상한3600초다. 인증 실패는 자동 재시도하지 않는다. 네트워크/비 fast-forward 충돌은 제한 내 재시도한다. 오류는 안전한 코드만 저장하며 Git stderr/토큰이 있는 traceback을 복사하지 않는다.

Git 작업은 매번 **새 임시 로컬 저장소**에서 수행한다. 원래 코드 checkout과 사용자 결과 checkout을 초기화/전환하지 않는다. 설정한 remote/branch 최신 상태를 읽고, 불변 Run 추가와 해당 카메라 index만 반영하여 일반 push한다. force/force-with-lease, merge/rebase로 다른 변경을 덮어쓰는 동작은 없다. push 후 remote가 해당 Commit을 포함하는지 확인한다. 이 확인 전후 프로세스가 죽어도 다음 시도는 remote Bundle을 먼저 확인하여 중복 commit을 만들지 않는다.

Index는 캐시다. 손상/누락 시 Run의 Bundle들을 스캔하여 이력을 복구하고, 다음 새 게시에서 index를 재작성한다. index에는 코드 Commit SHA와 content hash를 저장한다. **자신을 포함한 commit의 SHA를 같은 commit 안에 넣을 수 없으므로**, 실제 게시 Commit SHA는 Outbox receipt와 history의 Git 경로 조회로 제공한다. 이를 위해 별도의 중복 metadata commit을 만들지 않는다.

Outbox는 게시 대상을 바인딩한다. 같은 queue의 remote/branch를 바꿔 대기 항목을 몰래 다른 대상으로 보내지 않는다. GitLab 등 다른 대상으로 전환할 때는 **새 Outbox 경로**를 지정하고 기존 Session을 재사용해 enqueue한다.

Git의 일반 push/fast-forward 동작은 [Git push 문서](https://git-scm.com/docs/git-push), 인증 Helper 분리는 [Git credentials 문서](https://git-scm.com/docs/gitcredentials)를 근거로 했다.

## 인증/프라이버시

Git Credential Manager 등 기존 안전한 credential helper 또는 SSH agent를 사용한다. P3 자체는 토큰을 입력받거나 저장하지 않는다. 환경 토큰이 필요하면 Git의 외부 안전 helper에서만 처리한다. URL userinfo/토큰/query는 거부하고, Git trace/프롬프트를 비활성화한다. 실제 GitHub/GitLab 인증과 push는 별도 승인이 필요한 현장 작업이며 이번 검증에서는 수행하지 않았다.

허용 파일만 수집하고 원본이미지/bag/바이너리/cache/로그를 제외한다. 파일당10MiB/Bundle25MiB/32파일 제한, 경로 traversal/절대경로/symlink/Windows reparse 차단, 알려진 토큰·환경 secret·계정/절대경로 검사, YAML과 JSON의 K/D·해상도 일관성 확인을 적용한다. `.verification`, Outbox 기본 폴더, `.env`, `publisher.local.json`은 gitignore에 추가했다.

자동 검사는 임의 자유문장에 숨은 모든 개인정보를 보증하지 않는다. 외부 활성화 전 staging의 내용, 공개 가능한 Camera Model/SN, 결과 저장소 접근권한을 운영자가 확인해야 한다. P3는 결과가 생성된 원래 dirty 코드의 정확한 diff를 업로드하지 않으므로 `code_dirty=true`인 실행을 Commit SHA만으로 완전히 재현할 수는 없다.

## 이력 비교와 좌표계

history 기본값은 네트워크 없이 로컬 Outbox를 읽는다. 원격 이력이 필요하면 별도로 동기화한 결과 checkout을 `--repository`로 지정한다. 모델/SN, 해상도, distortion model, 좌표계/ROI가 호환되지 않거나 확인되지 않으면 K/D 수치 비교를 차단한다.

기존 출력에는 물리 ROI가 명확하지 않으므로 기본은 `coordinate_system=unknown`이다. 운영자가 실제 조건을 확인한 경우에만 최초 게시 전에 다음 선언을 사용한다.

```powershell
# 자동 run/live의 opt-in 게시 옵션
... --publish --publish-full-frame --publish-config publisher.local.json
... --publish --publish-roi 100 50 640 480 --publish-config publisher.local.json

# 기존 Session 게시
python -m automation publish --session SESSION --full-frame --publish-config publisher.local.json
python -m automation publish --session SESSION --roi 100 50 640 480 --publish-config publisher.local.json
```

ROI의 width/height는 실제 결과 해상도와 같아야 한다. 이미 게시한 Run의 ROI를 변경하여 덮어쓰지 않는다. 좌표계가 다른 결과는 새로운 명시적 Run으로 관리해야 한다.

호환되는 결과에는 fx/fy/cx/cy/D 변화, 판정/모델별 train/test/edge/P95/안정성 metric을 함께 표시한다. 같은 기준 광선63개(왼쪽 결과의10~90% 픽셀 격자)를 두 K/D로 투영한 차이의 RMS/P95/max도 제공한다. 파라미터나 투영 차이만으로 양불·승자·승인을 판정하지 않는다.
