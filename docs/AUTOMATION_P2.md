# P2 자동 Intrinsic 캘리브레이션 사용 방법

P2는 영상 수신 → 기존 패턴 검출 → 품질/새 자세 검사 → PNG 저장 → 준비도/사전 관측성 검사 → 기존 P1 4모델 계산 → 판정/추가 촬영을 제어합니다. 구현과 검증 범위는 [P2 보고서](AUTOMATION_P2_REPORT.md)를 확인하세요.

현재 정책2의 계산 가능성/품질 판정 분리와 실제 검증은 [게이트 개선 보고서](P2_GATE_IMPROVEMENT_REPORT.md)를 확인하세요. 낮은 Observability Score는 계산 전 차단 대신 결과 검토 WARN으로 전달하며, rank 부족/평가 불가 등은 계속 차단합니다.

## GUI

1. 기존 Live Capture 화면에서 P2 자동 캘리브레이션 버튼을 누릅니다. 수동 촬영 데이터는 보존되며, 수동 구독은 중지됩니다.
2. 카메라 모델과 실제 Serial Number를 입력하고 입력 종류를 선택합니다. ROS는 토픽/메시지 타입, 재생은 이미지 폴더 또는 rosbag 경로/토픽을 지정합니다.
3. 보드 사양을 확인합니다. GUI의 `default`는 현재 화면의 PatternConfig/해상도로 P2 프로파일을 생성합니다. 별도 YAML을 입력하면 해당 파일을 사용합니다.
4. 시작 후 검출 결과, 저장/제외 수, coverage/다양성/hold-out/관측성, 촬영 안내, 계산 단계, 최종 판정과 저장 경로를 확인합니다. 계산 진행은 P1 manifest의 단계이며 최적화기의 백분율은 아닙니다.
5. 취소는 확보한 이미지와 결과를 보존합니다. 복구 세션에 기존 P2 세션 폴더를 지정해 재시작할 수 있습니다. 프로파일·카메라 식별·입력 연결이 동일해야 합니다.

패널 자체에 영상 미리보기는 추가하지 않았습니다. 기존 수동 미리보기 기능은 유지되며 P2 패널은 상태/수치 중심입니다. ROS 영상 수신과 실제 카메라 환경은 별도 현장 검증이 필요합니다.

## CLI

```powershell
venv/Scripts/python.exe -m automation live --profile p2.yaml --camera-model CAMERA --serial SERIAL --ros --topic /camera/image_raw --msg-type sensor_msgs/msg/Image
venv/Scripts/python.exe -m automation live --profile p2.yaml --camera-model CAMERA --serial SERIAL --replay image --output-root outputs/p2
venv/Scripts/python.exe -m automation live --profile p2.yaml --camera-model CAMERA --serial SERIAL --bag recording.bag --topic /camera/image_raw
```

ROS는 실행 환경의 ROS Python 의존성과 토픽 접근이 필요합니다. `--resume <P2 세션 폴더>`로 복구합니다. CLI 취소는 Ctrl+C 또는 해당 세션의 `cancel.request` 파일 생성입니다. 재개 시 해당 제어 파일만 초기화합니다. 이미지 재생 기본 간격은 1.1초이며 `--replay-interval`로 조정합니다. 저장 간격보다 빠르게 재생하면 저장 간격 조건으로 프레임이 제외됩니다.

종료 코드 0은 제어 흐름 `COMPLETED`를 뜻합니다. 품질 PASS를 뜻하지 않으며 WARN/수동 검토 완료도 포함합니다. 130은 취소, 2는 실행 실패/입력 소진/제한 도달, 1은 입력·설정 오류입니다. 최종 P2 품질은 attempt별 `automation/p2_acceptance.json`과 P2 manifest를 확인하세요. 기존 P1 `acceptance.json`과 `run_manifest.json`은 P1 원판정으로 보존됩니다.

## 프로파일

다음 보드 치수는 물리 보드와 대조해서 교체해야 합니다. CLI `default`는 기존 P1 예시인 7×5 ChArUco이며 보드 자동 식별 기능이 아닙니다.

```yaml
live_profile_version: 1
p1:
  pattern:
    type: charuco
    squares_x: 7
    squares_y: 5
    square_size: 0.04
    marker_size: 0.03
    dictionary: DICT_5X5_100
  # acceptance는 기존 P1 스키마로 지정합니다.
capture:
  interval_sec: 1.0
  max_images: 120
readiness:
  min_images: 35
  coverage_pct: 75.0
  check_every_new_frames: 5
  observability_min: 50.0  # 결과 검토 경계이며 계산 시작을 위한 최소 Score가 아님
recovery:
  max_calculations: 3
  max_recaptures: 2
  min_new_frames: 5
```

생략한 항목은 [config.py](../automation/live/config.py)의 기본값을 사용합니다. 모든 P2 임계값은 `capture`, `readiness`, `recovery`로 조정하고 기존 선별/노출/품질 판정값은 `p1`에서 설정합니다. 네 가지 Intrinsic 모델과 P1 이미지 선별, pattern 노출 모드는 필수입니다. `readiness.min_images`는 P1 `image_selection.target_count`를 초과할 수 없습니다.

준비도 기본값은 촬영 안내용 실험 설정입니다. P1 운영 허용 기준은 여전히 null이며 미설정 항목을 PASS로 간주하지 않습니다. 촬영 준비도 통과와 제품 승인 기준 통과는 별도입니다.

## 저장 및 복구

```text
<출력 루트>/<시각_모델_serial_capture>/
  manifest.json
  p2_manifest.json                 # 원자적 체크포인트
  capture_events.jsonl             # 검출/제외/전이/복구/성능 이력
  captures/capture_000000_<uuid>.png
  tasks/<task-id>/request.json, response.json, task.log
  runs/attempt_001/<기존 P1 세션>/
    run_manifest.json
    automation/acceptance.json, run_summary.json, ...
    automation/p2_acceptance.json  # P2 최종 검토/판정 및 모델별 관측성
    reports/..., calibration.json, ...
```

원본 입력 파일은 읽기만 하며 P2는 별도 PNG를 저장합니다. 세션 OS 잠금으로 동시 재개를 막습니다. 재개 시 저장 이미지 경로·SHA256·프로파일·입력 연결·카메라 식별을 검사하고 완료된 task 응답이 있으면 재계산 없이 반영합니다. 취소·타임아웃은 해당 실행의 프로세스 트리를 종료합니다. 비정상 중단 작업은 재시도할 수 있으나 계산/관측성/재촬영 횟수 제한은 초기화하지 않습니다.

정책2는 이전 preflight 숫자 보고서를 현재 규칙으로 다시 판단합니다. 이전 정책의 P1-only 완료 응답은 새 정책의 PASS로 자동 승인하지 않습니다. 새 사진이 추가돼도 선택된 학습/검증 데이터가 같으면 같은 계산을 반복하지 않습니다.

저장 오류 중 체크포인트에 등록되지 못한 PNG는 자동 채택하지 않고 디스크에 보존합니다. 이런 세션은 로그와 저장 경로를 수동 점검하세요. 이미 완료된 세션을 재개해도 완료 상태를 임의로 다시 계산하지 않습니다.
