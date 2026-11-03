# Intrinsic Calibration 자동 실행 (automation, 1차)

이미지 폴더 하나로 **검출 → 이미지 선별 → 4개 모델 계산 → Hold-out 검증 →
모델 추천 → 품질 판정 → 결과 저장**을 한 번에 실행합니다. 계산은 기존
`app.cli`를 그대로 사용하며, 자동화 계층은 입력 검증·실행 기록·합격 판정만
추가합니다.

```bash
python -m automation check --profile default --camera-model SG3S-ISX031C
python -m automation run   --profile default \
    --images ./captures --serial SN123 --camera-model SG3S-ISX031C
```

`--camera-model`(별칭 `--model`)은 카메라 하드웨어 모델명입니다. 최종
calibration 모델을 강제로 고르는 기존 CLI 옵션 `--model`과는 다르며, 그
기능은 프로파일 `calibration.model`로만 지정합니다.

## 종료 코드

| 코드 | 의미 |
|---|---|
| 0 | 완료, 판정 PASS |
| 3 | 완료, 판정 WARN (검토 필요 또는 운영 기준 미정) |
| 4 | 완료, 판정 FAIL (품질 기준 미달) |
| 1 | 프로파일/입력 오류. 계산을 시작하지 않았고 세션도 만들지 않음 |
| 2 | 실행 실패. 세션의 `run_manifest.json`에 실패 단계와 원인이 기록됨 |
| 130 | 사용자 중단 (`run_manifest.json`에 failed/interrupted로 기록) |

## 프로파일

`--profile`에는 YAML 경로 또는 내장 프로파일 이름을 줍니다. 내장 `default`는
패키지 데이터(`automation/profiles/default.yaml`)라 `pip install .` 후 저장소
밖에서도 찾습니다. 우선순위는 **명령행 > 프로파일 > 기존 CLI 기본값**입니다.
`pattern`과 `calibration` 섹션의 키는 기존 `app.cli` 옵션 이름과 같고, 검증도 기존
CLI 파서와 패턴 검증 함수로 수행합니다.

- 패턴 치수와 dictionary는 실제 보드와 일치해야 합니다. 기본 파일의 값은
  저장소 예제(7x5 ChArUco, 40/30 mm, DICT_5X5_100)입니다.
- ChArUco/AprilGrid는 필요한 ID 개수가 dictionary 크기를 넘으면 실행 전에
  거부합니다.
- `acceptance.thresholds`의 기본값은 모두 `null`(운영 기준 미정)입니다. 필수
  (`required: true`) 지표의 기준이 비어 있으면 판정은 PASS가 될 수 없고
  WARN으로 남습니다.

## 실행 방식

자동화는 결과 세션 폴더를 먼저 확보(`OutputManager.reserve_session_directory`,
mkdir 실패 시 다음 이름)한 뒤, 기존 CLI를 자식 프로세스로 그대로 실행합니다.

```text
python -m app.cli --config <session>/automation/cli_config.yaml
```

`cli_config.yaml`은 기존 CLI `--config` 형식이며 `session_dir`(= `--session-dir`)을
포함합니다. 이미지 수백 장도 명령줄 길이 제한 없이 넘기고, 같은 파일로 그대로 재실행할
수 있습니다. 실행 전에 같은 옵션을 argv 형태로도 기존 파서에 통과시켜(type/choices 검사)
두 해석 결과가 같은지 대조합니다.

## 이미지 노출 판정 (`image_selection.exposure_mode`)

| 모드 | 기준 | 사용처 |
|---|---|---|
| `global` | 이미지 전체 평균 밝기 25~230, 전체 clipping(≤5 또는 ≥250) 비율 ≤ 15% | 기존 동작, GUI 자동 선별 |
| `pattern` | 보드 영역(코너 convex hull) 평균 밝기 25~230, 코너 주변 패치(코너 간격/4)의 ≥250 비율 중앙값 ≤ `corner_clip_max`(0.45) | 자동화 기본 프로파일 |

코너가 4개 미만이거나 이미지를 다시 읽지 못하면 `pattern` 모드도 `global` 기준으로
판단합니다. 두 모드는 선택되는 이미지가 달라질 수 있으므로(→ 모델 비교 결과도 달라질 수
있음) 모드와 설정이 `run_manifest.json`(`image_selection.config`)과
`image_selection_manifest.json`(`exposure_mode`, 이미지별 `corner_clip`, `roi_brightness`)에
기록됩니다.

근거 (실측, 2026-11-01): 실제 4K 6장과 합성 이미지에 노출 이득 1.5~5를 걸어 원본 대비
코너 위치 이동(p95 > 0.5 px 또는 코너 20% 이상 손실 = 품질 저하)을 기준으로 비교했습니다.

| 데이터 | global 정확도 | pattern 정확도 |
|---|---|---|
| 실제 4K × 이득 (31건) | 20/31 | 24/31 |
| 흰 배경 합성 (기존 CLI 테스트 fixture 방식, 24건) | 0/24 | 24/24 |
| 물리적으로 일관된 합성 (18건) | 12/18 | 16/18 |
| 실제 원본+이득 2.0 쌍 (11건) | 9/11 | 8/11 |

선택 결과 변화: 실제 55장 0장, 일관된 합성 24장 0장, 흰 배경 합성 16장은 0 → 16장.
한계: 과노출은 이득(gain)만으로 모의했으며 실제 센서 포화·블루밍·HDR 처리는 검증하지
않았습니다. 0.45는 위 실험에서 정한 값이고 과노출 검출력은 `global`과 비슷합니다
(장점은 밝은 배경 때문에 정상 이미지를 버리지 않는 것). 실제 과노출 촬영본을 모아 재검증이
필요합니다.

## 재현성과 허용오차

같은 입력·설정에서 Train/Test 분할 ID, 검출 결과(코너 좌표·이미지 지표)와 추천 모델은
항상 같았습니다. K/D/RMS는 OpenCV 내부 병렬 처리(스레드 스케줄링에 따른 부동소수점 합산
순서) 때문에 실행마다 달라질 수 있습니다. seed나 분할 문제는 아닙니다. 기존 CLI만
별도 프로세스로 반복 실행한 실측(2026-11-01):

| 데이터 | 비교 | 결과 |
|---|---|---|
| 일관된 합성 720p 24장 | 기본 스레드 2회 | K 최대 7e-8 (Rational), Test RMS 상대 ≤ 1.95e-6 |
| 일관된 합성 720p 24장 | `cv2.setNumThreads(1)` 2회 | 비트 단위 동일 |
| 실제 4K 35장 (`image/`) | 기본 스레드 3회 (자동화·CLI 직접 2회) + GUI 계산 경로 1회 | Pinhole/Brown 동일(≤1e-10), **Rational fx 2294~2387 px·Test RMS 5.05~6.32**, Fisheye Test RMS 3.99~4.16, 추천 모델은 모두 Brown-Conrady |
| 실제 4K 35장 | `cv2.setNumThreads(1)` 2회 | Rational 포함 비트 단위 동일 |

실제 데이터에서 Rational은 Train RMS가 Brown-Conrady(2.60 px)보다 커서 최적화가
수렴하지 못한 상태로 보이며, 이런 조건이 나쁜 계산에서는 스레드 잡음이 서로 다른 해로
이어질 수 있습니다 (기존 알고리즘 특성, 수정하지 않음).

그래서 자동화 기본 프로파일은 `execution.opencv_threads: 1`입니다 (CLI 실행 전
`cv2.setNumThreads(1)`만 호출, 계산 알고리즘은 동일). 실행 시간 영향은 아래 "성능" 절의
측정값을 참고하세요. `null`로 두면 기존 CLI/GUI와 같은 기본 스레드를 씁니다.

자동화 E2E 테스트의 기본-스레드 비교 허용오차(K/D 상대 1e-5·절대 1e-6, RMS 상대 1e-4)는
조건이 좋은 합성 데이터에서 실측 변동의 약 10~50배 여유를 둔 값입니다. 실제 데이터처럼
조건이 나쁜 경우에는 허용오차 비교가 의미가 없으므로 단일 스레드로 정확히 비교합니다
(`test_single_thread_opencv_runs_are_bit_identical`).

## 성능 (실측 2026-11-01, Windows 11 / Python 3.11 / OpenCV 5.0.0, 다른 부하 없음)

패턴 검출 (순차): 720p 42 ms/장, 1080p 82 ms/장, 4K 372 ms/장.

실제 4K 55장 (`image/`), 선별 결과 35장:

| 구간 | 단일 스레드 (`opencv_threads: 1`, 기본) | 기본 스레드 (`null`) |
|---|---|---|
| 사전 검증 (해상도 확인용 디코딩, 기록 밖) | 3.1초 | 3.1초 |
| 선별 단계 (55장 검출 + 선별) | 23초 | 24초 |
| 계산 단계 (기존 CLI: 35장 재검출 + 4모델 + Hold-out + Export) | 49초 | 46초 |
| 전체 (run_manifest 기준) | 73초 | 70초 |

선별을 켜면 선택된 이미지를 기존 CLI가 다시 검출합니다 (위 예에서 약 13초, 전체의
약 18%; 720p 24장에서는 약 1초). 서로 다른 프로세스의 검출 결과가 비트 단위로 같음은
확인했지만(35/35장), 재사용하려면 기존 CLI에 검출 결과 입력 경로와 캐시 무효화 키
(이미지 sha256·패턴 설정·검출 코드·OpenCV 버전)가 필요해 이번 단계에서는 적용하지
않았습니다.

## 실제 카메라 데이터 검증 절차

1. 실제 보드 사양(칸 수, 칸·마커 크기 m, dictionary)을 확인해 프로파일 `pattern`에 넣고
   `python -m automation check --profile <p> --images <dir>`로 확인합니다.
2. 해상도가 섞이지 않은 한 카메라의 이미지 폴더로 `run`을 실행합니다.
3. `run_manifest.json`의 `image_selection.counts`(검출 실패/노출/중복 제외 수)와
   `results.models`(모델별 K/D, Train/Test/Edge RMS)를 확인합니다.
4. 기존 CLI와 비교하려면 `automation/cli_config.yaml`을 복사해 `session_dir`,
   `save_project`, `json_summary`, `log_file` 네 경로를 **모두** 새 위치로 바꾼 뒤
   `python -m app.cli --config <복사본>`으로 실행합니다 (하나라도 그대로 두면 기존 결과를
   덮어씁니다 - `session_dir`만 같으면 CLI가 거부하지만 나머지 세 경로는 보호되지 않습니다).
   정확히 비교하려면 두 실행 모두 `cv2.setNumThreads(1)` 조건으로 맞춥니다.

저장소 `image/`의 55장(4K)은 rosbag 추출 이미지로, 보드 사양 기록이 없어 마커 ID 분포
(0~34 → 35개)와 검출 결과로 **10x7 ChArUco, DICT_5X5_100**을 추정했습니다(55/55장
검출). 칸 크기(m)는 알 수 없어 0.04/0.03 m로 두었으며, 보드 크기의 균일한 배율은 K/D에
영향이 없지만 마커/칸 비율은 실제 보드로 확인해야 합니다. 저장소에 있는 다른 YAML의
11x8 DICT_7X7_50 사양으로는 55장 모두 검출되지 않았습니다.

## 결과 폴더

기존 Output Session 구조(`docs/OUTPUT_SESSIONS.md`)를 그대로 쓰고 자동화
파일만 추가합니다.

```text
<output root>/<YYYY-MM-DD_HHMMSS>_<camera-model>_<serial>/
├── manifest.json                 # 기존 형식 + automation.* exports 항목
├── .session_claim                # 기존 CLI --session-dir가 세션을 연 표시 (중복 사용 방지)
├── run_manifest.json             # 실행 기록 (상태, 설정, 코드 버전, 입력 해시, 분할, 결과)
├── project/calibration.ccproj
├── intrinsic/camera_<model>.yaml, camera_info.yaml
├── reports/report.html, calibration.json, dataset.csv, image_selection_manifest.json
└── automation/
    ├── acceptance.json           # PASS/WARN/FAIL, fail/warn 사유와 종류(kind), 검사별 근거
    ├── run_summary.json          # 요약
    ├── profile_resolved.json     # 적용된 프로파일 전체
    ├── cli_config.yaml           # 기존 CLI --config (재실행용)
    ├── cli_summary.json          # 기존 CLI --json-summary
    ├── cli_output.log, cli_debug.log
```

실행마다 새 세션을 만들므로 이전 결과를 덮어쓰지 않습니다. 원본 이미지는
읽기만 합니다. 설치본(`pip install .`)에서 실행하면 git 정보 대신 설치 패키지 버전이
기록됩니다.

판정은 계산 결과의 품질 평가이며 차량 적용 승인이 아닙니다.

P2 실시간 자동 촬영/계산: [사용 방법](AUTOMATION_P2.md), [구현·검증 보고서](AUTOMATION_P2_REPORT.md).

P3 결과 전용 Git 게시/Outbox/카메라 이력: [사용 방법](AUTOMATION_P3.md), [구현·검증 보고서](AUTOMATION_P3_REPORT.md). 기본 비활성화이며 외부 remote에 대한 실제 게시 검증은 수행하지 않았다.
