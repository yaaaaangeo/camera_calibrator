# P2 자동 Intrinsic 캘리브레이션 구현·검증 보고서

검증일: 2026-11-02. 기존 미커밋 P1 변경을 보존하고 P2를 추가했다. 커밋·푸시·배포는 하지 않았다. Windshield 계산/가중치와 Intrinsic 알고리즘은 변경하지 않았다.

후속 실데이터·관측성 분석과 최소 버그 수정은 [P2 Observability 진단 보고서](P2_OBSERVABILITY_DIAGNOSIS.md)에 별도 기록했다. 아래 수치는 최초 P2 검증의 이력이며 후속 결과와 중복 합산하지 않는다.

현재 Readiness/Acceptance 분리 정책과 최신 검증은 [게이트 개선 보고서](P2_GATE_IMPROVEMENT_REPORT.md)를 참조한다.

## 1. 구현 결과와 검증 범위

ROS1/ROS2 어댑터와 이미지/rosbag 재생 입력, 자동 촬영, 단계별 준비도 판단, 사전 관측성 검사, P1 실행, 추가 촬영/수동 검토 분기, 세션 복구, Qt GUI 어댑터를 구현했다. 알려진 K를 가진 합성 이미지의 실제 subprocess 실행에서 자동 촬영부터 P1 4모델 계산·검증·저장까지 확인했다. 기본 운영 허용값 미설정에 따른 WARN은 재촬영하지 않고 검토 대상으로 완료한다.

실제 이미지 55장 재생은 관측성 기준 미달로 최종 계산에 진입하지 못했다. 실제 ROS 토픽/물리 카메라에서 전체 자동 흐름이 성공했다고 보고하지 않는다. 실제 ROS 미들웨어, 카메라 연결/재연결, Ubuntu/Jetson 환경은 미검증이다. 전이·재연결 로직의 단위 테스트와 생성한 ROS1/ROS2 bag 재생은 통과했다.

사용 방법: [AUTOMATION_P2.md](AUTOMATION_P2.md).

## 2. 신규·수정 파일

| 구분 | 파일 | 역할 |
|---|---|---|
| 신규 | `automation/live/__init__.py`, `config.py`, `default.yaml` | P2 패키지, 프로파일/기본값 검증 |
| 신규 | `automation/live/controller.py` | 촬영, 상태 머신, 결과 분기, 체크포인트/재개 |
| 신규 | `automation/live/readiness.py` | 기존 선별/분할 기반 준비도와 train-only 사전 관측성 |
| 신규 | `automation/live/sources.py` | ROS latest-only, 이미지/rosbag 재생 |
| 신규 | `automation/live/task.py`, `process.py`, `windows_job.py` | 격리 작업, timeout, owner 감시, 프로세스 트리 취소 |
| 신규 | `automation/live/storage.py` | 원자 JSON 저장, Windows 일시적 공유 오류 한정 재시도 |
| 신규 | `automation/live/cli.py` | 라이브/재생 실행 진입점 |
| 신규 | `ui/automatic_calibration_dialog.py` | QProcess 기반 얇은 GUI 패널 |
| 신규 | `tests/test_p2_live.py`, `test_p2_readiness.py`, `test_p2_runtime.py`, `test_p2_bag.py`, `test_p2_gui.py`, `test_p2_e2e.py` | 신규 고유 테스트 61개 |
| 신규 | `docs/AUTOMATION_P2.md`, `docs/AUTOMATION_P2_REPORT.md` | 사용 안내와 본 보고서 |
| 수정 | `automation/cli.py` | 기존 run/check와 분리된 live 디스패치/도움말 |
| 수정 | `ui/live_capture_dialog.py` | 자동 모드 버튼과 패널 연결 |
| 수정 | `pyproject.toml` | P2 YAML 패키지 데이터 포함 |
| 수정 | `docs/AUTOMATION.md` | P2 문서 링크 추가 |

P2 이전부터 있던 tracked 변경은 `app/cli.py`, `calibration/image_quality.py`, `calibration/image_selection.py`, `calibration/validation.py`, `docs/OUTPUT_SESSIONS.md`, `export/output_manager.py`, `pyproject.toml`이었다. 기존 untracked P1 automation/문서/테스트/검증 자료도 유지했다. 표의 pyproject 변경은 기존 변경 위에 P2 리소스 항목을 추가한 것이다. `.verification/p2/`에는 개별 로그·XML·체크포인트·패키징 자료를 저장했다.

## 3. 기존 코드 재사용과 실행 경계

| 기존 코드 | P2 사용 |
|---|---|
| `LiveTopicSubscriber`, `LatestFrameBuffer`, ROS image codec | 구독/디코딩, 최신 프레임 1개, 교체/전달 통계 |
| `build_detect_fn()` | ChArUco/Chessboard/Circle Grid/AprilGrid 검출 |
| `image_quality`의 pattern exposure, motion blur, pHash | 저장 전 품질 검사; 흰 보드 전체 포화를 과노출로 취급하지 않음 |
| `analyze_dataset_quality`, `compute_frame_quality_scores` | coverage·분포·거리·회전·가장자리 평가 |
| `select_calibration_images`, `split_train_test` | P1과 같은 후보 선별/seed/분할 규칙 |
| `evaluate_holdout_evidence` | 실제 분할된 hold-out 증거량 검사 |
| `calibrate_brown_conrady`, `compute_observability_report` | train-only 사전 계산; bootstrap 미실행 |
| `run_automation`, 기존 app CLI/추천/acceptance/manifest/export | 최종 4모델 계산과 결과 저장 그대로 호출 |
| `diagnose_calibration().capture_recommendations` | 결과에 따른 추가 촬영 안내 |
| `OutputManager`, project codecs, atomic JSON | 고유 출력 세션, 검출/프로젝트 복구, 기존 출력 호환 |

검출/품질 처리는 Qt 밖 제어 프로세스에서 실행한다. 사전 계산과 최종 계산은 별도 작업 프로세스에서 수행하며 그동안 새 검출/저장을 중지한다. ROS 구독은 최신 1프레임만 교체하고 계산 후 stale 프레임을 비운다. GUI는 JSON 상태를 읽고 표시한다. 원본 배열과 원본 파일은 변경하지 않는다.

최종 P1은 저장 PNG를 다시 검출한다. 기존 P1 경로와 결과 일치를 우선하여 중복 검출 캐시 전달 최적화는 적용하지 않았다. 사전 계산은 최종 4모델의 계산 결과를 대체하지 않는다.

## 4. 자동 촬영 판정

검출 실패는 사유를 기록하고 다음 프레임으로 진행한다. 검출 성공/코너 4개 이상, 저장 간격, 해상도 일치, Laplacian sharpness, motion blur, 보드 ROI contrast, ROI 밝기와 코너 주변 clipping을 검사한다. 검출 결과의 중심·크기·tilt·가장자리 정보와 품질 수치를 이벤트로 남긴다.

기존 수동 자동 캡처 기본값인 중심 이동 0.025×이미지 대각선, 상대 면적 변화 0.08, tilt 변화 5도를 사용한다. 이미 저장한 모든 자세와 비교해 세 조건 모두 비슷하면 제외한다. pHash도 기록/비교하며 유사 자세는 해시가 달라도 `duplicate_pose`로 제외한다. 저장 순서 접두어와 UUID로 파일명을 구성하고 독점 생성·fsync 후 등록한다. 해상도가 바뀌면 혼합하지 않고 ERROR로 중단한다.

ROI 노출은 P1 pattern 모드(기본 밝기 25..230, 코너 clipping 중앙값 ≤0.45)를 재사용한다. Blur는 P1 선별값을 사용한다. Motion ratio 3은 기존 경고 기준을 저장 게이트로 사용한 것이며 실제 모션 블러 정답 라벨에 대한 검증을 뜻하지 않는다. ROI contrast 10은 실험값이다.

## 5. 촬영 완료 조건과 수치 근거

매 프레임은 저장 판단만 수행한다. 최소 수량 도달 후 신규 저장 5장마다 선별/품질/분할을 평가한다. 유한 입력이 끝나면 주기에 못 미친 최종 신규 데이터도 한 번 평가한다. 가벼운 조건을 모두 만족할 때만 별도 프로세스에서 사전 Brown 계산과 관측성을 검사한다.

| 기본값 | 근거/의미 |
|---|---|
| 최소 선택 35장, 최대 저장 120장 | 35는 P1 target count; 120은 유한 자원 상한 |
| coverage ≥75%, 중심 3×3 중 ≥6칸, 네 모서리 모두 | 촬영 안내용 실험값; 실제 운영 승인 기준 아님 |
| distance/rotation/edge diversity 각각 ≥0.25 | 기존 점수 재사용, 임계값은 실험값; 실제 3D yaw/pitch 추정 보장 아님 |
| 후보 duplicate_scene 비율 ≤0.8 | 저장 후보에 P1 선별을 적용한 비율; 원시 영상의 중복 프레임 비율과 다름 |
| train ≥3장 | 기존 모델 최소 프레임 수 |
| hold-out ≥5장/60코너/coverage 40%/pose diversity 0.25 | 기존 holdout_evidence 기본값 |
| observability ≥50, full rank | 기존 WARNING 등급 경계 재사용; train-only Brown의 국소 관측성 |
| 관측성 검사 최대20회, 각120초 | 비용·무한 반복 제한용 실험 기본값 |
| 최종 계산 최대3회/회당1800초, 추가 촬영 최대2회/새5장 이상 | 재시도/자원 제한용 기본값 |
| stream timeout5초/재접속 간격3초/최대5회 | 연결 복구 정책 기본값; 현장 네트워크에서 미검증 |

모든 값은 프로파일에서 관리한다. P1 이미지 선별·노출·acceptance 값은 중첩 `p1`로 관리한다. 검증되지 않은 실험값을 운영 허용값으로 확정하지 않았다. 사전 관측성은 기존 fixed-pose local intrinsic Jacobian이며 프레임별 extrinsic 결합을 주변화하지 않는다. Brown 통과가 다른 모델의 관측성 보장을 뜻하지 않는다.

## 6. P1 연동·추가 촬영·복구

`IDLE → CONNECTING → CAPTURING → CHECKING_READINESS → CALIBRATING → EVALUATING`을 기록하고 결과에 따라 `COMPLETED`, 추가 `CAPTURING`, `ERROR`, `CANCELLED`로 이동한다. P1 결과는 attempt별 고유 세션에 원래 형식으로 저장한다. train/test seed, 모델 수식·가중치·추천은 변경하지 않았다.

PASS는 완료한다. 설정/시스템/유효하지 않은 결과는 운영 기준 미설정보다 우선하여 ERROR로 중단한다. 나머지 WARN/FAIL 중 운영 기준 미설정·필수 metric 누락은 촬영을 반복하지 않고 수동 검토 대상으로 완료한다. 품질 문제에 명시적 capture recommendation이 있으면 기존 정상 이미지를 유지하고 새 프레임을 추가하여 재평가한다. 분류되지 않은 P1 실행 실패도 보수적으로 ERROR 처리하므로 모든 검출 실패를 자동 재촬영으로 분류하지는 않는다. 반복 상한 도달 시 자동 중단한다.

체크포인트는 프로파일·식별·입력 연결·검출 결과·이미지 SHA256·작업 ID·카운터·결과를 저장한다. OS 파일 잠금으로 동일 세션 동시 실행을 차단한다. 완료 응답이 있으면 재개 시 그대로 반영하고 재계산하지 않는다. 중단된 응답 없는 작업은 제한 횟수를 유지한 채 다시 평가한다. 잘린 마지막 이벤트는 완료 캡처로 간주하지 않는다. 저장에 실패해 미등록된 PNG는 보존하지만 자동 채택하지 않는다.

취소/timeout은 자기 작업 트리만 종료한다. Windows는 KILL_ON_JOB_CLOSE Job Object와 worker 자체 가입으로 venv wrapper의 자식 생성 경합도 처리했다. 실제 손자 프로세스 종료 테스트를 통과했다. POSIX는 별도 process group을 사용하나 이번 환경에서 실행하지 않았다. Windows 동작 근거: [Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects). GUI는 취소 파일을 전달하고 응답이 없으면 10초 후 강제 종료하며 이전 실행의 타이머가 새 실행을 종료하지 않도록 세대 번호를 검사한다.

## 7. 테스트 명령과 정확한 집계

환경은 Windows/Python venv이며 GUI 테스트는 `QT_QPA_PLATFORM=offscreen`, `PYTHONUTF8=1`로 실행했다. 아래 최신 증거만 집계했다. 수정 검증을 위해 수행했던 이전 XML 및 단독 재확인 XML은 중복 합산하지 않았다.

| 증거 (`.verification/p2/`) | 고유 수 | PASS | FAIL | SKIP | XFAIL |
|---|---:|---:|---:|---:|---:|
| `final-frozen-unit.xml` P2 단위/런타임/GUI/bag | 59 | 59 | 0 | 0 | 0 |
| `final-e2e.xml` P2 실제 subprocess/CLI 일치 | 2 | 2 | 0 | 0 | 0 |
| `p1-regression.xml` P1 자동화/CLI 세션 | 138 | 138 | 0 | 0 | 0 |
| `validation-regression.xml` validation/result validity | 35 | 35 | 0 | 0 | 0 |
| `regressions/*.xml` 파일별 회귀 | 151 | 148 | 2 | 0 | 1 |
| **이번 고유 실행 합계** | **385** | **382** | **2** | **0** | **1** |

고유 ID별 원장은 `test-result-ledger.json`, 집계는 `test-summary.json`이다. 초기 존재하지 않는 `test_holdout_evidence.py` 실행은 exit4/0case였고 성공·실패 테스트 수에 넣지 않았다. 실제 hold-out 관련 `test_validation.py`, `test_result_validity.py`를 별도 실행하여 35개 통과했다. 단위 테스트의 mock 전이 검증과 E2E 실제 계산 검증은 구분했다.

```powershell
venv/Scripts/python.exe -m pytest -q tests/test_p2_live.py tests/test_p2_gui.py tests/test_p2_readiness.py tests/test_p2_runtime.py tests/test_p2_bag.py --basetemp=.verification/p2/final-frozen-unit --junitxml=.verification/p2/final-frozen-unit.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_e2e.py --junitxml=.verification/p2/final-e2e.xml
venv/Scripts/python.exe -m pytest -q tests/test_automation_acceptance.py tests/test_automation_profile.py tests/test_automation_runner.py tests/test_automation_e2e.py tests/test_cli_session_dir.py --junitxml=.verification/p2/p1-regression.xml
venv/Scripts/python.exe -m pytest -q tests/test_validation.py tests/test_result_validity.py --junitxml=.verification/p2/validation-regression.xml
venv/Scripts/python.exe .verification/p2/run_regressions.py
```

E2E는60.94초, P1 회귀군은100.35초, validation군은5.71초였다. P2 단위군은 마지막 오류 우선순위 보완 후 재확인했으며 시간은 `final-frozen-unit.xml`에 기록했다. 로그/실행 디렉터리는 개별 검증 자료에 남겼다. 장시간 테스트는 관련 변경의 검증에 필요한 경우만 재실행하고 완료한 E2E나 기존 Spline을 불필요하게 반복하지 않았다.

E2E는 알려진 K의 합성24이미지를 입력하고16이미지에서 시작하는 테스트 전용 완화 준비도를 사용했다. 운영 임계값을 낮춘 것이 아니다. 자동 저장→사전 관측성→P1 4모델→기본null임계값에 따른WARN→검토 완료를 확인했다. 같은 저장 이미지를 직접CLI에 전달하고 OpenCV 단일 스레드에서 K/D 배열, train/test/edge RMS, 분할ID, 추천 모델이 일치했다. PASS/FAIL, 품질 재촬영, 횟수 상한, 복구, 저장 장애 등의 분기는 단위 테스트로 확인했다. GUI는 실제QProcess의 시작→입력 부족ERROR→재조작 가능도 확인했다.

## 8. 기존 기준선과의 관계

P1 전체 기준선은1,376개 = PASS1,358 / FAIL4 / SKIP4 / XFAIL4 / 미완료6이다. 이번385개는 변경 관련 재실행 부분집합이며 전체1,376개 재검증이 아니다. 신규61개를 포함해도 과거 합격 결과를 이번 실행 결과에 더해 전체 재합격으로 간주하지 않는다.

| 기존 실패 | 원인·영향 | P2에서의 처리 |
|---|---|---|
| `test_cli_chessboard_does_not_require_marker_size` | 합성Chessboard 4장으로 전 모델 계산 실패, exit2. 작은 합성 데이터의 계산 성립에 영향 | 이번에도FAIL. P1에서HEAD 재현 확인 |
| `test_rows_and_charts_stack_at_narrow_width_without_overlap` | 800px에서 세로 배치 대신LeftToRight. 기존 좁은GUI 레이아웃에 영향 | 이번에도FAIL. P1에서HEAD 재현 확인 |
| `test_recommender_picks_extended_pinhole_when_distortion_is_clean` | Rational 기대에Brown 추천. 모델 선택 기대 불일치 | P1에서HEAD 재현 확인. 이번 미재실행 |
| `test_recommender_majority_correct_under_noise[true_fisheye_noisy]` | Rational/Brown/Brown으로 정답 과반 미충족. 잡음 합성 데이터의 추천 정확도에 영향 | P1에서HEAD 재현 확인. 이번 미재실행 |

이번 실행 범위에서 신규 실패는 확인하지 않았다. 모든 기존 테스트의 무영향을 보장하는 결과는 아니다. 기존 Spline 마지막6개는 미완료 상태를 유지하며 PASS로 바꾸지 않았다.

P1 단일 프로세스 전체 실행은790결과 뒤 비정상 종료했고 별도 GUI 연속 실행에서도 Windows heap corruption 0xc0000374가 있었다. 이는 P2의385개 집계와 별도 실행이다. P2는 GUI 포함 회귀를 파일별 프로세스로 실행하여 XML/로그를 회수했다. 단일 프로세스 전체 실행이 정상 완료했다고 보고하지 않는다. P1 상세는 [안정화 보고서](AUTOMATION_STABILIZATION_REPORT.md)를 참조한다.

## 9. 실제 이미지55장과 처리 시간

실행 증거는 `.verification/p2/final-real-runs/*/p2_manifest.json`, `capture_events.jsonl`이고 프로파일은 `.verification/p2/real-profile.yaml`이다. 입력은3840×2160 저장 이미지55장, ChArUco **10×7 squares / DICT_5X5_100 / square40mm / marker30mm（비0.75）**를 가정했다. 물리 보드 확인값이 아닌 이미지 기반 추정 사양이다. 잘못된 치수/사전/마커 비율은 대응점·스케일·자세·추정 결과에 영향을 주므로 정밀도나 실카메라 합격 확정에 사용할 수 없다. 균일 스케일의 모호성만으로 모든 사양 오류를 설명할 수는 없다.

| 최종 재생 결과 | 값 |
|---|---|
| 처리/저장/제외 | 55 / 54 / 1（duplicate_pose） |
| 준비도 검사/최종 계산/추가 촬영 | 5 / 0 / 0 |
| 최종 선택/coverage/중심 분포 | 35 / 75% / 7 of 9 |
| hold-out | 8장,332코너,coverage62.5%,diversity0.78125 |
| 사전 관측성 | score3.5,rank9/9,normalized condition100.011,최대 상관0.9833 |
| 종료 | ERROR/input_exhausted. 관측성 부족으로 최종P1 미실행 |

rank가 충분해도score 기준을 충족하지 않아 추가 촬영 안내를 유지했다. 실제 이미지의PASS, 최종4모델 성공, 실카메라 운영 준비 완료를 의미하지 않는다. 가속을 위해 저장 간격0.001초/재생 간격0.01초로 실행했다. 기본 저장 간격은1초이며 이를 실시간 수신fps로 해석하지 않는다.

검출 호출55장의 합계는5.457초였다. 5회 준비도 선별/ROI재읽기는 각각6.813/7.657/8.531/9.797/10.500초, 사전Brown＋관측성은0.281/0.281/0.437/0.360/0.329초였다. 다른 회귀 테스트도 실행하던 환경의 단일 관측값으로 성능 보장이 아니다. 프레임 품질 처리 전체의 추가 비용은 분리 측정하지 않았다.

이번 주요 대기는 fitting 없는 준비도 단계의4K재읽기/후보 평가였다. 5장 주기와 계산 중 촬영 정지로 부하를 제한하며 latest-only버퍼로 메모리 누적을 막는다. GUI는 별도 프로세스지만 이 동안 라이브 검출/촬영은 진행하지 않는다. 현장에서 허용 지연을 측정하고 결과 일치를 유지하는 캐시 등을 후속 검토해야 한다. 관측성을 매 프레임 계산하지 않는다.

## 10. 패키지와 원본 보존

`pip install . --no-deps --no-build-isolation --upgrade --target .verification/p2/installed`가 성공했다. 저장소 밖 임시 폴더에서 설치 경로만PYTHONPATH에 넣고 P2/GUI import, YAML리소스, `automation check --profile default`, `automation live --help`를 확인했다. 설치된P2관련14파일은 소스와 바이트 일치했다.

원본55장 집약SHA256은 P1감사 기록과 일치했다. 증거는 `.verification/p2/package-and-source-check.json`이다. 검증 출력은 전용 경로에 생성하여 P1결과나 원본을 덮어쓰지 않았다.

## 11. 한계와 후속 작업의 선행 조건

실ROS의 대역폭/드라이버/실기 연결 복구, 실제GUI의 시인성/응답성, Linux/Jetson 프로세스 종료, 모든 패턴의 실물 촬영은 미검증이다. 생성bag은 실ROS구독 검증을 대체하지 않는다. bag의1회poll이 오래 블로킹하면 협조 취소가 늦어져 GUI 강제 종료가 필요할 수 있다. 중단 후 체크포인트로 복구한다.

GitHub 업로드 전에는 차이·기존 실패·라이선스/비밀 정보·이미지 공개 가능 여부를 검토하고 사용자 지시를 받아야 한다. Factory/차량 적용 전에는 물리 보드 사양과Serial연결을 확정하고 실ROS/실기 연속 시험, 독립 검증 데이터, 모델별 허용 기준 승인, 적용/롤백 절차, EEPROM/API사양·권한을 확정해야 한다. 이번에는 해당 연계와 자동 적용을 구현/실행하지 않았다.
