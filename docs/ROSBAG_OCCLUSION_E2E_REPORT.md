# 실제 ROSbag + Occlusion Mask 전체 자동화 E2E 검증

2026-11-03 실행. **자동 실행 완료 / 결과 WARN / 3개 모델 계산 성공·Fisheye 실패 / 로컬 P3 게시 성공**. PASS 또는 차량 적용 승인이 아니다.

## 실행 조건과 보존

- 녹화: `rosbag2_2026_10_23-15_38_30`, 표준 ROS2 MCAP, `/video0/image_raw`, native BGR8 3840×2160.
- 이전 프로파일의 input/live 설정과 Polygon Mask를 그대로 재사용했다. `extract_fps=1`, `replay_interval=1.1`, target_count=35, test_ratio=0.25, seed=42, OpenCV threads=1.
- 보드: ChArUco 10×7, DICT_5X5_100, Square 100mm는 사용자 확인. **Marker 75mm는 UNCONFIRMED_ASSUMPTION**이며 정확도 검증이 완료됐다고 주장하지 않는다.
- 모델/SN `UNKNOWN / DIAG-video0`는 진단 식별자다. 실제 하드웨어 식별을 검증하지 않았다.
- 출력/로컬 Bare Git 대상만 분리한 프로파일: `.verification/e2e-mask/profile.yaml`. 원본 프로파일·마스크는 수정하지 않았다.
- Factory 참조가 없어 기존 정책대로 SKIPPED. 알고리즘·추천 가중치·판정 임계값·소스 코드는 변경하지 않았다.
- 원본 bag fingerprint는 앞선 감사 결과와 일치했다. 이전 저장 90장 및 새 저장 35장 픽셀 보존을 확인했다. 기존 소스 Python 192개 파일 SHA-256도 동일하다. 코드 저장소 HEAD는 `acb22f147b277a600649e9ece84729c66dee1fb1`로 유지된다.

## 자동 촬영 종료와 Coverage

- 샘플 처리 38개 → 저장 35장, 거부 3개(중복 카운터 2는 거부와 별도 합산하지 않음). 자동 계산 1회, 추가 촬영 0회.
- 이번 실제 자동 실행은 준비도를 처음 충족한 35장 시점에서 종료했다. 전체 1,142프레임 무결성 검사는 아니다.

|자료|Full-frame Coverage|Observable Coverage|공간 준비도|
|---|---:|---:|---|
|이전 전체 90장 기반 선정 35장 진단|75.00%|92.37%|충족|
|이번 자동 종료 35장|68.75%|77.03%|충족|

설정은 같지만 종료 시점과 입력 후보가 달라 두 35장 구성은 같지 않다. 이전 92.37%를 이번 실행 결과로 재사용하지 않았다. 실제 적용된 Polygon과 각 셀·면적·마스크 SHA·버전·출처는 P2 Manifest에 남는다. 보닛 영역 정확도를 검증했다는 의미가 아니다.

## 4개 모델 계산과 Hold-out

|모델|계산|최종 fit RMS(px)|Hold-out Train RMS(px)|Test RMS(px)|Edge RMS(px)|Test P95(px)|
|---|---|---:|---:|---:|---:|---:|
|pinhole|수치 계산 성공|20.819605|19.356177|12.939660|11.855998|29.329885|
|brown_conrady|수치 계산 성공|12.293655|2.584170|2.718053|2.720808|5.482956|
|extended_pinhole|수치 계산 성공|10.515849|1.500972|1.591968|1.471250|3.357127|
|fisheye|FAIL|N/A|N/A|N/A|N/A|N/A|

4개 모델 모두 동일 Train 28장/Test 7장 분할을 사용했고 교집합이 없다. 성공한 3개 모델의 Test pose 추정 실패는 0장이며 Hold-out 구성은 sufficient다.

Fisheye는 최종 fit에서 OpenCV `InitExtrinsics`, `fabs(norm_u1)>0` 오류로 실패했다. Hold-out Train fit도 `CALIB_CHECK_COND / ill-conditioned matrix`로 실패했다. 기존 알고리즘의 문제 프레임 제외 재시도 후에도 안정화되지 않았다. 자동 실행과 직접 CLI에서 같은 실패 메시지를 확인했다. 실패 모델에 K/D·RMS·검증 성공을 부여하거나 YAML을 만들지 않았다.

표의 최종 fit RMS와 Hold-out Train RMS는 서로 다른 fit의 값이다. 예를 들어 Rational 최종 RMS=10.515849px와 Hold-out Train=1.500972px/Test=1.591968px를 혼동하면 안 된다. 기존 검증 경로는 Train subset에 별도 fit 후 Test를 평가한다. 이 Test RMS가 Export된 최종 fit K/D 자체를 직접 평가한 수치라고 주장하지 않는다. 이 구조는 변경하지 않았다.

## 추천·Acceptance

- 추천/선택: **Rational (`extended_pinhole`)**, 기존 모델 추천 결과 그대로.
- P1 WARN, P2 WARN. 운영 지표 기준이 미설정되어 criteria_complete=false이며 PASS 불가.
- 기존 종합 등급 POOR, 최종 Confidence REJECT(score 약 6.9). 모델 간 상대 추천과 최종 신뢰도는 다르다.
- Rational 최종 RMS 약 10.516px, 왜곡 계수 최대 절대값 약 27.130으로 Sanity WARNING.
- 사전 Brown Rank 9/9, Condition 약 130.819, 최대 상관 0.983116(k2/k3), Score 3.6.
- 최종 Rational Rank 12/12, Condition 약 7,797,556.50, 최대 상관 0.999884(k1/k4), Score 0.0. 기존 정책에 따라 계산은 허용하지만 PASS 승인은 차단된다.
- 차량 적용/자동 승인/배포는 수행하지 않았다. 로컬 게시도 UNREVIEWED / NOT_DEPLOYED다.

## 기존 CLI 비교와 원본 좌표 검증

기존 `automation/cli_config.yaml`의 이미지 경로·설정·seed·분할 정책을 그대로 사용하고 출력 경로만 별도로 지정해 기존 `app.cli`를 한 번 실행했다. 마스크 옵션은 CLI 설정에 없다. 같은 입력의 불필요한 재계산은 하지 않았으며 이 한 번은 독립 비교 목적이다.

- Pinhole/Brown/Rational: K/D ndarray, RMS, Hold-out Train/Test/Edge RMS **비트 단위 동일**.
- Fisheye: 실패 상태·오류 메시지·분할이 동일. K/D/RMS는 비교 불가능(N/A).
- 모든 모델 Train/Test ID 동일·누수 없음, 검출 코너 배열 동일, 추천 모델 동일.
- 새 저장 35장 전부 bag의 이벤트 timestamp 대응 프레임과 대조하여 BGR 픽셀 배열 완전 일치. 리사이즈·크롭·마스킹 이미지 변조 없음.
- 원본 좌표 K/D가 같다는 재현성 검증이며 실제 정답 정확도 검증은 아니다.

## Export

프로파일의 OpenCV YAML, ROS CameraInfo YAML, JSON, HTML, CSV, project가 모두 생성됐고 비어 있지 않음을 확인했다. 기존 동작은 추천/최종 모델의 YAML을 생성하므로 모델 4개에 YAML 4개를 요구하지 않았다. 실패한 Fisheye를 성공 export로 처리하지 않았다.

- `intrinsic/camera_rational.yaml`: 최종 Rational K/D/RMS와 정확히 일치.
- `intrinsic/camera_info.yaml`: 3840×2160, rational_polynomial, K/D는 기존 소수점 8자리 반올림 범위 내 일치(검사 atol 5.1e-9).
- `reports/calibration.json`, `reports/report.html`, `reports/dataset.csv`, `project/calibration.ccproj` 및 Acceptance/선정 Manifest.
- 세션 결과 파일 총 18개. 전체 목록과 수치/선정 ID/분할은 `.verification/e2e-mask/comparison.json`에 기록.

최종 Rational K:

```text
5731.9735880779 0.0000000000 1949.6906629609
0.0000000000 5447.5181030351 960.6755131113
0.0000000000 0.0000000000 1.0000000000
```

최종 Rational D (OpenCV k1,k2,p1,p2,k3,k4,k5,k6):

```text
-0.6934019791, 10.0919696537, 0.0523458779, 0.0245290244, -22.5921427625, 1.0113437450, 10.3410459607, -27.1301600079
```

## 게시 실패·복구·중복 방지

첫 `GitPublisher.publish` 호출 직전에 재시도 가능한 오류를 의도적으로 주입했다. 실제 외부 네트워크 장애/인증 실패 재현은 아니다. 계산 완료 결과를 유지한 채 실행 exit=2, publication RETRY_WAIT, calibration_verdict WARN으로 구분됐다.

이후 동일 프로파일의 retry-publish 경로를 사용해 실제 P3 Git Publisher로 로컬 Bare Git에 게시했다. 종료 코드 3은 WARN 유지의 정상 완료이며 실행 실패를 뜻하지 않는다.

- 로컬 저장소: `.verification/e2e-results.git`
- 브랜치: `calibration-results`
- 게시 Commit SHA: **`ab16111e3c82bef733407065a6422bff4b3d6bcd`**
- Bundle hash: `15173ab81bcb8d1e0c05d4d8ba69812ef13dc0c7c5f5924892237a78bfbdc5b3`
- 게시 시도 2회(주입 오류 1회 + 성공 1회), 실제 저장소 Commit 1개.
- 재시도에서는 publication 단계만 실행했다. 자동 계산 1회 유지, 결과 18개 파일 해시 동일.
- 동일 세션 enqueue/process를 한 번 확인하여 기존 job/receipt를 재사용했고 Publisher 재호출 및 두 번째 Commit이 없음을 검증했다.
- 게시 번들 verify/security scan 통과. 원본 이미지·bag·project·raw log는 게시하지 않았다. 기존 허용 목록의 정제 JSON/YAML/HTML만 게시했다.
- 기존 P3 번들 스키마는 변경하지 않았다. 두 Coverage·마스크·Marker 가정의 별도 감사 문맥은 로컬 Experiment/E2E Manifest에 기록되며 P3 번들에 새 필드로 추가되지는 않았다.

## Manifest·로그·재실행

- `.verification/e2e-mask/E2E_VERIFICATION_MANIFEST.json`: Marker 75mm 가정, Board/원본/실행/게시/테스트 근거.
- `.verification/e/experiments/b/experiment.json`: 실행 상태·P2/P1 소요시간·마스크 metadata·validation_context(두 Coverage와 Marker 가정). P1의 immutable run_manifest는 수정하지 않았다.
- P2 capture session의 `p2_manifest.json`, `capture_events.jsonl`, `capture.log`: Readiness, Observability, 실제 선정/좌표/해시.
- `run.log`, `failed-experiment.json`, `retry.log`, `retry-verification.json`: 최초 게시 실패와 복구를 분리 보존.
- `direct-cli.log`, `direct-process.json`, `comparison.json`, `pixel-verification.json`, `publication-verification.json`, `evidence-tests.xml`.

```powershell
# 이번 완료 결과를 확인/재게시할 때: 재계산하지 않는 명령
venv\Scripts\python.exe -m automation experiment retry-publish --profile .verification/e2e-mask/profile.yaml --dry-run
venv\Scripts\python.exe -m automation experiment retry-publish --profile .verification/e2e-mask/profile.yaml
```

이미 완료된 실제 계산을 다시 시작하지 않는다. 다른 설정/새 녹화 실험에는 별도 job_id와 출력 경로를 사용한다. 외부 Publisher를 켜는 변경은 하지 않았다.

## 테스트 결과와 남은 항목

이번 실제 E2E 증거에 대한 자동 assertion 테스트: **10 PASS / 0 FAIL / 0 SKIP / 0 XFAIL**. 원본 픽셀, 독립 CLI 일치, Fisheye 실패 보존, 공유 분할/누수, Coverage와 WARN, Export, 게시 실패 복구, 로컬 번들/중복, 보드 가정, 기존 소스/HEAD 보존을 검사했다. 이 테스트 PASS는 모델 4개가 모두 성공했다는 뜻이 아니다. 이전 361 PASS/1 SKIP 회귀를 이번 실행 수에 합산하지 않았다. 이번에 제품 소스 변경이 없어 전체 회귀를 불필요하게 반복하지 않았다.

남은 검증: Marker 실제 크기 확인, 실제 카메라 모델/SN 확인, Fisheye 초기 자세 불안정 프레임 진단, 높은 Rational 최종 오차·계수 결합 원인 검토, 독립 외부 Test/정답 기반 정확도 검증, 실제 ROS/카메라 연결, 실제 GitHub 인증/네트워크 게시. 하단 마스크 영역 정확도 검증·차량 적용은 미수행이다. 지금 결과를 운영용으로 승인하지 않는다.

코드 저장소 Commit·Push, 실제 외부 GitHub Push, 배포는 하지 않았다. 이번 요청에서 명시적으로 허용된 별도 로컬 결과 저장소 Commit·Push만 수행했다. 원본과 기존 미커밋 변경을 보존했다.
