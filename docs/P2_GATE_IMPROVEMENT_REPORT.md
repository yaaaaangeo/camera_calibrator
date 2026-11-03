# P2 Readiness / Acceptance 분리 결과

검증일: 2026-11-02. 이전 [관측성 진단](P2_OBSERVABILITY_DIAGNOSIS.md)에 이어 계산 시작 가능성과 결과 신뢰성 판정을 분리했다. 기존 계산 수식·추천 가중치·Score 산식·촬영/분포/hold-out 임계값은 유지했다. 원본·이전 결과·미커밋 변경을 보존하고 커밋·푸시·배포하지 않았다.

## 1. 변경된 판정 구조

| 항목 | 이전 역할 | 변경 후 역할 |
|---|---|---|
| 이미지 수/품질/선별, coverage, 중심/모서리, 거리/회전/edge 다양성 | 계산 전 필수 조건 | 유지 |
| train 최소/hold-out 증거량 | 계산 전 필수 조건 | 유지 |
| 사전 Brown fit 실패 | 계산 차단 | 유지, 횟수/신규 데이터 제한 내 재평가 |
| Jacobian 없음/진단 불가, 비유한 condition/Score/상관 | 계산 불가 또는 낮은Score에 혼재 | 명시적 feasibility blocker |
| full rank 부족 | 계산 전 차단 | 기본값에서 유지 |
| 유한한 큰 condition, 높은 상관, 낮은 Score | 낮은 Score로 계산 자체 차단 | 계산은 허용, 결과 검토 WARN |
| 최종 선택 모델의 rank 부족/평가 불가 | P1 기존 신호 매핑 의존 | P2 최종 FAIL, 승인 불가 |
| P1의 기존 WARN/FAIL | 기존 판정 | 유지, P2가 PASS로 승격하지 않음 |
| P1 PASS + 관측성 경고 | 계산에 진입하지 못할 수 있음 | P2 WARN, 수동 검토로 종료 |

기본 `observability_min=50`은 **검토 경계로 유지**했다. 기존 observability의 condition 경고1e8, 상관 경고0.98을 동일하게 사용한다. 새 수치를 정해 통과시킨 것이 아니다. 바뀐 것은 낮은 Score의 의미이며, 계산 가능한 결과를 보존한 뒤 승인 여부를 별도 판단한다.

`ready=true`는 계산 가능을 의미하고 `review_required=true`가 동시에 존재할 수 있다. 사전 진단은 여전히 train-only Brown이며 hold-out을 fit에 섞지 않는다. 최종 계산은 기존 P1 Runner/CLI 네 모델을 그대로 호출한다. 최종 선택 모델의 관측성 및 사전 train 경고를 P2 Acceptance에 반영한다. 사전 Brown 경고만 있고 다른 최종 모델이 양호한 경우에도 보수적으로 검토 대상으로 남긴다.

P2 overlay는 모든 모델의 진단을 보존하되 **선택되지 않은 복잡한 모델의 낮은 Score만으로 선택 모델을 FAIL시키지 않는다.** 선택 모델의 평가 불가/rank 부족은 FAIL, 유한한 높은 결합/condition/낮은 Score는 WARN이다. `require_full_rank=false`라는 기존 옵션을 유지하더라도 최종 rank 부족을 PASS로 승인하지 않는다. 상관계수가 산출되지 않으면 진단 불가로 보수적으로 차단하므로 이 옵션이 모든 부족한 진단을 우회하는 것은 아니다.

## 2. 결과 저장과 사용상 변화

P1의 `automation/acceptance.json`, `run_manifest.json`, 추천/프로젝트/HTML 보고서의 원래 판정은 변경하지 않는다. 새 실행에 다음 파일을 추가한다.

```text
<P1 attempt 세션>/automation/p2_acceptance.json
  policy_version: 2
  p1_verdict / verdict / review_required
  chosen_model / model_observability / preflight
  blockers / warnings / effective acceptance
```

P2 체크포인트/응답에는 원래 `p1_outcome`, `p1_acceptance`와 적용된 P2 판정을 함께 저장한다. GUI는 P2 판정/P1 원판정과 P2 근거 파일 경로를 구분해 표시한다. P2에서 시작한 실행의 최종 품질은 **P2 manifest와 p2_acceptance.json**을 기준으로 읽어야 한다. P1 HTML/acceptance만 읽으면 P2 overlay 경고를 놓칠 수 있다.

CLI 제어 프로세스의 종료0은 여전히 COMPLETED이며 품질PASS가 아니다. 내부 attempt의 품질 exit code는 PASS0/WARN3/FAIL4를 사용한다. 미완료/런타임 실패는 별도로 유지한다.

## 3. 반복 촬영/계산과 복구

- 기존 중복 자세/간격/품질 제외, 최대120장/관측성20회/계산3회/재촬영2회, 추가5장 조건을 유지한다.
- 선택 경로와 train/test ID를 fingerprint로 저장한다. 새5장을 확보해도 선택된 계산 데이터/분할이 같으면 동일 계산을 반복하지 않는다. 이후 선택이 바뀌면 재평가할 수 있다. 이미 시작한 뒤 중단된 작업의 복구 재시도는 예외이며 제한 카운터를 초기화하지 않는다.
- P1 PASS에 관측성 경고만 추가된 경우 자동 재촬영하지 않고 WARN/수동 검토로 완료한다. P1의 실제 품질 문제와 구체적 capture recommendation은 기존 추가 촬영 분기로 처리한다. 운영 기준 미설정은 촬영으로 해결하지 않는다.
- 관측성 경고와 기존 위치별 coaching을 함께 보존한다. 낮은 Score라는 이유만으로 포화된 거리/회전 지표를 무조건 더 늘리라고 안내하지 않는다. 하단 부족 등 현재 데이터의 구체적 coaching을 계산/검토 단계에도 전달한다.
- 정책2를 체크포인트와 작업 요청에 기록한다. 이전 정책의 완료 preflight 응답은 **숫자 보고서를 재판정**하며 refit하지 않는다. 이 과정에서 생성된 계산 작업도 런타임이 실행하도록 복구 경로를 보완했다.
- 미완료 세션에서 발견한 P1-only 완료 응답을 현재 정책의 PASS로 자동 승인하지 않고 검토 대상으로 보존한다. 이전 response/P1 결과 파일은 덮어쓰지 않는다. 이미 COMPLETED인 역사적 세션은 재개만으로 소급 수정/재계산하지 않는다.

실시간 ROS에서 새 자세가 전혀 들어오지 않으면 유효 데이터 대기 상태는 계속될 수 있다. 중복 프레임을 저장하거나 같은 fit을 무한 반복하는 것은 아니다. 유한 재생이 부족한 상태로 종료되면 ERROR/input_exhausted이며 확보한 진단 결과를 성공으로 승격하지 않는다.

## 4. 동일 입력의 정책 전후 비교

이전 진단의 18개 실데이터 fit과 3개 알려진K 대조군 보고서에 같은 기본 설정을 적용했다. 보고서를 읽어 정책만 비교했으며 기존 fit을 불필요하게 반복하지 않았다. 근거: `.verification/p2-gate/same-input-policy-comparison.json`.

| 동일 보고서 | Rank | norm condition | 최대상관 | Score | 이전 계산 시작 | 변경 후 |
|---|---:|---:|---:|---:|---|---|
| 원본55 Brown train43 | 9/9 | 105.642 | .980031 | 5.2 | 차단 | 계산 가능/검토 |
| 선택35 Brown train27 | 9/9 | 100.011 | .983317 | 3.5 | 차단 | 계산 가능/검토 |
| 합성 상·중36 전체 fit | 9/9 | 529.259 | .980989 | 4.7 | 차단 | 계산 가능/검토 |
| 합성 하단 추가60 전체 fit | 9/9 | 489.156 | .983509 | 3.4 | 차단 | 계산 가능/검토 |
| 합성 거리/tilt/edge 추가96 전체 fit | 9/9 | 512.216 | .984004 | 3.2 | 차단 | 계산 가능/검토 |

위 합성 대응점은 알려진 K를0.0002px 이내로 복원했지만 다항식 기저의 높은 상관 때문에 낮은 Score를 얻는다. 데이터가 좋아지면 상관 기반 Score가 반드시 단조 증가하는 것은 아니다. 이전 게이트의 false-block 사례는 해소하지만, 이 결과를 실제 센서가 양호하다는 근거로 사용하지 않는다.

### 새 합성 train-only 대조군

추가로 알려진 K(fx=fy1000px), Brown D, seed214의36개 물리적으로 일관된 투영 뷰를 생성했다. 실데이터와 동일한 split ratio0.25/seed42, 기본 review threshold50을 사용했다. 대응점 실험이므로 blur/노출/선별 이미지 게이트를 검증한 것은 아니다. 별도의 합성 이미지 E2E가 실제 검출/저장/Runner 연결을 검증한다.

| 모델 | Rank | norm condition | 최대상관 | Score | 변경 후 |
|---|---:|---:|---:|---:|---|
| Brown | 9/9 | 4756.691 | .987082 | 1.5 | 계산 가능/WARN |
| Rational | 12/12 | 7.147e9 | 거의1 | 0.0 | 계산 가능/WARN |
| Fisheye | 8/8 | 124864.770 | .992776 | 0.0 | 계산 가능/WARN |

Brown의 K복원오차0.01px 이내/D오차0.001 이내 assertion을 통과했다. Rational/Fisheye가 생성에 사용한 실제 모델이라는 뜻은 아니다. 서로 다른 모델/기저의 Score를 정확도 순위로 사용하지 않았다.

## 5. 실제55장 재생 통합 결과

동일한 원본55장 디렉터리, 추정 ChArUco10×7/DICT_5X5_100/square40mm/marker30mm를 사용했다. 물리 보드 사양은 여전히 미확인이다. 기존 coverage75%, 중심6/9, diversity0.25, hold-out5장/60코너/40%/0.25를 유지했다. 시간을 줄이기 위한 저장간격0.001초/재생0.01초만 이전 감사와 동일하게 사용했다.

| 항목 | 결과 |
|---|---|
| 입력 | 기존55장 전체 디렉터리 |
| 실제 소비/저장/제외 | 36 / 35 / 1（중복 자세） |
| 준비도 | coverage75%, 중심6/9, distance/rotation1.0, edge0.6667 |
| hold-out | 7장/323코너/coverage62.5%/diversity0.78125 |
| 사전 Brown | rank9/9, condition123.571, correlation.982459, score4.0 |
| 제어 흐름 | 자동 계산1회, 추가 촬영0회, COMPLETED |
| 품질 | P1 WARN, P2 WARN/수동 검토 |
| 추천/선택 | Fisheye, 직접CLI와 동일 |

**55장을 모두 소비한 것은 아니다.** 변경 후에는35장 유효 데이터에서 계산 조건을 충족하여 먼저 계산하고 멈췄다. 이전에는 Score 때문에54장까지 계속 저장했다. 따라서 이전 최종 선택35장과 이번 초기35장은 서로 다르며, 추천 모델/숫자를 그대로 비교하면 안 된다. 추천 가중치 변경이 아니라 자동 종료 시점과 입력 집합의 변화다. 별도의 고정55/선택35 보고서 정책 비교가 이 영향을 분리한다.

### 이번 자동 계산과 같은 입력의 직접 CLI

자동 실행의 정확한 입력/설정을 직접 CLI에 한 번 더 전달했다. 네 모델 모두 **K/D 배열 완전 동일, 전체 fit RMS/test RMS/edge RMS 차이0**, 추천 모델도 동일했다. 원본 집약SHA256도 일치했다.

| 모델 | 전체 fit RMS(px) | hold-out train RMS | hold-out test RMS | edge RMS | 전체 fit Score |
|---|---:|---:|---:|---:|---:|
| Pinhole | 31.279863 | 32.280048 | 29.334292 | 26.567392 | 100.0 |
| Brown | 3.542105 | 3.673699 | 3.076082 | 3.003530 | 4.2 |
| Rational | 10.521003 | 8.581644 | 10.046234 | 9.326053 | 0.0 |
| Fisheye | 1.853063 | 2.080403 | 1.410111 | 1.276381 | 0.0 |

선택 Fisheye는 rank8/8, condition1196.963, 최대상관.991142로 계산은 가능하지만 검토 경고가 남는다. P1 기존 신뢰도REJECT/운영 기준 미설정도 유지했다. **정밀도 합격이나 차량 적용 가능을 의미하지 않는다.** P1 hold-out Fisheye가 불안정한 train 프레임4장을 제외한 경고도 양쪽 CLI에 동일하게 기록됐다.

실행 근거: `.verification/p2-gate/real/real55_gate0/`의 원본 해시 검사, stdout/stderr, 자동 세션, 직접CLI 결과 및 JUnit. 재생+자동 계산+직접CLI 비교2개 테스트는234.83초에 완료했다.

## 6. 테스트 및 파일 변경

신규 제품 모듈: `automation/live/assessment.py`.

수정 제품 파일: `automation/live/readiness.py`, `controller.py`, `task.py`, `cli.py`, `config.py`(기본값 주석), `default.yaml`(역할 주석), `ui/automatic_calibration_dialog.py`. P1 acceptance/runner, calibration 모델/observability 산식, 추천 코드는 변경하지 않았다. 이전 턴의 관측성 frame alignment 수정도 그대로 보존했다.

신규 테스트: `tests/test_p2_gate_policy.py`, `test_p2_gate_synthetic.py`, `test_p2_gate_real.py`. 기존 `test_p2_e2e.py`에서 관측성 최소0/full-rank 해제 override를 제거하여 기본 관측성 정책을 사용하게 했다. 기존 안내 회귀 `test_p2_observability_diagnostics.py`는 단순 낮은 Score가 아니라 실제 rank 부족을 차단 사례로 검증하도록 갱신했다. 합성 이미지 E2E의 나머지 공간/수량 완화값은 연결 시험용이며 운영 검증값이 아니다.

```powershell
venv/Scripts/python.exe -m pytest -q tests/test_p2_live.py tests/test_p2_readiness.py tests/test_p2_runtime.py tests/test_p2_gui.py tests/test_p2_bag.py tests/test_p2_observability_diagnostics.py tests/test_p2_gate_policy.py --basetemp=.verification/p2-gate/unit-final --junitxml=.verification/p2-gate/unit-final.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_gate_synthetic.py --basetemp=.verification/p2-gate/synthetic --junitxml=.verification/p2-gate/synthetic.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_gate_real.py --basetemp=.verification/p2-gate/real --junitxml=.verification/p2-gate/real.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_e2e.py --basetemp=.verification/p2-gate/e2e --junitxml=.verification/p2-gate/e2e.xml
venv/Scripts/python.exe -m pytest -q tests/test_automation_acceptance.py tests/test_automation_profile.py tests/test_automation_runner.py tests/test_automation_e2e.py tests/test_cli_session_dir.py --basetemp=.verification/p2-gate/p1 --junitxml=.verification/p2-gate/p1.xml
venv/Scripts/python.exe -m pytest -q tests/test_observability.py tests/test_observability_frame_alignment.py tests/test_quality_pose_distribution.py tests/test_diagnosis.py --basetemp=.verification/p2-gate/metrics --junitxml=.verification/p2-gate/metrics.xml
```

| 최신 증거 | 고유 PASS |
|---|---:|
| `unit-final.xml` P2/게이트/취소/복구/GUI/bag | 82 |
| `synthetic.xml` 알려진K/모델별 진단 | 5 |
| `real.xml` 실제55장 재생/CLI일치 | 2 |
| `e2e.xml` 합성 이미지 전체 흐름/CLI일치 | 2 |
| `p1.xml` 기존 P1 자동화/세션 | 138 |
| `metrics.xml` 관측성/품질/진단 | 29 |
| **합계** | **258** |

FAIL0/SKIP0/XFAIL0/미완료0. 원장은 `.verification/p2-gate/test-result-ledger.json`, 요약은 `test-summary.json`이다. 중간 재확인 XML은 합산하지 않았다. 실제 데이터 테스트는55장이 없는 환경에서는 명시적SKIP하며, 이번 로컬 실행에서는2개 모두PASS다. 기존 전체1376개/알려진FAIL4개/Spline미완료6개를 이번에 모두 재검증하거나 해결했다는 뜻은 아니다.

설치 확인도 `pip install . --no-deps --no-build-isolation --target .verification/p2-gate/installed`로 수행했다. 저장소 밖 임시 폴더에서 정책2/GUI import, 기존 `automation check --profile default`, `automation live --help`를 확인했고 설치된 live 모듈은 소스와 일치했다. 근거는 `install.log`, `package-check.json`이다. `final-integrity.json`은 원본55장 해시, P1/모델/추천 코드 보존 및 Capture/Readiness/Recovery 기본값의 이전 스냅샷 대비 동일성을 기록한다. `git diff --check`도 통과했다.

## 7. 남은 한계와 다음 단계

- 이번 변경은 **진단 계산의 오차단을 줄이고 낮은 신뢰도 승인을 막는 구조 개선**이다. 낮은 Score의 false-WARN은 여전히 가능하며, 추가 촬영하면 반드시 점수가 좋아진다고 보장하지 않는다.
- full rank와 유한 condition은 fixed-pose 국소 Jacobian의 수치 조건이다. extrinsic 결합을 주변화한 전체 식별가능성이나 독립 데이터의 정확도를 보장하지 않는다. 잘못된 모델도 높은 Score를 가질 수 있으므로 P1의 잔차/hold-out/validity와 운영 승인 기준을 계속 적용한다.
- 단일 실카메라 데이터 집합과 합성 대조군만으로 오탐/미탐율을 추정하지 않았다. 라벨이 있는 다양한 카메라/렌즈/보드/노이즈와 독립 검증 데이터가 필요하다.
- 기존 distance/tilt 지표는 면적/영상 내 회전의 대리값이며 minAreaRect 각도 wrap 한계가 남는다. 하단 관측 부족은 coaching에 표시하지만 단순 모서리 존재/coverage75%만으로 하단의 충분한 밀도를 보장하지 않는다.
- 이전 정책보다 빨리 계산을 시작하므로 뒤의 좋은 프레임을 아직 수집하지 못할 수 있다. 이를 승인으로 간주하지 않고 결과 품질/독립 검증/실제 결손에 따라 추가 촬영 또는 수동 검토해야 한다.
- 실제 ROS 구독/카메라·네트워크 재연결/현장 GUI와 Linux/Jetson은 **미검증**이다. 생성bag/이미지 replay와 단위 테스트를 하드웨어 성공으로 보고하지 않는다.
- 현재는 오프라인 자동 계산/검토 단계로 진행 가능하다. Factory/차량 자동 적용 전에 물리 보드/Serial/운영 허용값을 확정하고 실ROS/실기 시험 및 독립 hold-out 승인 절차를 거쳐야 한다.
