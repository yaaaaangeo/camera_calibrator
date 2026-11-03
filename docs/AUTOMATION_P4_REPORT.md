# P4 개발·검증 최종 보고서

작성일: 2026-11-02. 작업 브랜치 `windshield`.
HEAD `acb22f147b277a600649e9ece84729c66dee1fb1` 유지.

## 1. 완료 범위와 판정

**파일 기반 비교 소프트웨어 경로를 구현·검증했다. 실제 제조사 Factory 값과의 성능 비교 및
SENSING 하드웨어 연동은 미검증이다.** 제조사 원본 파일, 실제 S/N·촬영 좌표 확인 자료와
Jetson/카메라가 없어 해당 항목을 성공으로 처리하지 않았다.

File load → 검증 → Train-only Local/Seeded → 동일 Test 평가 → 별도 JSON/HTML →
P3 로컬 bundle/history 경로를 검증했다. SDK는 조회 불가 상태를 명시하는 Provider로
분리했다. 기존 P1/P2 판정, 추천, 계산 알고리즘은 변경하지 않았다.

## 2. 신규·수정 파일

신규 소스:

- `automation/factory/__init__.py`: 선택 기능 패키지
- `automation/factory/provider.py`: Provider 계약, File Provider, 검증, SDK unavailable
- `automation/factory/compare.py`: Train-only 후보 생성·공통 평가·보고서
- `automation/factory/cli.py`: lookup/compare
- `automation/factory/service.py`: 실패가 Local 결과에 전파되지 않는 후처리
- `tests/test_factory.py`: 48개 고유 사례
- `tests/test_factory_gui.py`: 읽기 전용 보고서 표시 1개

기존 파일 수정은 다음 5개로 한정했다.

- `automation/cli.py`: factory 명령 및 P1 opt-in 후처리
- `automation/live/cli.py`: P2 완료 후 opt-in 후처리
- `automation/publish/bundle.py`: Session에 바인딩된 정제 비교 JSON/해시 확장
- `automation/publish/cli.py`: 명시적 `--factory-report` 첨부
- `ui/external_compare_view.py`: 읽기 전용 P4 JSON 보기 버튼

신규 문서: `AUTOMATION_P4.md`, 이 보고서, `factory_reference.example.json`.
실행 증거는 `.verification/p4/`에 보존했다. 예제 JSON의 값은 합성이며 제조사 값이 아니다.

## 3. Provider와 지원 형식

`FactoryProvider.lookup(camera_model, serial)` 계약을 분리했다. File Provider는 JSON envelope의
inline K/D 또는 envelope가 참조하는 OpenCV YAML/기존 calibration 파일을 읽는다.
기존 `calibration_io`와 `benchmark_compatibility`를 재사용한다. 메타데이터 없는 YAML은
식별정보를 추정하지 않으며, JSON에 해당 정보가 있어야 한다.

지원 수학 모델은 OpenCV Brown 5, Rational 8, Fisheye 4다. 파일 경로는 SG3S-ISX031C,
SG8-ISX028C 및 추가 기종의 정확한 모델 문자열을 받을 수 있다. **파일 지원은 해당 기종의
SDK/드라이버 지원 확인과 다르다.** 실제 vendor export 파일 자체는 제공되지 않았다.

## 4. SDK·S/N 조회 조사

[공식 sgMIX 공개 헤더](https://raw.githubusercontent.com/SENSING-Technology/sgMIX/master/include/SG_CameraControl.h)에서
연결된 카메라의 `SG_GetCameraSerialNumber`, `SG_GetCameraIntrinsic`와 EEPROM 주소 선택
API의 선언을 확인했다. 이들은 카메라 객체·장치 초기화 경로이며, 임의 S/N을 인수로 받는
서버 조회 계약은 아니다. EEPROM 주소 선택과 EEPROM 내용 쓰기를 혼동하지 않았다.
이번 구현은 장치 API를 호출하거나 EEPROM을 쓰지 않는다.

[공식 Intrinsic 안내](https://wiki.sensing-world.com/zh-Hans/docs/1_1_Serdes_Camera/GMSL_Camera/Getting_Started/Camera_Intrinsic_Parameters)는
S/N을 사용하는 온라인 다운로드 경로와 장치 데이터 읽기를 안내한다. 공개된 안정적
서버 API 계약·인증 방식은 확인하지 못했다. 자동 웹 요청을 추측하여 구현하지 않았다.

[현재 공식 README](https://github.com/SENSING-Technology/sgMIX)는 SDK V2의 AGX Thor /
JetPack 7.2 / V4L2·SIPL 범위를 설명한다. 검색 색인에 남은 이전 설명의 Orin /
JetPack 6+ 범위와 다르며, 공개 헤더·Wiki의 구 API와 현재 README의 초기화/Calibration
API도 차이가 있다. 따라서 SDK 버전 고정 없이 바이너리 호환성을 단정하지 않는다.
SG3S-ISX031C 변형은 문서에 포함되지만 SG8-ISX028C는 조사한 지원 목록에서 확인하지 못했다.
사용자 장치의 JetPack·보드·드라이버 조합 및 두 기종의 실제 호출은 미검증이다.

[공개 타입 헤더](https://raw.githubusercontent.com/SENSING-Technology/sgMIX/master/include/SG_CameraTypes.h)의
Pinhole 필드 순서는 `k1,k2,p1,p2,k3,k4,k5,k6`다. `k1~k6,p1,p2` 가정에 따른 재정렬은
구현하지 않았다. Fisheye는 필드 이름만으로 투영 수식의 동등성을 보장할 수 없고
vendor Wiki 수식도 주의가 필요하다. 입력의 `projection_convention: opencv` 선언을
필수로 하며 미확인 vendor 모델을 자동 변환하지 않는다. 공개 자료는 조회일 기준이며
장치에 설치된 SDK의 헤더·샘플·실측으로 다시 확인해야 한다.

## 5. 입력 검증

Camera Model/SN, 양의 정수 해상도, 기준 해상도, full-frame ROI, binning, orientation,
왜곡 모델, 명시적 계수 순서·개수, K shape/구조, NaN/Inf, 양의 focal length와
principal point 범위를 검사한다. 기존 호환성 validator도 적용한다.
모델/해상도가 다른 YAML, inline/file 중복 소스, 경로 이탈, 잘못된 JSON도 거부한다.
임의 rescale/crop/flip 보정은 하지 않는다. Factory RMS는 유효한 선택 메타데이터로만
보존하고 정의·촬영 조건이 불명확하므로 순위에 사용하지 않는다.
출처와 좌표는 사용자 선언이며 `factory_authenticated=false`다.

## 6. 후보와 누수 방지

Local chosen model의 기존 분할을 재사용한다. Factory-fixed K/D는 변경하지 않는다.
Local은 기존 `refit_on_train_split`으로 Train만 재학습한다. Seeded는 별도 실험 경로에서
Factory K와 D를 모두 초기값으로 사용한다. Brown/Rational은 해당 OpenCV flags,
Fisheye는 기존 버전 호환 helper와 intrinsic-guess 경로를 사용한다.
OpenCV 5의 Fisheye 플래그 네임스페이스/입력 shape 문제는 이 신규 경로에서 수정했다.

Test에는 K/D 최적화를 하지 않는다. 공통 `_evaluate_side`에서 동일 검출 코너와
object points를 사용하고, 모델별 PnP로 자세만 계산한다. 중복 ID/교집합/알 수 없는 ID/
비활성·미검출 프레임 및 동일 이미지 bytes·동일 관측의 Train/Test 중복을 거부한다.
Test 코너 해시·개수, 분할 ID와 알고리즘 설명을 보고서에 기록한다.
기존 helper가 Frame에 기록하는 재투영 오차도 원본 객체에 남지 않게 복사본에서 수행한다.
파일과 메모리의 원본 보존, Test 코너 변경 시 Seeded Train 결과 불변을 검증했다.
제조사 학습 데이터와의 독립성은 파일만으로 인증할 수 없다.

## 7. 평가·결론

Test RMS(동일 프레임 가중), pooled-corner RMS, Median/P95/P99, edge, straightness,
spatial map, K/D 차이, nominal FOV, 자유도와 기존 파라미터 진단을 출력한다.
서로 다른 모델의 D는 직접 뺄셈하지 않는다. 같은 모델의 63개 공통 광선 투영 비교는
P3 evaluator를 재사용한다. 다른 모델은 해당 투영 비교의 제한을 명시한다.

결론은 Factory/Local/Seeded 우세, UNCLEAR, INCOMPARABLE로 분리한다. frame bootstrap
95% CI, edge/straightness 비열화, 최소 Test 수와 미세 수치 차이 방어를 함께 사용한다.
부분 실패는 INCOMPARABLE이며 실패 프레임·후보를 보존한다. 세부 규칙은 사용 문서에
기록했다. 비교 결과는 기존 추천이나 승인 상태를 대체하지 않는다.

파라미터 진단은 Test fixed-pose sensitivity/covariance의 근사다. 반복 촬영 안정성이나
P2 Train Observability와 같은 값으로 해석하면 안 된다. 계산 불가 시 UNAVAILABLE로
명시하며 다른 후보의 성공을 감추지 않는다.

## 8. 합성 대조군

정답 Brown K/D에서 Fixed/Local/Seeded가 모두 Test RMS <0.001px를 달성한다.
진짜 정답인 Factory를 고정하고 Train에만 노이즈를 넣으면 Fixed가 Local보다 낮은
Test 오차를 보인다. 잘못된 Factory 계수에서는 Local <0.001px, Fixed >0.1px다.
다만 중앙 관측만 있는 대조군은 edge 근거가 부족하므로 전체 우세를 주장하지 않고
UNCLEAR를 유지한다. 알려진 모델의 Rational/Fisheye Fixed·Seeded도 <0.001px다.
최적화가 개선을 보장하지 않는 경우와 세 가지 우세 상태의 판정 로직을 별도 검증했다.

## 9. 실제 이미지 55장 기반 검증

기존 P1 Session `2026-11-02_163000_AUDIT_REAL`을 읽었다. 원본 입력 55장 중 선택 35장,
Train 27 / Test 8, Test 코너 332개다. 실제 제조사 파일·정확한 하드웨어/SN은 없다.
따라서 **Train-only Brown 결과를 시험용 참조로 저장하여 연결을 검증했으며, 이를
SENSING Factory 성능이나 제조사 GT로 주장하지 않는다.**

| 후보 | Test RMS px | Median px | P95 px | P99 px | Edge px | Straightness px |
|---|---:|---:|---:|---:|---:|---:|
| Train proxy-fixed | 4.271153 | 1.812050 | 6.250566 | 15.220885 | 3.662547 | 1.084462 |
| Local Train refit | 4.271154 | 1.812045 | 6.250566 | 15.220885 | 3.662549 | 1.084462 |
| Train proxy-seeded | 4.271148 | 1.812007 | 6.250520 | 15.220938 | 3.662521 | 1.084456 |

결론 UNCLEAR, 후보 실패 없음. Seeded 개선 CI는 약 `[-0.000038%, 0.000649%]`로 0을
포함한다. 차이는 개선 근거가 아니다. Fixed와 Local의 공통 광선 RMS 차이는 약
`8.2e-13px`다. 기존 저장된 CLI hold-out RMS 4.271152px와 약 수 µpx 범위에서 일치한다.
OpenCV 다중 스레드의 미세 수치 차이를 의미 있는 변화로 처리하지 않았다.

추가 fixed-pose 진단은 condition 약 60802.5, 최대 상관계수 약 0.984617이며 약한
fx/fy 식별가능성 경고를 남겼다. 이 condition은 P2의 다른 Jacobian 진단과 직접 비교하지 않는다.

보드 설정은 **추정 ChArUco 10×7 squares, square 0.04m, marker 0.03m,
DICT_5X5_100**, 이미지 3840×2160이다. 물리 보드, 실제 기종/SN, ROI/binning/orientation은
확인되지 않았다. 좌표는 이 시험의 선언값일 뿐이다. 정답 Intrinsic 정확도·실제 거리의
절대 정확도·차량 적용 적합성을 입증하지 못한다. 실제 Factory 비교는 미검증이다.

요약·원시 결과: `.verification/p4/real/summary.json`, 그 파일의 comparison_directory에
있는 JSON/HTML, `.verification/p4/real/parameter-diagnostics.json`.

## 10. 통합·GUI·P3

P1/P2 opt-in 호출 실패가 기존 WARN/완료 종료 코드를 바꾸지 않음을 검증했다.
실패 진단은 Session 밖 별도 JSON으로 남기며 완료된 비교로 P3에 첨부하지 않는다.
P1/P2 기본 옵션 및 실제 합성 E2E는 기존 CLI와 일치한다.
GUI는 보고서 읽기만 수행하며 기존 수동 입력과 외부 비교 동작을 유지한다.

P3 첨부는 Session manifest/project SHA와 모델/SN 바인딩을 검사한다. JSON은 기존
privacy cleaner를 통과하고 안전한 bundle HTML 요약 및 history에 노출된다.
원본 보고서 HTML은 올리지 않는다. 내용 변경은 hash에 반영하고 시간 의존 바인딩 hash는
semantic dedup에서 제외한다. 다른 Session 보고서 거부, 첨부 없는 기존 bundle,
시간만 바뀐 재패키징의 동일 hash를 검증했다. 로컬 bundle/history만 검증했으며
외부 GitHub/GitLab 게시·임시 Git commit/push도 이번 작업에서 수행하지 않았다.

## 11. 테스트 집계·실행 증거

**고유 테스트 340개: PASS 340 / FAIL 0 / SKIP 0 / XFAIL 0. 신규 49개 모두 PASS.**
전체 저장소 모든 테스트를 재실행했다는 뜻은 아니다. 여러 결과 파일의 동일 node는
최신 결과 한 번만 집계했다. `.verification/p4/test-result-ledger.json`, `test-summary.json`이
정확한 테스트명·판정·증거 파일을 제공한다.

| 범위 | 고유 PASS |
|---|---:|
| 신규 Factory core / GUI | 48 / 1 |
| P1 profile·acceptance·runner | 111 |
| P2 gate·readiness·live·runtime | 74 |
| P3 bundle·CLI·로컬 outbox/history | 53 |
| 기존 External Compare | 32 |
| 기존 External Compare GUI / P2 GUI | 6 / 4 |
| P1 E2E / P2 E2E | 9 / 2 |

주요 명령(인터프리터는 `venv/Scripts/python.exe`):

```text
-m pytest tests/test_factory.py -q --junitxml=.verification/p4/factory-final-all.xml
-m pytest tests/test_factory_gui.py -q --junitxml=.verification/p4/gui-factory-final.xml
-m pytest tests/test_automation_profile.py tests/test_automation_acceptance.py tests/test_automation_runner.py ...
-m pytest tests/test_p2_gate_policy.py tests/test_p2_readiness.py tests/test_p2_live.py tests/test_p2_runtime.py ...
-m pytest tests/test_publish_bundle.py tests/test_publish_cli.py tests/test_publish_outbox_history.py -k "not push_then_crash" ...
-m pytest tests/test_external_compare.py ...
-m pytest tests/test_external_compare_view.py ...
-m pytest tests/test_p2_gui.py ...
-m pytest tests/test_automation_e2e.py ...
-m pytest tests/test_p2_e2e.py ...
```

추가된 P2 hook·진단 실패 사례와 변경 후 hook 재검증은 `p2-hook.xml`, `diagnostic-final.xml`,
`hook-final.xml`에 있다. 초기 개발 실패(테스트 fixture, Fisheye namespace/shape, 과도한
bit equality 기대 등)는 이전 로그에 남겼으며 수정 후 해당 테스트 결과로 대체했다.
미수정 신규 실패는 없다. `compileall`, `git diff --check`도 통과했다.

장시간 P1 E2E는 84.98초, P2 E2E는 44.15초에 완료했다. 각각 한 프로세스로 한 번만
실행했고 별도 로그/XML을 남겼다. 전체 suite 단일 프로세스는 재시도하지 않았다.
P3 실제 Git commit/push 테스트는 제한에 따라 제외했다(선택식에서 1개 deselected;
다른 Git 전용 파일도 미실행). 제외·미실행을 SKIP/PASS에 포함하지 않았다.

기존 알려진 실패 4개(CLI chessboard, narrow responsive layout, recommender 2개) 및
장시간 Spline 미완료 6개는 이번 범위에서 재실행하지 않았다. 기존 보고서의 실패·미완료
상태를 해소한 것으로 간주하지 않는다. 이전 전체 집계와 이번 340개 집계를 합산하지 않는다.

## 12. 원본 보존·제한사항

P4 시작 당시 파일 8603개의 해시를 비교했다. 기존 파일 삭제 0, 수정은 위의 5개뿐이며
원본 `image/`와 기존 결과 및 Calibration/추천 코드가 보존됐다. 증거는
`.verification/p4/before-files.json`, `final-integrity.json`이다.
원래 미커밋 변경을 유지했고 Git HEAD는 동일하다. 커밋·푸시·배포·EEPROM 쓰기는 없다.
실제 ROS·카메라, Jetson SDK, 실물 Factory 데이터, 차량 적용은 미검증 상태다.

## 13. 다음 단계

실제 S/N에 해당하는 제조사 원본 파일과 정확한 계수/투영 모델, 촬영 mode의 해상도·ROI·
binning·orientation, 보드 실측 사양을 확보하면 현재 File Provider로 실데이터 비교를
진행할 수 있다. 그 전에는 Factory 우월성이나 Seeded 효과를 결론내릴 수 없다.
SDK 연동은 설치 버전의 헤더·라이브러리·드라이버 조합을 고정하고 읽기 전용 장치 검증
이후 추가한다. crop/변환 지원과 반복 촬영 안정성, 독립적인 추가 Test 촬영은 후속 범위다.
