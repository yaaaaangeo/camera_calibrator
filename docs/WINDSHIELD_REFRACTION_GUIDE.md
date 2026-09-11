# Windshield Refraction 실전 가이드

이 문서는 Camera Intrinsic Calibration은 알고 있지만 Windshield 기능은 처음인 사용자를 위한 작업 매뉴얼입니다. UI의 영어 이름은 프로그램에 표시되는 그대로 적었습니다. 수식과 구현 상세는 [README_EXPERT.md](README_EXPERT.md#6-windshield-geometry)를 참고하세요.

> **실차 검증 상태**: 이 저장소에는 실차 캡처 데이터셋이 없습니다. 아래 절차로 얻은 결과도 실제 차량의 거리·온도·시야각 범위에서 별도로 검증해야 합니다.

## 30초 요약

```text
Camera Intrinsic Calibration 완료
        ↓
① Base Camera 불러오기
        ↓
② Dataset에서 windshield를 통과해 찍은 target 이미지 검출
        ↓
③ Baseline 실행 → Spherical 실행
        ↓
④ Comparison에서 Hold-out/Test 성능 비교
        ↓
필요할 때만 Residual Ray 또는 Spline 실행
        ↓
③에서 선택한 결과를 Export Windshield YAML...
```

가장 중요한 원칙은 다음 한 줄입니다.

> **Base K,D는 Windshield Calibration 중 다시 최적화하지 않습니다.**

이 도구는 `Camera intrinsic distortion + Windshield refraction correction`을 한 모델로 뒤섞지 않습니다. 먼저 렌즈 자체의 K(초점거리·주점)와 D(왜곡)를 확정하고, 그 값을 고정한 다음 앞유리가 더 만드는 광선/픽셀 이동만 별도 계층으로 맞춥니다.

## 1. 실험 전 데이터 준비

### Base Camera 데이터

Base Camera는 **앞유리 보정을 얹을 기준 intrinsic 결과**입니다. 가능하면 카메라·렌즈·초점·해상도가 실제 차량 운용 조건과 같은 상태에서 먼저 일반 Camera Intrinsic Calibration을 완료하세요.

| 버튼 | 가져오는 것 | 언제 사용하나 |
|---|---|---|
| `Load from current session` | 현재 실행에서 성공한 intrinsic 결과와 Camera/Pattern 설정 | 방금 intrinsic을 계산한 경우. **가장 간단한 추천 경로** |
| `Load from Library...` | Library에 저장된 run의 프로젝트와 성공한 모델 | 이전에 같은 카메라 조건으로 저장한 run을 재사용할 때 |
| `Load OpenCV YAML...` | camera matrix K와 distortion D, 감지 가능하면 camera model | 외부 OpenCV calibration을 가져올 때 |
| `Load .ccproj...` | 프로젝트의 intrinsic 결과, Camera/Pattern 설정 | 작업을 저장했다가 이어갈 때. **재현성에 유리한 추천 경로** |

여러 성공 모델이 있으면 모델 선택 창이 열립니다. 실제 운용에서 사용할 intrinsic 모델을 고르세요.

> **OpenCV YAML 주의**: YAML만으로는 이미지 해상도와 calibration pattern을 신뢰성 있게 복원할 수 없습니다. 현재 세션에 기존 정보가 있으면 유지하지만, 새 세션에서 YAML만 불러오면 Pattern 정보가 없어 `Load windshield images...` 검출을 시작할 수 있습니다. 이 경우 current session 또는 `.ccproj`를 사용하세요.

성공하면 K/D, 해상도와 함께 `🔒 Base K,D fixed during windshield calibration`이 표시됩니다. 이는 읽기 전용 표시가 아니라 계산 계약입니다. Windshield 모델은 이 K/D를 변경하지 않습니다.

### Windshield Dataset 촬영

Intrinsic용 이미지를 그대로 재사용하는 것이 아니라, **카메라를 실제 차량 설치 위치에 고정하고 앞유리를 통과해 동일한 calibration target을 촬영한 이미지**가 필요합니다. Dataset 검출은 Base Camera에서 가져온 `PatternConfig`의 target 종류와 치수/ID 설정을 그대로 사용합니다.

- 실제 운용 카메라, 렌즈, 초점, 해상도와 windshield 상태를 유지합니다.
- target을 중앙에만 두지 말고 좌·우·상·하·corner와 특히 edge를 포함합니다.
- 가까운 거리와 먼 거리, 다양한 yaw/pitch를 섞습니다.
- target이 너무 작거나 흐리거나 반사광에 가려지지 않게 합니다.
- 같은 포즈를 연사한 중복 사진 수보다 공간·거리·각도 다양성을 우선합니다.
- Base 프로젝트와 같은 target 종류 및 실제 치수를 사용합니다.

| 좋은 데이터 | 나쁜 데이터 |
|---|---|
| target이 화면 전역과 edge/corner에 분포 | target이 중앙에만 반복 |
| 여러 거리와 yaw/pitch | 거의 같은 거리·정면 포즈만 연사 |
| 코너/마커가 선명하고 완전히 식별됨 | 모션 블러, 과노출, 심한 반사로 target이 가려짐 |
| 실제 설치 위치와 앞유리를 유지 | 카메라를 탈거하거나 앞유리 없이 촬영 |
| 같은 PatternConfig와 target 사용 | 다른 board 치수/사전/ID 배열을 사용 |

## 2. GUI 클릭 순서

### STEP 1 — `① Base Camera`

처음 실험은 `Load from current session` 또는 `Load .ccproj...`를 권장합니다. 외부 YAML은 K/D만 있고 pattern/해상도가 빠질 수 있으므로 이미 Dataset이 준비된 고급 사용 경로에 가깝습니다.

Base 표시에서 다음을 확인합니다.

- `Camera Model`: intrinsic 투영/왜곡 모델이 의도한 모델인지
- `fx/fy/cx/cy`, `Distortion`: 불러온 calibration과 일치하는지
- `Image Size`: windshield 이미지 해상도와 일치하는지
- 잠금 문구: 이후 모델 실행 중 K/D가 고정된다는 뜻

### STEP 2 — `② Dataset`

`Load windshield images...`를 누르고 촬영 이미지를 선택하면 이미지 로딩과 현재 PatternConfig 기반 target detection이 실행됩니다.

| 표시 | 의미 |
|---|---|
| `Images` | 선택한 전체 이미지 수 |
| `Valid` | target 검출에 성공해 계산에 쓸 수 있는 이미지 수 |
| `Coverage` | `Valid / Images × 100%`인 검출 성공 비율. 화면 공간 coverage score가 아님 |

Detection 실패가 많으면 Base에 Pattern 정보가 있는지, target 종류·행/열·실제 치수·dictionary가 맞는지, 이미지가 흐리거나 target이 잘리지 않았는지 확인합니다. Valid 수뿐 아니라 포즈와 화면 영역의 다양성도 필요합니다.

### STEP 3 — `③ Windshield Model`

모델을 선택하고 `Run`을 누릅니다. 현재 config의 기본 분할은 Test 25%, seed 42이며 모든 비교는 고정 Base K/D를 사용합니다. 실행한 각 모델 결과는 `④ Comparison`에 누적됩니다. 화면에 마지막으로 표시된 성공 결과가 `Export Windshield YAML...`의 대상입니다.

| 모델 | 목적 | 난이도 | 데이터 요구량 | 계산량 | 언제 사용? |
|---|---|---:|---:|---:|---|
| Baseline | Windshield correction 없이 고정 Base K/D만 평가 | 낮음 | 낮음 | 낮음 | 모든 실험의 기준선 |
| Spherical | 앞유리를 동심 구면과 Snell 굴절로 근사 | 중간 | 중간 | 중간 | 매끄러운 전역 굴절이 주효할 때 첫 보정 후보 |
| Residual Ray — Grid | 이미지 위치별 3D ray correction을 grid+bilinear로 학습 | 중간 | 높음 | 높음 | 구면으로 남는 위치별 잔차가 있고 coverage가 충분할 때 |
| Residual Ray — RBF | 성긴 center의 ray correction을 thin-plate RBF로 보간 | 중간 | 높음 | 높음 | 코너 분포가 불규칙하거나 Grid와 다른 smoothness가 필요할 때 |
| Residual Ray — Neural | 작은 MLP가 ray correction을 학습 | 높음/실험적 | 매우 높음 | 매우 높음 | PyTorch가 있고 단순 모델의 Hold-out 한계를 확인한 연구용 |
| Spline `[Advanced]` | 고정 Base Sphere 위 bicubic B-spline 표면 변형과 Snell 굴절 | 높음 | 매우 높음 | 매우 높음 | 국소 표면 형상 보정이 물리 모델로 꼭 필요할 때 |

초보자 권장 순서:

1. Baseline을 실행해 기준 Hold-out 오차를 기록합니다.
2. Spherical을 기본값으로 실행합니다.
3. `④ Comparison`에서 Test/Hold-out 개선을 확인합니다.
4. Spherical로 부족하고 edge까지 데이터가 충분할 때 Residual Ray Grid/RBF를 시도합니다.
5. 더 복잡한 물리 surface correction이 필요하고 데이터가 충분할 때만 Spline을 검토합니다.

가장 복잡한 모델이 자동으로 가장 좋은 모델은 아닙니다.

## 3. Advanced 옵션

> **초보자 Recommended Settings: 대부분 AUTO/default 그대로 사용하세요.** Manual은 치수 측정값이나 반복 검증 근거가 있을 때만 사용합니다.

### Spherical

값을 0으로 두면 UI의 `(default)`/`(auto)` sentinel로 처리되어 config에 강제로 기록되지 않습니다.

| 옵션 | 뜻·단위 | 기본 동작 | Manual을 고려할 때 |
|---|---|---|---|
| `Glass refractive index` | 유리 굴절률, 무차원 | 약 1.52 | 재질 사양의 신뢰할 수 있는 값을 알 때 |
| `Glass thickness (mm)` | 유리 두께, mm | 약 5 mm | 실제 적층 유리 두께를 측정/확인했을 때 |
| `Initial sphere radius (m)` | 구면 최적화 초기 반지름, m | 5.0 m | 최적화 초기화가 반복 실패하고 형상 prior가 있을 때 |
| `Initial standoff distance (m)` | 카메라에서 초기 구면까지의 거리 근사, m | 1.0 m | 설치 기하의 신뢰할 수 있는 초기값이 있을 때 |

굴절률과 두께는 고정 입력이고 sphere center/radius는 데이터에서 맞춥니다. 굴절률을 공기와 같게 두면 굴절이 사라져 sphere 형상 식별이 불가능해질 수 있습니다.

### Residual Ray — Grid

- `AUTO (권장)`: 바깥 Train 안에서 다시 Hold-out해 3×4, 4×6, 6×8, 8×12 후보를 비교합니다. RMS가 5% 이내로 비슷하면 더 단순한 grid를 택합니다.
- `Manual`: `Grid Rows`와 `Grid Cols`를 그대로 사용합니다. 기본 표시는 6×8입니다.
- `Magnitude λ`: correction 크기를 0 쪽으로 누르는 정규화(기본 0.001). 너무 낮으면 과적합, 너무 높으면 보정 부족 위험이 있습니다.
- `Smoothness λ`: 이웃 node가 급격히 달라지지 않게 하는 정규화(기본 0.01).

### Residual Ray — RBF

- `Kernel`: 현재 UI와 runtime은 `Thin Plate Spline`을 사용합니다.
- `AUTO`: Train 내부에서 Centers 32/64/128/256과 Smoothing 1e-5/1e-4/1e-3 후보를 비교하며, 비슷하면 center가 적은 후보를 택합니다.
- `Manual`: `Centers`(기본 64)와 `Smoothing`(기본 0.0001)을 직접 적용합니다.
- Centers를 늘리면 국소 표현력과 데이터 요구량이 함께 증가합니다. Smoothing을 높이면 더 매끈하지만 세부 correction이 줄 수 있습니다.

### Residual Ray — Neural

Neural에는 architecture AUTO search가 없습니다. UI 값이 그대로 적용되며 구조는 `2 → 32 → 64 → 32 → 3`, SiLU로 고정입니다.

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `Epochs (max)` | 최대 학습 epoch | 500 |
| `Learning Rate` | optimizer step 크기 | 0.001 |
| `Weight Decay` | weight 정규화 | 0.0001 |
| `Magnitude λ` | correction 크기 정규화 | 0.01 |
| `Smoothness λ` | 인접 좌표 출력의 매끄러움 정규화 | 0.01 |
| `Patience` | validation 개선이 없을 때 early-stop 대기 epoch | 30 |
| `Seed` | 재현 가능한 초기화/분할 seed | 42 |
| `Batch Size` | 한 학습 step의 샘플 수 | 128 |

`Early Stop`은 항상 ON입니다. 처음에는 모든 값을 그대로 두세요. Neural은 선택적 PyTorch 의존성이 필요하고 CPU에서도 다른 모델보다 오래 걸릴 수 있습니다.

### Spline `[Advanced]`

- Base Surface는 먼저 fitting한 `Spherical (frozen)`입니다.
- `AUTO (권장)`는 4×4, 4×6, 6×8 control grid를 Train 내부 Hold-out으로 비교합니다.
- Manual `Rows/Cols`는 각각 최소 4이며 기본 표시는 4×6입니다.
- `Magnitude λ`(0.01), `Smoothness λ`(0.1), `Curvature λ`(0.1)는 변형 크기·인접 변화·곡률을 제어합니다.
- `Max deformation` 기본값은 10 mm이며 병적인 표면 변형을 제한합니다.

## 4. 결과 읽기

`RESULT (Train / Test)`는 계산에 사용한 Train과 떼어 둔 Hold-out/Test를 나란히 보여줍니다.

| 지표 | 해석 |
|---|---|
| Train / Test | fitting에 사용한 프레임 / 모델 fitting에는 쓰지 않은 프레임 |
| RMS | pixel residual 제곱평균제곱근. 큰 오차에 민감하며 낮을수록 좋음 |
| Median | 중앙 residual. 전형적인 오차 크기 |
| P95 / P99 | 각각 95%/99%가 이 값 이하인 tail 지표. 극단 영역 실패 확인 |
| Mean dx / dy | 평균 수평/수직 signed pixel residual. 0에서 지속적으로 벗어나면 방향성 bias 가능 |
| Regional Error | center/left/right/top/bottom/corner 영역별 오차 |
| Edge Avg | left/right/top/bottom/corner 지역 오차의 평균. 화면 가장자리 품질 판단 |
| Ray Angular Error | 관측 ray와 target 방향 사이 각도(도). 코드상 물리 Spherical 계열에서 의미 있게 제공되며 미지원 모델은 N/A일 수 있음 |

Radial Profile은 중심에서 edge로 갈수록 오차가 어떻게 변하는지, Vector Field는 공간별 평균 dx/dy 방향과 크기를 보여줍니다.

> **Train이 좋아지는 것만 보면 안 됩니다. Test/Hold-out도 함께 좋아져야 합니다.** Train만 크게 낮아지고 Test가 유지되거나 악화되면 모델이 촬영 프레임을 외웠을 가능성이 있습니다.

간단한 선택 규칙:

```text
Test RMS 낮음 + Edge Avg 낮음 + P95/P99 낮음
          + Train–Test gap이 과도하게 커지지 않음
          + 반복 Hold-out/Stability가 안정적
                         ↓
                 더 나은 후보
```

Baseline 대비 Test 개선이 작다면 단순한 Baseline/Spherical을 유지하는 편이 안전할 수 있습니다. 최저 Train RMS만으로 Residual/Neural/Spline을 고르지 마세요.

## 5. `④ Comparison`

실행해 둔 Baseline, Spherical, Residual Grid/RBF/Neural, Spline을 열로 누적 비교합니다. 표의 행은 `Hold-out RMS`, Hold-out `P95`, Hold-out `Edge RMS`, `Ray Angular Error`, Baseline 대비 `Improvement %`입니다.

1. 먼저 Baseline 열이 있는지 확인합니다.
2. 각 후보의 Hold-out RMS/P95/Edge RMS가 함께 낮아지는지 봅니다.
3. 한 지표만 좋아지고 tail/edge가 나빠지지 않았는지 확인합니다.
4. 복잡한 모델의 개선이 작거나 불안정하면 단순 모델을 선택합니다.

Comparison은 Geometry 전용입니다. Reflection/Ghost 점수는 이 표의 승자 결정에 들어가지 않습니다.

## 6. Export와 Runtime

성공한 결과가 화면에 표시되면 `Export Windshield YAML...`이 활성화됩니다. 이 버튼은 **현재 마지막으로 표시된 모델**을 저장합니다. 원하는 모델을 다시 선택해 `Run`하고 결과를 확인한 뒤 Export하세요.

저장 파일은 일반 OpenCV intrinsic YAML을 수정하거나 덮어쓰지 않는 별도 파일이며 실제 구조는 다음과 같습니다.

```yaml
format_version: 1
base_camera:
  camera_model: brown_conrady
  camera_matrix: ...
  distortion_coefficients: ...
  distortion_coefficient_order: ...
  image_width: 1920
  image_height: 1080
windshield:
  model: spherical
  train_rms: ...
  test_rms: ...
  fitted_param_names: ...
  fitted_param_<name>: ...
```

즉, 고정 Base K/D의 스냅샷과 선택한 Windshield correction 파라미터가 한 runtime artifact 안에 구분되어 저장됩니다. Neural Residual은 weight를 YAML에 수천 개 float로 펼치지 않고 같은 폴더에 `<yaml_stem>_neural.pt`를 만들며 YAML의 `neural_state_dict_file`이 그 파일을 가리킵니다. 배포할 때 두 파일을 함께 옮기세요.

Python runtime의 exact 모델 재구성:

```python
from export.windshield import windshield_model_from_yaml

model = windshield_model_from_yaml("vehicle_windshield.yml")
uv = model.project_point(x, y, z)
ray = model.unproject_pixel(u, v)
```

대량 포인트에는 LUT 근사 계층이 있습니다. 메모리에 `WindshieldCalibrationResult`가 있으면 `build_runtime_projector(result, width, height)`를 쓰고, Export YAML에서 시작하면 위 exact `model`을 `RuntimeWindshieldProjector(model, width, height)`로 감쌉니다. 가까운 3D 점의 parallax 및 LUT 오차가 있으므로 `validate_runtime_projector_vs_exact()`로 실제 거리/FOV 범위를 검증한 뒤 사용하세요. 정확도가 우선이면 exact API를 사용합니다.

## 7. Refraction, Reflection, Ghost 구분

| 기능 | 다루는 현상 | 판단/출력 |
|---|---|---|
| Geometry / Refraction | 유리를 통과한 광선 때문에 픽셀 위치가 기하학적으로 어디로 이동하는가 | Hold-out pixel/ray error, projection model |
| Reflection | 하늘·차내 조명·대시보드처럼 다른 빛/영상이 유리에 겹쳐 비치는가 | Reference 차이 또는 No-Reference likelihood |
| Ghost | 같은 외부 장면이 내부 다중 반사로 어긋나 두 개처럼 보이는가 | displacement dx/dy와 strength 또는 likelihood |

**Windshield Model 결과가 좋지 않다고 Reflection Suppression을 실행하는 것이 아닙니다.** Geometry, Reflection, Ghost는 독립 문제입니다. Reflection Suppression은 별도로 학습된 PyTorch 모델의 inference이고, Ghost Suppression은 GhostField 기반 결정론적 반복 복원입니다. 둘도 서로 교환해 쓸 수 없습니다.

## 8. Troubleshooting

| 증상 | 가능한 원인 | 확인할 것 | 해결 |
|---|---|---|---|
| Base Camera를 못 불러옴 | 성공 intrinsic 결과가 없음/파일 오류 | 현재 결과의 `success`, YAML/.ccproj 경로 | intrinsic을 먼저 성공시키거나 올바른 프로젝트를 선택 |
| Dataset 클릭 시 Pattern 정보 없음 | OpenCV YAML만 새로 불러옴 | Base 표시의 pattern 경고 | current session 또는 `.ccproj`로 Base를 다시 로드 |
| Valid 이미지가 너무 적음 | 설정 불일치, blur, target 가림 | target type/rows/cols/size/dictionary와 실패 이미지 | 설정을 맞추고 선명한 다양한 이미지를 재촬영 |
| Calibration 실패 | 유효 프레임/코너 부족 또는 비물리적 초기화 | 화면의 오류·warning, Dataset Valid | Baseline부터 확인하고 기본값 복원, 데이터 coverage 확대 |
| Spherical이 Baseline보다 나쁨 | 구면 근사가 부적합하거나 데이터/초기값 문제 | Hold-out RMS/P95/Edge, Train–Test gap | 기본값으로 재실행, 데이터 다양성 점검; 개선 없으면 Baseline 유지 |
| Train은 좋아졌지만 Test가 나쁨 | 과적합 또는 분포 차이 | Train/Test gap과 test edge/tail | 단순 모델/AUTO 사용, 중복 제거, 포즈·edge 데이터 추가 |
| Edge만 나쁨 | edge 촬영 부족 또는 edge에서 모델 실패 | Regional Error, Edge Avg, radial chart | target을 네 변/corner에 더 배치해 재촬영 |
| Residual Ray 과적합 의심 | grid/center 과다, 정규화 부족 | Hold-out, P95/P99, 반복 stability | AUTO 사용, 더 단순 후보, λ/smoothing 증가를 신중히 검증 |
| Neural이 오래 걸림 | 반복 학습/CPU 실행 | Epochs, Patience, PyTorch 환경 | 기본 early stop 유지; 먼저 Grid/RBF 사용, 필요 시 max epochs 축소 후 검증 |
| Export가 비활성 | 아직 성공 결과가 없음 | Run 결과와 error message | Base/Dataset 준비 후 모델을 성공적으로 실행 |
| 작은 화면에서 잘려 보임 | 긴 페이지의 아래쪽 내용 | 탭 내부 세로 scrollbar, tab scroll 화살표 | 창을 800×600 이상으로 두고 페이지를 스크롤; OS DPI가 크면 실제 모니터에서 추가 확인 |

## 9. 처음 실험용 체크리스트

- [ ] 같은 카메라/렌즈/초점/해상도의 intrinsic을 완료했다.
- [ ] current session 또는 `.ccproj`로 Base를 불러와 K/D 잠금을 확인했다.
- [ ] 실제 설치 위치에서 windshield를 통과한 target 이미지를 골고루 촬영했다.
- [ ] Dataset의 Valid와 촬영 다양성을 확인했다.
- [ ] Baseline과 Spherical을 기본값으로 실행했다.
- [ ] Comparison의 Hold-out RMS/P95/Edge를 함께 비교했다.
- [ ] 복잡한 모델은 실제 Test 개선이 있을 때만 선택했다.
- [ ] 원하는 결과를 다시 표시한 뒤 Windshield YAML을 Export했다.
- [ ] runtime exact/LUT를 실제 운용 거리와 FOV에서 검증했다.
