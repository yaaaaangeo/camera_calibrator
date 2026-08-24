# Evaluation Metric Spec — Cam–LiDAR Calibration Evaluation Tool
## 0. 목적

**이미 존재하는** camera–LiDAR extrinsic calibration(`T_CL`)의 품질을
GT(ground truth) 없이 평가한다. 새로운 calibration을 계산하지 않는다 —
이미 갖고 있는 calibration을 얼마나 신뢰할 수 있는지, GT 없이 판단하는
것이 유일한 목적이다.

실 운영 환경에서는 비교할 ground-truth `T_CL`을 거의 가질 수 없다.
GT 없이도 측정 가능한 것들이 이 tool의 근거가 된다:

- 이 calibration이 *지금* LiDAR 구조를 이미지 edge에 맞게 투영하는가? (M2)
- 여러 시간대에 걸쳐 일관되게 유지되는가, 아니면 특정 장면에서만
  우연히 좋았던 건가? (M3)
- 프레임별 정확도가 안정적인가, 아니면 예측 불가능하게 튀는가? (M4)

---

## v0.1 — Input Loader Spec

**대상 모듈**: `input/camera.py`, `input/lidar.py`, `input/extrinsic.py`,
`input/dataset.py`

### 책임 분리

- **Camera / LiDAR 로더는 PARSE ONLY.** 원본 config + 파일을 표준화된
  `CameraModel`/`LidarModel` + `Frame` 객체로 바꾸는 것까지만 한다.
  calibration 정확도를 검증하지 않고(그건 `verify_extrinsic`의 책임),
  어떤 평가 metric도 계산하지 않는다.
- **Extrinsic 로더는 두 가지 책임을 명확히 분리한다**:
  1. `load_extrinsic(...)`: 사용자가 준 임의의 포맷(rpy/quaternion/matrix,
     임의의 parent/child 방향, 임의의 단위)을 파싱해 단일 4×4 `T_CL`
     행렬로 정규화한다.
  2. `verify_extrinsic(...)`: 로드된 결과가 수학적으로 유효한지만
     검사한다(rotation 유효성, 유한성, 그럴듯한 단위/크기). Calibration
     "품질"을 재판단하는 게 아니라 — 그건 Evaluation Engine의 일 —
     "이게 애초에 well-formed transform인가"만 본다.

### 지원 포맷

- **이미지**: `image_dir` 소스, `.png/.jpg/.jpeg/.bmp/.tif/.tiff`
- **포인트클라우드**: `pcd_dir` 소스, PCD(ASCII/binary 공통 subset),
  PLY(ASCII). `rosbag`/`ros_topic` 소스는 스텁 처리(`NotImplementedError`)
  — 이 환경에 ROS 역직렬화 의존성이 없기 때문.
- **Extrinsic rotation format**: `rpy_deg` | `rpy_rad` | `quaternion` |
  `matrix3x3` | `matrix4x4`. `parent`/`child`는 `lidar`/`camera` 어느
  방향이든 지정 가능하며 로더가 자동으로 `T_CL`/`T_LC` 방향을 정규화한다.
  단위(`unit`)는 `m`/`cm`/`mm` 지원.

### Dataset 동기화

`input/dataset.py`가 camera + lidar + extrinsic을 하나의
`EvaluationDataset`으로 묶는다:

- 타임스탬프 동기화는 **최근접 이웃(nearest-neighbor) 매칭**, 허용
  오차는 `sync_max_time_diff_ms`(기본 50ms).
- M3(Hold-out Consistency)가 필요로 하는 **연속 시간 블록 분할**
  (`EvaluationDataset.time_blocks(n_blocks)`)도 이 모듈이 제공한다 —
  랜덤 셔플이 아니라 시간축 상에서 연속된 구간으로 분할한다(§ M3 참고).

---

## v0.2–v0.4 — M0. Projection Sanity Gate

**대상 모듈**: `evaluation/sanity_gate.py`

**점수화되지 않는 metric.** Data Quality Assessment 단계에 위치하며
M2/M3/M4보다 앞서 실행된다. M2~M4와 다른 질문에 답한다 — "이 calibration이
얼마나 정확한가"가 아니라 **"이 T/데이터 조합이 애초에 정확도를 의미
있게 측정할 수 있는 상태인가"**를 판단한다.

`input/extrinsic.py`의 `verify_extrinsic()`과는 다르다: `verify_extrinsic`은
`T_CL`이 그 자체로 well-formed transform인지만 본다(유효한 rotation,
유한한 translation, 그럴듯한 단위). M0는 `T_CL` + 실제 camera/LiDAR
**데이터**의 조합을 본다 — 이 T로 실제 포인트를 투영했을 때 말이 되는
그림이 나오는가, 아니면 바로 무너지는가(이미지에 아무것도 안 맺힘, depth가
말이 안 됨, occlusion 패턴이 터무니없음)?

### 체크 항목

1. **FOV coverage**: 이미지 안에 들어오는 LiDAR 포인트의 비율.
   기본 threshold: **`≥ 30%`** (`DEFAULT_MIN_FOV_COVERAGE = 0.30`)
2. **Depth distribution sanity**: NaN/Inf 없음, 필터링 후 음수 depth
   없음, depth가 주어진 `min_range_m`/`max_range_m` 센서 스펙 범위 내에
   있음.
3. **Occlusion violation (근사)**: 거친 픽셀 버킷 단위 depth buffer를
   이용해, 같은 이미지 영역을 이미 차지한 포인트보다 훨씬 뒤에 있는
   포인트를 flag한다 — 완전한 mesh/z-buffer 렌더러 없이 "이 장면이
   기본적인 occlusion과 부합하게 투영되는가"의 근사 지표.
   - 버킷 크기: **`8px`** (`DEFAULT_OCCLUSION_BUCKET_PX = 8.0`)
   - depth margin: **`0.5m`** (`DEFAULT_OCCLUSION_DEPTH_MARGIN_M = 0.5`)
   - 위반 허용 비율: **`≤ 20%`** (`DEFAULT_MAX_OCCLUSION_VIOLATION_RATIO = 0.20`)

최소 유효 포인트 수: **`500`** (`DEFAULT_MIN_VALID_POINTS = 500`)

Pass/Fail threshold는 의도적으로 거칠게 잡혀 있다(이건 게이트지 점수가
아니다) — 기본값은 인라인 문서화돼 있고 호출 시 조정 가능하다.

---

## v0.4 — M2. Edge Alignment (MVP 주 점수 metric)

**대상 모듈**: `evaluation/edge_alignment.py`

LiDAR depth-discontinuity("edge") 포인트를 기존 `T_CL`로 이미지에
투영했을 때, 실제 이미지 edge와 얼마나 잘 정렬되는지 측정한다.

### 파이프라인

1. 모든 LiDAR 포인트를 이미지에 투영한다(`geometry.projection`).
2. 투영된 포인트 중 LiDAR 쪽 depth discontinuity 위에 있는 포인트를
   식별한다(= 물체 실루엣/occlusion 경계에 해당하는 포인트).
3. 이미지 edge를 추출하고(Canny) distance-transform map을 만든다.
4. 각 LiDAR edge 포인트의 픽셀 위치에서 distance-transform 값을
   샘플링한다 — 이것이 포인트별 정렬 오차(픽셀 단위)다.
5. 집계(mean/median/P95)하고, sensor-relative `floor(Z)` threshold
   (`quality.noise_floor`)에 M2 전용 배수(**2× / 5×**)를 적용해
   GOOD/WARNING/BAD로 분류한다.

---

## v0.4 — M3. Hold-out Consistency

**대상 모듈**: `evaluation/holdout_consistency.py`

**고정된** 기존 `T_CL`이 데이터셋의 서로 다른 연속 시간 블록에 걸쳐
일관되게 동작하는지 측정한다 — 즉 calibration이 전체 시퀀스에 걸쳐
일반화되는지, 아니면 우연히 잘 맞는 특정 장면/시간대에서만 "어쩌다
괜찮은" 건지를 본다.

### 파이프라인

1. 동기화된 dataset을 N개의 **연속 시간 블록**으로 분할한다
   (`EvaluationDataset.time_blocks` — **랜덤 셔플 아님**, spec에 따름).
2. 각 블록마다, 블록에 속한 프레임별로 독립적으로 M2를 실행하고,
   블록 내 모든 프레임의 edge-point 오차를 pool한 뒤 블록 하나의
   집계 M2 결과를 계산한다.
3. 각 블록의 `mean_px`를 모아 블록 간 분포를 만들고, 블록 간
   Mean/STD/range를 계산한다.
4. STD를 sensor-relative `floor(Z)`에 대해 분류하되, M2의 2×/5× 체계가
   아니라 **STD 전용 배수(1× / 3×)**를 사용한다 — 여기서 측정하는 건
   포인트별 offset이 아니라 "퍼짐(spread)"이기 때문.

### 기본값 / 실패 조건

- `n_blocks` 기본값: **4**
- `min_frames_per_block` 기본값: **30**
- `min_frames_per_block`보다 프레임 수가 적은 블록은 (warning과 함께)
  **제외**된다 — 조용히 포함시키지 않는다.
- **유효 블록이 3개 미만이면 FAIL** (`MIN_VALID_BLOCKS = 3`) — 통계적으로
  의미가 없기 때문.

---

## v0.4 — M4. Multi-frame Consistency

**대상 모듈**: `evaluation/multiframe_consistency.py`

**고정된** 기존 `T_CL`이 전체 프레임 시퀀스에 걸쳐 안정적인 프레임별
오차를 내는지 측정한다 — calibration이 프레임 대 프레임으로 신뢰할
만한지, 아니면 특정 프레임에서 튀는지(순간적 misalignment, outlier
장면, sync glitch 등).

M3가 프레임을 연속 시간 블록으로 pool해서 "시간대에 걸친 일반화"를
보는 것과 달리, M4는 **각 프레임을 독립적으로 평가**하고 프레임별
오차의 분포를 직접 본다 — 이것이 블록 단위 평균이 뭉개버릴 "프레임
하나만 나쁜" 상황을 잡아낸다.

### 파이프라인

1. 동기화된 프레임마다 독립적으로 M2를 실행한다: `E_i = M2(T_fixed, Frame_i)`
2. 유효한 프레임별 `mean_px` 값을 모은다.
3. 프레임 간 Mean/STD/P95/Max를 집계한다.
4. Outlier 프레임 플래그: `mean_px_i > outlier_multiplier * median(전체 mean_px)`
   (spec 기준: **5× median**, `floor(Z)`와 무관하게 적용,
   `DEFAULT_OUTLIER_MULTIPLIER = 5.0`)
5. STD를 sensor-relative `floor(Z)`에 대해, M3와 동일한 **STD 배수
   체계(1× / 3×)**로 분류한다.

### 기본값 / 실패 조건

- 전체 프레임 수가 `min_frames`(기본 **30**) 미만이면 FAIL — spec의
  미결 항목 "통계적 유의성 위해 몇 프레임 이상 필요한지, 예: 최소 30"에
  따른 기본값.
- 유효(non-FAIL) 프레임이 2개 미만이면 FAIL(STD 계산 불가).

---

## v0.3 — Sensor-relative Noise Floor: `floor(Z)`

**대상 모듈**: `quality/noise_floor.py`

M2/M3/M4의 모든 GOOD/WARNING/BAD threshold는 고정된 절대 픽셀값이
아니라, 센서 성능으로부터 유도된 `floor(Z)`의 배수로 정의된다 —
sensor-relative threshold 전략의 핵심.

```
floor(Z) = sqrt( floor_angular^2 + floor_range(Z)^2 + floor_edge^2 )

floor_angular   = fx * theta_res
floor_range(Z)  = fx * b * sigma_r / Z^2
floor_edge      = 상수 (edge detector의 sub-pixel floor)
```

세 항 모두 독립적인 오차 원인으로 취급되어 **quadrature(제곱합의
제곱근)**로 결합된다.

- `fx`: 카메라 focal length(px)
- `theta_res`: LiDAR 각해상도(rad) — **수평/수직 중 worse case(더 큰
  값)**를 사용. 수직 해상도가 명시적으로 없으면
  `vertical_fov_deg / channels`로 근사하며, 이 근사는 수평 해상도가
  주어졌는지와 무관하게 항상 적용된다(그래야 더 거친 미해결 수직
  해상도가 조용히 가려지지 않는다).
- `b`: baseline(m) — `T_CL`의 translation 크기(`||t||`)
- `sigma_r`: LiDAR range 정확도(m, 1-sigma)
- `floor_edge`: edge localization의 sub-pixel floor(px)

### Fallback 상수 (센서 스펙이 불완전할 때)

| 상수 | 기본값 | 의미 |
|---|---|---|
| `DEFAULT_RANGE_ACCURACY_M` | 0.02 m | `sigma_r` fallback (1-sigma) |
| `DEFAULT_ANGULAR_RESOLUTION_DEG` | 0.2° | `theta_res` fallback (업계 평균 근사치) |
| `DEFAULT_EDGE_LOCALIZATION_FLOOR_PX` | 0.5 px | `floor_edge` fallback |

fallback이 쓰일 때마다 `FloorInputs.fallback_warnings`에 명시적으로
기록되어 리포트에 노출된다 — 조용히 근사치를 쓰지 않는다.

### 분류 배수 (multiplier-based threshold)

| Metric | GOOD | WARNING |
|---|---|---|
| M2 (포인트별 offset) | `M2_GOOD_MULTIPLIER = 2.0×` | `M2_WARNING_MULTIPLIER = 5.0×` |
| M3/M4 STD (퍼짐) | `STD_GOOD_MULTIPLIER = 1.0×` | `STD_WARNING_MULTIPLIER = 3.0×` |

이 fallback 상수들은 한곳(`quality/noise_floor.py`)에 모아두어, 나중에
reference dataset으로 anchor 검증이 끝나면 쉽게 조정할 수 있게 했다
("Input Loader 결정 필요 사항 #2": fallback 값을 constants로 분리).

---

## §17–18 — 0–100 점수화 & Quality Score 집계

**대상 모듈**: `quality/normalization.py`, `quality/quality_score.py`

### 점수 곡선 (`normalization.py`)

원시 metric 값(px)을 0–100 점수로 매핑하되, GOOD/WARNING/BAD 분류에
쓰는 것과 **동일한** sensor-relative `floor(Z)` 배수 threshold에
고정시킨다 — "점수"와 "분류"가 서로 다른 두 시스템이 되어 어긋나지
않도록 설계 단계부터 일치시킨다.

`r = value_px / floor_px`라 할 때 (classify()가 쓰는 것과 같은 비율),
다음을 만족하는 매끄럽고 단조 감소하는 곡선 `score(r)`을 원한다:

```
r = 0            -> score = 100  (완벽: floor 이상의 측정 가능한 오차 없음)
r = good_mult    -> score = 80   (GOOD/WARNING 경계)
r = warning_mult -> score = 50   (WARNING/BAD 경계, 자연스러운 중간점)
r -> ∞           -> score -> 0
```

일반화된 logistic(Hill-type) 곡선을 사용한다:

```
score(r) = 100 / (1 + (r / warning_mult)^p)
```

`r = warning_mult`를 50에 고정하면 `r0 = warning_mult`가 정확히
결정되고(`(warning_mult/warning_mult)^p = 1`이므로 `100/(1+1) = 50`),
지수 `p`는 `r = good_mult` 앵커로부터 풀린다:

```
100 / (1 + (good_mult/warning_mult)^p) = 80
=> (good_mult/warning_mult)^p = 0.25
=> p = ln(0.25) / ln(good_mult/warning_mult)
```

배수 쌍(M2의 2×/5× vs M3/M4 STD의 1×/3×)마다 한 번씩 풀어서 캐싱한다.

곡선이 `(good_mult, 80)`과 `(warning_mult, 50)`을 정확히 지나는
단조감소함수이므로: `r < good_mult`(GOOD)는 항상 `score > 80`,
`good_mult ≤ r < warning_mult`(WARNING)는 항상 `50 ≤ score < 80`,
`r ≥ warning_mult`(BAD)는 항상 `score < 50`으로 매핑된다 — "점수 92"를
보는 사람과 "분류 GOOD"을 보는 사람이 절대 모순되지 않는다.

### Quality Score 집계 (`quality_score.py`)

M2(Geometry) / M3(Generalization) / M4(Stability)를 하나의 Overall
Quality 점수로 집계한다 — spec section 17-18의
"Geometry / Generalization / Stability / Sensitivity" 분류 체계를
따르되, **Sensitivity 카테고리는 보류**한다(이 코드베이스에는 아직
독립된 Perturbation 기반 scored metric이 없음 — Perturbation
Sensitivity는 advanced/opt-in 진단으로만 존재).

- 카테고리 가중치는 **기본적으로 동일**(각 1/3, `DEFAULT_WEIGHTS`) —
  아직 하나를 더 중요하게 볼 데이터 기반 근거가 없다는 spec의 미결
  항목("Normalization")을 그대로 반영. 가중치는 하드코딩된 상수가
  아니라 파라미터이므로, 실제 평가 사례가 쌓이면 재검토할 수 있다.
- 카테고리 metric이 완전히 FAIL하면 그 카테고리는 제외되고 나머지
  가중치가 재정규화된다(조용히 0점 처리하지 않음 — "측정 못함"과
  "측정했더니 최악"은 다른 것이므로).
- 일부 카테고리만 유효할 때 Overall Quality의 classification은
  **WARNING을 넘지 못하도록 캡**된다(구현 세부는 CHANGELOG 참고) —
  부분 평가 결과가 완전한 평가 결과와 같은 신뢰도로 보이면 안 되기
  때문.

---

## §14 — Perturbation Sensitivity (Advanced / Phase-5, opt-in)

**대상 모듈**: `evaluation/perturbation.py`

기존 `T_CL`을 6-DOF(translation x/y/z, rotation roll/pitch/yaw) 각각을
따라 작은 크기로 양방향 nudge하고, 각 nudge된 T에서 M2를 재실행해서
**원래 T가 이미 모든 nudge 중 가장 낮은 오차를 갖는지** 확인한다.
근처의 어떤 T가 일관되게 더 낫다면, 이는 현재 calibration이 local
optimum에도 있지 않다는 증거다 — "정답" T를 몰라도, 근처 대안들이
더 나은지/나쁜지만 보면 되므로 GT 없이도 유용한 신호가 된다.

### 기본값

- Translation delta: **`(0.01m, 0.02m)`** 양방향
- Rotation delta: **`(0.1°, 0.2°)`** 양방향
- Local minimum 허용오차: **`0.05px`** — 이 이내의 개선은 노이즈로 간주
- 6축 × 2 delta × 2 방향 = 총 24회의 M2 재평가 (baseline 포함 25회)

Baseline M2 평가 자체가 FAIL하면 이 metric도 FAIL한다(비교 대상이
없으므로).

Quality Score에 **절대 영향을 주지 않는다** — opt-in 진단
(`--advanced`)일 뿐 MVP 점수 셋의 일부가 아니다.

---

## §15 — Plane Consistency (Advanced / Phase-5, opt-in)

**대상 모듈**: `evaluation/plane_consistency.py`

LiDAR 포인트클라우드에서 지배적인 평면(대개 지면이나 큰 벽)을
RANSAC으로 피팅하고, 그 inlier 포인트들을 이미지에 투영한 뒤, 투영된
영역의 **윤곽(convex hull 경계)**이 실제 이미지 edge와 맞는지 확인한다.

M2의 depth-discontinuity 기반 edge-point 추출을 의도적으로 재사용하지
않는다 — 하나의 평면은 정의상 내부 depth discontinuity가 없으므로(그게
평면이라는 것의 의미), M2의 edge-point 선택 방식을 평면 inlier에
그대로 적용하면 항상 0개를 찾게 된다. 단일 평면에서 의미 있는 것은
그 실루엣 — 이미지 안에서 그 표면이 시각적으로 끝나는 지점(다른
물체, 지평선 등과의 경계) — 이므로, 이 metric은 각 inlier가 투영된
포인트 집합의 2D convex hull까지의 거리로 boundary point를 추출한 뒤,
M2와 동일한 edge-map + distance-transform 샘플링을 재사용한다. 이렇게
하면 두 metric이 정렬을 "측정하는 방식"(distance-transform 샘플링)은
일관되게 유지하면서, 각 metric의 기하학적 특성에 맞는 "edge point"
정의를 쓸 수 있다.

### 기본값

- RANSAC plane distance threshold: **0.05m**
- RANSAC 반복 횟수: **300**
- 최소 inlier 비율: **10%**
- Boundary margin: **4px**
- 최소 boundary point 수: **30**

---

## §17 (Stability 카테고리 보강) — Temporal Drift (Advanced / Phase-5, opt-in)

**대상 모듈**: `evaluation/temporal_drift.py`

M4(Multi-frame Consistency)를 보완한다: M4는 퍼짐(STD/P95/Max)을
측정하고 개별 outlier 프레임을 flag하지만, **방향성**에 대해서는
아무것도 말하지 않는다 — rig flex나 열팽창 등으로 오차가 시퀀스 전체에
걸쳐 꾸준히 커지는 calibration은 STD 자체는 특별히 눈에 띄지 않으면서도
명백한 하강 궤적을 그릴 수 있다. 이 metric은 M4의 프레임별 오차
시퀀스에 선형 추세를 피팅하고, 그 추세가 노이즈와 통계적으로 구별되는지
보고한다.

이미 계산된 `MultiFrameConsistencyResult`에서 바로 계산되므로(새로운
투영이 필요 없음), M4가 이미 실행됐다면 계산 비용이 저렴하다.

### 기본값

- 유의수준(alpha): **0.05**
- 최소 프레임 수: **5**
- STD 분류와 동일한 **1×/3×** 배수 체계를 재사용

---

## 알려진 한계 / 스코프 밖 (§13 포함)

- **rosbag / ROS topic 소스**는 스텁 처리(`NotImplementedError`) — 이
  환경에 ROS 역직렬화 의존성이 없기 때문. `image_dir`/`pcd_dir` 소스는
  완전히 구현되어 있다.
- **§13 "Level 2" — Re-calibration repeatability**(subset별로
  calibration을 재수행해서 결과 T들을 비교)는 명시적으로 스코프
  밖이다 — 이 tool은 *기존* calibration을 평가할 뿐, 새로 계산하지
  않는다.
- **Photometric consistency**는 보류됐다 — illumination/exposure/
  reflectance가 calibration 품질과 너무 쉽게 섞여버려서 첫 버전에는
  부적합하다고 판단.
- **GT mode**(실제 ground-truth transform 대비 정확도, 연구/벤치마크용)는
  스펙에 별도 모드로 설명되었으나 여기서는 구현하지 않았다 — 이
  tool은 GT-free 모드 전용이다.
- Advanced metric(`--advanced`)들은 진단용이며 quality_score에 영향을
  주지 않는다 — MVP 셋만큼 검증되지 않았고, headline 점수에 기여하기
  보다 다른 질문(local optimality, 추세, 단일 표면 체크)에 답하기
  때문에 의도적으로 분리했다.
- **Reference dataset으로 anchor 검증**: `floor(Z)`의 fallback 상수
  (§ v0.3)는 아직 실제 reference dataset으로 검증되지 않았다 — 검증이
  끝나기 전까지는 fallback이 쓰일 때마다 명시적으로 warning을 남긴다.
