"""
calibration.windshield.ghost.config
==============================================================

STEP 8 - Ghost / Double Image 전용 순수 Python 상수.

이 패키지는 PyTorch를 전혀 필요로 하지 않는다(Evaluation/Suppression 모두
NumPy/OpenCV만으로 deterministic하게 구현한다 - 사용자 스펙 36-38번, "큰
CNN부터 만들지 않는다"). 임의로 고른 threshold는 실데이터로 조정될
initial default일 뿐이며 과학적으로 검증된 값이 아니다.
"""

from __future__ import annotations

# STEP 8 stabilization 라운드에서 pairing definition(global consensus)과
# dataset aggregation(robust median/MAD) 정의 자체가 바뀌었으므로 metric/
# model version을 올린다 - 단 project_io의 read-side는 여전히 버전 번호와
# 무관하게 존재하는 필드만 채우므로 v1으로 저장된 기존 프로젝트도 그대로
# 로드된다(스키마 자체는 전부 하위 호환 - 신규 필드는 Optional).
GHOST_METRIC_VERSION = 2
GHOST_MODEL_VERSION = 2

# ---------------------------------------------------------------------------
# Point-source detection
# ---------------------------------------------------------------------------
DEFAULT_BRIGHT_SOURCE_THRESHOLD = 200.0  # luminance(0-255) 이상을 "밝은 광원 후보"로 본다
DEFAULT_MIN_BLOB_AREA_PX = 2
DEFAULT_MAX_SEARCH_RADIUS_PX = 60.0      # main peak 주변 이 반경 안에서만 ghost peak를 찾는다
DEFAULT_GAUSSIAN_SIGMA = 0.4             # blob 검출 전 smoothing - 너무 크면 4px 근처의 가까운
                                          # ghost가 main peak에 뭉개져 사라진다(noise 억제와
                                          # close-ghost 분리 사이의 trade-off, 실데이터로 재조정 필요)
DEFAULT_MIN_PEAK_DISTANCE_PX = 1.5       # 이보다 가까운 두 local maxima는 하나의 peak로 합친다

# Multi-LED global consensus pairing(STEP 8 stabilization 1번) - 단순
# nearest-neighbor가 촘촘한 LED array에서 main-main을 잘못 짝짓는 문제를
# 막기 위해, 모든 plausible (main,ghost) candidate의 displacement vector
# 중 dominant cluster를 먼저 찾고 그 벡터와 일치하는 후보를 우선한다.
DEFAULT_PAIRING_CONSENSUS_RADIUS_PX = 3.0   # 이 반경 안의 displacement vector들을 "같은 cluster"로 본다
DEFAULT_MIN_CONSENSUS_CANDIDATES = 3        # 이보다 candidate pair가 적으면 global consensus를 신뢰하지 않고
                                             # local nearest-neighbor(+ghost<main energy 제약) fallback을 쓴다

# Pair score의 energy-ratio consistency 항(STEP 8 semantic/safety fix
# 5번) - "촘촘한 LED에서 약간 어두운 이웃 Main"이 순수 displacement
# consensus만으로는 걸러지지 않는 경우를 막기 위해, ghost/main energy
# ratio가 dataset 전체의 dominant ratio와 얼마나 다른지도 점수에 반영한다.
# 우선순위는 항상 displacement > energy > 단순 거리이며, 세 항 모두
# 서로 다른 단위(px, px, ratio)라 그대로 더하면 안 되므로 각 항을
# 정규화(consensus_radius_px, max_search_radius_px 기준)한 뒤 더한다.
# 초기 default일 뿐 실데이터로 재조정 필요.
PAIR_SCORE_WEIGHT_VECTOR = 1.0
PAIR_SCORE_WEIGHT_DISTANCE = 0.15
PAIR_SCORE_WEIGHT_ENERGY = 0.5

# 2-pass spatial pairing(Phase B-4 안정화) - PASS 1은 위 global consensus
# vector 하나로 이미지 전체를 커버하지만, windshield 곡률/장착 각도에 따라
# 실제 ghost displacement가 위치마다(예: 좌측 상단 vs 우측 하단) 달라질 수
# 있다. PASS 2는 PASS 1 결과로 만든 성긴 coarse grid를 이용해 main별 local
# dx(u,v)/dy(u,v)를 예측하고, 그 local vector로 재-pairing한다. 새 딥러닝
# 모델을 쓰지 않고 기존 robust median/MAD 집계(spatial_model.py)를 그대로
# 재사용한다. 초기 default일 뿐 실데이터로 재조정 필요.
DEFAULT_TWO_PASS_GRID_ROWS = 3
DEFAULT_TWO_PASS_GRID_COLS = 3
# 이보다 샘플(=PASS 1 detection)이 적은 cell은 local vector를 신뢰하지
# 않고 PASS 1의 global dominant vector로 fallback한다.
DEFAULT_TWO_PASS_MIN_CELL_SAMPLES = 3

# ---------------------------------------------------------------------------
# Edge-target detection
# ---------------------------------------------------------------------------
DEFAULT_EDGE_MIN_GRADIENT = 15.0
DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX = 30.0
DEFAULT_EDGE_MIN_SECONDARY_RATIO = 0.05  # 이보다 약한 2차 peak는 noise로 취급, ghost로 보지 않는다

# Image -> 1D profile 추출(STEP 8 stabilization 2번) - 고대비 직선
# calibration target 기준의 단순한 첫 버전(범용 line detector가 아님).
DEFAULT_EDGE_CANNY_LOW = 50.0
DEFAULT_EDGE_CANNY_HIGH = 150.0
DEFAULT_EDGE_HOUGH_THRESHOLD = 30
DEFAULT_EDGE_HOUGH_MIN_LINE_LENGTH_PX = 40.0
DEFAULT_EDGE_HOUGH_MAX_LINE_GAP_PX = 10.0
DEFAULT_EDGE_PROFILE_HALF_LENGTH_PX = 40.0  # 1D profile이 edge 중심 기준 양쪽으로 뻗는 길이
DEFAULT_EDGE_MAX_PROFILE_SAMPLES = 16        # dominant edge를 따라 몇 개의 profile을 뽑을지
DEFAULT_EDGE_ORIENTATION_TOLERANCE_DEG = 20.0  # edge_axis="vertical"/"horizontal" 필터링 허용 오차

# ---------------------------------------------------------------------------
# General(No-Reference) Likelihood mode - heuristic, ground truth 아님
# ---------------------------------------------------------------------------
DEFAULT_LIKELIHOOD_MIN_GRADIENT = 10.0
DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX = 20.0

# ---------------------------------------------------------------------------
# Spatial map / dataset-level robust aggregation
# ---------------------------------------------------------------------------
DEFAULT_SPATIAL_ROWS = 4
DEFAULT_SPATIAL_COLS = 6
DEFAULT_MAD_OUTLIER_K = 3.5   # |x-median(x)| <= k*1.4826*MAD 를 벗어나면 outlier로 제외

# ---------------------------------------------------------------------------
# Dataset directory input(STEP 8 stabilization 3번)
# ---------------------------------------------------------------------------
GHOST_DATASET_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------------------------
# Suppression(STEP 8B) - deterministic iterative reconstruction
# ---------------------------------------------------------------------------
DEFAULT_SUPPRESSION_ITERATIONS = 4
DEFAULT_MAX_CORRECTION = 0.5
# Over-suppression score의 "clean-region unnecessary change" 항을 얼마나
# 반영할지 / "main edge loss" 항을 얼마나 반영할지(STEP 8 stabilization
# 7-C번) - 단순 합으로 두되 나중에 실데이터로 재조정 가능하도록 상수화한다.
DEFAULT_OVER_SUPPRESSION_CLEAN_WEIGHT = 1.0
DEFAULT_OVER_SUPPRESSION_EDGE_WEIGHT = 1.0
