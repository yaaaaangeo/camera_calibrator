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

GHOST_METRIC_VERSION = 1
GHOST_MODEL_VERSION = 1

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

# ---------------------------------------------------------------------------
# Edge-target detection
# ---------------------------------------------------------------------------
DEFAULT_EDGE_MIN_GRADIENT = 15.0
DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX = 30.0
DEFAULT_EDGE_MIN_SECONDARY_RATIO = 0.05  # 이보다 약한 2차 peak는 noise로 취급, ghost로 보지 않는다

# ---------------------------------------------------------------------------
# General(No-Reference) Likelihood mode - heuristic, ground truth 아님
# ---------------------------------------------------------------------------
DEFAULT_LIKELIHOOD_MIN_GRADIENT = 10.0
DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX = 20.0

# ---------------------------------------------------------------------------
# Spatial map
# ---------------------------------------------------------------------------
DEFAULT_SPATIAL_ROWS = 4
DEFAULT_SPATIAL_COLS = 6

# ---------------------------------------------------------------------------
# Suppression(STEP 8B) - deterministic iterative reconstruction
# ---------------------------------------------------------------------------
DEFAULT_SUPPRESSION_ITERATIONS = 4
DEFAULT_MAX_CORRECTION = 0.5
