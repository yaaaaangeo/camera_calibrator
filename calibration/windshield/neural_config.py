"""
camera_calibrator.calibration.windshield.neural_config
==============================================================

Neural Residual(STEP 5)의 순수 Python 상수/타입 매핑만 모은 모듈.

**이 파일은 PyTorch를 import하지 않는다. PySide6도 import하지 않는다.**
그래야 `ui/windshield_workspace.py`(Neural 기본값 표시)와 `calibration.
windshield.neural_residual`(실제 학습/추론) 양쪽이 이 모듈 하나를 공유해도
"UI를 열기만 했는데 PyTorch가 로드된다" 같은 일이 생기지 않는다.

STEP 5 안정화 라운드(항목 1) 배경: 이전 라운드에는 이 상수들이
`neural_residual.py` 안에 있었고, 그 파일이 module 최상단에서
`import torch`를 실행했다 - `ui/windshield_workspace.py`가 UI 기본값
(hidden dims/lr/epochs 등)을 얻으려고 그 모듈을 top-level import하면서
결과적으로 "UI를 여는 것만으로 PyTorch가 로드"되는 문제가 있었다. 이제
UI는 이 모듈만 import한다 - PyTorch 유무와 완전히 무관해진다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Architecture(고정 - 이번 라운드에서도 Architecture Search는 하지 않는다)
# ---------------------------------------------------------------------------
DEFAULT_NEURAL_HIDDEN_DIMS: list[int] = [32, 64, 32]
DEFAULT_NEURAL_ACTIVATION = "silu"

# ---------------------------------------------------------------------------
# Training 기본값 - residual_ray_hint로 덮어쓸 수 있다(Grid/RBF와 동일 원칙).
# ---------------------------------------------------------------------------
DEFAULT_NEURAL_LEARNING_RATE = 1e-3
DEFAULT_NEURAL_WEIGHT_DECAY = 1e-4
DEFAULT_NEURAL_LAMBDA_MAG = 1e-2
DEFAULT_NEURAL_LAMBDA_SMOOTH = 1e-2
DEFAULT_NEURAL_MAX_EPOCHS = 500
DEFAULT_NEURAL_PATIENCE = 30
DEFAULT_NEURAL_SEED = 42
DEFAULT_NEURAL_BATCH_SIZE = 128
DEFAULT_NEURAL_VALIDATION_RATIO = 0.2

# STAGE B는 매 라운드 처음부터 재학습하지 않고 현재 weight에서 fine-tune을
# 계속한다(비용 억제) - 완전 재학습 대비 훨씬 적은 epoch 상한을 둔다.
NEURAL_STAGE_B_NUM_ROUNDS = 2
NEURAL_STAGE_B_MAX_EPOCHS = 100

# 정규화 좌표계([-1,1]) 기준 smoothness finite-neighbor epsilon.
NEURAL_SMOOTHNESS_EPS = 0.02

MIN_CORNERS_FOR_TRAINING = 20
MAX_ACCEPTABLE_CORNER_FAILURE_RATE = 0.10

# Runtime/training 공통 correction guard(RBF의 MAX_CORRECTION_MAGNITUDE와
# 동일 철학 - 병적인 extrapolation/NaN만 잡아낸다).
MAX_CORRECTION_MAGNITUDE = 1.0

DEFAULT_NEURAL_SEED_STABILITY_SEEDS: tuple[int, ...] = (1, 2, 3)

# activation 이름 <-> fitted_params에 저장 가능한 숫자 코드(flat float dict
# 제약) 매핑 - 향후 activation을 추가해도 이 표만 늘리면 된다.
ACTIVATION_TO_CODE: dict[str, float] = {"silu": 0.0, "tanh": 1.0, "relu": 2.0}
CODE_TO_ACTIVATION: dict[float, str] = {0.0: "silu", 1.0: "tanh", 2.0: "relu"}
