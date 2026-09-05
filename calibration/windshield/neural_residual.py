"""
camera_calibrator.calibration.windshield.neural_residual
==============================================================

STEP 5 - Residual Ray의 세 번째 variant: 작은 MLP(Tiny Neural Network) 기반의
`(u,v) -> ΔRay` 모델. Grid(bilinear)/RBF(thin plate spline)와 완전히 같은
최상위 개념(residual-ray correction)을 쓴다:

    Pixel -> 고정된 Base K,D -> Base Ray -> Tiny MLP ΔRay(u,v) -> Corrected Ray

    d_corrected = normalize(d_base + NN(u_n, v_n))

차이는 Delta_d(u,v)를 만드는 방법뿐이다 - Grid/RBF는 결정론적 보간이고, 이
모듈은 gradient-descent로 학습한 작은 MLP다. `WindshieldModelType`에 새
멤버(NEURAL)를 추가하지 않는다 - 여전히 `WindshieldModelType.RESIDUAL_RAY`
하나이고, `WindshieldConfig.residual_ray_hint["method"]`("grid"/"rbf"/"neural")
로만 구분된다(dispatch는 validation.py/projection.py/ui/windshield_worker.py에
있다). `fitted_params["residual_ray_method"]`에 숫자 코드(0.0=grid, 1.0=rbf,
2.0=neural)를 남긴다.

PyTorch는 선택적(soft) 의존성이다 - 이 모듈은 torch가 없어도 import 자체는
항상 성공한다(다른 Windshield 모델이 PyTorch import 실패로 실행 불가능해지면
안 된다는 요구사항). 실제로 Neural 기능을 "사용"하려는 순간(모델 생성/학습/
런타임 재구성)에만 `_require_torch()`가 명확한 ImportError를 낸다.

Architecture(고정, 이번 라운드에서는 Architecture Search를 하지 않는다):

    Input(2, normalized u,v) -> Linear(2,32) -> SiLU
                              -> Linear(32,64) -> SiLU
                              -> Linear(64,32) -> SiLU
                              -> Linear(32,3)  (Output = Delta Ray xyz)

정규화 좌표(Grid/RBF와 동일 convention, residual_common.normalize_pixel_
coordinates 재사용):

    u_n = 2*u/W - 1
    v_n = 2*v/H - 1

Loss:

    L_ray    = mean(||d_target - normalize(d_base + NN(u_n,v_n))||^2)
    L_mag    = mean(||NN(u_n,v_n)||^2)
    L_smooth = mean(||NN(u_n+eps,v_n)-NN(u_n,v_n)||^2 + ||NN(u_n,v_n+eps)-NN(u_n,v_n)||^2)
    L        = L_ray + lambda_mag * L_mag + lambda_smooth * L_smooth

Train/Validation/Early Stopping: Outer Train 코너를 (seed로 결정론적인)
random split으로 NN Train/NN Validation으로 나눈다. Validation ray loss가
`patience` epoch 동안 개선되지 않으면 조기 종료하고, 마지막 epoch가 아니라
**best validation epoch의 weight**(state_dict)를 최종으로 쓴다.

STAGE A/B/Pose prior/Repeated Hold-out(Outer Train subset만)/Ray Stability는
Grid/RBF와 완전히 동일한 정책을 `residual_common.py`의 공유 함수로 재사용한다.
Neural 고유 추가 진단은 Seed Stability(같은 split, 다른 학습 seed들의 corrected
ray 차이) - `compute_ray_stability_deg`를 그대로 재사용해서 "비교 대상이
split이냐 seed냐"만 다르게 호출한다.

직렬화(사용자 스펙 36-39번, "float key 수천 개로 펼치지 않는다"): 학습된
`state_dict`는 base64 문자열 하나로 인코딩해 `WindshieldCalibrationResult.
neural_state_dict_b64`(fitted_params와 별도 필드)에 담는다. fitted_params에는
아키텍처 메타데이터(hidden dims, activation, hyperparameter, 진단 값)만
float로 남긴다 - 재구성에 필요한 값은 전부 이 안에 있다(SciPy/PyTorch 내부
객체 상태를 직렬화하지 않는다는 기존 원칙과 동일).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import math
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from calibration.models.common import MIN_FRAMES_REQUIRED, infer_image_size
from calibration.types import CameraConfig, CameraModelType, Dataset, Frame
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.refraction import normalize
from calibration.windshield.residual_common import (
    DEFAULT_REPEATED_HOLDOUT_SEEDS,
    MAX_PROJECT_POINT_ANGULAR_ERROR_DEG,
    RepeatedHoldoutSummary,
    collect_corner_arrays,
    compute_ray_stability_deg,
    evaluate_residual_ray_model,
    normalize_pixel_coordinates,
    populate_pose_diagnostics,
    populate_repeated_holdout_diagnostics,
    refine_frame_pose_ray_domain,
    residual_ray_failure_result,
)

import os

# Anaconda(numpy/scipy가 끌어오는 Intel MKL)와 PyTorch CPU wheel이 각각 자체
# OpenMP 런타임(libiomp5md.dll)을 들고 있어, 같은 프로세스에 둘 다 로드되면
# Windows에서 "OMP: Error #15: ... already initialized"로 즉시 죽는 경우가
# 있다(이 프로젝트의 Anaconda 환경에서 실제로 재현됨). torch를 import하기
# *직전에만* 이미 설정돼 있지 않은 경우에 한해 완화 플래그를 켠다 - 다른
# 모델(Grid/RBF/Spline 등)은 이 모듈을 아예 import하지 않으므로 영향이 없고,
# Neural을 실제로 쓰는 프로세스에만 적용된다. Trade-off(사용자 스펙 24번):
# 이 플래그는 "중복 OpenMP 런타임이 있어도 그냥 진행"이라는 뜻이라 극단적으로
# 드문 경우 성능 저하/스레딩 이상이 있을 수 있지만, 이 모듈이 다루는
# 데이터/모델 규모(수백 코너, 수천 파라미터의 Tiny MLP)에서는 실질적인
# 위험이 거의 없다 - 아무 설정도 안 하면 아예 실행이 안 되는 쪽이 더 나쁘다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _e:  # pragma: no cover - exercised only in torch-less environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Neural Residual Windshield 모델을 사용하려면 PyTorch가 필요합니다. "
            "`pip install torch`(또는 프로젝트 환경에 맞는 CPU/GPU wheel)로 설치한 뒤 "
            "다시 시도하세요. 다른 Windshield 모델(Baseline/Spherical/Residual Grid/"
            "Residual RBF/Spline)은 PyTorch 없이도 정상 동작합니다."
        ) from _TORCH_IMPORT_ERROR


# ---------------------------------------------------------------------------
# 고정 상수 - residual_ray_hint로 덮어쓸 수 있다(Grid/RBF와 동일 원칙).
# ---------------------------------------------------------------------------
DEFAULT_NEURAL_HIDDEN_DIMS: list[int] = [32, 64, 32]
DEFAULT_NEURAL_ACTIVATION = "silu"
DEFAULT_NEURAL_LEARNING_RATE = 1e-3
DEFAULT_NEURAL_WEIGHT_DECAY = 1e-4
DEFAULT_NEURAL_LAMBDA_MAG = 1e-2
DEFAULT_NEURAL_LAMBDA_SMOOTH = 1e-2
DEFAULT_NEURAL_MAX_EPOCHS = 500
DEFAULT_NEURAL_PATIENCE = 30
DEFAULT_NEURAL_SEED = 42
DEFAULT_NEURAL_BATCH_SIZE = 128
DEFAULT_NEURAL_VALIDATION_RATIO = 0.2

# STAGE B는 매 라운드 처음부터 재학습하지 않고 현재 weight에서 fine-tune
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

_ACTIVATION_TO_CODE: dict[str, float] = {"silu": 0.0, "tanh": 1.0, "relu": 2.0}
_CODE_TO_ACTIVATION: dict[float, str] = {0.0: "silu", 1.0: "tanh", 2.0: "relu"}


def _activation_classes() -> dict[str, "type[nn.Module]"]:
    return {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_mlp(hidden_dims: list[int], activation: str) -> "nn.Module":
    """고정 architecture: Input(2) -> hidden_dims... -> Output(3). 첫 버전은
    Tiny MLP 하나만 지원한다(대규모 architecture search를 하지 않는다는
    요구사항) - hidden_dims/activation만 hint로 바꿀 수 있다."""
    _require_torch()
    act_cls = _activation_classes().get(activation, nn.SiLU)
    layers: list[nn.Module] = []
    prev = 2
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, 3))
    return nn.Sequential(*layers)


def _encode_state_dict(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_state_dict(state_dict_b64: str) -> dict:
    raw = base64.b64decode(state_dict_b64.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _evaluate_net_delta(net: "nn.Module", u: float, v: float, image_width: float, image_height: float) -> np.ndarray:
    """학습된 net 하나로 한 픽셀의 ΔRay를 평가한다 - runtime API
    (NeuralResidualWindshieldModel._delta)와 STAGE B pose refinement의
    delta_fn 콜백이 이 함수 하나를 공유한다(Training과 Runtime이 서로 다른
    correction policy를 쓰지 않는다는 요구사항)."""
    un, vn = normalize_pixel_coordinates(u, v, image_width, image_height)
    with torch.no_grad():
        raw = net(torch.tensor([[un, vn]], dtype=torch.float32)).numpy()[0].astype(np.float64)
    if not np.all(np.isfinite(raw)):
        return np.zeros(3)
    norm = float(np.linalg.norm(raw))
    if norm > MAX_CORRECTION_MAGNITUDE:
        raw = raw * (MAX_CORRECTION_MAGNITUDE / norm)
    return raw


# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

class NeuralResidualWindshieldModel(WindshieldModel):
    """작은 MLP가 학습한 `(u,v) -> ΔRay`를 Base Ray에 더하는 모델.
    project_point()/unproject_pixel() 모두 이 보정을 실제로 반영한다. CPU
    inference만으로 동작한다(GPU 불필요, 사용자 스펙 R번)."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        model: CameraModelType,
        state_dict: dict,
        hidden_dims: list[int],
        activation: str,
        image_width: float,
        image_height: float,
    ):
        _require_torch()
        self._net = _build_mlp(hidden_dims, activation)
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._hidden_dims = list(hidden_dims)
        self._activation = activation
        self._image_width = float(image_width)
        self._image_height = float(image_height)
        self._baseline = BaselineWindshieldModel(camera_matrix, distortion, model)

    def _delta(self, u: float, v: float) -> np.ndarray:
        return _evaluate_net_delta(self._net, u, v, self._image_width, self._image_height)

    def evaluate_delta(self, u: float, v: float) -> np.ndarray:
        return self._delta(u, v)

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """3D(카메라 좌표) -> 픽셀. Closed-form이 아니다(NN 보정이 픽셀에 따라
        달라지므로) - Base K,D 투영을 초기값으로 삼아 작은 2변수 root-solve로
        푼다. Grid/RBF의 project_point()와 동일한 구조."""
        from scipy.optimize import least_squares

        target_dir = normalize(np.array([x, y, z], dtype=np.float64))
        initial_uv = np.asarray(self._baseline.project_point(x, y, z), dtype=np.float64)

        def residual(uv: np.ndarray) -> np.ndarray:
            d_base = np.asarray(self._baseline.unproject_pixel(float(uv[0]), float(uv[1])), dtype=np.float64)
            corrected = normalize(d_base + self._delta(float(uv[0]), float(uv[1])))
            return target_dir - corrected

        result = least_squares(residual, x0=initial_uv, method="lm", max_nfev=50)
        if not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.cost):
            raise ValueError("project_point(): local root-solve did not converge to a finite result.")

        residual_norm = float(np.linalg.norm(result.fun))
        angle_rad = 2.0 * math.asin(min(1.0, residual_norm / 2.0))
        if math.degrees(angle_rad) > MAX_PROJECT_POINT_ANGULAR_ERROR_DEG:
            raise ValueError(
                "Could not find a valid corrected projection for this point "
                "(Neural residual correction may not cover this region well)."
            )
        return float(result.x[0]), float(result.x[1])

    def ray_angular_error_deg(self, u: float, v: float, target_point_cam: np.ndarray) -> Optional[float]:
        d_base = np.asarray(self._baseline.unproject_pixel(u, v), dtype=np.float64)
        corrected = normalize(d_base + self._delta(u, v))
        target = np.asarray(target_point_cam, dtype=np.float64)
        norm = np.linalg.norm(target)
        if norm < 1e-9:
            return None
        cos_angle = float(np.clip(np.dot(target / norm, corrected), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))


def build_neural_residual_model_from_fitted_params(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    fitted_params: dict[str, float],
    state_dict_b64: Optional[str],
) -> NeuralResidualWindshieldModel:
    """fitted_params + neural_state_dict_b64에서 runtime 모델을 재구성하는
    유일한 지점(projection.py::build_projector/repeated-holdout stability가
    재사용). fitted_params에는 architecture 메타데이터만 있고, 실제 weight는
    별도 base64 문자열(WindshieldCalibrationResult.neural_state_dict_b64)에
    있다 - 이 함수가 그 둘을 합쳐 실행 가능한 모델로 되돌리는 유일한 지점."""
    _require_torch()
    if not state_dict_b64:
        raise ValueError("Neural Residual 모델의 state_dict가 없습니다 (neural_state_dict_b64 누락).")
    fp = fitted_params
    n_hidden = int(fp["neural_num_hidden_layers"])
    hidden_dims = [int(fp[f"neural_hidden_dim_{i}"]) for i in range(n_hidden)]
    activation = _CODE_TO_ACTIVATION.get(fp.get("neural_activation_code", 0.0), DEFAULT_NEURAL_ACTIVATION)
    state_dict = _decode_state_dict(state_dict_b64)
    return NeuralResidualWindshieldModel(
        camera_matrix, distortion, model, state_dict, hidden_dims, activation,
        fp["image_width"], fp["image_height"],
    )


# ---------------------------------------------------------------------------
# Hyperparameter 설정
# ---------------------------------------------------------------------------

@dataclass
class _NeuralSettings:
    hidden_dims: list[int]
    activation: str
    learning_rate: float
    weight_decay: float
    lambda_mag: float
    lambda_smooth: float
    max_epochs: int
    patience: int
    seed: int
    batch_size: int
    validation_ratio: float


def _neural_settings(config: WindshieldConfig) -> _NeuralSettings:
    hint = config.residual_ray_hint or {}
    hidden_dims = [int(h) for h in hint.get("neural_hidden_dims", DEFAULT_NEURAL_HIDDEN_DIMS)]
    if not hidden_dims:
        raise ValueError("residual_ray_hint neural_hidden_dims는 최소 1개 이상의 hidden layer가 필요합니다.")
    activation = str(hint.get("neural_activation", DEFAULT_NEURAL_ACTIVATION)).lower()
    if activation not in _ACTIVATION_TO_CODE:
        raise ValueError(f"지원하지 않는 neural_activation: {activation} (지원: {sorted(_ACTIVATION_TO_CODE)}).")
    return _NeuralSettings(
        hidden_dims=hidden_dims,
        activation=activation,
        learning_rate=float(hint.get("neural_learning_rate", DEFAULT_NEURAL_LEARNING_RATE)),
        weight_decay=float(hint.get("neural_weight_decay", DEFAULT_NEURAL_WEIGHT_DECAY)),
        lambda_mag=float(hint.get("neural_lambda_mag", DEFAULT_NEURAL_LAMBDA_MAG)),
        lambda_smooth=float(hint.get("neural_lambda_smooth", DEFAULT_NEURAL_LAMBDA_SMOOTH)),
        max_epochs=int(hint.get("neural_max_epochs", DEFAULT_NEURAL_MAX_EPOCHS)),
        patience=int(hint.get("neural_patience", DEFAULT_NEURAL_PATIENCE)),
        seed=int(hint.get("neural_seed", DEFAULT_NEURAL_SEED)),
        batch_size=int(hint.get("neural_batch_size", DEFAULT_NEURAL_BATCH_SIZE)),
        validation_ratio=float(hint.get("neural_validation_ratio", DEFAULT_NEURAL_VALIDATION_RATIO)),
    )


def _split_train_validation(n: int, validation_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Outer Train 코너를 NN Train/NN Validation으로 나눈다(사용자 스펙
    22번) - Outer Test는 이 함수 어디에도 등장하지 않는다(호출부가 항상
    Outer Train 코너만 넘긴다). n이 너무 작아 validation이 0개가 되면(작은
    합성 fixture 등) train 전체를 validation으로도 재사용하는 fallback은
    `_train_mlp`가 담당한다(여기서는 인덱스만 나눈다)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * validation_ratio))
    n_val = min(max(n_val, 0), max(n - 1, 0))
    return idx[n_val:], idx[:n_val]


@dataclass
class _NeuralTrainingOutcome:
    state_dict: dict
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    stopped_early: bool


def _train_mlp(
    train_uv: np.ndarray, train_d_base: np.ndarray, train_target: np.ndarray,
    val_uv: np.ndarray, val_d_base: np.ndarray, val_target: np.ndarray,
    settings: _NeuralSettings,
    max_epochs: Optional[int] = None,
    initial_state_dict: Optional[dict] = None,
) -> _NeuralTrainingOutcome:
    """STAGE A(from-scratch) / STAGE B(fine-tune continuation)이 공유하는
    학습 루프. Validation ray loss 기준 early stopping + best checkpoint
    restore(사용자 스펙 23번) - 마지막 epoch weight를 쓰지 않는다."""
    _require_torch()
    _set_all_seeds(settings.seed)
    net = _build_mlp(settings.hidden_dims, settings.activation)
    if initial_state_dict is not None:
        net.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)

    train_uv_t = torch.from_numpy(train_uv).float()
    train_d_base_t = torch.from_numpy(train_d_base).float()
    train_target_t = torch.from_numpy(train_target).float()
    has_val = val_uv.shape[0] > 0
    if has_val:
        val_uv_t = torch.from_numpy(val_uv).float()
        val_d_base_t = torch.from_numpy(val_d_base).float()
        val_target_t = torch.from_numpy(val_target).float()

    n_train = train_uv_t.shape[0]
    eff_batch = max(1, min(settings.batch_size, n_train))
    epochs = settings.max_epochs if max_epochs is None else max_epochs

    def _loss(uv_t: "torch.Tensor", d_base_t: "torch.Tensor", target_t: "torch.Tensor"):
        delta = net(uv_t)
        corrected = d_base_t + delta
        corrected = corrected / corrected.norm(dim=1, keepdim=True).clamp_min(1e-9)
        ray_loss = ((target_t - corrected) ** 2).sum(dim=1).mean()
        mag_loss = (delta ** 2).sum(dim=1).mean()
        offset_u = torch.zeros_like(uv_t)
        offset_u[:, 0] = NEURAL_SMOOTHNESS_EPS
        offset_v = torch.zeros_like(uv_t)
        offset_v[:, 1] = NEURAL_SMOOTHNESS_EPS
        delta_du = net(uv_t + offset_u)
        delta_dv = net(uv_t + offset_v)
        smooth_loss = ((delta_du - delta) ** 2).sum(dim=1).mean() + ((delta_dv - delta) ** 2).sum(dim=1).mean()
        total = ray_loss + settings.lambda_mag * mag_loss + settings.lambda_smooth * smooth_loss
        return total, ray_loss

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_loss = float("nan")
    stopped_early = False

    rng = np.random.default_rng(settings.seed)

    for epoch in range(epochs):
        net.train()
        perm = rng.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, eff_batch):
            idx = perm[start:start + eff_batch]
            idx_t = torch.from_numpy(idx.copy()).long()
            optimizer.zero_grad()
            loss, _ray_loss = _loss(train_uv_t[idx_t], train_d_base_t[idx_t], train_target_t[idx_t])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        final_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        net.eval()
        with torch.no_grad():
            if has_val:
                _total, ray_loss_eval = _loss(val_uv_t, val_d_base_t, val_target_t)
            else:
                # Validation split이 0개가 될 만큼 데이터가 작으면(합성
                # fixture 등) train ray loss를 조기 종료 기준으로 대신
                # 쓴다(명시적 fallback, 값을 억지로 만들지 않는다).
                _total, ray_loss_eval = _loss(train_uv_t, train_d_base_t, train_target_t)
            val_metric = float(ray_loss_eval.item())

        if val_metric < best_val - 1e-12:
            best_val = val_metric
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                stopped_early = True
                break

    return _NeuralTrainingOutcome(
        state_dict=best_state, best_epoch=best_epoch,
        final_train_loss=final_train_loss, final_val_loss=best_val,
        stopped_early=stopped_early,
    )


def _fit_neural_stage_a(
    observed_pixels: np.ndarray, d_obs: np.ndarray, p_cam: np.ndarray,
    image_width: float, image_height: float, settings: _NeuralSettings,
) -> _NeuralTrainingOutcome:
    """STAGE A - pose 고정 상태에서 NN을 처음부터(from-scratch) 학습한다."""
    target_dirs = np.array([normalize(p) for p in p_cam])
    uv_norm = np.array([
        normalize_pixel_coordinates(u, v, image_width, image_height) for u, v in observed_pixels
    ])
    n = uv_norm.shape[0]
    train_idx, val_idx = _split_train_validation(n, settings.validation_ratio, settings.seed)
    return _train_mlp(
        uv_norm[train_idx], d_obs[train_idx], target_dirs[train_idx],
        uv_norm[val_idx], d_obs[val_idx], target_dirs[val_idx],
        settings,
    )


# ---------------------------------------------------------------------------
# STAGE B - Neural + per-frame pose joint refinement (ray-domain, alternating)
# ---------------------------------------------------------------------------

@dataclass
class _JointNeuralRefinementOutcome:
    state_dict: dict
    rvecs: list
    tvecs: list
    converged_cleanly: bool
    best_epoch: int
    final_train_loss: float
    final_val_loss: float


def _joint_refine_neural_and_poses(
    ok_frames: list[Frame],
    observed_pixels_per_frame: list[np.ndarray],
    d_obs_per_frame: list[np.ndarray],
    initial_rvecs: list[np.ndarray],
    initial_tvecs: list[np.ndarray],
    image_width: float,
    image_height: float,
    settings: _NeuralSettings,
    initial_state_dict: dict,
    num_rounds: int = NEURAL_STAGE_B_NUM_ROUNDS,
) -> _JointNeuralRefinementOutcome:
    """STAGE B - alternating(block-coordinate) 방식으로 NN과 프레임별 pose를
    번갈아 refine한다. Grid/RBF의 alternating 구조와 동일 - pose refine은
    공유 `refine_frame_pose_ray_domain`에 위임한다. NN 쪽은 매 라운드
    처음부터 재학습하지 않고 현재 weight에서 fine-tune을 계속한다(비용
    억제, `NEURAL_STAGE_B_MAX_EPOCHS`로 라운드당 상한을 짧게 둔다)."""
    rvecs = [np.asarray(r, dtype=np.float64).copy() for r in initial_rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).copy() for t in initial_tvecs]
    state_dict = initial_state_dict
    converged_cleanly = True
    best_epoch, final_train_loss, final_val_loss = 0, float("nan"), float("nan")

    for _ in range(num_rounds):
        net = _build_mlp(settings.hidden_dims, settings.activation)
        net.load_state_dict(state_dict)
        net.eval()

        def delta_fn(u: float, v: float, _net=net) -> np.ndarray:
            return _evaluate_net_delta(_net, u, v, image_width, image_height)

        for i, frame in enumerate(ok_frames):
            pose_fit = refine_frame_pose_ray_domain(
                frame, observed_pixels_per_frame[i], d_obs_per_frame[i], delta_fn,
                initial_rvecs[i], initial_tvecs[i],
            )
            if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                rvecs[i] = pose_fit.x[:3].reshape(3, 1)
                tvecs[i] = pose_fit.x[3:6].reshape(3, 1)
            else:
                converged_cleanly = False

        obs_pixel_list, p_cam_list = [], []
        for i, frame in enumerate(ok_frames):
            R, _ = cv2.Rodrigues(rvecs[i])
            obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
            cam_pts = (R @ obj.T).T + tvecs[i].reshape(1, 3)
            p_cam_list.append(cam_pts)
            obs_pixel_list.append(observed_pixels_per_frame[i])
        p_cam_arr = np.concatenate(p_cam_list, axis=0)
        obs_pixel_arr = np.concatenate(obs_pixel_list, axis=0)
        d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)

        target_dirs = np.array([normalize(p) for p in p_cam_arr])
        uv_norm = np.array([
            normalize_pixel_coordinates(u, v, image_width, image_height) for u, v in obs_pixel_arr
        ])
        n = uv_norm.shape[0]
        train_idx, val_idx = _split_train_validation(n, settings.validation_ratio, settings.seed)

        outcome = _train_mlp(
            uv_norm[train_idx], d_obs_arr[train_idx], target_dirs[train_idx],
            uv_norm[val_idx], d_obs_arr[val_idx], target_dirs[val_idx],
            settings, max_epochs=NEURAL_STAGE_B_MAX_EPOCHS, initial_state_dict=state_dict,
        )
        state_dict = outcome.state_dict
        best_epoch, final_train_loss, final_val_loss = outcome.best_epoch, outcome.final_train_loss, outcome.final_val_loss

    return _JointNeuralRefinementOutcome(
        state_dict=state_dict, rvecs=rvecs, tvecs=tvecs, converged_cleanly=converged_cleanly,
        best_epoch=best_epoch, final_train_loss=final_train_loss, final_val_loss=final_val_loss,
    )


def _failure_result(config: WindshieldConfig, train_ids: list[str], test_ids: list[str], message: str) -> WindshieldCalibrationResult:
    return residual_ray_failure_result(config, train_ids, test_ids, message)


def calibrate_neural_residual(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Residual Ray Neural 모델을 fitting한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않는다.

    흐름: STAGE A(NN만, pose 고정, NN Train/Validation split + early
    stopping) -> STAGE B(NN+pose joint, ray-domain alternating fine-tune) ->
    두 stage의 실제 pixel RMS를 비교해 더 나은 쪽을 최종으로 채택 -> Train
    평가 -> Test는 최종 NN을 완전히 고정한 채 자기 pose만 별도로 refine한 뒤
    평가(leakage 없음, 사용자 스펙 15번).
    """
    _require_torch()
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)
    width, height = image_size

    try:
        settings = _neural_settings(config)
    except ValueError as e:
        return _failure_result(config, train_ids, test_ids, str(e))

    train_frames = _subset_frames(windshield_dataset, train_ids)
    if len(train_frames) < MIN_FRAMES_REQUIRED:
        return _failure_result(
            config, train_ids, test_ids,
            f"Train 프레임이 {len(train_frames)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요).",
        )

    ok_frames, rvecs, tvecs, failed_ids = solve_poses_fixed_intrinsics(train_frames, K, D, model)
    if not ok_frames:
        return _failure_result(config, train_ids, test_ids, "Train 프레임에서 pose를 하나도 구하지 못했습니다.")

    baseline_model = BaselineWindshieldModel(K, D, model)
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(
        ok_frames, rvecs, tvecs, baseline_model
    )

    total_corners = sum(len(a) for a in d_obs_per_frame)
    if total_corners < MIN_CORNERS_FOR_TRAINING:
        return _failure_result(
            config, train_ids, test_ids,
            f"Neural 학습에 코너 수가 부족합니다 (코너 {total_corners}개, "
            f"최소 {MIN_CORNERS_FOR_TRAINING}개 필요).",
        )

    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    # --- STAGE A: NN만, pose 고정, from-scratch ---
    stage_a_training = _fit_neural_stage_a(observed_pixels_arr, d_obs_arr, p_cam_arr, width, height, settings)
    stage_a_model = NeuralResidualWindshieldModel(
        K, D, model, stage_a_training.state_dict, settings.hidden_dims, settings.activation, width, height,
    )
    stage_a_outcome = evaluate_residual_ray_model(ok_frames, rvecs, tvecs, stage_a_model, image_size)

    # --- STAGE B: NN + per-frame pose joint refinement (ray-domain, fine-tune) ---
    joint = _joint_refine_neural_and_poses(
        ok_frames, observed_pixels_per_frame, d_obs_per_frame, rvecs, tvecs,
        width, height, settings, stage_a_training.state_dict,
    )

    stage_used_is_joint_refined = False
    final_state_dict = stage_a_training.state_dict
    final_rvecs, final_tvecs = rvecs, tvecs
    final_outcome = stage_a_outcome
    final_best_epoch, final_train_loss, final_val_loss = (
        stage_a_training.best_epoch, stage_a_training.final_train_loss, stage_a_training.final_val_loss,
    )
    refinement_note = ""

    stage_b_model = NeuralResidualWindshieldModel(
        K, D, model, joint.state_dict, settings.hidden_dims, settings.activation, width, height,
    )
    stage_b_outcome = evaluate_residual_ray_model(ok_frames, joint.rvecs, joint.tvecs, stage_b_model, image_size)

    stage_a_rmse = stage_a_outcome.residual_stats.rmse
    stage_b_rmse = stage_b_outcome.residual_stats.rmse
    improved = (
        stage_b_outcome.residual_stats.n > 0
        and stage_a_rmse is not None
        and stage_b_rmse is not None
        and stage_b_rmse <= stage_a_rmse
    )

    if improved:
        stage_used_is_joint_refined = True
        final_state_dict = joint.state_dict
        final_rvecs, final_tvecs = joint.rvecs, joint.tvecs
        final_outcome = stage_b_outcome
        final_best_epoch, final_train_loss, final_val_loss = joint.best_epoch, joint.final_train_loss, joint.final_val_loss
        if not joint.converged_cleanly:
            refinement_note = "STAGE B 일부 sub-fit이 수렴하지 않아 해당 프레임/라운드는 이전 값을 유지했습니다. "
    else:
        refinement_note = (
            "STAGE B(ray-domain alternating Neural/pose refinement)가 STAGE A(Neural-only initial fit)보다 "
            "실제 pixel RMS를 개선하지 못해 STAGE A 결과를 최종으로 사용했습니다. "
        )

    final_model = NeuralResidualWindshieldModel(
        K, D, model, final_state_dict, settings.hidden_dims, settings.activation, width, height,
    )

    total_train_points = final_outcome.num_points_ok + final_outcome.num_points_failed
    train_failure_rate = (final_outcome.num_points_failed / total_train_points) if total_train_points else 1.0
    if total_train_points == 0 or train_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
        return _failure_result(
            config, train_ids, test_ids,
            f"최종 Neural 모델로 Train 코너의 {train_failure_rate*100:.0f}%에서 유효한 pixel 예측을 "
            "계산하지 못했습니다.",
        )

    param_count = sum(p.numel() for p in _build_mlp(settings.hidden_dims, settings.activation).parameters())
    pose_param_count_train = len(ok_frames) * 6

    fitted_params: dict[str, float] = {
        "residual_ray_method": 2.0,  # 0.0=Grid, 1.0=RBF, 2.0=Neural
        "image_width": float(width),
        "image_height": float(height),
        "neural_input_dim": 2.0,
        "neural_output_dim": 3.0,
        "neural_num_hidden_layers": float(len(settings.hidden_dims)),
        "neural_activation_code": _ACTIVATION_TO_CODE[settings.activation],
        "neural_seed": float(settings.seed),
        "neural_learning_rate": settings.learning_rate,
        "neural_weight_decay": settings.weight_decay,
        "neural_lambda_mag": settings.lambda_mag,
        "neural_lambda_smooth": settings.lambda_smooth,
        "neural_max_epochs": float(settings.max_epochs),
        "neural_patience": float(settings.patience),
        "neural_validation_ratio": settings.validation_ratio,
        "neural_best_epoch": float(final_best_epoch),
        "neural_final_train_loss": float(final_train_loss),
        "neural_final_val_loss": float(final_val_loss),
        "neural_param_count": float(param_count),
        "num_fit_points": float(total_corners),
        "runtime_param_count": float(param_count),
        "pose_param_count_train": float(pose_param_count_train),
        "stage_used_is_joint_refined": 1.0 if stage_used_is_joint_refined else 0.0,
    }
    for i, h in enumerate(settings.hidden_dims):
        fitted_params[f"neural_hidden_dim_{i}"] = float(h)
    populate_pose_diagnostics(fitted_params, rvecs, tvecs, final_rvecs, final_tvecs)

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=model,
        base_camera_matrix=K,
        base_distortion=D,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        failed_frame_ids=list(failed_ids),
        per_frame_error=final_outcome.per_frame_error,
        residual_stats=final_outcome.residual_stats,
        regional_error=final_outcome.regional_error,
        radial_profile=final_outcome.radial_profile,
        radial_bands=final_outcome.radial_bands,
        spatial_error_map=final_outcome.spatial_error_map,
        mean_dx=final_outcome.mean_dx,
        mean_dy=final_outcome.mean_dy,
        ray_angular_error_deg=final_outcome.ray_angular_error_deg,
        fitted_params=fitted_params,
        success=True,
        warning_message=(refinement_note or None),
        neural_state_dict_b64=_encode_state_dict(final_state_dict),
    )

    if test_ids:
        test_frames = _subset_frames(windshield_dataset, test_ids)
        if test_frames:
            t_ok_frames, t_init_rvecs, t_init_tvecs, t_failed = solve_poses_fixed_intrinsics(test_frames, K, D, model)
            if t_ok_frames:
                # Test pose는 Standard solvePnP를 초기값으로 삼아, 최종(고정된)
                # Neural 기준으로 pose만 다시 refine한다 - Neural weight/K/D는
                # 여기서 절대 건드리지 않는다(leakage 없음, 사용자 스펙 15번).
                t_rvecs, t_tvecs = [], []
                t_obs_pixels, t_d_obs, _t_p_cam = collect_corner_arrays(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, baseline_model
                )

                for frame, init_rvec, init_tvec, obs_px, d_obs in zip(
                    t_ok_frames, t_init_rvecs, t_init_tvecs, t_obs_pixels, t_d_obs
                ):
                    pose_fit = refine_frame_pose_ray_domain(
                        frame, obs_px, d_obs, final_model.evaluate_delta, init_rvec, init_tvec, regularize=True,
                    )
                    if pose_fit.success and np.all(np.isfinite(pose_fit.x)):
                        t_rvecs.append(pose_fit.x[:3].reshape(3, 1))
                        t_tvecs.append(pose_fit.x[3:6].reshape(3, 1))
                    else:
                        t_rvecs.append(init_rvec)
                        t_tvecs.append(init_tvec)

                test_outcome = evaluate_residual_ray_model(t_ok_frames, t_rvecs, t_tvecs, final_model, image_size)
                result.test_residual_stats = test_outcome.residual_stats
                result.test_regional_error = test_outcome.regional_error
                result.test_radial_profile = test_outcome.radial_profile
                result.test_radial_bands = test_outcome.radial_bands
                result.test_spatial_error_map = test_outcome.spatial_error_map
                result.test_mean_dx = test_outcome.mean_dx
                result.test_mean_dy = test_outcome.mean_dy
                result.test_ray_angular_error_deg = test_outcome.ray_angular_error_deg

                total_test_points = test_outcome.num_points_ok + test_outcome.num_points_failed
                test_failure_rate = (
                    test_outcome.num_points_failed / total_test_points if total_test_points else 1.0
                )
                if test_failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE:
                    result.warning_message = (
                        (result.warning_message or "")
                        + f"Test 코너의 {test_failure_rate*100:.0f}%에서 유효한 pixel 예측을 계산하지 "
                        "못했습니다 (Test 결과의 신뢰도가 낮을 수 있습니다)."
                    )
            for fid in t_failed:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = (result.warning_message or "") + "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result


# ---------------------------------------------------------------------------
# Repeated Hold-out (Outer Train subset만, 사용자 스펙 34번)
# ---------------------------------------------------------------------------

def run_repeated_holdout_neural_residual(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    test_ratio: float = 0.25,
) -> RepeatedHoldoutSummary:
    """windshield_dataset을 여러 (다른 seed의) Train/Test로 나눠 반복
    평가한다 - Grid/RBF의 run_repeated_holdout_residual_*와 평행한 구조. 매
    split마다 `calibrate_neural_residual`을 새로 호출하므로, NN도 매번
    새로 초기화되고 새로 학습된다(같은 trained model을 재사용하지 않는다,
    사용자 스펙 34번)."""
    from calibration.models.common import regional_edge_average

    test_rmses: list[float] = []
    test_p95s: list[float] = []
    edge_rmses: list[float] = []
    models: list[WindshieldModel] = []
    successful_seeds: list[int] = []

    K, D, model_name = config.base_camera_matrix, config.base_distortion, config.base_model_name
    width, height = infer_image_size(windshield_dataset, camera_config)

    for seed in seeds:
        train_ids, test_ids = split_train_test(windshield_dataset, camera_config, test_ratio, seed)
        result = calibrate_neural_residual(windshield_dataset, config, camera_config, train_ids, test_ids)
        if not result.success or result.test_residual_stats is None:
            continue
        successful_seeds.append(seed)
        test_rmses.append(result.test_residual_stats.rmse)
        if result.test_residual_stats.p95 is not None:
            test_p95s.append(result.test_residual_stats.p95)
        edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
        if edge is not None:
            edge_rmses.append(edge)
        models.append(
            build_neural_residual_model_from_fitted_params(K, D, model_name, result.fitted_params, result.neural_state_dict_b64)
        )

    ray_stability_mean_deg, ray_stability_p95_deg = compute_ray_stability_deg(models, width, height)

    return RepeatedHoldoutSummary(
        seeds_used=successful_seeds,
        n_successful=len(successful_seeds),
        mean_test_rmse=float(np.mean(test_rmses)) if test_rmses else None,
        std_test_rmse=float(np.std(test_rmses)) if test_rmses else None,
        mean_test_p95=float(np.mean(test_p95s)) if test_p95s else None,
        mean_edge_rms=float(np.mean(edge_rmses)) if edge_rmses else None,
        grid_stability_l2=None,  # Grid 전용 legacy metric - Neural에는 적용되지 않음
        ray_stability_mean_deg=ray_stability_mean_deg,
        ray_stability_p95_deg=ray_stability_p95_deg,
    )


# ---------------------------------------------------------------------------
# Seed Stability (Neural 전용, 사용자 스펙 35-B번) - 같은 split, 다른 학습 seed
# ---------------------------------------------------------------------------

def compute_seed_stability_deg(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
    seeds: tuple[int, ...] = DEFAULT_NEURAL_SEED_STABILITY_SEEDS,
) -> tuple[Optional[float], Optional[float]]:
    """같은 (train_ids, test_ids) split을 고정한 채 학습 seed만 바꿔 여러
    모델을 새로 학습하고, `compute_ray_stability_deg`(Grid/RBF와 동일한
    sampling 철학)로 corrected ray 차이를 측정한다. Split Stability(여러
    split)와는 별개의 축(같은 split, 다른 seed)이다."""
    K, D, model_name = config.base_camera_matrix, config.base_distortion, config.base_model_name
    width, height = infer_image_size(windshield_dataset, camera_config)
    hint = config.residual_ray_hint or {}

    models: list[WindshieldModel] = []
    for seed in seeds:
        cfg = dataclasses.replace(
            config, residual_ray_hint={**hint, "method": "neural", "neural_seed": float(seed)},
        )
        result = calibrate_neural_residual(windshield_dataset, cfg, camera_config, train_ids, test_ids)
        if not result.success:
            continue
        models.append(
            build_neural_residual_model_from_fitted_params(K, D, model_name, result.fitted_params, result.neural_state_dict_b64)
        )

    return compute_ray_stability_deg(models, width, height)


# ---------------------------------------------------------------------------
# 진단 오케스트레이터 - UI/worker가 호출하는 단일 진입점
# ---------------------------------------------------------------------------

def run_neural_residual_calibration_with_diagnostics(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
    *,
    compute_repeated_holdout: bool = True,
    repeated_holdout_seeds: tuple[int, ...] = DEFAULT_REPEATED_HOLDOUT_SEEDS,
    repeated_holdout_test_ratio: float = 0.25,
    compute_seed_stability: bool = True,
    seed_stability_seeds: tuple[int, ...] = DEFAULT_NEURAL_SEED_STABILITY_SEEDS,
) -> WindshieldCalibrationResult:
    """calibrate_neural_residual()에 Repeated Hold-out + Ray Stability +
    Seed Stability 진단을 더한 결과를 반환한다 - Grid/RBF의 run_residual_*_
    calibration_with_diagnostics와 평행 구조(UI/worker의 단일 진입점).

    이번 라운드에서는 Neural hyperparameter/architecture AUTO search를
    구현하지 않는다(사용자 스펙 27/28번, "처음부터 Architecture Search 하지
    마라") - `diag_selection_mode_is_auto`는 항상 0.0(Manual)이다.

    Repeated Hold-out은 Spline/Grid/RBF와 동일하게 Outer Train subset만
    받는다(`outer_train_dataset`) - `windshield_dataset` 전체를 절대 그대로
    넘기지 않는다(leakage 방지)."""
    result = calibrate_neural_residual(windshield_dataset, config, camera_config, train_ids, test_ids)
    if not result.success:
        return result

    result.fitted_params["diag_selection_mode_is_auto"] = 0.0

    if compute_repeated_holdout:
        outer_train_dataset = Dataset(frames=_subset_frames(windshield_dataset, train_ids))
        hint = config.residual_ray_hint or {}
        resolved_hint = {
            **hint,
            "method": "neural",
            "neural_hidden_dims": [int(result.fitted_params[f"neural_hidden_dim_{i}"]) for i in range(int(result.fitted_params["neural_num_hidden_layers"]))],
            "neural_activation": _CODE_TO_ACTIVATION.get(result.fitted_params["neural_activation_code"], DEFAULT_NEURAL_ACTIVATION),
            "neural_learning_rate": result.fitted_params["neural_learning_rate"],
            "neural_weight_decay": result.fitted_params["neural_weight_decay"],
            "neural_lambda_mag": result.fitted_params["neural_lambda_mag"],
            "neural_lambda_smooth": result.fitted_params["neural_lambda_smooth"],
            "neural_max_epochs": result.fitted_params["neural_max_epochs"],
            "neural_patience": result.fitted_params["neural_patience"],
            "neural_validation_ratio": result.fitted_params["neural_validation_ratio"],
            "neural_seed": result.fitted_params["neural_seed"],
        }
        resolved_config = dataclasses.replace(config, residual_ray_hint=resolved_hint)
        summary = run_repeated_holdout_neural_residual(
            outer_train_dataset, resolved_config, camera_config,
            seeds=repeated_holdout_seeds, test_ratio=repeated_holdout_test_ratio,
        )
        populate_repeated_holdout_diagnostics(result.fitted_params, summary, len(repeated_holdout_seeds))

    if compute_seed_stability:
        seed_mean_deg, seed_p95_deg = compute_seed_stability_deg(
            windshield_dataset, config, camera_config, train_ids, test_ids, seeds=seed_stability_seeds,
        )
        if seed_mean_deg is not None:
            result.fitted_params["diag_seed_stability_mean_deg"] = seed_mean_deg
        if seed_p95_deg is not None:
            result.fitted_params["diag_seed_stability_p95_deg"] = seed_p95_deg

    return result
