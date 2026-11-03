"""Explicit model-aware initialization; never reinterpret fisheye coefficients."""
import numpy as np
from calibration.types import CameraModelType as M


def compatible_seed(result, target):
    if not result.success or result.camera_matrix is None:
        raise ValueError('Initialization requires a successful calibration')
    k = np.array(result.camera_matrix, dtype=np.float64, copy=True)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid intrinsic initialization')
    sizes = {M.PINHOLE: 5, M.BROWN_CONRADY: 5, M.EXTENDED_PINHOLE: 8, M.FISHEYE: 4}
    d = np.zeros((sizes[target], 1), dtype=np.float64)
    source = result.model_name
    # Only identical distortion models, or the exact Brown subset of Rational.
    if source == target and target != M.PINHOLE and result.distortion is not None:
        values = np.asarray(result.distortion, dtype=float).reshape(-1)
        d[:, 0] = values[:sizes[target]]
    elif source == M.BROWN_CONRADY and target == M.EXTENDED_PINHOLE:
        d[:5, 0] = np.asarray(result.distortion).reshape(-1)[:5]
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite distortion initialization')
    return k, d
