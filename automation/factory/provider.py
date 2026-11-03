"""Explicit, versioned reference envelope. No inferred vendor coefficient order."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import cv2

from calibration.calibration_io import load_standard_calibration
from calibration.types import CameraModelType


class FactoryError(ValueError):
    pass


ORDERS = {
    'brown_conrady': ['k1', 'k2', 'p1', 'p2', 'k3'],
    'extended_pinhole': ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'fisheye': ['k1', 'k2', 'k3', 'k4'],
}


@dataclass
class FactoryReference:
    metadata: dict
    model: CameraModelType
    K: np.ndarray
    D: np.ndarray


class FactoryProvider(Protocol):
    def lookup(self, camera_model: str, serial: str) -> FactoryReference: ...


def validate_coordinates(value, width, height):
    if not isinstance(value, dict):
        raise FactoryError('Explicit coordinate_system required')
    if value.get('reference_resolution') != [width, height]:
        raise FactoryError('Calibration reference resolution mismatch')
    if value.get('binning') != [1, 1] or value.get('orientation') != 'identity':
        raise FactoryError('Only explicitly declared unbinned identity orientation is supported')
    if value.get('roi') != [0, 0, width, height]:
        raise FactoryError('Only explicit full-frame coordinates are supported; no automatic rescaling')


def validate_parameters(model, K, D, width, height):
    if model.value not in ORDERS:
        raise FactoryError('Unsupported distortion model')
    if K.shape != (3, 3) or D.size != len(ORDERS[model.value]):
        raise FactoryError('Invalid K shape or distortion coefficient count')
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise FactoryError('Nonfinite K/D')
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12) or K[0, 1] != 0 or K[1, 0] != 0:
        raise FactoryError('Unsupported camera matrix structure/skew')
    if not (0 < K[0, 0] <= 100 * width and 0 < K[1, 1] <= 100 * height):
        raise FactoryError('Invalid focal length (plausibility bound; not quality approval)')
    if not (0 <= K[0, 2] < width and 0 <= K[1, 2] < height):
        raise FactoryError('Principal point outside reference image')


class FileProvider:
    """JSON envelope, optionally referencing existing OpenCV YAML K/D parser.

    Metadata remains mandatory even if YAML lacks identity/coordinate information.
    File provenance is a user declaration, never manufacturer authentication.
    """
    def __init__(self, path):
        self.path = Path(path)

    def lookup(self, camera_model, serial):
        try:
            m = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if m.get('schema_version') != 1:
                raise FactoryError('Unsupported factory reference schema')
            if m.get('camera_model') != camera_model or m.get('serial_number') != serial:
                raise FactoryError('Camera model or serial mismatch')
            if not isinstance(m.get('source'), str) or not m['source'].strip():
                raise FactoryError('Factory data source required')
            if m.get('factory_rms') is not None and (not isinstance(m['factory_rms'], (int,float))
                    or not np.isfinite(m['factory_rms']) or m['factory_rms'] < 0):
                raise FactoryError('Invalid optional factory RMS')
            w, h = m['width'], m['height']
            if type(w) is not int or type(h) is not int or w <= 0 or h <= 0:
                raise FactoryError('Invalid resolution')
            model = CameraModelType(m['distortion_model'])
            if m.get('projection_convention') != 'opencv':
                raise FactoryError('Explicit OpenCV projection convention required; vendor fisheye definition is not assumed equivalent')
            if model.value not in ORDERS or m.get('coefficient_order') != ORDERS[model.value]:
                raise FactoryError('Unknown coefficient order; vendor reorder is not verified')
            validate_coordinates(m.get('coordinate_system'), w, h)
            if 'intrinsic_file' in m:
                if 'K' in m or 'D' in m:
                    raise FactoryError('Ambiguous inline and file intrinsic sources')
                p = (self.path.parent / m['intrinsic_file']).resolve()
                if not p.is_relative_to(self.path.parent.resolve()):
                    raise FactoryError('Intrinsic file must remain inside reference directory')
                cal = load_standard_calibration(str(p))
                if (cal.width, cal.height) != (w, h) or cal.model_name != model:
                    raise FactoryError('YAML model/resolution mismatch')
                K, D = cal.camera_matrix, cal.distortion
            else:
                K, D = np.asarray(m['K'], dtype=float), np.asarray(m['D'], dtype=float)
            K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float).reshape(-1)
            validate_parameters(model, K, D, w, h)
            from calibration.calibration_io import StandardCalibration
            from calibration.benchmark_compatibility import validate_single_calibration
            cal = StandardCalibration('Factory reference', K, D, model_name=model, width=w, height=h)
            issues = validate_single_calibration(cal, side='Factory', validation_image_size=(w,h))
            if any(issue.severity.value == 'error' for issue in issues):
                raise FactoryError('Existing calibration compatibility validation failed')
            m['validation_warnings'] = [issue.message for issue in issues]
            return FactoryReference(m, model, K.copy(), D.copy())
        except FactoryError:
            raise
        except (ValueError, KeyError, TypeError, OSError, cv2.error, SystemError) as exc:
            raise FactoryError(f'Invalid factory reference: {type(exc).__name__}') from None


class SensingSDKProvider:
    def lookup(self, camera_model, serial):
        raise FactoryError('SDK_UNAVAILABLE: Jetson SDK/device ABI and coefficient definition unverified; no serial server API configured')
