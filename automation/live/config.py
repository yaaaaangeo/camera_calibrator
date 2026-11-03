from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from importlib import resources
import math
from pathlib import Path

import yaml

from automation.profile import AutomationProfile, ProfileError, resolve_profile
from automation.run_manifest import canonical_sha256


@dataclass(frozen=True)
class CaptureConfig:
    interval_sec: float = 1.0
    center_delta: float = 0.025
    relative_area_delta: float = 0.08
    tilt_delta_deg: float = 5.0
    max_images: int = 120
    min_corners: int = 4
    motion_blur_max: float = 3.0
    roi_contrast_min: float = 10.0
    stream_timeout_sec: float = 5.0
    reconnect_interval_sec: float = 3.0
    max_reconnects: int = 5


@dataclass(frozen=True)
class ReadinessConfig:
    min_images: int = 35
    coverage_pct: float = 75.0
    min_center_cells: int = 6
    require_corner_regions: bool = True
    distance_diversity: float = 0.25
    tilt_diversity: float = 0.25
    edge_coverage: float = 0.25
    max_duplicate_ratio: float = 0.8
    min_test_frames: int = 5
    min_test_corners: int = 60
    test_coverage_pct: float = 40.0
    test_pose_diversity: float = 0.25
    observability_min: float = 50.0  # review boundary, not a compute-start score gate
    require_full_rank: bool = True
    check_every_new_frames: int = 5
    max_checks: int = 20
    timeout_sec: float = 120.0


@dataclass(frozen=True)
class RecoveryConfig:
    max_calculations: int = 3
    max_recaptures: int = 2
    min_new_frames: int = 5
    calculation_timeout_sec: float = 1800.0


@dataclass
class LiveProfile:
    p1: AutomationProfile
    capture: CaptureConfig
    readiness: ReadinessConfig
    recovery: RecoveryConfig
    raw: dict
    coverage_mask: dict | None = None

    @property
    def fingerprint(self):
        data = {"p1": self.p1.to_dict(), "capture": asdict(self.capture),
                "readiness": asdict(self.readiness), "recovery": asdict(self.recovery)}
        if self.coverage_mask is not None:
            data['coverage_mask'] = self.coverage_mask
        return canonical_sha256(data)


def _settings(cls, section):
    if not isinstance(section, dict):
        raise ProfileError(f"{cls.__name__}: expected a mapping")
    defaults = cls()
    valid = {f.name for f in fields(cls)}
    if set(section) - valid:
        raise ProfileError(f"{cls.__name__}: unknown keys {set(section) - valid}")
    for key, value in section.items():
        default = getattr(defaults, key)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        else:
            ok = (not isinstance(value, bool) and isinstance(value, (int, float))
                  and math.isfinite(value) and value >= 0)
            if isinstance(default, int):
                ok = ok and isinstance(value, int)
        if not ok:
            raise ProfileError(f"{cls.__name__}.{key}: invalid value {value!r}")
    result = cls(**section)
    for key in valid:
        value = getattr(result, key)
        if key.endswith('pct') or key == 'observability_min':
            if value > 100:
                raise ProfileError(f"{key}: expected 0..100")
        if key in {'distance_diversity', 'tilt_diversity', 'edge_coverage', 'max_duplicate_ratio'} and value > 1:
            raise ProfileError(f"{key}: expected 0..1")
        if key.endswith('_sec') and value <= 0:
            raise ProfileError(f"{key}: expected positive seconds")
    return result


def resolve_live_profile(raw: dict) -> LiveProfile:
    if not isinstance(raw, dict) or raw.get('live_profile_version') != 1:
        raise ProfileError('live_profile_version must be 1')
    if set(raw) - {'live_profile_version', 'p1', 'capture', 'readiness', 'recovery', 'coverage_mask'}:
        raise ProfileError('Unknown live profile section')
    base = yaml.safe_load(resources.files('automation').joinpath('profiles/default.yaml').read_text(encoding='utf-8'))
    overrides = raw.get('p1', {})
    if not isinstance(overrides, dict):
        raise ProfileError('p1 must be a mapping')
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    profile = resolve_profile(base, source_path='p2:p1')
    if not profile.image_selection_enabled:
        raise ProfileError('P2 requires P1 image_selection.enabled=true')
    if profile.image_selection.exposure_mode != 'pattern':
        raise ProfileError('P2 requires the P1 pattern exposure gate')
    models = {'rational' if str(m) == 'extended_pinhole' else str(m)
              for m in profile.calibration.get('models', [])}
    if models != {'pinhole', 'brown_conrady', 'rational', 'fisheye'}:
        raise ProfileError('P2 requires the existing four intrinsic models')
    capture = _settings(CaptureConfig, raw.get('capture', {}))
    readiness = _settings(ReadinessConfig, raw.get('readiness', {}))
    recovery = _settings(RecoveryConfig, raw.get('recovery', {}))
    if capture.min_corners < 4 or capture.max_images < readiness.min_images or readiness.min_images < 3:
        raise ProfileError('Require min_corners>=4 and max_images>=min_images>=3')
    if readiness.min_center_cells > 9:
        raise ProfileError('min_center_cells must be <=9')
    if readiness.min_images > profile.image_selection.target_count:
        raise ProfileError('readiness.min_images cannot exceed P1 image_selection.target_count')
    if readiness.min_test_frames < 1 or readiness.min_test_corners < 4:
        raise ProfileError('Hold-out requires at least one frame and four corners')
    if min(readiness.check_every_new_frames, readiness.max_checks, recovery.max_calculations,
           recovery.min_new_frames) < 1:
        raise ProfileError('Check/attempt intervals and calculation limits must be positive')
    from calibration.occlusion import resolve_mask
    try:
        mask = resolve_mask(raw.get('coverage_mask'))
    except ValueError as exc:
        raise ProfileError(str(exc)) from exc
    return LiveProfile(profile, capture, readiness, recovery, raw, mask)


def load_live_profile(path: str | Path = 'default') -> LiveProfile:
    try:
        text = (resources.files('automation.live').joinpath('default.yaml').read_text(encoding='utf-8')
                if str(path) == 'default' else Path(path).expanduser().read_text(encoding='utf-8'))
        return resolve_live_profile(yaml.safe_load(text))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(str(exc)) from exc
