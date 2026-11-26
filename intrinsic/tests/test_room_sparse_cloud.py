"""
tests/test_room_sparse_cloud.py
===============================

CAMERA–LiDAR: 녹화 전에 복셀 다운샘플된 점군(표본 간격 ~10 cm)에서도 보드를 찾는다.

- 보드 구멍(지름 24 cm)이 표본 1~2개 빈자리로만 보이는 점군에서 template 정합으로 찾고,
  중심이 몇 cm 안으로 맞는다
- 벽에 뚫린 구멍 4개(창문 등)는 패널 둘레가 막혀 있어 보드로 받지 않는다
- 녹화 중 차량이 움직인 구간은 피해서 정지 구간을 고른다 (still_moment)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOM = Path(__file__).resolve().parents[2] / "extrinsic"
if str(ROOM) not in sys.path:
    sys.path.append(str(ROOM))

from gui.core import room  # noqa: E402
from gui.core.project import Target  # noqa: E402

TARGET = Target()  # stock CAD board: holes 0.50 x 0.40 m apart, radius 0.12 m
LEAF = 0.15  # gives a sample spacing of ~9 cm, like the PV5 recordings (10-11 cm)


def _frame(centre, yaw_deg):
    """Board axes (right, up, normal) for a board at `centre` turned `yaw_deg` from facing the origin."""
    n = -np.asarray(centre, float)
    n[2] = 0.0
    n /= np.linalg.norm(n)
    a = np.radians(yaw_deg)
    n = np.array([np.cos(a) * n[0] - np.sin(a) * n[1], np.sin(a) * n[0] + np.cos(a) * n[1], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, n)
    return right / np.linalg.norm(right), up, n


def _surface(rng, centre, yaw_deg, w, h, holes, n=60000):
    """Dense random samples of a w x h panel with circular holes, in world coordinates."""
    right, up, _ = _frame(centre, yaw_deg)
    uv = rng.uniform([-w / 2, -h / 2], [w / 2, h / 2], size=(n, 2))
    for hx, hy, r in holes:
        uv = uv[np.hypot(uv[:, 0] - hx, uv[:, 1] - hy) > r]
    return np.asarray(centre) + uv[:, :1] * right + uv[:, 1:] * up


def _voxel_means(xyz, leaf=LEAF, offset=0.0):
    """What a voxel-grid filter publishes: one mean per occupied voxel."""
    key = np.floor((xyz + offset) / leaf).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((inv.max() + 1, 3))
    np.add.at(sums, inv, xyz)
    return (sums / np.bincount(inv)[:, None]).astype(np.float32)


def _sweeps(scene, k=10, seed=0):
    """`k` downsampled sweeps: each samples the surfaces afresh, as a real sweep's returns do."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        dense = np.vstack([f(rng) for f in scene])
        dense = dense + rng.normal(0, 0.005, dense.shape)
        out.append(_voxel_means(dense))
    return out


def _holes(target):
    w, h, r = target.delta_width_circles / 2, target.delta_height_circles / 2, target.circle_radius
    return [(-w, h, r), (w, h, r), (w, -h, r), (-w, -h, r)]


BOARD_AT = np.array([8.0, 1.5, 1.6])


def _board_scene():
    pw, ph = 1.40, 0.96
    return [
        lambda rng: _surface(rng, BOARD_AT, 12.0, pw, ph, _holes(TARGET)),
        # the stand: a thin pole below the panel
        lambda rng: _surface(rng, BOARD_AT - [0, 0, 0.9], 12.0, 0.06, 0.8, [], n=3000),
        # the wall far behind, off the board's plane
        lambda rng: _surface(rng, np.array([16.0, 0.0, 1.5]), 0.0, 12.0, 3.0, [], n=200000),
    ]


def test_sparse_board_is_found_and_centred():
    sweeps = _sweeps(_board_scene(), k=10)
    cloud, single = np.vstack(sweeps), sweeps[0]
    assert room.sweep_spacing(single) >= room.SPARSE_SPACING
    boards, stats = room.find_lidar_boards(cloud, single, TARGET)
    assert stats["sparse"]
    near = [b for b in boards if np.linalg.norm(b.centers.mean(axis=0) - BOARD_AT) < 0.3]
    assert len(near) == 1, [b.centers.mean(axis=0) for b in boards]
    assert np.linalg.norm(near[0].centers.mean(axis=0) - BOARD_AT) < 0.04
    assert near[0].det.method == "template"


def test_holes_in_a_wall_are_not_a_board():
    """Four empty discs in the board's pattern, but wall on every side of them."""
    at = np.array([9.0, -2.0, 1.6])
    scene = [lambda rng: _surface(rng, at, 0.0, 4.0, 2.6, _holes(TARGET), n=150000)]
    sweeps = _sweeps(scene, k=10, seed=1)
    cloud, single = np.vstack(sweeps), sweeps[0]
    right, up, n = _frame(at, 0.0)
    box = room.box_for_patch(at, np.column_stack([right, up, n]), TARGET)
    det = room.fit_board_template(cloud, box, TARGET, room.sweep_spacing(single))
    assert not det.ok
    assert "막혀" in det.reason


def test_dense_cloud_is_not_sparse():
    rng = np.random.default_rng(3)
    dense = _surface(rng, BOARD_AT, 0.0, 1.4, 0.96, _holes(TARGET), n=40000).astype(np.float32)
    assert room.sweep_spacing(dense) < room.SPARSE_SPACING
    assert room.auto_sweeps(dense)[0] == 1
    sparse = _voxel_means(dense.astype(float))
    n, note = room.auto_sweeps(sparse)
    assert n == room.SPARSE_SWEEPS and "다운샘플" in note


class _FakeSource:
    """A recording at 1 Hz: parked for 0-15 s, driving 16-40 s, parked again 41-45 s."""

    def __init__(self):
        rng = np.random.default_rng(5)
        self.base = rng.uniform([-10, -10, 0], [10, 10, 3], size=(5000, 3)).astype(np.float32)
        self.base = self.base[np.linalg.norm(self.base, axis=1) > 2]
        self.stamps = [int(s * 1e9) for s in range(46)]

    def timestamps(self, topic):
        return list(self.stamps)

    def first_after(self, topic, t):
        s = t / 1e9
        shift = 0.0 if s <= 15 else (min(s, 40) - 15) * 0.5  # 0.5 m/s, then stopped
        jitter = np.random.default_rng(int(s)).normal(0, 0.005, self.base.shape)
        return t, self.base + [shift, 0, 0] + jitter


@pytest.fixture
def fake_clouds(monkeypatch):
    monkeypatch.setattr(room, "_sample_cloud", lambda msg, cap=20000: np.asarray(msg, np.float32))
    return _FakeSource()


def test_still_moment_moves_off_the_drive(fake_clouds):
    t, note = room.still_moment(fake_clouds, "/lidar", int(28e9))
    assert 4e9 <= t <= 11e9  # the middle of the long parked stretch, not the short one at the end
    assert "정지 구간" in note


def test_still_moment_keeps_a_still_request(fake_clouds):
    t, note = room.still_moment(fake_clouds, "/lidar", int(6e9))
    assert t == int(6e9) and note == ""
