"""
tests/test_room_beam_rescue.py
==============================

CAMERA–LiDAR: 빔통과 검사에만 걸린 대응을 보드가 분명히 가리키면 되살린다
(sensor_calibration rear 카메라 모호성 수정, room._rescue_beam_dropped).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOM = Path(__file__).resolve().parents[2] / "extrinsic"
if str(ROOM) not in sys.path:
    sys.path.append(str(ROOM))

from gui.core import room  # noqa: E402

BEAM = "빔통과: 카메라 자리를 LiDAR 빔 42개가 지나감"


def _d(n, per_board_m, why=BEAM, sol="sol"):
    return (n, n * per_board_m, sol, [], why)


class _Check:
    def __init__(self, wrong_way=()):
        self.wrong_way = set(wrong_way)

    def facing(self, sol):
        return "방향: 반대" if sol in self.wrong_way else ""


def test_many_boards_dropped_for_beams_only_are_taken_back():
    cands = [(7, 7 * 0.05, "x", [])]
    good = _d(10, 0.06)
    assert room._rescue_beam_dropped([good], cands, _Check()) == [good]


def test_not_taken_back_without_a_clear_board_margin_or_when_loose():
    cands = [(7, 7 * 0.05, "x", [])]
    assert room._rescue_beam_dropped([_d(8, 0.04)], cands, _Check()) == []  # 7 + 2 미만
    assert room._rescue_beam_dropped([_d(10, 0.09)], cands, _Check()) == []  # 보드당 8 cm 초과
    assert room._rescue_beam_dropped([_d(2, 0.01)], [], _Check()) == []  # 약한 카메라는 그대로


def test_direction_still_applies_and_other_reasons_stay_dropped():
    cands = [(7, 7 * 0.05, "x", [])]
    assert room._rescue_beam_dropped([_d(10, 0.05, sol="back")], cands, _Check({"back"})) == []
    assert room._rescue_beam_dropped([_d(10, 0.05, why="방향: 반대")], cands, _Check()) == []


def test_best_first():
    cands = [(3, 0.15, "x", [])]
    a, b, c = _d(6, 0.05, sol="a"), _d(8, 0.06, sol="b"), _d(8, 0.03, sol="c")
    assert room._rescue_beam_dropped([a, b, c], cands, _Check()) == [c, b, a]
