"""Project state — everything the user builds up across the steps.

Split in two files on purpose:

  <name>.calib.yaml    camera, target, scene picks, filters. Small, portable,
                       meant to be committed and synced between machines.
  <name>.local.yaml    where the bag actually lives on *this* machine.

Scenes are identified by timestamp, so opening the project on another machine
only needs the bag to be located again -- everything else carries over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path



# The package's own folders, resolved from this file so they hold wherever the
# repository is checked out.
#
#   calib_result  results, per vehicle and session. Kept out of this repository;
#                 the results the team consumes go to the results repository with
#                 the Git upload button (step 7).
PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = PACKAGE_DIR / "calib_result"


@dataclass
class Camera:
    """Camera intrinsics, in one of two projection models.

    `model="pinhole"` covers plumb_bob (5 coefficients) and rational_polynomial
    (8). k3 is included; the original tool forced it to zero. k4..k6 are the
    rational model's denominator terms, and OpenCV picks between the two from how
    many coefficients it is handed -- so `dist()` returning eight is all
    solvePnP, projectPoints and undistort need to use them.

    `model="fisheye"` is OpenCV's equidistant model: four radial terms, no
    tangential ones, and they live in k1..k4. p1, p2, k5 and k6 mean nothing
    there and are left out of `dist()`. A fisheye camera cannot be projected with
    the pinhole calls at all -- see `gui.core.camera_ops`, which every call site
    goes through for exactly that reason.

    Neither difference is a small approximation. Truncating a rational
    calibration to five coefficients makes the denominator disappear, so the
    curve changes shape rather than losing a high-order term; projecting a
    fisheye as a pinhole swaps f·θ for f·tan θ. Both run to hundreds of pixels of
    error near the edges.
    """

    # "pinhole" or "fisheye". First field so it reads before the numbers it
    # governs, and so a saved project says up front what it is.
    model: str = "pinhole"

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    # Rational-model denominator, and the fisheye model's fourth radial term.
    # For a pinhole camera all three zero means plumb_bob, and then the rational
    # form is identical to it -- so a 5-coefficient camera stays 5-coefficient
    # everywhere, byte for byte.
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    @property
    def is_fisheye(self) -> bool:
        return self.model == "fisheye"

    @property
    def rational(self) -> bool:
        """Does this camera need the 8-coefficient pinhole form?"""
        return not self.is_fisheye and bool(self.k4 or self.k5 or self.k6)

    @property
    def model_name(self) -> str:
        """What to call this model in a file someone else will read.

        The names are the ones ROS and the sibling camera_calibration tool use,
        so a distortion_model written here means the same thing there.
        """
        if self.is_fisheye:
            return "fisheye"
        return "rational_polynomial" if self.rational else "plumb_bob"

    @property
    def model_label(self) -> str:
        """Model and coefficient count together, for showing to a person."""
        return f"{self.model_name}({len(self.dist_names())})"

    @property
    def is_set(self) -> bool:
        return self.fx > 0 and self.fy > 0

    def matrix(self):
        import numpy as np

        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def dist(self):
        import numpy as np

        if self.is_fisheye:
            return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)
        coeffs = [self.k1, self.k2, self.p1, self.p2, self.k3]
        if self.rational:
            coeffs += [self.k4, self.k5, self.k6]
        return np.array(coeffs, dtype=np.float64)

    def dist_names(self) -> tuple[str, ...]:
        """Coefficient names in the order `dist()` returns them."""
        if self.is_fisheye:
            return ("k1", "k2", "k3", "k4")
        base = ("k1", "k2", "p1", "p2", "k3")
        return (base + ("k4", "k5", "k6")) if self.rational else base


@dataclass
class Target:
    """Calibration board geometry, in metres. Defaults are the stock CAD."""

    marker_size: float = 0.20
    delta_width_qr_center: float = 0.55
    delta_height_qr_center: float = 0.35
    delta_width_circles: float = 0.50
    delta_height_circles: float = 0.40
    circle_radius: float = 0.12

    def scaled(self, factor: float) -> "Target":
        return Target(**{k: v * factor for k, v in asdict(self).items()})


@dataclass
class FilterBox:
    """The region kept for detection: six bounds, plus an optional rotation.

    An axis-aligned box cannot isolate a board held at an angle. It has to span
    the board's full diagonal, and the wall behind fills the corners that leaves
    empty -- which is exactly what stops the holes being found. Turning the box
    to sit square with the board lets it be thin again.

    Rotation is about the box centre so the board does not slide off screen
    while being adjusted, and the bounds stay in the box's own frame.
    """

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    yaw: float = 0.0  # degrees, about the sensor's z
    pitch: float = 0.0  # about the box's y after yaw
    roll: float = 0.0  # about the box's x after pitch

    @property
    def is_set(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min and self.z_max > self.z_min

    @property
    def rotated(self) -> bool:
        return bool(self.yaw or self.pitch or self.roll)

    def as_tuple(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def center(self):
        import numpy as np

        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        ])

    def rotation(self):
        """Box axes as columns: sensor frame = R @ box frame."""
        import numpy as np

        cy, sy = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        cp, sp = np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch))
        cr, sr = np.cos(np.radians(self.roll)), np.sin(np.radians(self.roll))
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def mask(self, xyz):
        """Which points fall inside, taking the rotation into account."""
        import numpy as np

        if self.rotated:
            # The cheap axis-aligned bound of the rotated box first: a box is a
            # board's worth of a room-sized cloud, and the rotation per point is
            # most of the cost (4.7 s of a 10-sweep board search).
            raw = np.asarray(xyz)
            c = self.corners()
            lo, hi = c.min(axis=0), c.max(axis=0)
            cand = np.flatnonzero(np.all((raw >= lo) & (raw <= hi), axis=1))
            out = np.zeros(len(raw), bool)
            if len(cand):
                pts = (np.asarray(raw[cand], np.float64) - self.center()) @ self.rotation() + self.center()
                out[cand] = ((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max)
                             & (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max)
                             & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max))
            return out
        pts = np.asarray(xyz, np.float64)
        return (
            (pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max)
            & (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max)
            & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        )

    def corners(self):
        """The eight corners in sensor coordinates, for drawing."""
        import numpy as np

        lo = np.array([self.x_min, self.y_min, self.z_min])
        hi = np.array([self.x_max, self.y_max, self.z_max])
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        if self.rotated:
            c = self.center()
            pts = (pts - c) @ self.rotation().T + c
        return pts
