"""
Stereo workflow session state and small controller-style operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calibration.stereo import (
    StereoCalibrationResult,
    StereoPairObservation,
    reject_pairs_by_id,
)


@dataclass
class StereoSession:
    pairs: list[StereoPairObservation] = field(default_factory=list)
    result: StereoCalibrationResult | None = None

    def outlier_pair_ids(self) -> set[str]:
        if self.result is None:
            return set()
        return {row.pair_id for row in self.result.pair_validations if row.status == "Outlier"}

    def visible_pairs(self, *, outliers_only: bool = False) -> list[StereoPairObservation]:
        if not outliers_only:
            return self.pairs
        outliers = self.outlier_pair_ids()
        return [pair for pair in self.pairs if pair.pair_id in outliers]

    def reject_outliers(self) -> int:
        return reject_pairs_by_id(self.pairs, self.outlier_pair_ids())
