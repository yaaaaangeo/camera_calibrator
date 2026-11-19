import numpy as np

from calibration.image_selection import ImageSelectionRecord, _preserve_coverage
from calibration.types import DetectionResult, Frame, ImageInfo


def frame(name, points):
    return Frame(image_info=ImageInfo(image_id=name, path=name, width=400, height=400),
                 detection=DetectionResult(image_id=name, success=True,
                    corners=np.array(points, dtype=np.float32).reshape(-1, 1, 2),
                    num_corners=len(points)))


def records(ids):
    return {i: ImageSelectionRecord(i, i, position='center', distance='far', tilt='front',
                                  cluster_id=n, score=1.) for n, i in enumerate(ids)}


def test_swap_preserves_covered_cells_and_replaces_duplicate_representative():
    frames = [frame('a', [(150, 150)] * 100), frame('b', [(150, 150)] * 100),
              frame('c', [(150, 150)] * 90 + [(350, 250)] * 10)]
    rec = records(['a', 'b', 'c'])
    rec['c'].cluster_id = rec['b'].cluster_id
    chosen = _preserve_coverage(frames, rec, {'a', 'b'})
    assert chosen == {'a', 'c'}
    assert _preserve_coverage(frames, rec, {'a', 'b'}) == chosen


def test_coverage_cannot_override_distance_tilt_or_duplicate_constraints():
    frames = [frame('a', [(150, 150)] * 100), frame('b', [(150, 150)] * 100),
              frame('c', [(150, 150)] * 90 + [(350, 250)] * 10)]
    for field, value in [('distance', 'close'), ('tilt', 'left_tilt')]:
        rec = records(['a', 'b', 'c'])
        setattr(rec['c'], field, value)
        assert _preserve_coverage(frames, rec, {'a', 'b'}) == {'a', 'b'}
    rec = records(['a', 'b', 'c'])
    rec['c'].cluster_id = rec['a'].cluster_id
    rec['a'].position = 'left'  # cannot replace a; replacing b would add a duplicate
    assert _preserve_coverage(frames, rec, {'a', 'b'}) == {'a', 'b'}


def test_empty_pool_and_no_new_spatial_support_keep_original_choice():
    assert _preserve_coverage([], {}, set()) == set()
    frames = [frame('a', [(150, 150)] * 20), frame('b', [(150, 150)] * 20)]
    assert _preserve_coverage(frames, records(['a', 'b']), {'a'}) == {'a'}


def test_readiness_distinguishes_missing_input_cells_from_selection_loss(monkeypatch):
    from types import SimpleNamespace
    from tests.test_p2_live import profile, detection
    from calibration.types import Dataset, CameraConfig, DiversityScores, CoverageCell, HoldoutEvidenceGate
    import automation.live.readiness as module
    data = Dataset(frames=[Frame(ImageInfo(str(i), f'{i}.png', 160, 120), detection()) for i in range(8)])
    calls = []
    def analyze(dataset, camera):
        selected = bool(calls)
        calls.append(1)
        dataset.coverage_grid = [CoverageCell(r, c, int(r < 3), float(r < 3 and not (selected and (r, c) == (2, 3))))
                                 for r in range(4) for c in range(4)]
        dataset.diversity = DiversityScores(1., 1., 1., 1.)
        return []
    monkeypatch.setattr(module, 'analyze_dataset_quality', analyze)
    monkeypatch.setattr(module, 'compute_frame_quality_scores', lambda *a, **k: None)
    monkeypatch.setattr(module, 'select_calibration_images', lambda *a: SimpleNamespace(
        selected_paths=[f.image_info.path for f in data.frames], counts={}))
    monkeypatch.setattr(module, 'split_train_test', lambda *a: (['0', '1', '2', '3'], ['6', '7']))
    monkeypatch.setattr(module, 'evaluate_holdout_evidence', lambda *a, **k: HoldoutEvidenceGate(status='sufficient'))
    result, _ = module.check_readiness(data, CameraConfig(160, 120), None, profile(readiness={'min_images': 3}))
    assert not result['ready']
    assert any('없는 셀' in reason and '(4, 1)' in reason for reason in result['reasons'])
    assert any('선별 후' in reason and '(3, 4)' in reason for reason in result['reasons'])
