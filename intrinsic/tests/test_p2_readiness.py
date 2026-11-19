"""Gate policy isolation; real fitting/selection is covered by test_p2_e2e."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tests.test_p2_live import detection, profile
from calibration.types import CameraConfig, Dataset, Frame, ImageInfo, DiversityScores, CoverageCell, HoldoutEvidenceGate
from automation.live.readiness import check_readiness


@pytest.mark.parametrize('failure', [None,'count','coverage','position','corners','distance','tilt','edge','duplicates','holdout','train'])
def test_readiness_checks_each_requirement(monkeypatch,failure):
    import automation.live.readiness as module
    data=Dataset(frames=[Frame(ImageInfo(str(i),f'{i}.png',160,120),detection()) for i in range(8)])
    for i,f in enumerate(data.frames): f.detection.board_center_px=((i%3+.5)*160/3,(i//3+.5)*120/3)
    if failure=='position':
        for f in data.frames: f.detection.board_center_px=(80,60)
    cells=[CoverageCell(r,c,corner_count=1,coverage_score=1) for r in range(4) for c in range(4)]
    if failure=='corners': cells=[]
    diversity=DiversityScores(1.,0. if failure=='distance' else 1.,0. if failure=='tilt' else 1.,0. if failure=='edge' else 1.)
    def analyze(dataset,_camera): dataset.coverage_grid=cells; dataset.diversity=diversity; return []
    monkeypatch.setattr(module,'analyze_dataset_quality',analyze)
    monkeypatch.setattr(module,'compute_frame_quality_scores',lambda *_a,**_k:None)
    monkeypatch.setattr(module,'select_calibration_images',lambda *_:SimpleNamespace(
        selected_paths=[f.image_info.path for f in data.frames],counts={'duplicate_scene':8 if failure=='duplicates' else 0}))
    monkeypatch.setattr(module,'coverage_percentage',lambda _:0 if failure=='coverage' else 100)
    monkeypatch.setattr(module,'split_train_test',lambda *_:([str(i) for i in range(2 if failure=='train' else 4)],['6','7']))
    monkeypatch.setattr(module,'evaluate_holdout_evidence',lambda *_a,**_k:HoldoutEvidenceGate(
        status='insufficient_evidence' if failure=='holdout' else 'sufficient',
        reasons=['holdout insufficient'] if failure=='holdout' else []))
    p=profile(readiness={'min_images':12 if failure=='count' else 3})
    result,_=check_readiness(data,CameraConfig(160,120),None,p)
    assert result['ready']==(failure is None),result
    assert bool(result['reasons'])==(failure is not None)
