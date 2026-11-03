from __future__ import annotations

import copy
from dataclasses import asdict
import time

from calibration.frame_quality import compute_frame_quality_scores
from calibration.holdout_evidence import evaluate_holdout_evidence
from calibration.image_selection import select_calibration_images
from calibration.models.common import MIN_FRAMES_REQUIRED
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.types import Dataset
from calibration.validation import split_train_test
from automation.live.assessment import assess_observability


def check_readiness(dataset, camera, pattern, profile):
    """Cheap gates on the SAME candidate selection/split rules as P1. No fitting."""
    start = time.monotonic()
    data = copy.deepcopy(dataset)
    analyze_dataset_quality(data, camera)
    input_grid = list(data.coverage_grid)
    compute_frame_quality_scores(data, pattern, (camera.width, camera.height), use_reprojection=False)
    selection = select_calibration_images(data, profile.p1.image_selection)
    chosen = set(selection.selected_paths)
    data = Dataset(frames=[f for f in data.frames if f.image_info.path in chosen])
    coaching = analyze_dataset_quality(data, camera)
    cfg = profile.readiness
    diversity = data.diversity
    coverage = coverage_percentage(data.coverage_grid)
    full_frame_coverage = coverage
    mask = getattr(profile, 'coverage_mask', None)
    observable = input_observable = None
    if mask is not None:
        from calibration.occlusion import observable_coverage
        observable = observable_coverage(data, camera, mask)
        input_observable = observable_coverage(dataset, camera, mask)
        coverage = observable['coverage_pct']
        # Full-frame coaching would request structurally hidden regions. Keep
        # other gates below unchanged and give visible-region coaching only.
        coaching = [f'관측 가능 영역의 부족 셀(4×4, 행/열 1부터): '
                    f'{[(c["row"]+1,c["col"]+1) for c in observable["grid"] if c["visible_pixels"] and not c["sufficient"]]}'] if coverage < cfg.coverage_pct else []
    centers = {(min(2, int(f.detection.board_center_px[0] / camera.width * 3)),
                min(2, int(f.detection.board_center_px[1] / camera.height * 3)))
               for f in data.frames if f.detection.board_center_px is not None}
    corner_regions = {(c.row, c.col) for c in data.coverage_grid if c.corner_count > 0}
    required_corners = {(0, 0), (0, 3), (3, 0), (3, 3)}
    if observable is not None:
        required_corners = {tuple(rc) for rc in observable['required_corner_cells']}
        corner_regions = {(c['row'],c['col']) for c in observable['grid'] if c['corner_count'] > 0}
    train, test = split_train_test(data, camera, profile.p1.calibration.get('test_ratio', .25),
                                   profile.p1.calibration.get('seed', 42))
    holdout = evaluate_holdout_evidence(
        Dataset(frames=[f for f in data.frames if f.image_info.image_id in set(test)]), camera,
        min_test_frames=cfg.min_test_frames, min_test_corners=cfg.min_test_corners,
        min_coverage_pct=cfg.test_coverage_pct, min_pose_diversity=cfg.test_pose_diversity)
    duplicate_ratio = selection.counts.get('duplicate_scene', 0) / max(1, len(dataset.frames))
    reasons = []
    def require(ok, text):
        if not ok:
            reasons.append(text)
    require(len(data.frames) >= cfg.min_images, f'유효 이미지가 부족합니다 ({len(data.frames)}/{cfg.min_images}).')
    require(coverage >= cfg.coverage_pct, f'{"관측 가능 영역" if mask else "화면"} coverage가 부족합니다 ({coverage:.1f}/{cfg.coverage_pct}%).')
    require(len(centers) >= cfg.min_center_cells, '보드 중심을 좌/중/우·상/중/하로 다양하게 이동해 주세요.')
    require(not cfg.require_corner_regions or required_corners <= corner_regions,
            '화면 모서리의 검출 데이터가 부족합니다.')
    if coverage < cfg.coverage_pct or (cfg.require_corner_regions and
            not required_corners <= corner_regions):
        # Report grid coordinates as 1-based row/column, with an explicit frame
        # of reference. Missing observations and selection losses differ.
        absent = [(c.row + 1, c.col + 1) for c in input_grid if c.corner_count == 0]
        input_covered = {(c.row, c.col) for c in input_grid if coverage_percentage([c]) > 0}
        selected_covered = {(c.row, c.col) for c in data.coverage_grid if coverage_percentage([c]) > 0}
        lost = sorted((r + 1, c + 1) for r, c in input_covered - selected_covered)
        if observable is not None:
            absent = [(c['row']+1,c['col']+1) for c in input_observable['grid'] if c['visible_pixels'] and not c['corner_count']]
            input_covered = {(c['row'],c['col']) for c in input_observable['grid'] if c['sufficient']}
            selected_covered = {(c['row'],c['col']) for c in observable['grid'] if c['sufficient']}
            lost = sorted((r+1,c+1) for r,c in input_covered-selected_covered)
        if absent:
            reasons.append(f'전체 입력에 검출 코너가 없는 셀(4×4, 위→아래 행/좌→우 열, 1부터): {absent}. 해당 화면 영역에 보드 코너를 추가 촬영해 주세요.')
        if lost:
            reasons.append(f'자동 선별 후 coverage 기준에서 탈락한 셀(4×4, 행/열 1부터): {lost}. 보존된 입력의 선정 결과를 검토해 주세요.')
    require(diversity.distance_diversity >= cfg.distance_diversity, '보드를 더 가까이/멀리 이동해 주세요.')
    require(diversity.rotation_diversity >= cfg.tilt_diversity, '다양한 기울기에서 촬영해 주세요.')
    require(diversity.edge_coverage >= cfg.edge_coverage, '이미지 가장자리의 검출 데이터가 부족합니다.')
    require(duplicate_ratio <= cfg.max_duplicate_ratio, '저장 후보의 중복 자세 비율이 높습니다.')
    require(len(train) >= MIN_FRAMES_REQUIRED, 'Train 학습 프레임이 부족합니다.')
    reasons.extend(holdout.reasons)
    result = {'ready': not reasons, 'reasons': reasons, 'coaching': coaching,
            'selected_paths': selection.selected_paths, 'train_ids': train, 'test_ids': test,
            'coverage_pct': coverage, 'center_cells': len(centers), 'diversity': asdict(diversity),
            'duplicate_ratio': duplicate_ratio, 'holdout': asdict(holdout),
            'cheap_seconds': time.monotonic() - start}
    if observable is not None:
        result.update(coverage_basis='observable',full_frame_coverage_pct=full_frame_coverage,
                      observable_coverage=observable, input_observable_coverage=input_observable)
    return result, data


def preflight_observability(dataset, camera, train_ids, config):
    """Train-only preliminary Brown fit; never changes P1 fit or weights."""
    from calibration.models.brown_conrady import calibrate_brown_conrady
    from calibration.observability import compute_observability_report
    start = time.monotonic()
    train = Dataset(frames=[f for f in dataset.frames if f.image_info.image_id in set(train_ids)])
    result = calibrate_brown_conrady(train, camera, estimate_uncertainty_bootstrap=False)
    if not result.success:
        return {'ready': False, 'reasons': ['사전 Calibration 실패: 다양한 자세의 데이터를 추가해 주세요.'],
                'error': result.error_message, 'seconds': time.monotonic() - start}
    obs = compute_observability_report(result, train)
    assessment = assess_observability(asdict(obs), config)
    return {**assessment, 'seconds': time.monotonic() - start}
