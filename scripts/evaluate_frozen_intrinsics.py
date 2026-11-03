"""Evaluate frozen K/D on a declared new capture using existing fixed-K benchmark API.

No calibration, seed selection, pruning, recommendation update, or publication.
Exact pixel disjointness is checked; temporal/scene independence still needs review.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import cv2
from calibration.project_io import load_project
from calibration.detector import detect_dataset
from calibration.calibration_io import load_standard_calibration
from calibration.external_compare import compare_reference_candidate_calibrations
from calibration.json_utils import json_safe


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(path, expected_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != tuple(expected_size):
        raise ValueError(f'Unreadable image or native resolution mismatch: {path}')
    return hashlib.sha256(str((image.shape, str(image.dtype))).encode()+image.tobytes()).hexdigest()


def assert_disjoint(training_paths, evaluation_paths, size):
    known = {pixel_digest(p, size) for p in training_paths}
    evaluated = []
    for path in evaluation_paths:
        h = pixel_digest(path, size)
        if h in known:
            raise ValueError('Evaluation image duplicates development/training pixels: '+str(path))
        if any(x['pixel_sha256'] == h for x in evaluated):
            raise ValueError('Duplicate evaluation pixels: '+str(path))
        evaluated.append({'name':Path(path).name,'file_sha256':digest(path),'pixel_sha256':h})
    return evaluated


def evaluate(source_project, reference, candidate, images, output, capture_id,
             source_capture_id='rosbag2_2026_10_23-15_38_30', training_manifests=()):
    if not capture_id.strip() or capture_id.strip() == source_capture_id:
        raise ValueError('Declare a distinct new capture ID; source recording is not independent')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; existing results are preserved')
    locked = {str(p):digest(p) for p in (source_project, reference, candidate)}
    project, missing = load_project(str(source_project))
    if missing:
        raise ValueError('Development images are missing; pixel-disjointness cannot be checked')
    paths = sorted(p for p in Path(images).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
    training = [f.image_info.path for f in project.dataset.frames]
    for manifest in training_manifests:
        training.extend(json.loads(Path(manifest).read_text(encoding='utf-8'))['hashes'])
    size = (project.camera_config.width, project.camera_config.height)
    evidence = assert_disjoint(sorted(set(training)), paths, size)
    if len(evidence) < 10:
        raise ValueError('At least 10 new images required; independent evaluation will not fall back to old data')
    dataset = detect_dataset([str(p) for p in paths], project.pattern_config, parallel=False)
    if dataset.num_detected < 10:
        raise ValueError('Fewer than 10 detected new views; collect more data, no fallback')
    ref = load_standard_calibration(str(reference));cand = load_standard_calibration(str(candidate))
    frozen_parameters = {name:{'model':cal.model_name.value, 'K':cal.camera_matrix.tolist(),
                               'D':cal.distortion.tolist(), 'width':cal.width, 'height':cal.height,
                               'parameter_scope':'frozen exported K/D; no intrinsic refit'}
                         for name,cal in [('reference',ref),('candidate',cand)]}
    original_arrays = [(cal.camera_matrix.tobytes(),cal.distortion.tobytes()) for cal in (ref,cand)]
    result = compare_reference_candidate_calibrations(
        project.dataset, project.camera_config, project.pattern_config, ref, cand, [],
        independent_benchmark_dataset=dataset, evaluation_mode='independent_benchmark',
        benchmark_kfold=5, benchmark_bootstrap=1000)
    if result.evaluation_source != 'independent_benchmark' or result.benchmark_status != 'ok':
        raise ValueError('Independent benchmark rejected; no internal-data fallback accepted')
    if original_arrays != [(cal.camera_matrix.tobytes(),cal.distortion.tobytes()) for cal in (ref,cand)]:
        raise ValueError('Frozen K/D arrays changed during pose-only evaluation')
    if any(digest(p) != h for p,h in locked.items()):
        raise ValueError('Frozen calibration/source project changed during evaluation')
    payload = {'approval':'NOT_APPROVED','capture_id':capture_id,'source_capture_id':source_capture_id,
               'independence':'DECLARED_NEW_CAPTURE; exact pixel overlap excluded; temporal/scene independence requires review',
               'intrinsics_refitted':False,'frozen_parameters':frozen_parameters,'locked_files':locked,'evaluated_images':evidence,
               'board':json_safe(asdict(project.pattern_config),ndarray_wrapper=False),
               'marker_size_status':'UNCONFIRMED_ASSUMPTION; verify physical board before interpreting accuracy',
               'result':json_safe(asdict(result),ndarray_wrapper=False)}
    output.mkdir(parents=True)
    (output/'independent_result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-project','reference','candidate','images','output','capture-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--training-manifest',action='append',default=[])
    parser.add_argument('--source-capture-id',default='rosbag2_2026_10_23-15_38_30')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    try:
        evaluate(args.source_project,args.reference,args.candidate,args.images,args.output,args.capture_id,
                 args.source_capture_id,args.training_manifest)
    except (ValueError,OSError) as exc:
        parser.exit(2,str(exc)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
