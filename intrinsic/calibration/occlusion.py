"""Coverage diagnostics in native distorted-image coordinates; never alter fits."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math

import cv2
import numpy as np

from calibration.quality import coverage_percentage
from calibration.types import CoverageCell


def resolve_mask(raw):
    if raw is None:
        return None
    required = {'schema_version', 'version', 'source', 'camera', 'width', 'height',
                'coordinate_system', 'polygons'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('coverage_mask requires exactly: ' + ', '.join(sorted(required)))
    if type(raw['schema_version']) is not int or raw['schema_version'] != 1:
        raise ValueError('Unsupported coverage_mask schema_version')
    if raw['coordinate_system'] != 'native_distorted_pixel_edges':
        raise ValueError('Mask coordinates must be native_distorted_pixel_edges; no implicit scaling/crop')
    for key in ('version', 'source'):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f'Mask {key} must identify its provenance')
    if (not isinstance(raw['camera'], dict) or set(raw['camera']) != {'model', 'serial'} or
            any(not isinstance(v, str) or not v.strip() for v in raw['camera'].values())):
        raise ValueError('Mask requires a camera model/serial binding')
    for key in ('width', 'height'):
        if type(raw[key]) is not int or not 4 <= raw[key] <= 16384:
            raise ValueError('Mask requires native integer width/height in [4,16384]')
    polygons = raw['polygons']
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('Mask requires exclusion polygons (ROI: four rectangle vertices)')
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError('Mask polygon requires at least three vertices')
        for point in polygon:
            if (not isinstance(point, list) or len(point) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point) or
                    not 0 <= point[0] <= raw['width'] or not 0 <= point[1] <= raw['height']):
                raise ValueError('Invalid/out-of-frame mask vertex')
        pts = np.asarray(polygon, dtype=np.float32)
        if abs(cv2.contourArea(pts)) < 1 or not cv2.isContourConvex(pts):
            raise ValueError('Use nondegenerate convex polygons; compose concave masks from multiple polygons')
    result = json.loads(json.dumps(raw, sort_keys=True))
    visible = _visible(json.dumps(result, sort_keys=True))
    if not visible.any():
        raise ValueError('Mask excludes the entire image')
    return result


def validate_binding(mask, width=None, height=None, identity=None):
    if mask is None:
        return
    if width is not None and (width, height) != (mask['width'], mask['height']):
        raise ValueError('Occlusion mask/native image resolution mismatch')
    if identity is not None and mask['camera'] != {'model': identity.camera_model, 'serial': identity.serial_number}:
        raise ValueError('Occlusion mask belongs to another camera')


@lru_cache(maxsize=4)
def _visible(serialized):
    mask = json.loads(serialized)
    w, h = mask['width'], mask['height']
    visible = np.ones((h, w), dtype=np.uint8)
    # Pixel centers are x+.5,y+.5; evaluate the declared edge coordinates
    # exactly, rather than rounding a polygon onto a shifted pixel grid.
    for polygon in mask['polygons']:
        pts = np.asarray(polygon, dtype=np.float64)
        area = cv2.contourArea(pts.astype(np.float32), oriented=True)
        x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(pts.max(axis=0)).astype(int), [w, h])
        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = np.ones((y1-y0, x1-x0), dtype=bool)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            cross = (b[0]-a[0])*(yy+.5-a[1]) - (b[1]-a[1])*(xx+.5-a[0])
            inside &= cross >= 0 if area > 0 else cross <= 0
        visible[y0:y1, x0:x1][inside] = 0
    visible.flags.writeable = False
    return visible


def mask_metadata(mask):
    if mask is None:
        return None
    serialized = json.dumps(mask, sort_keys=True)
    return {**mask, 'sha256': hashlib.sha256(serialized.encode()).hexdigest(),
            'evaluation_version': 'observable-area-v1', 'grid': [4, 4],
            'accuracy_validated': False}


def observable_coverage(dataset, camera, mask):
    validate_binding(mask, camera.width, camera.height)
    visible = _visible(json.dumps(mask, sort_keys=True))
    h, w = visible.shape
    counts = np.zeros((4, 4), dtype=int)
    for frame in dataset.enabled_frames:
        if (frame.image_info.width, frame.image_info.height) != (w, h):
            raise ValueError('Mixed image resolutions are incompatible with coverage mask')
        det = frame.detection
        if det is None or not det.success or det.corners is None:
            continue
        for x, y in det.corners.reshape(-1, 2):
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h and visible[int(y), int(x)]:
                counts[min(int(y*4/h), 3), min(int(x*4/w), 3)] += 1
    areas = np.zeros((4, 4), dtype=int)
    yy = np.ceil(np.arange(5)*h/4).astype(int)
    xx = np.ceil(np.arange(5)*w/4).astype(int)
    for r in range(4):
        for c in range(4):
            areas[r,c] = visible[yy[r]:yy[r+1], xx[c]:xx[c+1]].sum()
    total_area, total_corners = int(areas.sum()), int(counts.sum())
    cells=[]
    for r in range(4):
        for c in range(4):
            area, n = int(areas[r,c]), int(counts[r,c])
            expected = total_corners*area/total_area
            score = min(1., n/expected) if expected > 0 else 0.
            sufficient = bool(area and coverage_percentage([CoverageCell(r,c,n,score)]) > 0)
            cells.append({'row':r,'col':c,'visible_pixels':area,
                          'visible_fraction':area/((yy[r+1]-yy[r])*(xx[c+1]-xx[c])),
                          'corner_count':n,'coverage_score':score,'sufficient':sufficient})
    active = [c for c in cells if c['visible_pixels']]
    required = set()
    for r, col in [(0,0),(0,3),(3,0),(3,3)]:
        nearest = min(active, key=lambda c: ((c['row']-r)**2+(c['col']-col)**2,c['row'],c['col']))
        required.add((nearest['row'], nearest['col']))
    required = sorted(required)
    missing = [list(rc) for rc in required if counts[rc] == 0]
    return {'coverage_pct':100*sum(c['visible_pixels'] for c in active if c['sufficient'])/total_area,
            'visible_area_fraction':total_area/(w*h),'visible_corner_count':total_corners,
            'grid':cells,'required_corner_cells':[list(rc) for rc in required],
            'missing_corner_cells':missing,'mask':mask_metadata(mask),
            'accuracy_validated':False}
