"""Read-only history; immutable bundles are authoritative, indexes are caches."""
from __future__ import annotations
import json
from pathlib import Path

from automation.publish.bundle import verify_bundle
from automation.publish.base import PublishError


def list_history(root,serial=None,camera_model=None):
    root=Path(root).resolve();records=[]
    patterns=['cameras/*/*/runs/*/bundle.json','items/*/bundle/bundle.json']
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            meta=verify_bundle(path.parent);camera=meta['camera']
            if serial is not None and camera['serial_number']!=serial:continue
            if camera_model is not None and camera['camera_model']!=camera_model:continue
            publication='UNKNOWN';receipt=None
            if path.parent.name=='bundle':
                try:
                    # Recovery logic also handles a damaged state or interrupted push.
                    from automation.publish.outbox import Outbox
                    from automation.publish.config import PublishConfig
                    state=next(s for s in Outbox(root,PublishConfig()).list() if s['job_id']==path.parent.parent.name)
                    publication=state['status'];receipt=state.get('receipt')
                except (OSError,ValueError):publication='STATE_UNAVAILABLE'
            commit=receipt.get('commit_sha') if receipt else None
            if path.parent.name!='bundle' and (root/'.git').exists():
                import os,subprocess
                try:
                    p=subprocess.run(['git','log','-1','--format=%H','--',path.relative_to(root).as_posix()],cwd=root,
                        env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),capture_output=True,text=True,timeout=10)
                    if p.returncode==0 and len(p.stdout.strip()) in {40,64}:
                        commit=p.stdout.strip();publication='PUBLISHED'
                except (OSError,subprocess.SubprocessError):pass
            records.append({**meta['summary'],'run_id':meta['run_id'],'date':meta['date'],
                            'content_hash':meta['content_hash'],'publication_status':publication,
                            'receipt':receipt,'publication_commit_sha':commit,'run_path':meta['run_path']})
    return sorted(records,key=lambda r:(r['date'],r['run_id']))


def compare_runs(left,right):
    reasons=[]
    if left.get('camera')!=right.get('camera'):reasons.append('CAMERA_IDENTITY_MISMATCH')
    if left.get('resolution')!=right.get('resolution'):reasons.append('RESOLUTION_MISMATCH')
    if left.get('chosen_model')!=right.get('chosen_model'):reasons.append('DISTORTION_MODEL_MISMATCH')
    a,b=left.get('coordinate_system') or {},right.get('coordinate_system') or {}
    if a.get('kind') not in {'full_frame','roi'} or b.get('kind') not in {'full_frame','roi'}:reasons.append('ROI_UNCONFIRMED')
    elif a!=b:reasons.append('ROI_MISMATCH')
    result={'comparable':not reasons,'limitations':reasons,
            'left_verdict':left.get('acceptance_verdict'),'right_verdict':right.get('acceptance_verdict'),
            'interpretation':'Parameter or projection differences are not a quality verdict.'}
    if reasons:return result
    import cv2
    import numpy as np
    try:
        ka,kb=np.asarray(left['K'],float),np.asarray(right['K'],float)
        da,db=np.asarray(left['D'],float).ravel(),np.asarray(right['D'],float).ravel()
        if ka.shape!=(3,3) or kb.shape!=(3,3) or da.shape!=db.shape:raise ValueError()
        if not all(np.isfinite(x).all() for x in [ka,kb,da,db]):raise ValueError()
        if min(ka[0,0],ka[1,1],kb[0,0],kb[1,1])<=0:raise ValueError()
        result['intrinsic_delta']={name:float(y-x) for name,x,y in zip(['fx','fy','cx','cy'],[ka[0,0],ka[1,1],ka[0,2],ka[1,2]],[kb[0,0],kb[1,1],kb[0,2],kb[1,2]])}
        result['distortion_delta']=(db-da).tolist()
        width,height=left['resolution']
        pixels=np.array([(x,y) for y in np.linspace(.1*height,.9*height,7) for x in np.linspace(.1*width,.9*width,9)],float).reshape(-1,1,2)
        fish=left['chosen_model']=='fisheye'
        rays=(cv2.fisheye.undistortPoints(pixels,ka,da) if fish else cv2.undistortPoints(pixels,ka,da)).reshape(-1,2)
        xyz=np.column_stack([rays,np.ones(len(rays))]).reshape(-1,1,3)
        def project(k,d):
            fn=cv2.fisheye.projectPoints if fish else cv2.projectPoints
            return fn(xyz,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
        delta=np.linalg.norm(project(ka,da)-project(kb,db),axis=1)
        if not np.isfinite(delta).all():raise ValueError()
        result['shared_ray_projection_px']={'samples':len(delta),'rms':float(np.sqrt(np.mean(delta**2))),
            'p95':float(np.percentile(delta,95)),'max':float(delta.max()),'reference':'left 10%-90% pixel grid'}
    except (ValueError,TypeError,KeyError,cv2.error):
        return {**result,'comparable':False,'limitations':['INVALID_OR_UNSUPPORTED_CALIBRATION']}
    result['metrics']={'left':left.get('models'),'right':right.get('models')}
    return result
