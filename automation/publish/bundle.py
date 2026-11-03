"""Allowlisted, privacy-reviewed derivative of an existing Output Session."""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import re

from automation.identity import build_identity
from automation.publish.base import PublishError
from automation.publish.security import clean, component, redact, scan
from automation.publish.storage import confined, write_json

MAX_FILE = 10 * 1024 * 1024
MAX_TOTAL = 25 * 1024 * 1024
MAX_FILES = 32
MODELS = {'pinhole','brown_conrady','rational','fisheye'}


def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def read_file(root, name):
    try:path=confined(root,name)
    except PublishError as exc:
        if exc.code=='REQUIRED_FILE_MISSING':raise PublishError(exc.code,details={'missing':[name]}) from None
        raise
    if path.stat().st_size>MAX_FILE: raise PublishError('FILE_TOO_LARGE')
    return path.read_bytes()


def read_json(root,name):
    try: return json.loads(read_file(root,name).decode('utf-8'))
    except PublishError:raise
    except (ValueError,UnicodeError): raise PublishError('INVALID_RESULT_JSON') from None


def resolve_session(session):
    root=Path(session).expanduser().resolve()
    if (root/'p2_manifest.json').is_file():
        data=read_json(root,'p2_manifest.json')
        attempts=data.get('attempts') or []
        if not attempts or data.get('state')!='COMPLETED': raise PublishError('P2_NOT_COMPLETED')
        child=Path(attempts[-1]['outcome']['session_dir']).resolve()
        if not child.is_relative_to(root/'runs'): raise PublishError('UNSAFE_P2_SESSION')
        # Validate every ancestor for redirecting links.
        confined(root,child.relative_to(root).as_posix()+'/run_manifest.json')
        return child
    return root


def build_bundle(session, staging, *, full_frame=False, roi=None, factory_report=None):
    root=resolve_session(session); staging=Path(staging)
    if staging.exists(): raise PublishError('STAGING_ALREADY_EXISTS')
    manifest=read_json(root,'run_manifest.json')
    if manifest.get('status')!='completed': raise PublishError('CALIBRATION_NOT_COMPLETED')
    run_id=manifest.get('run_id','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}',run_id): raise PublishError('INVALID_RUN_ID')
    camera=manifest.get('camera') or {}
    try:identity=build_identity(camera.get('camera_model'),camera.get('serial_number')).to_dict()
    except ValueError:raise PublishError('CAMERA_IDENTITY_MISSING_OR_INVALID') from None
    if identity['serial_number'].upper() in {'UNKNOWN','UNSET','NONE','NULL','NA','N-A','CHECK-ONLY'}:
        raise PublishError('SERIAL_UNCONFIRMED')
    # A malicious output entry must never redirect collection outside a session.
    for entry in manifest.get('outputs',[]):
        if not isinstance(entry,dict) or not isinstance(entry.get('path'),str): raise PublishError('INVALID_OUTPUT_ENTRY')
        confined(root,entry['path'],must_exist=False)
    required=['reports/calibration.json','reports/report.html','automation/acceptance.json','intrinsic/camera_info.yaml']
    missing=[name for name in required if not confined(root,name,must_exist=False).is_file()]
    if not any((root/f'intrinsic/camera_{m}.yaml').is_file() for m in MODELS):missing.append('intrinsic/camera_<model>.yaml')
    if missing:raise PublishError('REQUIRED_FILES_MISSING',details={'missing':missing})
    source={}
    def take(name):
        data=read_file(root,name); source[name]=digest(data);return data
    take('run_manifest.json')
    try:
        calibration=json.loads(take('reports/calibration.json'))
        p1=json.loads(take('automation/acceptance.json'))
    except (ValueError,UnicodeError):raise PublishError('INVALID_RESULT_JSON') from None
    take('reports/report.html')
    run_summary=json.loads(take('automation/run_summary.json')) if (root/'automation/run_summary.json').exists() else {}
    p2=None
    if (root/'automation/p2_acceptance.json').exists():
        p2=json.loads(take('automation/p2_acceptance.json'))
        acceptance=p2.get('acceptance') or {}
        if p2.get('verdict')!=acceptance.get('verdict'): raise PublishError('INCONSISTENT_ACCEPTANCE')
    else: acceptance=p1
    verdict=acceptance.get('verdict')
    if verdict not in {'PASS','WARN','FAIL'}: raise PublishError('INVALID_ACCEPTANCE')
    fingerprint=(manifest.get('inputs') or {}).get('fingerprint') or {}
    if not re.fullmatch(r'[a-f0-9]{64}',fingerprint.get('aggregate_sha256','')): raise PublishError('INPUT_FINGERPRINT_MISSING')
    image_hashes=[];aliases={}
    for image in fingerprint.get('files',[]):
        sha=image.get('sha256','')
        if not re.fullmatch(r'[a-f0-9]{64}',sha): raise PublishError('INVALID_INPUT_FINGERPRINT')
        image_hashes.append(sha)
        name=image.get('name') or Path(image.get('path','')).name
        aliases[name]='frame_'+sha[:20];aliases[Path(name).stem]='frame_'+sha[:20]
    input_semantic=sorted(image_hashes) if image_hashes else fingerprint['aggregate_sha256']
    # No source paths, executable names, argv, raw environment or original HTML.
    safe_manifest=clean({k:manifest[k] for k in ['schema_version','status','camera','code_version',
        'dataset','split','results','acceptance'] if k in manifest},aliases)
    code=manifest.get('code_version') or {}
    safe_manifest['code_version']={k:code.get(k) for k in ['available','commit_sha','branch','dirty']}
    safe_manifest['camera']={**identity,'image_width':camera.get('image_width'),'image_height':camera.get('image_height')}
    safe_manifest['inputs']={'original_aggregate_sha256':fingerprint['aggregate_sha256'],
                            'count':fingerprint.get('count'), 'image_sha256':sorted(image_hashes)}
    safe_calibration=clean(calibration,aliases)
    safe_acceptance=clean(acceptance,aliases)
    safe_p1=clean(p1,aliases)
    results=clean(manifest.get('results') or {},aliases)
    recommendation=results.get('recommendation') or {}
    chosen=recommendation.get('chosen_model')
    chosen_result=(results.get('models') or {}).get(chosen) or {}
    if not chosen or chosen not in (results.get('models') or {}): raise PublishError('CHOSEN_RESULT_MISSING')
    import numpy as np
    try:
        k=np.asarray(chosen_result['camera_matrix'],float);d=np.asarray(chosen_result['distortion'],float)
        if not chosen_result.get('success') or k.shape!=(3,3) or not d.size or not np.isfinite(k).all() or not np.isfinite(d).all():raise ValueError()
    except (ValueError,TypeError,KeyError):raise PublishError('INVALID_CHOSEN_CALIBRATION') from None
    width,height=camera.get('image_width'),camera.get('image_height')
    if not isinstance(width,int) or not isinstance(height,int) or min(width,height)<=0: raise PublishError('IMAGE_SIZE_MISSING')
    if full_frame and roi is not None: raise PublishError('INCONSISTENT_ROI')
    if roi is not None and (len(roi)!=4 or any(not isinstance(x,int) or isinstance(x,bool) for x in roi)
                            or min(roi[:2])<0 or min(roi[2:])<=0 or list(roi[2:])!=[width,height]):
        raise PublishError('INVALID_ROI')
    coordinate={'kind':'full_frame' if full_frame else 'roi' if roi is not None else 'unknown',
                'roi':list(roi) if roi is not None else None,'source':'user_declared' if full_frame or roi is not None else 'unconfirmed'}
    yaml_files={};yaml_semantics={}
    from calibration.calibration_io import load_standard_calibration
    for name in [f'intrinsic/camera_{m}.yaml' for m in sorted(MODELS) if (root/f'intrinsic/camera_{m}.yaml').exists()]+['intrinsic/camera_info.yaml']:
        raw=take(name)
        try:
            cal=load_standard_calibration(str(confined(root,name)))
            values={'K':cal.camera_matrix.tolist(),'D':cal.distortion.ravel().tolist(),
                    'width':cal.width,'height':cal.height,'model':cal.model_name.value if cal.model_name else cal.distortion_model}
            text=raw.decode('utf-8')
        except Exception:raise PublishError('INVALID_INTRINSIC_YAML') from None
        text='\n'.join(redact(line) for line in text.splitlines() if not line.lstrip().startswith('#'))+'\n'
        if re.search(r'(?i)(?:token|password|secret|authorization|api_key)\s*:',text): raise PublishError('SENSITIVE_YAML_FIELD')
        if '!!python' in text: raise PublishError('UNSAFE_YAML')
        scan(text);yaml_files[name]=text;yaml_semantics[name]=values
    if len(yaml_files)<2:raise PublishError('INTRINSIC_YAML_MISSING')
    for values in yaml_semantics.values():
        if values['width']!=width or values['height']!=height:raise PublishError('EXPORT_RESOLUTION_MISMATCH')
    selected_yaml='intrinsic/camera_'+('rational' if chosen=='extended_pinhole' else chosen)+'.yaml'
    if selected_yaml not in yaml_semantics:raise PublishError('CHOSEN_INTRINSIC_MISSING')
    for name in [selected_yaml,'intrinsic/camera_info.yaml']:
        values=yaml_semantics[name]
        if not np.allclose(values['K'],k,rtol=0,atol=1e-6) or np.asarray(values['D']).shape!=d.ravel().shape or not np.allclose(values['D'],d.ravel(),rtol=0,atol=1e-6):
            raise PublishError('EXPORT_PARAMETER_MISMATCH')
    summary={'schema_version':1,'camera':identity,'identity_verified':False,
        'calibration_status':'completed','acceptance_verdict':verdict,
        'publication_status':'PACKAGED','approval_status':'UNREVIEWED','deployment_status':'NOT_DEPLOYED',
        'chosen_model':chosen,'recommended_model':recommendation.get('recommended_model'),
        'recommendation':recommendation,'models':results.get('models'),
        'resolution':[width,height],'coordinate_system':coordinate,
        'K':chosen_result.get('camera_matrix'),'D':chosen_result.get('distortion'),
        'metrics':clean(run_summary.get('metrics') or chosen_result,aliases),
        'stability_pct':(run_summary.get('metrics') or {}).get('stability_pct'),
        'code_commit_sha':code.get('commit_sha'),'code_dirty':code.get('dirty')}
    semantic={'hash_version':1,'identity':identity,'inputs':input_semantic,
              'manifest':{k:v for k,v in safe_manifest.items() if k not in {'inputs'}},
              'calibration':safe_calibration,'acceptance':safe_acceptance,'p1_acceptance':safe_p1,
              'intrinsic':yaml_semantics,'coordinate_system':coordinate}
    safe_factory = None
    if factory_report is not None:
        comparison_path = Path(factory_report)
        if comparison_path.stat().st_size > MAX_FILE:
            raise PublishError('FILE_TOO_LARGE')
        try:
            comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            raise PublishError('INVALID_FACTORY_COMPARISON') from None
        if (comparison.get('schema_version') != 1 or comparison.get('status') != 'COMPLETED'
                or comparison.get('session_manifest_sha256') != digest((root/'run_manifest.json').read_bytes())
                or comparison.get('project_sha256') != digest((root/'project/calibration.ccproj').read_bytes())
                or any(comparison.get('camera', {}).get(key) != identity[key] for key in ('camera_model','serial_number'))):
            raise PublishError('FACTORY_SESSION_MISMATCH')
        safe_factory = clean(comparison, aliases)
        # Bind raw artifacts above, but exclude timestamps/RunID-dependent source
        # hashes from semantic deduplication, just like the existing P3 manifest.
        for key in ('session_manifest_sha256','project_sha256','factory_file_sha256'):
            safe_factory.pop(key, None)
        source['factory_comparison.json'] = digest(comparison_path.read_bytes())
        semantic['factory_comparison'] = safe_factory
        summary['factory_comparison'] = {'conclusion': comparison.get('conclusion'), 'authenticated': False}
    content_hash=digest(canonical(semantic))
    date=str(manifest.get('started_at',''))[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date): raise PublishError('RUN_DATE_MISSING')
    run_path=f"cameras/{component(identity['camera_model'])}/{component(identity['serial_number'])}/runs/{date}_{run_id}"
    # Rebuild a safe report from structured results; never carry scripts, links,
    # embedded images, arbitrary prose or local paths from the original HTML.
    report='<!doctype html><meta charset="utf-8"><title>Calibration result</title><h1>Sanitized calibration report</h1><p>Publication is not approval or deployment.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    staging.mkdir(parents=True)
    files={'run_manifest.json':{**safe_manifest,'run_id':run_id,'started_at':manifest.get('started_at'),'finished_at':manifest.get('finished_at')},
           'calibration.json':safe_calibration,'acceptance.json':safe_acceptance,'p1_acceptance.json':safe_p1,
           'summary.json':summary,'semantic.json':semantic}
    if p2 is not None:files['p2_acceptance.json']=clean(p2,aliases)
    if safe_factory is not None:
        files['factory_comparison.json'] = safe_factory
    for name,value in files.items():write_json(staging/name,value)
    (staging/'report.html').write_bytes(report.encode('utf-8'))
    for name,text in yaml_files.items():
        p=staging/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(text.replace('\r\n','\n').encode('utf-8'))
    hashes={p.relative_to(staging).as_posix():digest(p.read_bytes()) for p in staging.rglob('*') if p.is_file()}
    meta={'schema_version':1,'hash_version':1,'run_id':run_id,'date':date,'run_path':run_path,
          'content_hash':content_hash,'camera':identity,'files':hashes,'source_file_sha256':source,
          'summary':summary,'sanitization':'structured JSON redaction; HTML regenerated without source assets'}
    write_json(staging/'bundle.json',meta)
    verify_bundle(staging)
    return meta


def verify_bundle(root):
    root=Path(root)
    if root.is_symlink() or (root.exists() and getattr(root.lstat(),'st_file_attributes',0)&0x400):raise PublishError('UNSAFE_LINK')
    root=root.resolve();meta=read_json(root,'bundle.json')
    if meta.get('schema_version')!=1 or meta.get('hash_version')!=1:raise PublishError('BUNDLE_VERSION')
    identity=build_identity(meta['camera']['camera_model'],meta['camera']['serial_number']).to_dict()
    expected=f"cameras/{component(identity['camera_model'])}/{component(identity['serial_number'])}/runs/{meta['date']}_{meta['run_id']}"
    if expected!=meta.get('run_path'):raise PublishError('BUNDLE_PATH_MISMATCH')
    confined(root,meta['run_path'],must_exist=False)
    expected_files=set(meta.get('files') or {})
    required={'run_manifest.json','calibration.json','acceptance.json','summary.json','semantic.json','report.html','p1_acceptance.json','intrinsic/camera_info.yaml'}
    allowed=required|{'p2_acceptance.json','factory_comparison.json'}|{f'intrinsic/camera_{m}.yaml' for m in MODELS}
    if not expected_files<=allowed:raise PublishError('UNEXPECTED_BUNDLE_FILE')
    if not required<=expected_files or not any(re.fullmatch(r'intrinsic/camera_(?:pinhole|brown_conrady|rational|fisheye)\.yaml',n) for n in expected_files): raise PublishError('BUNDLE_INCOMPLETE')
    entries=list(root.rglob('*'))
    for entry in entries:
        confined(root,entry.relative_to(root).as_posix(),must_exist=False)
        if entry.is_dir() and entry.relative_to(root).as_posix()!='intrinsic':raise PublishError('UNEXPECTED_BUNDLE_DIRECTORY')
    actual={p.relative_to(root).as_posix() for p in entries if p.is_file()}
    if actual!=expected_files|{'bundle.json'} or len(actual)>MAX_FILES:raise PublishError('UNEXPECTED_BUNDLE_FILE')
    total=0
    for name in actual:
        if not (name.endswith(('.json','.yaml','.html'))):raise PublishError('FILE_TYPE_FORBIDDEN')
        raw=read_file(root,name);total+=len(raw)
        # Windows checkouts can convert Git's LF text blobs to CRLF. Only this
        # lossless line-ending normalization is allowed; semantic edits fail.
        if name!='bundle.json' and digest(raw)!=meta['files'][name] and digest(raw.replace(b'\r\n',b'\n'))!=meta['files'][name]:
            raise PublishError('BUNDLE_HASH_MISMATCH')
        try:scan(raw.decode('utf-8'))
        except UnicodeError:raise PublishError('BINARY_FILE_FORBIDDEN') from None
    if total>MAX_TOTAL:raise PublishError('BUNDLE_TOO_LARGE')
    semantic=read_json(root,'semantic.json')
    if ('factory_comparison.json' in expected_files) != ('factory_comparison' in semantic):
        raise PublishError('SEMANTIC_FILE_MISMATCH')
    if 'factory_comparison' in semantic and semantic['factory_comparison'] != read_json(root,'factory_comparison.json'):
        raise PublishError('SEMANTIC_FILE_MISMATCH')
    if digest(canonical(semantic))!=meta.get('content_hash') or semantic.get('identity')!=identity:
        raise PublishError('CONTENT_HASH_MISMATCH')
    if read_json(root,'summary.json')!=meta.get('summary'):raise PublishError('SUMMARY_MISMATCH')
    for key,name in [('calibration','calibration.json'),('acceptance','acceptance.json'),('p1_acceptance','p1_acceptance.json')]:
        if semantic.get(key)!=read_json(root,name):raise PublishError('SEMANTIC_FILE_MISMATCH')
    return meta
