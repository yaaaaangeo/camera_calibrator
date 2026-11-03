"""Read-only checks by default; dry-run never creates an output/outbox/lock."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re

from automation.publish.config import validate_remote, validate_results_tree
from automation.publish.security import redact


def git_read(*args, timeout=15):
    env={k:v for k,v in os.environ.items() if not k.startswith(('GIT_TRACE','GIT_CONFIG_'))
         and k not in {'GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_CURL_VERBOSE'}}
    env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_OPTIONAL_LOCKS='0',
               GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes')
    return subprocess.run(['git',*args],capture_output=True,text=True,encoding='utf8',errors='replace',
        env=env,timeout=timeout)


def nearest_existing(path):
    p=Path(path)
    while not p.exists() and p!=p.parent:p=p.parent
    return p


def preflight(exp, *, network=False, dry_run=False, verify_write=False, publish_only=False):
    checks=[];topics=[]
    def add(name,status,detail):checks.append({'check':name,'status':status,'detail':redact(detail)})
    add('python','PASS' if sys.version_info>=(3,10) else 'FAIL',sys.version.split()[0])
    identity=exp.identity.to_dict()
    placeholder=any('REPLACE_WITH' in str(v) for v in identity.values())
    add('camera_identity','FAIL' if placeholder else 'WARN','Replace example camera/serial before experiment' if placeholder else 'Operator-declared identity; hardware serial not queried')
    modules=['numpy','scipy','yaml','cv2','PySide6']+(['rosbags'] if exp.source['kind']=='bag' and not publish_only else [])
    for name in modules:
        try:
            m=importlib.import_module(name)
            if name=='cv2' and (not hasattr(m,'aruco') or not hasattr(m.fisheye,'solvePnP')):
                raise RuntimeError('OpenCV contrib/modern fisheye API missing')
            version=getattr(m,'__version__',None)
            if version is None:
                try:version=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:version=None
            minimum={'numpy':(1,24),'scipy':(1,11),'yaml':(6,0),'cv2':(4,10),'PySide6':(6,5,3),'rosbags':(0,9)}[name]
            if version and tuple(int(x) for x in re.findall(r'\d+',version)[:3])<minimum:
                raise RuntimeError('Version below required minimum')
            add('library:'+name,'PASS' if version else 'WARN',version or 'Importable; version unavailable')
        except Exception as exc:add('library:'+name,'FAIL',type(exc).__name__)
    try:
        from automation.profile import build_cli_options,options_to_argv,parse_cli_argv
        options=build_cli_options(exp.live.p1,image_paths=['preflight'],sensor_name=exp.identity.label,
            save_project='unused',json_summary='unused',log_file='unused')
        _,pattern=parse_cli_argv(options_to_argv(options))
        from calibration.detector import build_detect_fn
        build_detect_fn(pattern)  # board construction only; no image detection
        add('pattern','PASS','Pattern configuration constructs successfully')
        add('physical_pattern','WARN','Physical board dimensions/dictionary must be confirmed by operator')
    except Exception as exc:add('pattern','FAIL',str(exc))
    source=Path(exp.source['path'])
    add('sampling','PASS',f"bag-time extract_fps={exp.source.get('extract_fps','all')}; wall-clock replay_interval={exp.source.get('replay_interval')}; P2 capture interval={exp.live.capture.interval_sec}")
    try:
        if publish_only:
            add('input','PASS','Not required for retry-publish; no bag/image read or calibration')
            image=None
        elif exp.source['kind']=='bag':
            if not (source.is_file() or source.is_dir()):raise ValueError('Bag input missing')
            if source.is_file() and source.suffix.lower() not in {'.bag','.db3','.mcap'}:
                raise ValueError('Unsupported bag filename; use ROS1 .bag or ROS2 recording directory')
            from calibration.rosbag_reader import _open_reader
            from calibration.ros_image_codec import decode_image_message
            with _open_reader(str(source)) as reader:
                topics=[{'name':c.topic,'msg_type':c.msgtype,'count':c.msgcount} for c in reader.connections
                    if c.msgtype in {'sensor_msgs/msg/Image','sensor_msgs/msg/CompressedImage'}]
                connections=[c for c in reader.connections if c.topic==exp.source['topic']
                    and c.msgtype in {'sensor_msgs/msg/Image','sensor_msgs/msg/CompressedImage'}]
                if not connections:raise ValueError('Configured image topic not present')
                first=next(reader.messages(connections=connections),None)
                if first is None:raise ValueError('Image topic is empty')
                c,_,raw=first
                image=decode_image_message(reader.deserialize(raw,c.msgtype),c.msgtype)
                if image is None:raise ValueError('First image message cannot be decoded')
                add('bag_format_topic','PASS',f'{connections[0].msgtype}; first image decoded; {sum(c.msgcount for c in connections)} messages')
                add('bag_full_decode','WARN','Only first image decoded; remaining messages verified during replay')
        else:
            from automation.live.sources import ImageReplay
            src=ImageReplay(source);image,_=src.poll();src.stop()
            add('image_input','PASS',f'{len(src.paths)} images; first image decoded')
        expected=exp.live.p1.camera
        if not publish_only and expected.get('width') and (image.shape[1],image.shape[0])!=(expected['width'],expected['height']):
            raise ValueError('Input resolution differs from profile')
        if exp.live.coverage_mask is not None:
            from calibration.occlusion import validate_binding, mask_metadata
            validate_binding(exp.live.coverage_mask, identity=exp.identity)
            if not publish_only:
                validate_binding(exp.live.coverage_mask, image.shape[1], image.shape[0])
            add('coverage_mask','PASS',json.dumps(mask_metadata(exp.live.coverage_mask),ensure_ascii=False))
    except Exception as exc:add('input','FAIL',str(exc))
    root=exp.output
    ancestor=nearest_existing(root)
    try:
        if ancestor.is_file():raise ValueError('Output ancestor is a file')
        if root==source or root.is_relative_to(source) or source.is_relative_to(root/'experiments'):
            raise ValueError('Input and managed output overlap')
        outbox=Path(exp.publisher.outbox)
        if outbox==source or outbox.is_relative_to(source):raise ValueError('Outbox overlaps input path')
        for p in [root,*root.parents,root/'experiments',exp.folder,outbox,*outbox.parents]:
            if p.is_symlink() or (p.exists() and getattr(p.lstat(),'st_file_attributes',0)&0x400):
                raise ValueError('Output link/reparse point is not supported')
        if not os.access(ancestor,os.W_OK):raise PermissionError('Output ancestor not writable')
        if verify_write and not dry_run:
            import tempfile
            with tempfile.TemporaryFile(dir=ancestor) as probe:
                probe.write(b'camera-calibrator-preflight');probe.flush();os.fsync(probe.fileno())
            add('output_permission','PASS','Temporary write/fsync check passed; only owned probe removed')
        else:add('output_permission','WARN','Metadata access check passed; actual write not exercised (read-only preflight)')
        required=exp.raw['output'].get('min_free_gb',5)*1024**3
        free=shutil.disk_usage(ancestor).free
        add('disk_space','PASS' if free>=required else 'FAIL',f'{free/1024**3:.2f} GiB free; required {required/1024**3:.2f} GiB; estimate excludes final capture sizing')
        box_ancestor=nearest_existing(outbox)
        if not box_ancestor.is_dir() or not os.access(box_ancestor,os.W_OK):
            raise PermissionError('Outbox ancestor not writable')
        if verify_write and not dry_run and box_ancestor!=ancestor:
            with tempfile.TemporaryFile(dir=box_ancestor) as probe:
                probe.write(b'outbox-preflight');probe.flush();os.fsync(probe.fileno())
        add('outbox_permission','PASS' if verify_write and not dry_run else 'WARN',
            'Write probe confirmed' if verify_write and not dry_run else 'Metadata check only; no write exercised')
        box_free=shutil.disk_usage(box_ancestor).free
        add('outbox_space','PASS' if box_free>=required else 'FAIL',f'{box_free/1024**3:.2f} GiB free; same minimum budget applies')
        manifest=exp.folder/'experiment.json'
        if exp.folder.exists():
            if not manifest.is_file():raise ValueError('Existing job folder has no manifest; collision, no overwrite')
            state=json.loads(manifest.read_text(encoding='utf8'))
            if state['fingerprint']!=exp.fingerprint:raise ValueError('Job ID belongs to another configuration')
            if publish_only and not state.get('calibration_session'):raise ValueError('No completed calibration to republish')
            add('output_collision','PASS','Existing matching job; completed stages will be reused')
        elif publish_only:raise ValueError('No existing experiment to republish')
        else:add('output_collision','PASS','New job path')
    except Exception as exc:add('output','FAIL',str(exc))
    add('git_cli','PASS' if shutil.which('git') else 'FAIL','Available' if shutil.which('git') else 'Git not installed')
    pub=exp.publisher
    declared=exp.raw.get('publish',{}).get('results_repository',False)
    add('publisher_enabled','PASS' if pub.enabled else 'WARN','Enabled explicitly' if pub.enabled else 'Disabled; local outbox only, no Push')
    if pub.enabled and not declared:add('results_repository','FAIL','Explicit results_repository=true required')
    else:add('results_repository','PASS' if declared else 'WARN','Results-only declaration' if declared else 'Not declared; publication disabled')
    if pub.remote:
        try:
            validate_remote(pub.remote)
            effective=git_read('ls-remote','--get-url',pub.remote)
            if effective.returncode:raise ValueError('Unable to resolve Git remote')
            validate_remote(effective.stdout.strip())
            local='://' not in pub.remote and not pub.remote.startswith('git@')
            if local:
                result=git_read('--git-dir='+pub.remote,'rev-parse','--is-bare-repository')
                if result.returncode or result.stdout.strip()!='true':raise ValueError('Local remote must be an existing bare repository')
                tree=git_read('--git-dir='+pub.remote,'ls-tree','--name-only','HEAD')
                if tree.returncode==0:validate_results_tree(set(tree.stdout.splitlines()))
            add('git_remote','PASS','Remote syntax and destination guards passed; configuration is not ownership proof')
            if network or local:
                probe=git_read('ls-remote',pub.remote,timeout=min(pub.timeout_sec,30))
                add('git_read_access','PASS' if probe.returncode==0 else 'WARN',
                    'Read access confirmed; public access may be anonymous' if probe.returncode==0 else 'Read access failed; credentials/network/host trust not confirmed')
            else:add('git_read_access','WARN','Not probed; use preflight --network for read-only remote probe')
        except Exception as exc:add('git_remote','FAIL',getattr(exc,'code',str(exc)))
    else:add('git_remote','WARN','No remote configured; local outbox only')
    add('git_push_authorization','WARN','UNVERIFIED: no Push performed; read access does not prove write permission, branch policy or authenticated identity')
    if not exp.factory.get('reference'):
        add('factory_reference','WARN','SKIPPED: no Factory reference; Local Intrinsic only, no seeded optimization')
    elif not Path(exp.factory['reference']).is_file():
        add('factory_reference','WARN','SKIPPED: Factory file not found; Local Intrinsic preserved')
    else:
        try:
            from automation.factory.provider import FileProvider
            FileProvider(exp.factory['reference']).lookup(exp.identity.camera_model,exp.identity.serial_number)
            if not Path(exp.factory['coordinates']).is_file():raise ValueError('Session coordinates file missing')
            add('factory_reference','PASS','Reference loads; manufacturer authenticity/physical coordinates remain declarations')
        except Exception as exc:add('factory_reference','WARN','Optional Factory unavailable; Local calculation remains enabled: '+str(exc))
    status='FAIL' if any(c['status']=='FAIL' for c in checks) else 'WARN' if any(c['status']=='WARN' for c in checks) else 'PASS'
    return {'schema_version':1,'status':status,'dry_run':dry_run,'network_probe':network,
        'image_source_settings':{'extract_fps':exp.source.get('extract_fps'),'replay_interval':exp.source.get('replay_interval'),
            'expected_width':exp.live.p1.camera.get('width'),'expected_height':exp.live.p1.camera.get('height'),'resize':False},
        'image_topics':topics,
        'write_probe':bool(verify_write and not dry_run),
        'checks':checks,'plan':['input fingerprint','P2 replay/detection/selection/readiness','P1 four-model calculation/comparison/YAML export',
            'optional P4 comparison','P3 sanitized bundle/outbox','Git Push (enabled only)' if pub.enabled else 'Git Push DISABLED'],
        'job_id':exp.job,'calibration_will_run':False if dry_run or publish_only else 'on explicit run only'}
