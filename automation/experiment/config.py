from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import yaml

from automation.identity import build_identity
from automation.live.config import resolve_live_profile
from automation.publish.config import resolve_config
from automation.run_manifest import canonical_sha256


@dataclass
class Experiment:
    raw: dict
    identity: object
    live: object
    publisher: object
    source: dict
    output: Path
    job: str
    factory: dict

    @property
    def folder(self):
        return self.output/'experiments'/self.job

    @property
    def fingerprint(self):
        # Publication can be enabled/retried without recalibrating immutable inputs.
        return canonical_sha256({'identity':self.identity.to_dict(),'live':self.live.fingerprint,
            'source':self.source,'output':str(self.output)})


def load_experiment(path):
    path=Path(path).expanduser().resolve()
    raw=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw,dict) or raw.get('experiment_version')!=1:
        raise ValueError('experiment_version must be 1')
    if set(raw)-{'experiment_version','job_id','camera','input','live','output','publish','factory'}:
        raise ValueError('Unknown experiment section')
    def absolute(value):
        p=Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent/p).resolve())
    job=raw['job_id']
    if not isinstance(job,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',job):
        raise ValueError('Invalid job_id')
    identity=build_identity(raw['camera']['model'],raw['camera']['serial'])
    source=dict(raw['input'])
    if set(source)-{'kind','path','topic','replay_interval','extract_fps'} or source.get('kind') not in {'bag','images'}:
        raise ValueError('input.kind must be bag or images')
    source['path']=absolute(source['path'])
    if source.get('kind')=='bag' and Path(source['path']).suffix.lower()=='.db3':
        # ROS2 SQLite recordings require their metadata. Bind the whole recording
        # directory so the execution fingerprint covers every referenced segment.
        bag_file=Path(source['path']);metadata=bag_file.parent/'metadata.yaml'
        if not bag_file.is_file() or not metadata.is_file():
            raise ValueError('ROS2 .db3 requires its original metadata.yaml in the same directory')
        info=yaml.safe_load(metadata.read_text(encoding='utf8')).get('rosbag2_bagfile_information',{})
        if bag_file.name not in info.get('relative_file_paths',[]):
            raise ValueError('The .db3 is not referenced by its recording metadata')
        source['path']=str(bag_file.parent)
    if source['kind']=='bag' and (not isinstance(source.get('topic'),str) or not source['topic'].startswith('/')):
        raise ValueError('An explicit absolute image topic is required')
    interval=source.get('replay_interval',1.1)
    if isinstance(interval,bool) or not isinstance(interval,(int,float)) or not 0<interval<=60:
        raise ValueError('replay_interval must be >0 and <=60')
    source['replay_interval']=interval
    if 'extract_fps' in source:
        import math
        fps=source['extract_fps']
        if source['kind']!='bag' or isinstance(fps,bool) or not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:
            raise ValueError('extract_fps requires a bag and a finite positive FPS')
    live=resolve_live_profile(raw.get('live',{'live_profile_version':1}))
    output=raw['output']
    if set(output)-{'root','min_free_gb'}:
        raise ValueError('Unknown output setting')
    free=output.get('min_free_gb',5.)
    if isinstance(free,bool) or not isinstance(free,(int,float)) or not 0<=free<1e6:
        raise ValueError('Invalid min_free_gb')
    pub=dict(raw.get('publish',{}))
    if set(pub)-{'results_repository','full_frame','config','retry_in_process'}:
        raise ValueError('Unknown publish setting')
    for key in ('results_repository','full_frame','retry_in_process'):
        if key in pub and type(pub[key]) is not bool:raise ValueError('Publish declarations must be boolean')
    config=dict(pub.get('config',{}))
    if config.get('outbox'):config['outbox']=absolute(config['outbox'])
    else:config['outbox']=str(Path(absolute(output['root']))/'outbox')
    if config.get('remote') and '://' not in config['remote'] and not config['remote'].startswith('git@'):
        config['remote']=absolute(config['remote'])
    publisher=resolve_config(config)
    factory=dict(raw.get('factory',{}))
    if set(factory)-{'reference','coordinates','seeded'}:raise ValueError('Unknown factory setting')
    if 'seeded' in factory and type(factory['seeded']) is not bool:raise ValueError('factory.seeded must be boolean')
    for key in ('reference','coordinates'):
        if factory.get(key):factory[key]=absolute(factory[key])
    return Experiment(raw,identity,live,publisher,source,Path(absolute(output['root'])),job,factory)
