from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import time

from automation.publish.base import PublishError
from automation.publish.bundle import build_bundle, verify_bundle
from automation.publish.storage import Lock, write_json, confined


class Outbox:
    def __init__(self,root,config,*,clock=time.time):
        self.root=Path(root).expanduser().resolve();self.config=config;self.clock=clock
        self.root.mkdir(parents=True,exist_ok=True)

    def _folder(self,job):
        import re
        if not re.fullmatch('[a-f0-9]{64}',job):raise PublishError('INVALID_JOB_ID')
        path=self.root/'items'/job
        confined(self.root,'items/'+job,must_exist=False)
        confined(self.root,'items/'+job+'/bundle',must_exist=False)
        return path

    def _save(self,folder,state):
        for name in ['state.json','state.backup.json','events.jsonl']:
            confined(self.root,(folder/name).relative_to(self.root).as_posix(),must_exist=False)
        current=folder/'state.json'
        if current.is_file():
            try:write_json(folder/'state.backup.json',json.loads(current.read_text(encoding='utf-8')))
            except ValueError:pass
        state['updated_at']=self.clock();state['revision']=state.get('revision',0)+1
        write_json(current,state)
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(state,sort_keys=True)+'\n');f.flush()

    def _load(self,folder):
        for name in ['state.json','state.backup.json','events.jsonl']:
            confined(self.root,(folder/name).relative_to(self.root).as_posix(),must_exist=False)
        state=None;candidates=[]
        for name in ['state.json','state.backup.json']:
            try:
                candidate=json.loads((folder/name).read_text(encoding='utf-8'))
                if candidate['job_id']==folder.name and candidate['status'] in {'PENDING','PUBLISHING','PUBLISHED','FAILED','RETRY_WAIT'}:
                    candidates.append(candidate)
            except (OSError,ValueError,KeyError,TypeError):pass
        try:
            for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:
                    candidate=json.loads(line)
                    if candidate.get('job_id')==folder.name:candidates.append(candidate)
                except (ValueError,AttributeError):pass
        except (OSError,UnicodeError):pass
        valid=[c for c in candidates if isinstance(c.get('attempts'),int) and c['attempts']>=0
               and isinstance(c.get('revision',0),int) and c.get('status') in {'PENDING','PUBLISHING','PUBLISHED','FAILED','RETRY_WAIT'}]
        if valid:state=max(valid,key=lambda c:c.get('revision',0))
        if state is None:
            verify_bundle(folder/'bundle')
            state={'job_id':folder.name,'status':'FAILED','attempts':0,'retryable':False,
                   'error':'STATE_RECOVERY_REQUIRES_REVIEW','next_attempt_at':None}
        if state['status']=='PUBLISHING':
            state.update(status='RETRY_WAIT',retryable=True,next_attempt_at=0,error='INTERRUPTED_PUBLICATION')
        return state

    def enqueue(self,session,**options):
        with Lock(self.root/'.lock'):
            with tempfile.TemporaryDirectory(prefix='staging-',dir=self.root) as temp:
                stage=Path(temp)/'bundle';meta=build_bundle(session,stage,**options)
                folder=self._folder(meta['content_hash'])
                if folder.exists():
                    verify_bundle(folder/'bundle');return self._load(folder)
                entry=Path(temp)/'entry';entry.mkdir();stage.rename(entry/'bundle')
                state={'job_id':meta['content_hash'],'status':'PENDING','attempts':0,
                       'retryable':True,'next_attempt_at':0,'error':None,'receipt':None,
                       'target':{'remote':self.config.remote,'branch':self.config.branch} if self.config.remote else None}
                self._save(entry,state);folder.parent.mkdir(parents=True,exist_ok=True)
                entry.rename(folder);return state

    def list(self):
        with Lock(self.root/'.lock'):
            return [self._load(p) for p in sorted((self.root/'items').glob('*')) if p.is_dir()]

    def process(self,job,publisher,*,reset_failed=False):
        with Lock(self.root/'.lock'):
            folder=self._folder(job);verify_bundle(folder/'bundle');state=self._load(folder)
            target={'remote':self.config.remote,'branch':self.config.branch}
            if state.get('target') and state['target']!=target:raise PublishError('OUTBOX_TARGET_MISMATCH')
            if state['status']=='PUBLISHED':return state
            if reset_failed and state['status']=='FAILED':state.update(status='PENDING',attempts=0,retryable=True,next_attempt_at=0)
            if not self.config.enabled:return state
            if not state.get('target'):state['target']=target
            if state['status']=='FAILED' or (state.get('next_attempt_at') or 0)>self.clock():return state
            state.update(status='PUBLISHING',attempts=state['attempts']+1,error=None)
            self._save(folder,state)
            try:
                receipt=publisher.publish(folder/'bundle')
                write_json(folder/'receipt.json',receipt.to_dict())
                state.update(status='PUBLISHED',receipt=receipt.to_dict(),retryable=False,next_attempt_at=None)
            except Exception as exc:
                error=exc if isinstance(exc,PublishError) else PublishError('PUBLISHER_INTERNAL_ERROR',False)
                retry=error.retryable and state['attempts']<self.config.max_attempts
                delay=min(self.config.max_backoff_sec,self.config.base_backoff_sec*2**(state['attempts']-1))
                state.update(status='RETRY_WAIT' if retry else 'FAILED',retryable=retry,error=error.code,
                             next_attempt_at=self.clock()+delay if retry else None)
            self._save(folder,state);return state

    def record_error(self,code,details=None):
        import uuid
        with Lock(self.root/'.lock'):
            write_json(self.root/'errors'/(uuid.uuid4().hex+'.json'),
                       {'calibration_status':'ERROR','publication_status':'NOT_QUEUED',
                        'approval_status':'UNREVIEWED','deployment_status':'NOT_DEPLOYED','error':code,'details':details or {}})
