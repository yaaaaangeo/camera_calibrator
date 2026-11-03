"""Fresh isolated worktree + normal fast-forward push, for GitHub or GitLab."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from automation.publish.base import PublishError, PublishReceipt
from automation.publish.bundle import verify_bundle
from automation.publish.config import validate_remote, validate_results_tree
from automation.publish.storage import confined, write_json


def classify(stderr):
    text=stderr.lower()
    if any(x in text for x in ['authentication failed','permission denied','could not read username','access denied','repository not found','401','403']):
        return PublishError('AUTHENTICATION_FAILED',False)
    if any(x in text for x in ['non-fast-forward','fetch first','rejected','cannot lock ref']):
        return PublishError('REMOTE_CONFLICT',True)
    return PublishError('GIT_TRANSPORT_FAILED',True)


class GitPublisher:
    def __init__(self,config):
        self.config=config

    def _git(self,repo,*args,allowed=(0,)):
        env={k:v for k,v in os.environ.items() if not k.startswith(('GIT_TRACE','GIT_CONFIG_')) and k not in {'GIT_CURL_VERBOSE','GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'}}
        env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_OPTIONAL_LOCKS='0')
        try:
            p=subprocess.run(['git','-c','core.hooksPath='+str(repo.parent/'no-hooks'),
                '-c','core.autocrlf=false','-c','commit.gpgsign=false','-c','tag.gpgsign=false',
                '-c','user.name=Calibration Publisher','-c','user.email=calibration-publisher@localhost',
                *args],cwd=repo,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=self.config.timeout_sec)
        except subprocess.TimeoutExpired:raise PublishError('GIT_TIMEOUT',True) from None
        except OSError:raise PublishError('GIT_UNAVAILABLE',False) from None
        if p.returncode not in allowed:raise classify(p.stderr)
        return p

    def publish(self,bundle):
        if not self.config.enabled:raise PublishError('PUBLICATION_DISABLED')
        meta=verify_bundle(bundle);remote=validate_remote(self.config.remote)
        with tempfile.TemporaryDirectory(prefix='calibration-publish-') as work:
            repo=Path(work)/'repo';repo.mkdir();(Path(work)/'no-hooks').mkdir()
            # Init ONLY this newly-created empty temporary directory, never a
            # caller checkout or the source repository.
            self._git(repo,'init','--quiet')
            effective=self._git(repo,'ls-remote','--get-url',remote).stdout.strip()
            validate_remote(effective)  # Git URL rewrites must not bypass source-repo guards.
            if '://' not in remote and not remote.startswith('git@'):
                p=self._git(repo,'--git-dir='+remote,'rev-parse','--is-bare-repository')
                if p.stdout.strip()!='true':raise PublishError('LOCAL_REMOTE_NOT_BARE')
            self._git(repo,'remote','add','origin',remote)
            ref='refs/heads/'+self.config.branch
            latest=self._git(repo,'ls-remote','--heads','origin',ref).stdout.strip()
            if latest:
                self._git(repo,'fetch','--quiet','--no-tags','origin',ref)
                self._git(repo,'checkout','--quiet','--detach','FETCH_HEAD')
                inspect_ref='HEAD'
            else:
                head=self._git(repo,'ls-remote','origin','HEAD').stdout.strip()
                inspect_ref=None
                if head:
                    self._git(repo,'fetch','--quiet','--no-tags','origin','HEAD')
                    inspect_ref='FETCH_HEAD'
            if inspect_ref:
                names=set(self._git(repo,'ls-tree','--name-only',inspect_ref).stdout.splitlines())
                validate_results_tree(names)
            if latest:
                # A clean results branch must not disguise a code repository
                # whose default branch contains the application source.
                head=self._git(repo,'ls-remote','origin','HEAD').stdout.strip()
                if head and head.split()[0]!=latest.split()[0]:
                    self._git(repo,'fetch','--quiet','--no-tags','origin','HEAD')
                    validate_results_tree(set(self._git(repo,'ls-tree','--name-only','FETCH_HEAD').stdout.splitlines()))
            camera_path='/'.join(meta['run_path'].split('/')[:3])
            camera=confined(repo,camera_path,must_exist=False)
            existing=[]
            if (camera/'runs').exists():
                for path in sorted((camera/'runs').glob('*/bundle.json')):
                    confined(repo,path.relative_to(repo).as_posix())
                    other=verify_bundle(path.parent)
                    if other['camera']!=meta['camera']:raise PublishError('REMOTE_IDENTITY_MISMATCH')
                    if other['content_hash']==meta['content_hash']:
                        commit=self._git(repo,'log','-1','--format=%H','--',other['run_path']+'/bundle.json').stdout.strip()
                        if not commit:raise PublishError('REMOTE_COMMIT_UNAVAILABLE')
                        return PublishReceipt(commit,remote,self.config.branch,other['run_path'],meta['content_hash'],True)
                    if other['run_id']==meta['run_id']:raise PublishError('RUN_ID_COLLISION')
                    existing.append(other)
            destination=confined(repo,meta['run_path'],must_exist=False)
            if destination.exists():raise PublishError('RUN_PATH_COLLISION')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(bundle,destination)
            entries=[]
            for record in existing+[meta]:
                entries.append({'run_id':record['run_id'],'date':record['date'],'content_hash':record['content_hash'],
                    'path':record['run_path'],'code_commit_sha':record['summary'].get('code_commit_sha'),
                    'acceptance_verdict':record['summary']['acceptance_verdict'],
                    'chosen_model':record['summary']['chosen_model']})
            # Rebuild from immutable run records, never trust an existing index.
            index=confined(repo,camera_path+'/index.json',must_exist=False)
            write_json(index,{'schema_version':1,'camera':meta['camera'],'runs':sorted(entries,key=lambda e:(e['date'],e['run_id']))})
            self._git(repo,'add','--',meta['run_path'],camera_path+'/index.json')
            self._git(repo,'commit','--quiet','-m','Calibration result '+meta['content_hash'])
            commit=self._git(repo,'rev-parse','HEAD').stdout.strip()
            self._git(repo,'push','--porcelain','origin','HEAD:'+ref)
            # Success means the commit is confirmed in the remote branch.
            # Network loss here remains retryable; next attempt deduplicates.
            self._git(repo,'fetch','--quiet','--no-tags','origin',ref)
            check=self._git(repo,'merge-base','--is-ancestor',commit,'FETCH_HEAD',allowed=(0,1))
            if check.returncode:raise PublishError('REMOTE_CONFIRMATION_FAILED',True)
            return PublishReceipt(commit,remote,self.config.branch,meta['run_path'],meta['content_hash'])
