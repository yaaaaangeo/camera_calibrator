from __future__ import annotations
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from automation.publish.base import PublishError


@dataclass(frozen=True)
class PublishConfig:
    enabled: bool = False
    remote: str = ''
    branch: str = 'calibration-results'
    outbox: str = ''
    max_attempts: int = 5
    base_backoff_sec: float = 5.
    max_backoff_sec: float = 3600.
    timeout_sec: float = 90.


def validate_results_tree(names):
    allowed={'cameras','README.md','.gitignore','.gitattributes','RESULTS_REPOSITORY.json'}
    # Preserve existing collaborators' plain-text notes; reject source trees,
    # dependency manifests, images and bags before creating any result commit.
    if any(name not in allowed and not (name.endswith('.txt') and not name.lower().startswith(('requirements','cmake'))) for name in names):
        raise PublishError('REMOTE_NOT_RESULTS_REPOSITORY')


def validate_remote(remote):
    from automation.publish.security import TOKEN,secrets
    if TOKEN.search(remote) or any(s in remote for s in secrets()):raise PublishError('CREDENTIAL_IN_URL')
    if not remote or remote.startswith('-') or any(c in remote for c in '\r\n\x00'):
        raise PublishError('INVALID_REMOTE')
    if '://' in remote:
        parsed = urlsplit(remote)
        if parsed.scheme not in {'https','ssh'} or not parsed.hostname or parsed.password or parsed.query or parsed.fragment:
            raise PublishError('INVALID_REMOTE')
        if any(part in {'.','..'} for part in parsed.path.split('/')):raise PublishError('INVALID_REMOTE')
        if parsed.scheme == 'https' and parsed.username: raise PublishError('CREDENTIAL_IN_URL')
        if parsed.scheme == 'ssh' and parsed.username not in {None,'git'}: raise PublishError('CREDENTIAL_IN_URL')
        canonical = (parsed.hostname+'/'+parsed.path.strip('/')).lower().removesuffix('.git')
    elif re.fullmatch(r'git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+',remote):
        canonical = remote[4:].replace(':','/',1).lower().removesuffix('.git')
    else:
        path = Path(remote).expanduser()
        if not path.is_absolute(): raise PublishError('LOCAL_REMOTE_MUST_BE_ABSOLUTE')
        source = Path(__file__).resolve().parents[2]
        if path.resolve() in {source, source/'.git'}: raise PublishError('CODE_REPOSITORY_FORBIDDEN')
        return str(path.resolve())
    if canonical == 'github.com/yaaaaangeo/camera_calibrator': raise PublishError('CODE_REPOSITORY_FORBIDDEN')
    return remote


def resolve_config(raw):
    try:
        if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(PublishConfig)}: raise ValueError()
        cfg=PublishConfig(**raw)
        if not isinstance(cfg.enabled,bool): raise ValueError()
        if not isinstance(cfg.remote,str) or not isinstance(cfg.branch,str) or not isinstance(cfg.outbox,str): raise ValueError()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,127}',cfg.branch) or any(x in cfg.branch for x in ['..','//','@{']) or cfg.branch.endswith(('/','.','.lock')): raise ValueError()
        for value in [cfg.max_attempts,cfg.base_backoff_sec,cfg.max_backoff_sec,cfg.timeout_sec]:
            import math
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0: raise ValueError()
        if not isinstance(cfg.max_attempts,int) or cfg.max_attempts>100: raise ValueError()
        if cfg.remote: validate_remote(cfg.remote)
        if cfg.enabled and not cfg.remote: raise ValueError()
        return cfg
    except (ValueError,TypeError,OSError):
        raise PublishError('INVALID_PUBLISH_CONFIG') from None


def load_config(path=None):
    if path is None: return PublishConfig()
    try:
        return resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))
    except (ValueError,TypeError,OSError):
        raise PublishError('INVALID_PUBLISH_CONFIG') from None
