from __future__ import annotations
import getpass
import hashlib
import os
import re

from automation.publish.base import PublishError

TOKEN = re.compile(r'(?i)(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+|glpat-[A-Za-z0-9_-]+|bearer\s+[A-Za-z0-9._~-]+)')
PATH = re.compile(r'(?<![\w])(?:[A-Za-z]:[\\/][^\s<>"\x27,;]+|\\\\[^\s<>"\x27,;]+|/(?:[^/\s<>"\x27,;]+/)+[^\s<>"\x27,;]*)')
URL = re.compile(r'(?:https?|ssh)://[^\s<>"\x27]+')
EMAIL = re.compile(r'(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}')
PRIVATE_KEYS = re.compile(r'(?i)(?:token|password|secret|authorization|cookie|email|username|credential|api_key|private_key)')
VOLATILE = {'run_id', 'generated_at', 'started_at', 'finished_at', 'created_at', 'timestamp',
            'elapsed_sec', 'seconds', 'session_id', 'session_dir', 'source_path', 'source_root',
            'python_executable', 'path', 'file', 'outputs', 'environment', 'argv', 'command'}


def secrets():
    return [v for k,v in os.environ.items() if re.search(r'(?i)(TOKEN|PASSWORD|SECRET|API_KEY)$',k) and len(v)>=8]


def redact(value):
    value = str(value)
    for secret in secrets(): value = value.replace(secret, '[REDACTED]')
    value = TOKEN.sub('[REDACTED]', value)
    value = URL.sub('[URL_REMOVED]', value)
    value = EMAIL.sub('[ACCOUNT_REMOVED]', value)
    value = PATH.sub('[PATH_REMOVED]', value)
    user = getpass.getuser()
    if len(user) >= 3: value = value.replace(user, '[USER_REMOVED]')
    return value


def clean(value, aliases=None, key=''):
    aliases = aliases or {}
    if isinstance(value, dict):
        return {aliases.get(str(k),redact(k)):clean(v,aliases,str(k)) for k,v in value.items()
                if str(k) not in VOLATILE and not PRIVATE_KEYS.search(str(k))}
    if isinstance(value, list): return [clean(v,aliases,key) for v in value]
    if isinstance(value, str):
        if key.endswith('_ids') or key in {'train_ids','test_ids','image_id','frame_id'}:
            return aliases.get(value, 'frame_' + hashlib.sha256(value.encode()).hexdigest()[:20])
        return redact(value)
    return value


def scan(text):
    if TOKEN.search(text) or PATH.search(text) or URL.search(text) or EMAIL.search(text) or any(x in text for x in secrets()):
        raise PublishError('SENSITIVE_CONTENT')


def component(value):
    from automation.identity import build_identity
    build_identity(value, value)
    # Preserve exact case identity on case-insensitive filesystems too.
    return value + '__' + hashlib.sha256(value.encode()).hexdigest()[:12]
