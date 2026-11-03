from __future__ import annotations
import json
import os
from pathlib import Path
import uuid

from automation.publish.base import PublishError


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Lock:
    def __init__(self, path): self.path = Path(path)
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b'); self.stream.seek(0)
        self.stream.write(b'0'); self.stream.flush(); self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close(); raise PublishError('OUTBOX_BUSY', True) from None
        return self
    def __exit__(self, *_): self.stream.close()


def confined(root, relative, *, must_exist=True):
    """Reject traversal, symlinks and Windows junction/reparse redirects."""
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(x in {'.', '..'} for x in rel.parts) or ':' in str(rel) or '\\' in str(relative):
        raise PublishError('UNSAFE_PATH')
    path = root
    for component in rel.parts:
        path = path / component
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise PublishError('UNSAFE_LINK')
    if not path.resolve().is_relative_to(root): raise PublishError('UNSAFE_PATH')
    if must_exist and not path.is_file(): raise PublishError('REQUIRED_FILE_MISSING')
    return path
