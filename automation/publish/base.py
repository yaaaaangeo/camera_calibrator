from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class PublishError(ValueError):
    """Only safe categorical messages cross the publication boundary."""
    def __init__(self, code: str, retryable: bool = False, details=None):
        super().__init__(code)
        self.code, self.retryable = code, retryable
        self.details = details or {}


@dataclass(frozen=True)
class PublishReceipt:
    commit_sha: str
    remote: str
    branch: str
    run_path: str
    content_hash: str
    duplicate: bool = False

    def to_dict(self):
        return asdict(self)


class ResultPublisher(Protocol):
    def publish(self, bundle: Path) -> PublishReceipt: ...
