"""파일·폴더 이름 한 칸으로 쓸 수 있게 정리한다 (세 기능이 각자 쓰던 것을 하나로)."""

from __future__ import annotations

import re


def safe_name(text: str, *, fallback: str = "unnamed", ascii_only: bool = False, keep_dot: bool = False,
              strip: str = "_") -> str:
    """경로 한 칸으로 쓸 수 있는 이름.

    기본은 글자(모든 문자 체계)·숫자·``-``·``_`` 만 남기고 나머지는 ``_`` 로 바꾼다.
    ``ascii_only`` 면 영문 글자만, ``keep_dot`` 이면 ``.`` 도 남긴다. 앞뒤의 ``strip``
    글자를 지우고, 남은 게 없으면 ``fallback``.
    """
    allowed = r"A-Za-z0-9_" if ascii_only else r"\w"
    allowed += r"\-" + (r"." if keep_dot else "")
    s = re.sub(rf"[^{allowed}]+", "_", (text or "").strip(), flags=0 if ascii_only else re.UNICODE).strip(strip)
    return s or fallback
