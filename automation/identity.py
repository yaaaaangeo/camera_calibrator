"""Camera identity attached to every automated run.

1차 버전에서는 Serial Number와 카메라 모델명을 사람이 직접 입력한다. 이후
제조사 조회(SENSING sgMIX 등)나 EEPROM 연동이 추가되면 ``source`` 값만 다른
CameraIdentity를 만들어 같은 경로로 넘기면 된다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

# Serial/모델명은 세션 폴더 이름과 향후 결과 저장소 경로에 들어가므로 경로
# 구분자나 공백이 섞이면 안 된다. 제조사 S/N 형식이 확정되지 않았으므로
# 영숫자와 일반적인 구분 기호만 허용하고 나머지는 거부한다(조용히 치환하지
# 않는다 - 서로 다른 S/N이 같은 문자열로 합쳐지는 것을 막기 위해).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class IdentityError(ValueError):
    """Serial/모델명 형식 오류."""


@dataclass(frozen=True)
class CameraIdentity:
    camera_model: str
    serial_number: str
    source: str = "manual"

    @property
    def label(self) -> str:
        """세션 폴더/리포트 제목용 라벨 (기존 CLI의 --sensor-name으로 전달)."""
        return f"{self.camera_model}_{self.serial_number}"

    def to_dict(self) -> dict:
        return asdict(self)


def build_identity(camera_model: str | None, serial_number: str | None, source: str = "manual") -> CameraIdentity:
    problems = []
    model = (camera_model or "").strip()
    serial = (serial_number or "").strip()
    if not model:
        problems.append("카메라 모델명이 없습니다 (--camera-model 또는 profile camera.model).")
    elif not _IDENTIFIER_RE.match(model):
        problems.append(
            f"카메라 모델명 형식이 올바르지 않습니다: {model!r} "
            "(영문/숫자로 시작, 영문/숫자/._- 만 허용, 최대 64자)."
        )
    if not serial:
        problems.append("Serial Number가 없습니다 (--serial).")
    elif not _IDENTIFIER_RE.match(serial):
        problems.append(
            f"Serial Number 형식이 올바르지 않습니다: {serial!r} "
            "(영문/숫자로 시작, 영문/숫자/._- 만 허용, 최대 64자)."
        )
    if problems:
        raise IdentityError("\n".join(problems))
    return CameraIdentity(camera_model=model, serial_number=serial, source=source)
