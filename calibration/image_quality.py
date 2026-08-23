"""
camera_calibrator.calibration.image_quality
==============================================

설계 문서 3-1번 - 이미지 품질 검사.

detector.py가 이미 sharpness(Laplacian 분산)/brightness(평균 밝기)를
ImageInfo에 채워주고 있었지만, "절대적으로 이상한 이미지"(새까만 사진,
극단적으로 흔들린 사진, 완전히 똑같은 사진 두 장 등)를 걸러내는 절대 기준
검사는 없었다. 이 모듈이 그 역할을 한다.

설계 원칙 - 이 모듈은 frame_quality.py와 역할이 다르다:
    - frame_quality.py: 데이터셋 "안에서 상대적으로" 더 나은/나쁜 프레임에
      점수를 매긴다 (문서 6번, min-max 정규화 - 절대 기준 하드코딩 회피).
    - image_quality.py(이 모듈): "이 사진 한 장만 놓고 봐도 명백히 문제가
      있는가"를 절대 기준으로 잡아낸다 (문서 3-1번 - 새까만 사진은 다른
      사진과 비교할 필요도 없이 그 자체로 문제다).
    두 모듈은 서로 대체하지 않고 보완한다 - 여기서 걸러지지 않은 "미묘하게
    덜 좋은" 사진들의 순위를 frame_quality.py가 매긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from calibration.types import Dataset, Frame, ImageInfo

# ---------------------------------------------------------------------------
# 절대 기준값들 (전부 "명백한 문제"만 잡도록 보수적으로 설정 - 애매한 경우는
# frame_quality.py의 상대 점수에 맡긴다)
# ---------------------------------------------------------------------------

MIN_RESOLUTION_WIDTH = 640
MIN_RESOLUTION_HEIGHT = 480

_BRIGHTNESS_TOO_DARK = 25.0     # 0~255 스케일, 평균 밝기가 이보다 낮으면 "너무 어두움"
_BRIGHTNESS_TOO_BRIGHT = 230.0  # 평균 밝기가 이보다 높으면 "너무 밝음"

_SATURATION_CLIP_WARN = 0.15    # 픽셀의 15% 이상이 완전히 검거나(<=5) 흰(>=250) 경우 경고
_CLIP_LOW_THRESHOLD = 5
_CLIP_HIGH_THRESHOLD = 250

_SHARPNESS_ABSOLUTE_FLOOR = 15.0  # Laplacian 분산이 이 밑이면 "명백히 초점이 안 맞음"
_MOTION_BLUR_RATIO_WARN = 3.0     # 가로/세로 Sobel 분산 비율이 이 이상이면 방향성 블러 의심

_CONTRAST_TOO_LOW = 15.0  # grayscale 표준편차가 이 밑이면 "밋밋한/뿌연" 이미지

_PHASH_EXACT_DUPLICATE_DISTANCE = 4     # 256비트 해시 기준 - 이 이하 해밍 거리 = 사실상 동일 이미지
_PHASH_NEAR_DUPLICATE_DISTANCE = 14     # 이 이하 = 거의 동일(연속 촬영 등)


class ImageQualitySeverity(str, Enum):
    ERROR = "error"      # 캘리브레이션에 쓰기엔 명백히 부적합 (예: 해상도 미달)
    WARNING = "warning"  # 문제 가능성 - 사람이 확인 권장


@dataclass
class ImageQualityIssue:
    code: str
    severity: ImageQualitySeverity
    message: str


@dataclass
class ImageQualityReport:
    image_id: str
    issues: list[ImageQualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ImageQualitySeverity.ERROR for i in self.issues)


# ---------------------------------------------------------------------------
# 개별 지표 계산 (detector.py의 detect_image_file()이 이 함수들을 호출해서
# ImageInfo를 채운다 - 이 모듈은 "계산"과 "판정(threshold 비교)"을 분리해서,
# 계산 결과는 ImageInfo에 저장해 재사용하고 판정만 여기서 한다)
# ---------------------------------------------------------------------------

def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def compute_saturation(gray: np.ndarray) -> float:
    """명부/암부 clipping 픽셀 비율 (0~1). 사진 용어의 "채도"가 아니라
    "노출이 한쪽 끝에 몰려 정보가 날아간 정도"를 뜻한다 - 설계 문서 3-1번
    "saturation 검사" 항목이 노출 클리핑을 의도한 것으로 해석했다(평균 밝기
    기반의 "너무 어두운/밝은" 검사와는 별개로, 국소적으로 날아간 영역이
    있는지를 본다 - 평균은 정상인데 하이라이트만 날아간 경우를 잡아낸다).
    """
    total = gray.size
    if total == 0:
        return 0.0
    clipped = int(np.count_nonzero(gray <= _CLIP_LOW_THRESHOLD)) + int(
        np.count_nonzero(gray >= _CLIP_HIGH_THRESHOLD)
    )
    return float(clipped / total)


def compute_motion_blur_score(gray: np.ndarray) -> float:
    """방향성(모션) 블러 의심도. 등방성(out-of-focus) 블러는 가로/세로 방향
    그래디언트가 비슷하게 줄어들지만, 모션 블러는 흔들린 방향으로만 그래디언트가
    크게 줄어드는 경향이 있다 - 그 비대칭성을 Sobel 그래디언트 분산의
    가로/세로 비율로 근사한다.

    한계(정직하게 명시): 이건 완벽한 모션 블러 검출기가 아니다 - 보드 패턴
    자체가 방향성을 가지므로(체스보드 격자는 가로/세로 대칭이라 비교적
    안전하지만) 장면 구도에 따라 오탐이 날 수 있다. 그래서 절대 기준
    reject가 아니라 WARNING으로만 쓰고, 최종 판단은 사람에게 맡긴다.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x, var_y = float(gx.var()), float(gy.var())
    if max(var_x, var_y) < 1e-6:
        return 1.0  # 그래디언트가 사실상 없음(완전 평면 이미지) - 등방성으로 취급
    if min(var_x, var_y) < 1e-6:
        return 1000.0  # 한쪽 방향 그래디언트가 사실상 0 - 완전히 한쪽으로만 몰린 극단적 방향성
    return float(max(var_x, var_y) / min(var_x, var_y))


def compute_phash(gray: np.ndarray, hash_size: int = 16) -> str:
    """단순 average-hash(aHash). 외부 라이브러리 의존 없이 cv2/numpy만으로
    구현한다 - 정확한 지각적 해시(DCT 기반 pHash)보다는 거칠지만, "완전히
    같은 사진" 또는 "거의 같은 사진(연속 촬영)"을 잡아내는 데는 충분하다.

    hash_size=16(256비트)을 기본값으로 쓴다 - 8(64비트)로 실측했을 때, 보드가
    화면의 일부(예: 5% 미만)만 차지하고 나머지가 균일한 배경(캘리브레이션
    보드 촬영에서 흔한 구도)인 이미지들은 평균 임계값 기준 비트 패턴이 거의
    똑같아져서 서로 다른 자세인데도 해밍 거리가 비정상적으로 가깝게 나오는
    문제가 실측으로 확인됐다 - 해상도를 높이면 이 오탐이 크게 줄어든다.

    한계(정직하게 명시): 그래도 보드가 화면 대부분을 균일한 배경(흰 벽 등)
    안에서 아주 작게 차지하는 극단적인 경우엔 여전히 오탐 위험이 있다 -
    이 함수는 "완전 삭제"가 아니라 "경고"만 하므로, 최종 판단은 사람이
    직접 확인하고 내려야 한다.
    """
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size*hash_size//4}x")


def hamming_distance(hash_a: str, hash_b: str) -> int:
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except (ValueError, TypeError):
        return 999  # 파싱 실패 - 비교 불가로 취급 (절대 중복으로 오판하지 않도록 큰 값)


# ---------------------------------------------------------------------------
# 판정 (threshold 비교 -> 경고 목록)
# ---------------------------------------------------------------------------

def check_resolution(width: int, height: int) -> list[ImageQualityIssue]:
    issues = []
    if width < MIN_RESOLUTION_WIDTH or height < MIN_RESOLUTION_HEIGHT:
        issues.append(ImageQualityIssue(
            "resolution_too_low", ImageQualitySeverity.ERROR,
            f"해상도 {width}x{height}가 권장 최소치({MIN_RESOLUTION_WIDTH}x"
            f"{MIN_RESOLUTION_HEIGHT})보다 낮습니다 - 코너 검출 정밀도가 떨어질 수 있습니다.",
        ))
    return issues


def check_exposure(brightness: float | None) -> list[ImageQualityIssue]:
    issues = []
    if brightness is None:
        return issues
    if brightness < _BRIGHTNESS_TOO_DARK:
        issues.append(ImageQualityIssue(
            "too_dark", ImageQualitySeverity.WARNING,
            f"평균 밝기 {brightness:.1f}(0~255) - 너무 어둡습니다.",
        ))
    elif brightness > _BRIGHTNESS_TOO_BRIGHT:
        issues.append(ImageQualityIssue(
            "too_bright", ImageQualitySeverity.WARNING,
            f"평균 밝기 {brightness:.1f}(0~255) - 너무 밝습니다.",
        ))
    return issues


def check_saturation(saturation: float | None) -> list[ImageQualityIssue]:
    issues = []
    if saturation is not None and saturation > _SATURATION_CLIP_WARN:
        issues.append(ImageQualityIssue(
            "exposure_clipping", ImageQualitySeverity.WARNING,
            f"픽셀의 {saturation:.0%}가 완전히 검거나 흰 값으로 clipping됐습니다 "
            "- 하이라이트/섀도우 디테일이 날아갔을 수 있습니다.",
        ))
    return issues


def check_blur(sharpness: float | None) -> list[ImageQualityIssue]:
    issues = []
    if sharpness is not None and sharpness < _SHARPNESS_ABSOLUTE_FLOOR:
        issues.append(ImageQualityIssue(
            "too_blurry", ImageQualitySeverity.ERROR,
            f"선명도(Laplacian 분산) {sharpness:.1f} - 명백히 초점이 맞지 않습니다.",
        ))
    return issues


def check_motion_blur(motion_blur_score: float | None) -> list[ImageQualityIssue]:
    issues = []
    if motion_blur_score is not None and motion_blur_score > _MOTION_BLUR_RATIO_WARN:
        issues.append(ImageQualityIssue(
            "possible_motion_blur", ImageQualitySeverity.WARNING,
            f"가로/세로 그래디언트 비율 {motion_blur_score:.1f}배 - 한쪽 방향으로만 "
            "흔들린 모션 블러가 의심됩니다 (오탐 가능성 있음, 육안 확인 권장).",
        ))
    return issues


def check_contrast(contrast: float | None) -> list[ImageQualityIssue]:
    issues = []
    if contrast is not None and contrast < _CONTRAST_TOO_LOW:
        issues.append(ImageQualityIssue(
            "low_contrast", ImageQualitySeverity.WARNING,
            f"명암 대비(grayscale 표준편차) {contrast:.1f} - 대비가 낮아 뿌옇게 보이는 이미지입니다.",
        ))
    return issues


def evaluate_image_quality(info: ImageInfo) -> ImageQualityReport:
    """ImageInfo 하나에 대해 절대 기준 검사를 전부 실행."""
    issues: list[ImageQualityIssue] = []
    issues += check_resolution(info.width, info.height)
    issues += check_exposure(info.brightness)
    issues += check_saturation(info.saturation)
    issues += check_blur(info.sharpness)
    issues += check_motion_blur(info.motion_blur_score)
    issues += check_contrast(info.contrast)
    return ImageQualityReport(image_id=info.image_id, issues=issues)


# ---------------------------------------------------------------------------
# 중복 / 거의 동일 이미지 검사 (설계 문서 3-1번 마지막 두 항목)
# ---------------------------------------------------------------------------

@dataclass
class DuplicateGroup:
    image_ids: list[str]
    exact: bool  # True=완전 동일(해밍거리<=_PHASH_EXACT_DUPLICATE_DISTANCE), False=near-duplicate


def find_duplicate_groups(dataset: Dataset) -> list[DuplicateGroup]:
    """phash가 채워진 프레임들 중 서로 가까운(near-duplicate 포함) 이미지를
    그룹으로 묶는다. O(n^2) 비교라 프레임 수가 아주 많으면(수천 장) 느려질 수
    있지만, 일반적인 캘리브레이션 데이터셋 규모(수십~수백 장)에서는 충분히
    빠르다.
    """
    frames = [
        f for f in dataset.enabled_frames
        if f.image_info.phash
    ]
    n = len(frames)
    visited = [False] * n
    groups: list[DuplicateGroup] = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        min_dist_in_cluster = None
        for j in range(i + 1, n):
            if visited[j]:
                continue
            dist = hamming_distance(frames[i].image_info.phash, frames[j].image_info.phash)
            if dist <= _PHASH_NEAR_DUPLICATE_DISTANCE:
                cluster.append(j)
                min_dist_in_cluster = dist if min_dist_in_cluster is None else min(min_dist_in_cluster, dist)

        if len(cluster) > 1:
            for idx in cluster:
                visited[idx] = True
            exact = all(
                hamming_distance(frames[cluster[0]].image_info.phash, frames[k].image_info.phash)
                <= _PHASH_EXACT_DUPLICATE_DISTANCE
                for k in cluster[1:]
            )
            groups.append(DuplicateGroup(
                image_ids=[frames[k].image_info.image_id for k in cluster], exact=exact,
            ))
        else:
            visited[i] = True

    return groups


def format_duplicate_groups(groups: list[DuplicateGroup]) -> str:
    if not groups:
        return "중복/거의 동일한 이미지가 발견되지 않았습니다."
    lines = [f"중복 의심 그룹 {len(groups)}개 발견:"]
    for g in groups:
        kind = "완전 동일" if g.exact else "거의 동일(near-duplicate)"
        lines.append(f"  [{kind}] {', '.join(g.image_ids)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 데이터셋 전체 실행
# ---------------------------------------------------------------------------

def evaluate_dataset_image_quality(
    dataset: Dataset,
) -> tuple[dict[str, ImageQualityReport], list[DuplicateGroup]]:
    """detect_dataset() 직후 호출. 개별 이미지 판정 + 중복 그룹 탐지를 한 번에 실행."""
    reports = {
        f.image_info.image_id: evaluate_image_quality(f.image_info)
        for f in dataset.frames
    }
    duplicate_groups = find_duplicate_groups(dataset)
    return reports, duplicate_groups


def format_image_quality_summary(
    reports: dict[str, ImageQualityReport], duplicate_groups: list[DuplicateGroup]
) -> str:
    lines = []
    flagged = {k: r for k, r in reports.items() if r.issues}
    if flagged:
        lines.append(f"이미지 품질 경고 {len(flagged)}장:")
        for image_id, report in flagged.items():
            for issue in report.issues:
                mark = "\u2716" if issue.severity == ImageQualitySeverity.ERROR else "\u26a0"
                lines.append(f"  {mark} [{image_id}] {issue.message}")
    lines.append(format_duplicate_groups(duplicate_groups))
    return "\n".join(lines)
