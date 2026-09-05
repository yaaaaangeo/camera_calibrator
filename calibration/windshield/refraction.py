"""
camera_calibrator.calibration.windshield.refraction
========================================================

Windshield 굴절 계산에 필요한 순수 벡터 수학 - Snell의 법칙(vector form)과
Ray-Sphere Intersection. calibration.types나 다른 windshield 모듈에 의존하지
않는 독립 모듈이다 - Spherical뿐 아니라 향후 Residual Ray/Spline이 필요하면
그대로 재사용할 수 있다.

좌표계 관례(반드시 지킬 것): OpenCV 카메라 좌표계 - +x 오른쪽, +y 아래쪽,
+z 전방(장면 쪽). 이 모듈에 넘기는 모든 벡터(광선 방향, 구 중심, 표면 법선)는
이 좌표계 기준이어야 한다 - 이 모듈 자체는 좌표 변환을 하지 않는다(임의
변환을 넣지 말라는 설계 원칙).

벡터 방향 관례:
    incident_direction : 빛이 "진행하는" 방향(광선의 시작점 -> 표면), 단위 벡터.
    surface_normal      : 어느 쪽을 향해도 상관없다 - refract_ray가 내부적으로
                           incident_direction과 반대가 되도록 자동 정렬한다
                           (GLSL의 refract() 내장 함수와 동일한 관례).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def normalize(v) -> np.ndarray:
    """벡터를 단위 벡터로 만든다. 길이가 0에 가까우면(정의 불가) ValueError."""
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        raise ValueError("Cannot normalize a zero-length (or near-zero-length) vector.")
    return arr / norm


def refract_ray(
    incident_direction,
    surface_normal,
    n_from: float,
    n_to: float,
):
    """Snell의 법칙을 벡터 형태로 계산한다(GLSL refract() 공식과 동일).

        n_from * sin(theta1) = n_to * sin(theta2)

    Args:
        incident_direction: 입사광선 방향(진행 방향), 자동으로 정규화된다.
        surface_normal: 표면 법선. 방향(입사광선과 같은 쪽/반대 쪽)은 상관없다 -
            내부에서 항상 incident_direction과 반대가 되도록(cos_i >= 0)
            자동으로 뒤집는다. 이래야 표면의 "앞면"/"뒷면" 어느 쪽에서 광선이
            들어오든 같은 공식이 성립한다.
        n_from: 입사 매질의 굴절률(예: 공기 1.0).
        n_to: 굴절 매질의 굴절률(예: 유리 1.52).

    Returns:
        굴절된 방향의 단위 벡터. 전반사(Total Internal Reflection)가 일어나면
        None (예외를 던지지 않는다 - 굴절이 "없다"는 것도 유효한 물리적 결과이므로
        호출부가 판단하게 한다).
    """
    if n_from <= 0.0 or n_to <= 0.0:
        raise ValueError(f"Refractive indices must be positive (n_from={n_from}, n_to={n_to}).")

    incident = normalize(incident_direction)
    normal = normalize(surface_normal)

    cos_i = -float(np.dot(normal, incident))
    if cos_i < 0.0:
        # 법선이 입사광선과 "같은 쪽"을 향하고 있었다는 뜻 - 표면 뒷면에서 들어온
        # 경우도 같은 공식이 성립하도록 법선을 반대로 뒤집는다.
        normal = -normal
        cos_i = -cos_i

    eta = n_from / n_to
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None  # Total internal reflection - 이 매질쌍/각도에서는 굴절이 불가능

    cos_t = float(np.sqrt(max(0.0, 1.0 - sin2_t)))
    refracted = eta * incident + (eta * cos_i - cos_t) * normal
    return normalize(refracted)


def intersect_ray_sphere(
    ray_origin,
    ray_direction,
    sphere_center,
    sphere_radius: float,
):
    """직선(ray_origin에서 ray_direction 방향)과 구(sphere_center, sphere_radius)의
    교차점 중, t > 0(광선이 실제로 진행하는 전방)인 가장 가까운 유효 교차점을
    반환한다.

    ray_direction은 자동으로 정규화되므로 이차방정식의 a(=dot(D,D))는 항상 1이고,
    따라서 t1 = (-b - sqrt(disc))/2 <= t2 = (-b + sqrt(disc))/2가 항상 성립한다 -
    카메라가 구 "밖"에 있든(첫 번째 양의 근이 진입점) "안"에 있든(첫 번째 근이
    음수이므로 두 번째 양의 근이 탈출점) 이 하나의 규칙("t1보다 크면 t1, 아니면
    t2")으로 두 경우 모두 올바르게 처리된다 - 카메라 안/밖을 따로 분기하지 않는다.

    Returns:
        (intersection_point, t) 또는 교차/유효 해가 없으면 None:
          * 0개 교차 (discriminant < 0)
          * 교차가 있어도 전부 광선의 뒤쪽(t <= epsilon)인 경우
    """
    if sphere_radius <= 0.0:
        raise ValueError(f"Sphere radius must be positive (got {sphere_radius}).")

    origin = np.asarray(ray_origin, dtype=np.float64)
    direction = normalize(ray_direction)
    center = np.asarray(sphere_center, dtype=np.float64)
    radius = float(sphere_radius)

    oc = origin - center
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    discriminant = b * b - 4.0 * c  # a = dot(direction, direction) = 1 (정규화됨)

    if discriminant < 0.0:
        return None  # 0 intersections (완전히 빗나감)

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0  # t1 <= t2 항상 성립 (위 docstring 참고)

    if t1 > _EPS:
        t = float(t1)
    elif t2 > _EPS:
        t = float(t2)
    else:
        return None  # 교차는 있지만 전부 광선 뒤쪽(t<=0)

    point = origin + t * direction
    return point, t
