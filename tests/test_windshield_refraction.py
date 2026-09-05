"""
tests/test_windshield_refraction.py
========================================

calibration.windshield.refraction의 순수 벡터 수학 검증(Snell의 법칙,
ray-sphere intersection). 여기 있는 기대값은 refract_ray/intersect_ray_sphere
자체 코드를 쓰지 않고 손으로 계산한다 - 구현과 같은 코드로 정답을 만들면
공유된 버그를 잡아내지 못하기 때문이다(사용자 스펙 28-1~28-4번).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration.windshield.refraction import intersect_ray_sphere, normalize, refract_ray


def test_normalize_raises_on_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.array([0.0, 0.0, 0.0]))


def test_normalize_returns_unit_vector():
    v = normalize(np.array([3.0, 4.0, 0.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v[0] == pytest.approx(0.6)
    assert v[1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Snell's Law
# ---------------------------------------------------------------------------

def test_refract_matches_hand_computed_snell_angle():
    """n1=1.0 -> n2=1.5, 알려진 입사각 theta1에서 굴절각 theta2를
    theta2 = asin(n1/n2 * sin(theta1))로 손으로 계산해 비교한다."""
    n1, n2 = 1.0, 1.5
    theta1 = math.radians(30.0)
    incident = np.array([math.sin(theta1), 0.0, math.cos(theta1)])
    normal = np.array([0.0, 0.0, -1.0])  # 표면 법선이 입사광선 반대쪽을 향함

    refracted = refract_ray(incident, normal, n1, n2)
    assert refracted is not None

    theta2_expected = math.asin(n1 / n2 * math.sin(theta1))
    theta2_actual = math.atan2(refracted[0], refracted[2])
    assert theta2_actual == pytest.approx(theta2_expected, abs=1e-9)
    assert np.linalg.norm(refracted) == pytest.approx(1.0)


def test_refract_normal_incidence_direction_unchanged():
    """입사광선이 표면에 수직(normal incidence)이면 방향이 바뀌지 않는다."""
    incident = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 0.0, -1.0])
    refracted = refract_ray(incident, normal, 1.0, 1.5)
    assert refracted == pytest.approx(incident, abs=1e-9)

    # 매질이 반대(유리->공기)여도 수직 입사는 여전히 방향 불변.
    refracted_back = refract_ray(incident, normal, 1.5, 1.0)
    assert refracted_back == pytest.approx(incident, abs=1e-9)


def test_refract_auto_flips_normal_regardless_of_input_orientation():
    """법선을 반대로 줘도(표면의 "뒷면"에서 준 것처럼) 같은 결과가 나와야 한다."""
    n1, n2 = 1.0, 1.5
    theta1 = math.radians(20.0)
    incident = np.array([math.sin(theta1), 0.0, math.cos(theta1)])

    result_a = refract_ray(incident, np.array([0.0, 0.0, -1.0]), n1, n2)
    result_b = refract_ray(incident, np.array([0.0, 0.0, 1.0]), n1, n2)
    assert result_a == pytest.approx(result_b, abs=1e-9)


def test_refract_total_internal_reflection_returns_none():
    """유리(1.5)->공기(1.0)로 나갈 때 임계각보다 큰 입사각이면 전반사(None)."""
    n1, n2 = 1.5, 1.0
    critical_angle = math.asin(n2 / n1)
    theta1 = critical_angle + math.radians(5.0)  # 임계각보다 확실히 크게
    incident = np.array([math.sin(theta1), 0.0, math.cos(theta1)])
    normal = np.array([0.0, 0.0, -1.0])

    assert refract_ray(incident, normal, n1, n2) is None


def test_refract_below_critical_angle_does_not_reflect():
    n1, n2 = 1.5, 1.0
    critical_angle = math.asin(n2 / n1)
    theta1 = critical_angle - math.radians(5.0)  # 임계각보다 확실히 작게
    incident = np.array([math.sin(theta1), 0.0, math.cos(theta1)])
    normal = np.array([0.0, 0.0, -1.0])

    assert refract_ray(incident, normal, n1, n2) is not None


def test_refract_rejects_nonpositive_refractive_index():
    with pytest.raises(ValueError):
        refract_ray(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), 0.0, 1.5)


# ---------------------------------------------------------------------------
# Ray-Sphere Intersection
# ---------------------------------------------------------------------------

def test_intersect_ray_sphere_camera_outside_sphere():
    """카메라(원점)가 구 밖에 있을 때 - 더 가까운(작은 t) 교차점을 반환."""
    hit = intersect_ray_sphere(
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([0.0, 0.0, 1.0]),
        sphere_center=np.array([0.0, 0.0, 5.0]),
        sphere_radius=2.0,
    )
    assert hit is not None
    point, t = hit
    assert point == pytest.approx([0.0, 0.0, 3.0])
    assert t == pytest.approx(3.0)


def test_intersect_ray_sphere_origin_inside_sphere_returns_exit_point():
    """광선의 시작점이 구 안에 있으면(예: 두 번째 표면 굴절 계산) 탈출점(더 큰 t)을
    반환해야 한다 - Spherical의 두 번째 표면(바깥쪽)이 이 경우에 해당한다."""
    hit = intersect_ray_sphere(
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([0.0, 0.0, 1.0]),
        sphere_center=np.array([0.0, 0.0, 0.0]),
        sphere_radius=2.0,
    )
    assert hit is not None
    point, t = hit
    assert point == pytest.approx([0.0, 0.0, 2.0])
    assert t == pytest.approx(2.0)


def test_intersect_ray_sphere_tangent_case():
    """광선이 구에 접하는(tangent) 경우 - discriminant가 0에 가까운 단일 해."""
    hit = intersect_ray_sphere(
        ray_origin=np.array([0.0, 2.0, 0.0]),
        ray_direction=np.array([0.0, 0.0, 1.0]),
        sphere_center=np.array([0.0, 0.0, 5.0]),
        sphere_radius=2.0,
    )
    assert hit is not None
    point, t = hit
    assert point[1] == pytest.approx(2.0, abs=1e-6)
    assert point[2] == pytest.approx(5.0, abs=1e-6)


def test_intersect_ray_sphere_no_intersection_returns_none():
    """광선이 구를 완전히 빗나가면 None (crash 없이)."""
    hit = intersect_ray_sphere(
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([1.0, 0.0, 0.0]),
        sphere_center=np.array([0.0, 0.0, 5.0]),
        sphere_radius=1.0,
    )
    assert hit is None


def test_intersect_ray_sphere_behind_ray_returns_none():
    """교차점이 있어도 전부 광선의 뒤쪽(t<=0)이면 None."""
    hit = intersect_ray_sphere(
        ray_origin=np.array([0.0, 0.0, 10.0]),
        ray_direction=np.array([0.0, 0.0, 1.0]),  # 구에서 멀어지는 방향
        sphere_center=np.array([0.0, 0.0, 0.0]),
        sphere_radius=2.0,
    )
    assert hit is None


def test_intersect_ray_sphere_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        intersect_ray_sphere(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 5.0]), 0.0
        )
