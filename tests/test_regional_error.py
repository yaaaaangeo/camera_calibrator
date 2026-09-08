import pytest

from calibration.models.common import compute_regional_error


def test_regional_error_uses_true_point_rms_not_mean_error():
    regional = compute_regional_error(
        xs=[50, 60, 70, 80],
        ys=[300, 300, 300, 300],
        errors=[0, 0, 0, 4],
        image_size=(900, 600),
    )

    assert regional.left == pytest.approx(2.0)
    assert regional.left != pytest.approx(1.0)


def test_regional_error_pools_uneven_point_counts():
    regional = compute_regional_error(
        xs=[50] * 10,
        ys=[300] * 10,
        errors=[1, 1] + [3] * 8,
        image_size=(900, 600),
    )

    assert regional.left == pytest.approx((74 / 10) ** 0.5)
    assert regional.left != pytest.approx(2.0)


def test_regional_error_empty_returns_none_fields():
    regional = compute_regional_error([], [], [], image_size=(900, 600))

    assert regional.center is None
    assert regional.left is None
    assert regional.right is None
    assert regional.top is None
    assert regional.bottom is None
    assert regional.corner is None


def test_regional_error_classifies_each_corner_position():
    regional = compute_regional_error(
        xs=[450, 50],
        ys=[300, 300],
        errors=[1, 5],
        image_size=(900, 600),
    )

    assert regional.center == pytest.approx(1.0)
    assert regional.left == pytest.approx(5.0)
