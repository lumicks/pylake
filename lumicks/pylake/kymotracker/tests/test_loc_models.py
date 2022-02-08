import numpy as np
from lumicks.pylake.kymotracker.detail.loc_models import (
    LocalizationModel,
    GaussianLocalizationModel,
)


def test_default_model():
    m = LocalizationModel(0.15, np.arange(5) * 0.2)
    np.testing.assert_allclose(m.pixel_coordinate, [0.0, 1.33333333, 2.66666667, 4.0, 5.33333333])
    np.testing.assert_equal(m.loc_variance, np.full(5, np.nan))


def test_gaussian_model():
    m = GaussianLocalizationModel(
        0.15,
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.full(3, False),
    )

    np.testing.assert_allclose(m.pixel_coordinate, [0.0, 1.33333333, 2.66666667])
    ref_var = [0.5036671516579478, 0.6536440319302808, 0.4152688415531396]
    np.testing.assert_allclose(m.loc_variance, ref_var)


def test_gaussian_model_overlap():
    m = GaussianLocalizationModel(
        0.15,
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.array([False, True, False]),
    )

    np.testing.assert_allclose(m.pixel_coordinate, [0.0, 1.33333333, 2.66666667])
    ref_var = [0.5036671516579478, np.nan, 0.4152688415531396]
    np.testing.assert_allclose(m.loc_variance, ref_var)
