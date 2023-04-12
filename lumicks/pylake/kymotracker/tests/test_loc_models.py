import numpy as np
from lumicks.pylake.kymotracker.detail.localization_models import *


def test_default_model():
    m = LocalizationModel(0.15 * np.arange(5) * 0.2)
    np.testing.assert_allclose(m.position, [0.0, 0.03, 0.06, 0.09, 0.12])


def test_gaussian_model():
    m = GaussianLocalizationModel(
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.full(3, False),
    )
    np.testing.assert_allclose(m.position, [0.0, 0.2, 0.4])


def test_gaussian_model_overlap():
    m = GaussianLocalizationModel(
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.array([False, True, False]),
    )
    np.testing.assert_allclose(m.position, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(m.total_photons, [5, 6, 8])
    np.testing.assert_allclose(m.sigma, [0.5, 0.5, 0.6])
    np.testing.assert_allclose(m.background, [1, 2, 1])
    np.testing.assert_allclose(m._overlap_fit, [False, True, False])


def test_gaussian_model_with_position():
    m = GaussianLocalizationModel(
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.array([False, True, False]),
    )

    m2 = m.with_position(np.asarray([0.5, 0.6, 0.7]))
    np.testing.assert_allclose(m2.position, [0.5, 0.6, 0.7])
    np.testing.assert_allclose(m2.total_photons, [5, 6, 8])
    np.testing.assert_allclose(m2.sigma, [0.5, 0.5, 0.6])
    np.testing.assert_allclose(m2.background, [1, 2, 1])
    np.testing.assert_allclose(m2._overlap_fit, [False, True, False])

    m3 = m2._flip(5.0)
    np.testing.assert_allclose(m3.position, [5.0 - 0.5, 5.0 - 0.6, 5.0 - 0.7])
