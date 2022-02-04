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
