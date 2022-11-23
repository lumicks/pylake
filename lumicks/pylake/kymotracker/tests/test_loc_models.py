import pytest
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


@pytest.mark.parametrize(
    "num_points",
    (3, 2, 5, 7),
)
def test_localization_model_split(num_points):
    test_data = {
        "position": np.array([0.2, 0.3, 0.4, 0.5, 0.8]),
        "total_photons": np.array([2, 3, 4, 5, 6]),
        "sigma": np.array([0.6, 0.5, 0.4, 0.3, 0.2]),
        "background": np.array([1, 2, 1, 3, 4]),
        "_overlap_fit": np.array([False, False, True, True, True]),
    }

    def validate_split(localization_model, fields):
        indices = range(num_points, localization_model.position.size, num_points)
        splits = localization_model._split(indices)
        assert len(splits) == np.ceil(localization_model.position.size / num_points)

        for field in fields:
            ref_data = getattr(localization_model, field)
            for chunk, ref in zip(
                splits, np.array_split(ref_data, indices)
            ):
                np.testing.assert_equal(getattr(chunk, field), ref)

    validate_split(LocalizationModel(test_data["position"]), ["position"])
    validate_split(GaussianLocalizationModel(**test_data), test_data.keys())
