import numpy as np
import pytest

from lumicks.pylake.kymotracker.detail.localization_models import *
from lumicks.pylake.kymotracker.detail.localization_models import CentroidLocalizationModel


def test_default_model():
    m = LocalizationModel(0.15 * np.arange(5) * 0.2)
    np.testing.assert_allclose(m.position, [0.0, 0.03, 0.06, 0.09, 0.12])


def test_gaussian_model_evaluate():
    m = GaussianLocalizationModel(
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.full(3, False),
    )
    np.testing.assert_allclose(m.position, [0.0, 0.2, 0.4])

    position_axis = np.arange(0.0, 5 * 0.2, 0.2)
    ref_data = [
        [1.79788456, 1.73654028, 1.57938311, 1.38837211, 1.22184167],
        [2.88384834, 2.95746147, 2.88384834, 2.69525973, 2.46604653],
        [1.85186135, 2.00635527, 2.06384608, 2.00635527, 1.85186135],
    ]

    for idx, ref in enumerate(ref_data):
        np.testing.assert_allclose(m.evaluate(position_axis, idx, 0.2), ref)


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


def test_gaussian_model_add():
    m1 = GaussianLocalizationModel(
        np.arange(3) * 0.2,
        np.array([5, 6, 8]),
        np.array([0.5, 0.5, 0.6]),
        np.array([1, 2, 1]),
        np.array([False, True, False]),
    )

    m2 = GaussianLocalizationModel(
        np.arange(5, 8, 1) * 0.2,
        np.array([3, 2, 1]),
        np.array([0.3, 0.4, 0.2]),
        np.array([2, 1, 3]),
        np.array([False, False, True]),
    )

    added = m1 + m2
    np.testing.assert_allclose(added.position, [0.0, 0.2, 0.4, 1.0, 1.2, 1.4])
    np.testing.assert_allclose(added.total_photons, [5, 6, 8, 3, 2, 1])
    np.testing.assert_allclose(added.sigma, [0.5, 0.5, 0.6, 0.3, 0.4, 0.2])
    np.testing.assert_allclose(added.background, [1, 2, 1, 2, 1, 3])
    np.testing.assert_allclose(added._overlap_fit, [False, True, False, False, False, True])


def test_incompatible_add():
    m1 = GaussianLocalizationModel(
        np.arange(3), np.arange(3), np.arange(3), np.arange(3), np.zeros(3, dtype=bool)
    )
    m2 = LocalizationModel(np.arange(3))

    with pytest.raises(
        TypeError,
        match="Incompatible localization models GaussianLocalizationModel and LocalizationModel.",
    ):
        m1 + m2


@pytest.mark.parametrize(
    "slc",
    [
        slice(None, None, None),
        slice(1, None, None),
        slice(1, 1, None),
        slice(1, 2, None),
        slice(1, 3, None),
        slice(None, 2, None),
        slice(1, -1, None),
        slice(None, -1, None),
        slice(1, 4, 2),
    ],
)
def test_gaussian_model_getitem(slc):
    all_fields = {
        "position": np.arange(3) * 0.2,
        "total_photons": np.array([5, 6, 8]),
        "sigma": np.array([0.5, 0.5, 0.6]),
        "background": np.array([1, 2, 1]),
        "_overlap_fit": np.array([False, True, False]),
    }

    for model, fields in (
        (LocalizationModel, {"position": all_fields["position"]}),
        (CentroidLocalizationModel, dict(list(all_fields.items())[:2])),
        (GaussianLocalizationModel, all_fields),
    ):
        m1 = model(**fields)

        for f in fields.keys():
            np.testing.assert_allclose(getattr(m1[slc], f), getattr(m1, f)[slc])
