from inspect import getfullargspec

import numpy as np
import pytest

import lumicks.pylake.fitting.detail.model_implementation as model_impl
from lumicks.pylake.fitting.models import *


def test_deprecated_models():
    """Check that functions are deprecated and point to the corresponding new ones"""
    old_new = [
        (marko_siggia_simplified, wlc_marko_siggia_force),
        (inverted_marko_siggia_simplified, wlc_marko_siggia_distance),
        (marko_siggia_ewlc_distance, ewlc_marko_siggia_distance),
        (marko_siggia_ewlc_force, ewlc_marko_siggia_force),
        (odijk, ewlc_odijk_distance),
        (inverted_odijk, ewlc_odijk_force),
        (twistable_wlc, twlc_distance),
        (inverted_twistable_wlc, twlc_force),
        (freely_jointed_chain, efjc_distance),
        (inverted_freely_jointed_chain, efjc_force),
    ]
    for old, new in old_new:
        with pytest.deprecated_call():
            old("DEPRECATION_IS_FUN")
            assert repr(old("RENAMING_IS_EVEN_BETTER")) == repr(new("RENAMING_IS_EVEN_BETTER"))


def test_models():
    independent = np.arange(0.15, 2, 0.25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    assert ewlc_odijk_distance("WLC").verify_jacobian(independent, params)
    assert ewlc_odijk_force("iWLC").verify_jacobian(independent, params, atol=1e-5)
    assert efjc_distance("FJC").verify_jacobian(independent, params, dx=1e-4, atol=1e-6)
    assert wlc_marko_siggia_force("MS").verify_jacobian(independent, [5, 5, 4.11], atol=1e-6)
    assert wlc_marko_siggia_distance("iMS").verify_jacobian(
        independent, [38.18281266, 0.37704827, 4.11], atol=1e-5
    )

    assert ewlc_odijk_distance("WLC").verify_derivative(independent, params)
    assert ewlc_odijk_force("iWLC").verify_derivative(independent, params)
    assert efjc_distance("FJC").verify_derivative(independent, params, atol=1e-6)
    assert wlc_marko_siggia_force("MS").verify_derivative(independent, [5, 5, 4.11], atol=1e-6)

    assert ewlc_marko_siggia_force("MSF").verify_jacobian(independent, params, dx=1e-4, rtol=1e-4)
    assert ewlc_marko_siggia_distance("MSD").verify_jacobian(independent, params, dx=1e-4)
    assert ewlc_marko_siggia_force("MSF").verify_derivative(independent, params, dx=1e-4)
    assert ewlc_marko_siggia_distance("MSD").verify_derivative(independent, params, dx=1e-4)
    assert wlc_marko_siggia_distance("iMS").verify_derivative(
        independent, [38.18281266, 0.37704827, 4.11], atol=1e-5
    )

    # The finite differencing version of the FJC performs very poorly numerically, hence the less
    # stringent tolerances and larger dx values.
    assert efjc_force("iFJC").verify_derivative(independent, params, dx=1e-3, rtol=1e-2, atol=1e-6)
    assert efjc_force("iFJC").verify_jacobian(independent, params, dx=1e-3, atol=1e-2, rtol=1e-2)

    # Check the tWLC and inverted tWLC model
    params = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert twlc_distance("tWLC").verify_jacobian(independent, params)
    assert twlc_force("itWLC").verify_jacobian(independent, params)

    # Check whether the twistable wlc model manipulates the data order
    np.testing.assert_allclose(
        twlc_distance("tWLC")._raw_call(independent, params),
        np.flip(twlc_distance("tWLC")._raw_call(np.flip(independent), params)),
    )

    # Check whether the inverse twistable wlc model manipulates the data order
    np.testing.assert_allclose(
        twlc_force("itWLC")._raw_call(independent, params),
        np.flip(twlc_force("itWLC")._raw_call(np.flip(independent), params)),
    )

    # Check whether the inverted models invert correctly
    d = np.array([3.0, 4.0])
    params = [5.0, 5.0, 5.0]
    np.testing.assert_allclose(
        model_impl.ewlc_odijk_distance(model_impl.ewlc_odijk_force(d, *params), *params), d
    )
    params = [5.0, 15.0, 1.0, 4.11]
    np.testing.assert_allclose(
        model_impl.efjc_distance(model_impl.efjc_solve_force(independent, *params), *params),
        independent,
    )
    params = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    np.testing.assert_allclose(
        model_impl.twlc_distance(model_impl.twlc_solve_force(independent, *params), *params),
        independent,
    )

    d = np.arange(0.15, 2, 0.5)
    (Lp, Lc, St, kT) = (38.18281266, 0.37704827, 278.50103452, 4.11)
    params = [Lp, Lc, St, kT]
    m_fwd = ewlc_marko_siggia_force("fwd")
    m_bwd = ewlc_marko_siggia_distance("bwd")
    force = m_fwd._raw_call(d, params)
    np.testing.assert_allclose(m_bwd._raw_call(force, params), d)

    # Determine whether they actually fulfill the model
    lhs = force * Lp / kT
    rhs = 0.25 * (1.0 - (d / Lc) + (force / St)) ** (-2) - 0.25 + (d / Lc) - (force / St)
    np.testing.assert_allclose(lhs, rhs)

    # Test inverted simplified model
    d = np.arange(0.15, 0.377, 0.05)
    (Lp, Lc, kT) = (38.18281266, 0.37704827, 4.11)
    params = [Lp, Lc, kT]
    m_fwd = wlc_marko_siggia_force("fwd")
    m_bwd = wlc_marko_siggia_distance("bwd")
    force = m_fwd._raw_call(d, params)
    np.testing.assert_allclose(m_bwd._raw_call(force, params), d)

    # This model is nonsense about the contour length, so warn the user about this.
    with pytest.warns(RuntimeWarning):
        m_fwd._raw_call(np.array([Lc + 0.1]), params)


def test_model_reprs():
    assert ewlc_odijk_distance("test").__repr__()
    assert ewlc_odijk_force("test").__repr__()
    assert efjc_distance("test").__repr__()
    assert wlc_marko_siggia_force("test").__repr__()
    assert wlc_marko_siggia_distance("test").__repr__()
    assert ewlc_marko_siggia_force("test").__repr__()
    assert ewlc_marko_siggia_distance("test").__repr__()
    assert efjc_force("test").__repr__()
    assert twlc_distance("test").__repr__()
    assert twlc_force("test").__repr__()
    assert (ewlc_odijk_distance("test") + distance_offset("test")).__repr__()
    assert (ewlc_odijk_distance("test") + distance_offset("test")).invert().__repr__()
    assert (
        (ewlc_odijk_distance("test") + distance_offset("test"))
        .subtract_independent_offset()
        .__repr__()
    )

    assert ewlc_odijk_distance("test")._repr_html_()
    assert ewlc_odijk_force("test")._repr_html_()
    assert efjc_distance("test")._repr_html_()
    assert wlc_marko_siggia_force("test")._repr_html_()
    assert wlc_marko_siggia_distance("test")._repr_html_()
    assert ewlc_marko_siggia_force("test")._repr_html_()
    assert ewlc_marko_siggia_distance("test")._repr_html_()
    assert efjc_force("test")._repr_html_()
    assert twlc_distance("test")._repr_html_()
    assert twlc_force("test")._repr_html_()
    assert (ewlc_odijk_distance("test") + distance_offset("test"))._repr_html_()
    assert (ewlc_odijk_distance("test") + distance_offset("test")).invert()._repr_html_()
    assert (
        (ewlc_odijk_distance("test") + distance_offset("test"))
        .subtract_independent_offset()
        ._repr_html_()
    )
    assert (force_offset("a_b_c") + force_offset("b_c_d")).invert()._repr_html_().find(
        "offset_{b\\_c\\_d}"
    ) > 0


@pytest.mark.parametrize(
    "model, test_params",
    [
        (model_impl.ewlc_odijk_force, ["Lp", "Lc", "St", "kT"]),
        (model_impl.ewlc_odijk_distance, ["Lp", "Lc", "St", "kT"]),
        (model_impl.efjc_distance, ["Lp", "Lc", "St", "kT"]),
        (model_impl.efjc_solve_force, ["Lp", "Lc", "St", "kT"]),
        (model_impl.wlc_marko_siggia_force, ["Lp", "Lc", "kT"]),
        (model_impl.wlc_marko_siggia_distance, ["Lp", "Lc", "kT"]),
        (model_impl.ewlc_marko_siggia_force, ["Lp", "Lc", "St", "kT"]),
        (model_impl.ewlc_marko_siggia_distance, ["Lp", "Lc", "St", "kT"]),
        (model_impl.twlc_distance, ["Lp", "Lc", "St", "kT"]),
        (model_impl.twlc_solve_force, ["Lp", "Lc", "St", "kT"]),
    ],
)
def test_invalid_params_models(model, test_params):
    for test_param in test_params:
        params = {p: 1 for p in getfullargspec(model).args}
        for value in (0, -1):
            with pytest.raises(ValueError, match="must be bigger than 0"):
                params[test_param] = value
                model(**params)


@pytest.mark.parametrize(
    "convenience_model, ref_model, ref_params",
    [
        [
            dsdna_ewlc_odijk_distance,
            ewlc_odijk_distance,
            {"m/Lc": 100 * 0.34, "m/Lp": 50.0, "m/St": 1200.0, "kT": 4.11},
        ],
        [
            ssdna_efjc_distance,
            efjc_distance,
            {"m/Lc": 100 * 0.56, "m/Lp": 0.7, "m/St": 750.0, "kT": 4.11},
        ],
    ],
)
def test_convenience_models(convenience_model, ref_model, ref_params):
    x = np.arange(1.0, 5.0)
    model = convenience_model("m", 100)
    params = dict(model.defaults)

    for param in ref_params.keys():
        np.testing.assert_allclose(params[param].value, ref_params[param])

    # The convenience part is in the parameters, with the old parameters they should produce the
    # exact same as what they are supposed to be based on
    np.testing.assert_allclose(model(x, params), ref_model("m")(x, params))


def test_model_get_item():
    m = ewlc_odijk_force("m")
    m["m/Lc"].value = 3.1415
    np.testing.assert_equal(float(m.defaults["m/Lc"]), 3.1415)
