from lumicks.pylake.fitting.models import *
import lumicks.pylake.fitting.detail.model_implementation as model_impl
from inspect import getfullargspec
import numpy as np
import pytest


def test_models():
    independent = np.arange(0.15, 2, 0.25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    assert odijk("WLC").verify_jacobian(independent, params)
    assert inverted_odijk("iWLC").verify_jacobian(independent, params, atol=1e-5)
    assert freely_jointed_chain("FJC").verify_jacobian(independent, params, dx=1e-4, atol=1e-6)
    assert marko_siggia_simplified("MS").verify_jacobian(independent, [5, 5, 4.11], atol=1e-6)
    assert inverted_marko_siggia_simplified("iMS").verify_jacobian(
        independent, [38.18281266, 0.37704827, 4.11], atol=1e-5
    )

    assert odijk("WLC").verify_derivative(independent, params)
    assert inverted_odijk("iWLC").verify_derivative(independent, params)
    assert freely_jointed_chain("FJC").verify_derivative(independent, params, atol=1e-6)
    assert marko_siggia_simplified("MS").verify_derivative(independent, [5, 5, 4.11], atol=1e-6)

    assert marko_siggia_ewlc_force("MSF").verify_jacobian(independent, params, dx=1e-4, rtol=1e-4)
    assert marko_siggia_ewlc_distance("MSD").verify_jacobian(independent, params, dx=1e-4)
    assert marko_siggia_ewlc_force("MSF").verify_derivative(independent, params, dx=1e-4)
    assert marko_siggia_ewlc_distance("MSD").verify_derivative(independent, params, dx=1e-4)
    assert inverted_marko_siggia_simplified("iMS").verify_derivative(
        independent, [38.18281266, 0.37704827, 4.11], atol=1e-5
    )

    # The finite differencing version of the FJC performs very poorly numerically, hence the less
    # stringent tolerances and larger dx values.
    assert inverted_freely_jointed_chain("iFJC").verify_derivative(
        independent, params, dx=1e-3, rtol=1e-2, atol=1e-6
    )
    assert inverted_freely_jointed_chain("iFJC").verify_jacobian(
        independent, params, dx=1e-3, atol=1e-2, rtol=1e-2
    )

    # Check the tWLC and inverted tWLC model
    params = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert twistable_wlc("tWLC").verify_jacobian(independent, params)
    assert inverted_twistable_wlc("itWLC").verify_jacobian(independent, params)

    # Check whether the twistable wlc model manipulates the data order
    np.testing.assert_allclose(
        twistable_wlc("tWLC")._raw_call(independent, params),
        np.flip(twistable_wlc("tWLC")._raw_call(np.flip(independent), params)),
    )

    # Check whether the inverse twistable wlc model manipulates the data order
    np.testing.assert_allclose(
        inverted_twistable_wlc("itWLC")._raw_call(independent, params),
        np.flip(inverted_twistable_wlc("itWLC")._raw_call(np.flip(independent), params)),
    )

    # Check whether the inverted models invert correctly
    d = np.array([3.0, 4.0])
    params = [5.0, 5.0, 5.0]
    np.testing.assert_allclose(model_impl.WLC(model_impl.invWLC(d, *params), *params), d)
    params = [5.0, 15.0, 1.0, 4.11]
    np.testing.assert_allclose(
        model_impl.FJC(model_impl.invFJC(independent, *params), *params), independent
    )
    params = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    np.testing.assert_allclose(
        model_impl.tWLC(model_impl.invtWLC(independent, *params), *params), independent
    )

    d = np.arange(0.15, 2, 0.5)
    (Lp, Lc, St, kT) = (38.18281266, 0.37704827, 278.50103452, 4.11)
    params = [Lp, Lc, St, kT]
    m_fwd = marko_siggia_ewlc_force("fwd")
    m_bwd = marko_siggia_ewlc_distance("bwd")
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
    m_fwd = marko_siggia_simplified("fwd")
    m_bwd = inverted_marko_siggia_simplified("bwd")
    force = m_fwd._raw_call(d, params)
    np.testing.assert_allclose(m_bwd._raw_call(force, params), d)

    # This model is nonsense about the contour length, so warn the user about this.
    with pytest.warns(RuntimeWarning):
        m_fwd._raw_call(np.array([Lc + 0.1]), params)


def test_model_reprs():
    assert odijk("test").__repr__()
    assert inverted_odijk("test").__repr__()
    assert freely_jointed_chain("test").__repr__()
    assert marko_siggia_simplified("test").__repr__()
    assert marko_siggia_ewlc_force("test").__repr__()
    assert marko_siggia_ewlc_distance("test").__repr__()
    assert inverted_freely_jointed_chain("test").__repr__()
    assert twistable_wlc("test").__repr__()
    assert inverted_twistable_wlc("test").__repr__()
    assert (odijk("test") + distance_offset("test")).__repr__()
    assert (odijk("test") + distance_offset("test")).invert().__repr__()
    assert (odijk("test") + distance_offset("test")).subtract_independent_offset().__repr__()

    assert odijk("test")._repr_html_()
    assert inverted_odijk("test")._repr_html_()
    assert freely_jointed_chain("test")._repr_html_()
    assert marko_siggia_simplified("test")._repr_html_()
    assert marko_siggia_ewlc_force("test")._repr_html_()
    assert marko_siggia_ewlc_distance("test")._repr_html_()
    assert inverted_freely_jointed_chain("test")._repr_html_()
    assert twistable_wlc("test")._repr_html_()
    assert inverted_twistable_wlc("test")._repr_html_()
    assert (odijk("test") + distance_offset("test"))._repr_html_()
    assert (odijk("test") + distance_offset("test")).invert()._repr_html_()
    assert (odijk("test") + distance_offset("test")).subtract_independent_offset()._repr_html_()
    assert (force_offset("a_b_c") + force_offset("b_c_d")).invert()._repr_html_().find(
        "offset_{b\\_c\\_d}"
    ) > 0


@pytest.mark.parametrize(
    "model, test_params",
    [
        (model_impl.WLC, ["Lp", "Lc", "St", "kT"]),
        (model_impl.invWLC, ["Lp", "Lc", "St", "kT"]),
        (model_impl.FJC, ["Lp", "Lc", "St", "kT"]),
        (model_impl.invFJC, ["Lp", "Lc", "St", "kT"]),
        (model_impl.marko_siggia_simplified, ["Lp", "Lc", "kT"]),
        (model_impl.marko_siggia_ewlc_solve_force, ["Lp", "Lc", "St", "kT"]),
        (model_impl.marko_siggia_ewlc_solve_distance, ["Lp", "Lc", "St", "kT"]),
        (model_impl.tWLC, ["Lp", "Lc", "St", "kT"]),
        (model_impl.invtWLC, ["Lp", "Lc", "St", "kT"]),
    ],
)
def test_invalid_params_models(model, test_params):
    for test_param in test_params:
        params = {p: 1 for p in getfullargspec(model).args}
        for value in (0, -1):
            with pytest.raises(ValueError, match="must be bigger than 0"):
                params[test_param] = value
                model(**params)
