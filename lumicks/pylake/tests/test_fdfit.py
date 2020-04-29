from ..fitting.parameters import Parameters
from ..fitting.detail.utilities import parse_transformation
from ..fitting.detail.utilities import unique_idx
from ..fitting.fitdata import Condition, FitData
from ..fitting.detail.link_functions import generate_conditions
from ..fitting.fit import Fit
from ..fitting.models import *
from ..fitting.detail.parameter_trace import parameter_trace
from ..fitting.model import Model, InverseModel

import numpy as np
from collections import OrderedDict
import pytest


def test_transformation_parser():
    pars = ['blip', 'foo']
    assert parse_transformation(pars, {'foo': 'new_foo'}) == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))
    assert parse_transformation(pars, {'foo': 5}) == OrderedDict((('blip', 'blip'), ('foo', 5)))

    with pytest.raises(KeyError):
        parse_transformation(pars, {'blap': 'new_foo'}) == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))

    parameter_names = ['gamma', 'alpha', 'beta', 'delta']
    parameters = OrderedDict(zip(parameter_names, parameter_names))
    post_parameters = parse_transformation(parameters, {'gamma': 'gamma_specific', 'beta': 'beta_specific'})
    assert (post_parameters['gamma'] == 'gamma_specific')
    assert (post_parameters['alpha'] == 'alpha')
    assert (post_parameters['beta'] == 'beta_specific')
    assert (post_parameters['delta'] == 'delta')

    with pytest.raises(KeyError):
        parse_transformation(parameters, {'doesnotexist': 'yep'})


def test_parameters():
    params = Parameters()
    params._set_parameters(['alpha', 'beta', 'gamma'], [None]*3)
    assert (params['beta'].value == 0.0)

    params['beta'].value = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 0.0]))

    params._set_parameters(['alpha', 'beta', 'gamma', 'delta'], [None]*4)
    assert (params['beta'].value == 5.0)
    assert (np.allclose(params.values, [0.0, 5.0, 0.0, 0.0]))

    params['gamma'].value = 6.0
    params['delta'] = 7.0
    params['gamma'].lower_bound = -4.0
    params['gamma'].upper_bound = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 6.0, 7.0]))
    assert (np.allclose(params.lower_bounds, [-np.inf, -np.inf, -4.0, -np.inf]))
    assert (np.allclose(params.upper_bounds, [np.inf, np.inf, 5.0, np.inf]))

    assert(len(params) == 4.0)
    params._set_parameters(['alpha', 'beta', 'delta'], [None]*3)
    assert (np.allclose(params.values, [0.0, 5.0, 7.0]))
    assert ([p for p in params] == ['alpha', 'beta', 'delta'])
    assert(len(params) == 3.0)

    for i, p in params.items():
        p.value = 1.0

    assert (np.allclose(params.values, [1.0, 1.0, 1.0]))

    params = Parameters()
    params._set_parameters(['alpha', 'beta', 'gamma'], [Parameter(2), Parameter(3), Parameter(4)])
    params2 = Parameters()
    params2._set_parameters(['gamma', 'potato', 'beta'], [Parameter(10), Parameter(11), Parameter(12)])
    params << params2
    assert np.allclose(params.values, [2, 12, 10])

    params2 = Parameters()
    params2._set_parameters(['spaghetti'], [Parameter(10), Parameter(12)])
    with pytest.raises(RuntimeError):
        params << params2


def test_build_conditions():
    parameter_names = ['a', 'b', 'c']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    d1 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d2 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d3 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))

    assert generate_conditions([d1, d2, d3], parameter_lookup, parameter_names)

    # Tests whether we pick up when a parameter that's generated in a transformation doesn't actually exist in the
    # combined model
    d4 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, {'c': 'i_should_not_exist'}))
    with pytest.raises(AssertionError):
        assert generate_conditions([d1, d2, d4], parameter_lookup, parameter_names)

    # Tests whether we pick up on when a parameter exists in the model, but there's no transformation for it.
    d5 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    parameter_names = ['a', 'b', 'c', 'i_am_new']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    with pytest.raises(AssertionError):
        assert generate_conditions([d1, d2, d5], parameter_lookup, parameter_names)

    # Verify that the data gets linked up to the correct conditions
    d1 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d2 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d6 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, {'c': 'i_am_new'}))
    conditions, data_link = generate_conditions([d1, d2, d6], parameter_lookup, parameter_names)
    assert np.all(data_link[0] == [0, 1])
    assert np.all(data_link[1] == [2])

    # Test whether a parameter transformation to a value doesn't lead to an error
    d4 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, {'c': 5}))
    assert generate_conditions([d1, d2, d4], parameter_lookup, parameter_names)


def test_condition_struct():
    parameter_names = ['gamma', 'alpha', 'beta', 'delta', 'gamma_specific', 'beta_specific', 'zeta']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    parameter_trafos = parse_transformation(['gamma', 'alpha', 'beta', 'delta', 'zeta'],
                                            {'gamma': 'gamma_specific', 'delta': 5, 'beta': 'beta_specific'})
    parameter_vector = np.array([2, 4, 6, 8, 10, 12, 14])

    c = Condition(parameter_trafos, parameter_lookup)
    assert (np.all(c.p_local == [None, None, None, 5, None]))
    assert (np.allclose(parameter_vector[c.p_indices], [10, 4, 12, 14]))
    assert (np.all(c.p_external == np.array([0, 1, 2, 4])))
    assert (list(c.transformed) == ['gamma_specific', 'alpha', 'beta_specific', 5, 'zeta'])
    assert (np.allclose(c.get_local_parameters(parameter_vector), [10, 4, 12, 5, 14]))


def test_model_calls():
    def model_function(x, b, c, d):
        return b + c * x + d * x * x

    t = np.array([1.0, 2.0, 3.0])
    model = Model("m", model_function)
    y_ref = model._raw_call(t, [2.0, 3.0, 4.0])

    assert np.allclose(model(t, Parameters(m_a=Parameter(1), m_b=Parameter(2), m_c=Parameter(3), m_d=Parameter(4))),
                       y_ref)

    assert np.allclose(model(t, Parameters(m_d=Parameter(4), m_c=Parameter(3), m_b=Parameter(2))), y_ref)

    with pytest.raises(IndexError):
        assert np.allclose(model(t, Parameters(m_a=Parameter(1), m_b=Parameter(2), m_d=Parameter(4))), y_ref)


def test_unique_idx():
    uiq, inv = unique_idx(['str', 'str', 'hmm', 'potato', 'hmm', 'str'])
    assert(uiq == ['str', 'hmm', 'potato'])
    assert(inv == [0, 0, 1, 2, 1, 0])


def test_model_defaults():
    """Test whether model defaults propagate to the fit object correctly"""
    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    m = Model("M", g, f=Parameter(5))
    m._add_data("test", [1, 2, 3], [2, 3, 4])
    m._add_data("test2", [1, 2, 3], [2, 3, 4], params={'M_f': 'f_new'})
    f = Fit(m)
    f._build_fit()

    assert (f["M_a"].value == Parameter().value)
    assert (f.parameters["M_a"].value == Parameter().value)
    assert (f.parameters["f_new"].value == 5)
    assert (f.parameters["M_f"].value == 5)

    # Check whether each parameter is actually unique
    f.parameters["f_new"] = 6
    assert (f.parameters["f_new"].value == 6)
    assert (f.parameters["M_f"].value == 5)

    # Test whether providing a default for a parameter that doesn't exist throws
    with pytest.raises(AssertionError):
        Model("M", g, z=Parameter(5))

    # Verify that the defaults are in fact copies
    default = Parameter(5)
    m = Model("M", g, f=default)
    m._parameters["M_f"].value = 6
    assert default.value == 5


def test_model_build_status():
    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_f", "M_q"]

    m = Model("M", g, d=Parameter(4))
    m._add_data("test", [1, 2, 3], [2, 3, 4], {"M_c": 4})
    assert not m._built

    mock_fit_object = 1
    m._build_model(OrderedDict(zip(all_parameters, np.arange(len(all_parameters)))), mock_fit_object)
    assert m._built

    # Make sure that we detect invalidated builds
    assert m.built_against(mock_fit_object)
    assert not m.built_against(2)

    # Loading new data should invalidate the build
    m._add_data("test2", [1, 2, 3], [2, 3, 4], {'M_c': 5, 'M_f': 'f_new'})
    assert not m._built


def test_model_fit_object_linking():
    def fetch_parameters(keys, indices):
        p_list = list(f.parameters.keys)
        return [p_list[x] if x is not None else None for x in indices]

    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    def h(data, mu, e, q, c, r):
        return (data - mu) * 2

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_f", "M_q"]
    m = Model("M", g, d=Parameter(4))
    m2 = Model("M", h)
    m._add_data("test", [1, 2, 3], [2, 3, 4], {'M_c': 4})

    # Model should not be built
    f = Fit(m, m2)
    assert f.dirty

    # Asking for the parameters should have triggered a build
    f.parameters
    assert not f.dirty
    assert set(f.parameters.keys) == set(all_parameters)

    # Check the parameters included in the model
    assert np.allclose(f.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(f.parameters, f.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    # Loading data should make it dirty again
    m._add_data("test2", [1, 2, 3], [2, 3, 4], {'M_c': 4, 'M_e': 'M_e_new'})
    assert f.dirty

    # Check the parameters included in the model
    f._rebuild()
    assert np.allclose(f.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(f.parameters, f.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    assert np.allclose(f.models[0]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.models[0]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(f.parameters, f.models[0]._conditions[1]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e_new", "M_f", "M_q"]

    # Load data into model 2
    m2._add_data("test", [1, 2, 3], [2, 3, 4], {'M_c': 4, 'M_r': 6})
    assert f.dirty

    # Since M_r is set fixed in that model, it should not appear as a parameter
    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_e_new", "M_f", "M_q"]
    assert set(f.parameters.keys) == set(all_parameters)

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_e_new", "M_f", "M_q", "M_r"]
    m2._add_data("test", [1, 2, 3], [2, 3, 4], {'M_c': 4, 'M_e': 5})
    assert set(f.parameters.keys) == set(all_parameters)
    assert np.allclose(f.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(f.parameters, f.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    assert np.allclose(f.models[0]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.models[0]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(f.parameters, f.models[0]._conditions[1]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e_new", "M_f", "M_q"]

    assert np.allclose(f.models[1]._conditions[0].p_external, [0, 1, 2])
    assert np.all(f.models[1]._conditions[0].p_local == [None, None, None, 4, 6])
    assert fetch_parameters(f.parameters, f.models[1]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_e", "M_q", None, None]

    assert fetch_parameters(f.parameters, f.models[1]._conditions[1]._p_global_indices) == \
           ["M_mu", None, "M_q", None, "M_r"]


def test_jacobian_test_fit():
    def f(x, a, b):
        return a + b * x

    def f_jac(x, a, b):
        return np.vstack((np.ones((1, len(x))), x))

    def f_der(x, a, b):
        return b * np.ones((len(x)))

    def f_jac_wrong(x, a, b):
        return np.vstack((2.0*np.ones((1, len(x))), x))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a_true, b_true = (5.0, 5.0)
    f_data = f(x, a_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der)
    model._add_data("test", x, f_data)
    fit_object = Fit(model)
    fit_object.parameters["f_a"].value = a_true
    fit_object.parameters["f_b"].value = b_true
    assert fit_object.verify_jacobian(fit_object.parameters.values)

    model_bad = Model("f", f, jacobian=f_jac_wrong, derivative=f_der)
    model_bad._add_data("test", x, f_data)
    fit_object_bad = Fit(model_bad)
    fit_object_bad.parameters["f_a"].value = a_true
    fit_object_bad.parameters["f_b"].value = b_true
    assert not fit_object_bad.verify_jacobian(fit_object_bad.parameters.values)


def test_integration_test_fitting():
    def linear(x, a, b):
        f = a * x + b
        return f

    def linear_jac(x, a, b):
        jacobian = np.vstack((x, np.ones(len(x))))
        return jacobian

    def linear_jac_wrong(x, a, b):
        jacobian = np.vstack((np.ones(len(x)), x))
        return jacobian

    assert Model("M", linear, jacobian=linear_jac).has_jacobian
    assert not Model("M", linear).has_jacobian

    model = Model("M", linear, jacobian=linear_jac_wrong)
    assert not model.verify_jacobian([1, 2, 3], [1, 1])

    model = Model("M", linear, jacobian=linear_jac)
    x = np.arange(3)
    for i in np.arange(3):
        y = 4.0*x*i + 5.0
        model._add_data(f"test {i}", x, y, params={'M_a': f'slope_{i}'})
    fit = Fit(model)

    y = 4.0*x + 10.0
    model._add_data("test x", x, y, params={'M_a': 'slope_1', 'M_b': 'M_b_2'})

    # Test whether fixed parameters are not fitted
    fit["slope_2"].vary = False
    fit.fit()
    assert (np.isclose(fit["slope_2"].value, 0))

    fit["slope_2"].vary = True
    fit.fit()
    assert(len(fit.parameters.values) == 5)
    assert(len(fit.parameters) == 5)
    assert(fit.n_residuals == 12)
    assert(fit.n_parameters == 5)

    assert(np.isclose(fit.parameters["slope_0"].value, 0))
    assert(np.isclose(fit.parameters["slope_1"].value, 4))
    assert(np.isclose(fit.parameters["slope_2"].value, 8))
    assert(np.isclose(fit.parameters["M_b"].value, 5))
    assert(np.isclose(fit.parameters["M_b_2"].value, 10))

    # Verify that fixed parameters are correctly removed from sub-models
    model = Model("M", linear, jacobian=linear_jac)
    model._add_data("test1", x, 4.0*x + 5.0, {'M_a': 4})
    model._add_data("test2", x, 8.0*x + 10.0, {'M_b': 10})
    fit = Fit(model)
    fit.fit()
    assert (np.isclose(fit.parameters["M_b"].value, 5))
    assert (np.isclose(fit.parameters["M_a"].value, 8))


def test_models():
    independent = np.arange(0.15, 2, .25)
    parameters = [38.18281266, 0.37704827, 278.50103452, 4.11]
    assert(odijk("WLC").verify_jacobian(independent, parameters))
    assert(inverted_odijk("iWLC").verify_jacobian(independent, parameters, atol=1e-5))
    assert(freely_jointed_chain("FJC").verify_jacobian(independent, parameters, dx=1e-4, atol=1e-6))
    assert(marko_siggia_simplified("MS").verify_jacobian(independent, [5, 5, 4.11], atol=1e-6))

    assert(odijk("WLC").verify_derivative(independent, parameters))
    assert(inverted_odijk("iWLC").verify_derivative(independent, parameters))
    assert(freely_jointed_chain("FJC").verify_derivative(independent, parameters, atol=1e-6))
    assert(marko_siggia_simplified("MS").verify_derivative(independent, [5, 5, 4.11], atol=1e-6))

    assert(marko_siggia_ewlc_force("MSF").verify_jacobian(independent, parameters, dx=1e-4, rtol=1e-4))
    assert(marko_siggia_ewlc_distance("MSD").verify_jacobian(independent, parameters, dx=1e-4))
    assert(marko_siggia_ewlc_force("MSF").verify_derivative(independent, parameters, dx=1e-4))
    assert(marko_siggia_ewlc_distance("MSD").verify_derivative(independent, parameters, dx=1e-4))

    # The finite differencing version of the FJC performs very poorly numerically, hence the less stringent
    # tolerances and larger dx values.
    assert(inverted_freely_jointed_chain("iFJC").verify_derivative(independent, parameters, dx=1e-3, rtol=1e-2, atol=1e-6))
    assert(inverted_freely_jointed_chain("iFJC").verify_jacobian(independent, parameters, dx=1e-3, atol=1e-5, rtol=1e-2))

    # Check the tWLC and inverted tWLC model
    parameters = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert(twistable_wlc("tWLC").verify_jacobian(independent, parameters))
    assert(inverted_twistable_wlc("itWLC").verify_jacobian(independent, parameters))

    # Check whether the inverted models invert correctly
    from ..fitting.detail.model_implementation import WLC, invWLC, FJC, invFJC, tWLC, invtWLC

    d = np.array([3.0, 4.0])
    parameters = [5.0, 5.0, 5.0]
    assert (np.allclose(WLC(invWLC(d, *parameters), *parameters), d))
    parameters = [5.0, 15.0, 1.0, 4.11]
    assert (np.allclose(FJC(invFJC(independent, *parameters), *parameters), independent))
    parameters = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    assert(np.allclose(tWLC(invtWLC(independent, *parameters), *parameters), independent))

    d = np.arange(0.15, 2, .5)
    (Lp, Lc, St, kT) = (38.18281266, 0.37704827, 278.50103452, 4.11)
    parameters = [Lp, Lc, St, kT]
    m_fwd = marko_siggia_ewlc_force("fwd")
    m_bwd = marko_siggia_ewlc_distance("bwd")
    force = m_fwd._raw_call(d, parameters)
    assert np.allclose(m_bwd._raw_call(force, parameters), d)

    # Determine whether they actually fulfill the model
    lhs = (force*Lp/kT)
    rhs = 0.25 * (1.0 - (d/Lc) + (force/St))**(-2) - 0.25 + (d/Lc) - (force/St)
    assert np.allclose(lhs, rhs)


def test_model_composition():
    def f(x, a, b):
        return a + b * x

    def f_jac(x, a, b):
        return np.vstack((np.ones((1, len(x))), x))

    def f_jac_wrong(x, a, b):
        return np.vstack((np.zeros((1, len(x))), x))

    def g(x, a, d, b):
        return a - b * x + d * x * x

    def g_jac(x, a, d, b):
        return np.vstack((np.ones((1, len(x))), x * x, -x))

    def f_der(x, a, b):
        return b * np.ones((len(x)))

    def f_der_wrong(x, a, b):
        return np.ones((len(x)))

    def g_der(x, a, d, b):
        return - b * np.ones((len(x))) + 2.0 * d * x

    m1 = Model("M", f, dependent="x", jacobian=f_jac, derivative=f_der)
    m2 = Model("M", g, dependent="x", jacobian=g_jac, derivative=g_der)
    t = np.arange(0, 2, .5)

    # Check actual composition
    # (a + b * x) + a - b * x + d * x * x = 2 * a + d * x * x
    assert np.allclose((m1 + m2)._raw_call(t, np.array([1.0, 2.0, 3.0])), 2.0 + 3.0 * t * t), \
        "Model composition returns invalid function evaluation (parameter order issue?)"

    # Check correctness of the Jacobians and derivatives
    assert (m1 + m2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m1 + m2).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (m2 + m1).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m2 + m1).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (m2 + m1 + m2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m2 + m1 + m2).verify_derivative(t, [1.0, 2.0, 3.0])

    m1_wrong_jacobian = Model("M", f, dependent="x", jacobian=f_jac_wrong, derivative=f_der)
    assert not (m1_wrong_jacobian + m2).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)
    assert not (m2 + m1_wrong_jacobian).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(m1) + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(m1 + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(m1 + m2 + m1).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(m1) + m2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(m1 + m2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(m1 + m2 + m1).verify_derivative(t, [-1.0, 2.0, 3.0])

    m1_wrong_derivative = Model("M", f, dependent="x", jacobian=f_jac, derivative=f_der_wrong)
    assert not (InverseModel(m1_wrong_derivative) + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert not (InverseModel(m1_wrong_jacobian) + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert not (InverseModel(m1_wrong_derivative) + m2).verify_derivative(t, [-1.0, 2.0, 3.0])

    assert m1.subtract_independent_offset("d_offset").verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert m1.subtract_independent_offset("d_offset").verify_derivative(t, [-1.0, 2.0, 3.0])

    m1 = inverted_odijk("DNA").subtract_independent_offset("d_offset") + force_offset("f")
    m2 = (odijk("DNA") + distance_offset("DNA_d")).invert() + force_offset("f")
    t = np.array([.19, .2, .3])
    p1 = np.array([.1, 4.9e1, 3.8e-1, 2.1e2, 4.11, 1.5])
    p2 = np.array([4.9e1, 3.8e-1, 2.1e2, 4.11, .1, 1.5])
    assert np.allclose(m1._raw_call(t, p1), m2._raw_call(t, p2))

    # Check whether incompatible variables are found
    with pytest.raises(AssertionError):
        distance_offset("d") + force_offset("f")

    composite = (distance_offset("d") + odijk("DNA"))
    assert composite.dependent == "d"
    assert composite.independent == "f"

    inverted = composite.invert()
    assert inverted.dependent == "f"
    assert inverted.independent == "d"


def test_parameter_inversion():
    def f(x, a, b):
        return a + b * x

    def f_jac(x, a, b):
        return np.vstack((np.ones((1, len(x))), x))

    def g(x, a, d, b):
        return a - b * x + d * x * x

    def g_jac(x, a, d, b):
        return np.vstack((np.ones((1, len(x))), x * x, -x))

    def f_der(x, a, b):
        return b * np.ones((len(x)))

    def g_der(x, a, d, b):
        return - b * np.ones((len(x))) + 2.0 * d * x

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a_true = 5.0
    b_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_data = f(x, a_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der)
    model._add_data("test", x, f_data)
    fit_object = Fit(model)
    fit_object.parameters["f_a"].value = a_true
    fit_object.parameters["f_b"].value = 1.0
    assert np.allclose(parameter_trace(model, fit_object.parameters, 'f_b', x, f_data), b_true)

    a_true = 5.0
    b_true = 3.0
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_plus_g_data = f(x, a_true, b_true) + g(x, a_true, d_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der) + Model("f", g, jacobian=g_jac, derivative=g_der)
    model._add_data("test", x, f_data)
    fit_object = Fit(model)
    fit_object.parameters["f_a"].value = a_true
    fit_object.parameters["f_b"].value = b_true
    fit_object.parameters["f_d"].value = 1.0
    assert np.allclose(parameter_trace(model, fit_object.parameters, 'f_d', x, f_plus_g_data), d_true)


def test_uncertainty_analysis():
    x = np.arange(10)
    y = np.array([8.24869073, 7.77648717, 11.9436565, 14.85406276, 22.73081526, 20.39692261, 32.48962353, 31.4775862,
                  37.63807819, 40.50125925])

    def quad(x, a=1, b=1, c=1):
        return a * x * x + b * x + c

    def quad_jac(x, a=1, b=1, c=1):
        return np.vstack((x*x, x, np.ones(len(x))))

    def linear(x, a=1, b=1):
        f = a * x + b
        return f

    def linear_jac(x, a, b):
        J = np.vstack((x, np.ones(len(x))))
        return J

    linear_model = Model("linear", linear, jacobian=linear_jac)
    linear_model._add_data("test", x, y)
    linear_fit = Fit(linear_model).fit()
    model_quad = Model("quad", quad, jacobian=quad_jac)
    model_quad._add_data("test", x, y)
    quad_fit = Fit(model_quad).fit()

    assert np.allclose(linear_fit.cov, np.array([[0.06819348, -0.30687066], [-0.30687066,  1.94351415]]))
    assert np.allclose(quad_fit.cov, np.array([[0.00973206, -0.08758855,  0.11678473],
                                               [-0.08758855,  0.85058215, -1.33134597],
                                               [0.11678473, -1.33134597,  3.17654476]]))

    assert np.allclose(linear_fit.aic, 49.652690269143434)
    assert np.allclose(linear_fit.aicc, 51.36697598342915)
    assert np.allclose(linear_fit.bic, 50.25786045513153)

    assert np.allclose(quad_fit.aic, 50.74643780988533)
    assert np.allclose(quad_fit.aicc, 54.74643780988533)
    assert np.allclose(quad_fit.bic, 51.654193088867466)


def test_parameter_availability():
    x = np.arange(10)
    y = np.array([8.24869073, 7.77648717, 11.9436565, 14.85406276, 22.73081526, 20.39692261, 32.48962353, 31.4775862,
                  37.63807819, 40.50125925])

    def linear(x, a=1, b=1):
        f = a * x + b
        return f

    def linear_jac(x, a, b):
        J = np.vstack((x, np.ones(len(x))))
        return J

    linear_model = Model("linear", linear, jacobian=linear_jac)
    linear_fit = Fit(linear_model)

    with pytest.raises(IndexError):
        linear_fit.parameters["linear_a"]

    linear_model._add_data("test", x, y, {'linear_a': 5})
    linear_fit = Fit(linear_model)

    # Parameter linear_a is not actually a parameter in the fit object at this point (it was set to 5)
    with pytest.raises(IndexError):
        linear_fit.parameters["linear_a"]

    linear_model._add_data("test", x, y)
    assert linear_fit.parameters["linear_a"]


def test_data_loading():
    m = Model("M", lambda x, a: a*x)
    m._add_data("test", [1, np.nan, 3], [2, np.nan, 4])
    assert np.allclose(m._data[0].x, [1, 3])
    assert np.allclose(m._data[0].y, [2, 4])

    with pytest.raises(AssertionError):
        m._add_data("test2", [1, 3], [2, 4, 5])

    with pytest.raises(AssertionError):
        m._add_data("test3", [1, 3, 5], [2, 4])

    with pytest.raises(AssertionError):
        m._add_data("test4", [[1, 3, 5]], [[2, 4, 5]])


def test_parameter_slicing():
    # Tests whether parameters coming from a Fit can be sliced by a data handle,
    # i.e. fit.parameters[data_handle]

    def dummy(t, p1, p2, p3):
        return t * p1 + t * p2 * p2 + t * p3 * p3 * p3

    model = Model("dummy", dummy, p2=Parameter(2), p3=Parameter(3), p1=Parameter(1))
    fit = Fit(model)
    data_set = model._add_data("data1", [1, 1, 1], [1, 2, 3], {'dummy_p2': 'dummy_p2_b'})
    parameter_slice = fit.parameters[data_set]
    assert (parameter_slice["dummy_p1"].value == 1)
    assert (parameter_slice["dummy_p2"].value == 2)
    assert (parameter_slice["dummy_p3"].value == 3)

    data_set2 = model._add_data("data1", [1, 1, 1], [1, 2, 3], {'dummy_p2': 'dummy_p2_c'})
    fit.parameters["dummy_p2_c"] = 5
    parameter_slice = fit.parameters[data_set]
    assert (parameter_slice["dummy_p2"].value == 2)
    parameter_slice = fit.parameters[data_set2]
    assert (parameter_slice["dummy_p2"].value == 5)


def test_reprs():
    assert odijk('test').__repr__()
    assert inverted_odijk('test').__repr__()
    assert freely_jointed_chain('test').__repr__()
    assert marko_siggia_simplified('test').__repr__()
    assert marko_siggia_ewlc_force('test').__repr__()
    assert marko_siggia_ewlc_distance('test').__repr__()
    assert inverted_freely_jointed_chain('test').__repr__()
    assert (odijk('test') + distance_offset('test')).__repr__()
    assert (odijk('test') + distance_offset('test')).invert().__repr__()
    assert (odijk('test') + distance_offset('test')).subtract_independent_offset("test_offset").__repr__()

    assert odijk('test')._repr_html_()
    assert inverted_odijk('test')._repr_html_()
    assert freely_jointed_chain('test')._repr_html_()
    assert marko_siggia_simplified('test')._repr_html_()
    assert marko_siggia_ewlc_force('test')._repr_html_()
    assert marko_siggia_ewlc_distance('test')._repr_html_()
    assert inverted_freely_jointed_chain('test')._repr_html_()
    assert (odijk('test') + distance_offset('test'))._repr_html_()
    assert (odijk('test') + distance_offset('test')).invert()._repr_html_()
    assert (odijk('test') + distance_offset('test')).subtract_independent_offset("test_offset")._repr_html_()

    assert (force_offset("a_b_c") + force_offset("b_c_d")).invert()._repr_html_().find(r"b_{c\_d\_offset") > 0

    m = odijk('DNA')
    d1 = m._add_data("data_1", [1, 2, 3], [2, 3, 4])
    assert d1.__repr__() == 'FitData(data_1, N=3)'

    d2 = m._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {'DNA_Lc': 'DNA_Lc_2'})
    assert d2.__repr__() == 'FitData(dataset_2, N=3, Transformations: DNA_Lc → DNA_Lc_2)'

    f = Fit(m)
    assert f.__repr__()
    assert f._repr_html_()

    d3 = m._add_data("data_3", [1, 2, 3], [2, 3, 4], {'DNA_Lc': 5})
    assert f.__repr__()
    assert f._repr_html_()
