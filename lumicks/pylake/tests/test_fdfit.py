from ..fitting.parameters import Parameters
from ..fitting.detail.utilities import parse_transformation
from ..fitting.detail.utilities import unique_idx
from ..fitting.fitdata import Condition, FitData
from ..fitting.detail.link_functions import generate_conditions
from ..fitting.fitobject import FitObject
from ..fitting.fdmodels import *
from ..fitting.detail.parameter_trace import parameter_trace

import numpy as np
from collections import OrderedDict
import pytest


def test_transformation_parser():
    pars = ['blip', 'foo']
    assert parse_transformation(pars, foo='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))
    assert parse_transformation(pars, foo = 5) == OrderedDict((('blip', 'blip'), ('foo', 5)))

    with pytest.raises(KeyError):
        parse_transformation(pars, blap='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))

    parameter_names = ['gamma', 'alpha', 'beta', 'delta']
    parameters = OrderedDict(zip(parameter_names, parameter_names))
    post_parameters = parse_transformation(parameters, gamma='gamma_specific', beta='beta_specific')
    assert (post_parameters['gamma'] == 'gamma_specific')
    assert (post_parameters['alpha'] == 'alpha')
    assert (post_parameters['beta'] == 'beta_specific')
    assert (post_parameters['delta'] == 'delta')

    with pytest.raises(KeyError):
        parse_transformation(parameters, doesnotexist='yep')


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
    params['gamma'].lb = -4.0
    params['gamma'].ub = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 6.0, 7.0]))
    assert (np.allclose(params.lb, [-np.inf, -np.inf, -4.0, -np.inf]))
    assert (np.allclose(params.ub, [np.inf, np.inf, 5.0, np.inf]))

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
    d4 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c='i_should_not_exist'))
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
    d6 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c='i_am_new'))
    conditions, data_link = generate_conditions([d1, d2, d6], parameter_lookup, parameter_names)
    assert np.all(data_link[0] == [0, 1])
    assert np.all(data_link[1] == [2])

    # Test whether a parameter transformation to a value doesn't lead to an error
    d4 = FitData("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c=5))
    assert generate_conditions([d1, d2, d4], parameter_lookup, parameter_names)


def test_condition_struct():
    parameter_names = ['gamma', 'alpha', 'beta', 'delta', 'gamma_specific', 'beta_specific', 'zeta']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    parameter_trafos = parse_transformation(['gamma', 'alpha', 'beta', 'delta', 'zeta'],
                                            gamma='gamma_specific', delta=5, beta='beta_specific')
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

    M = Model("M", g, f=Parameter(5))
    M.load_data([1, 2, 3], [2, 3, 4])
    M.load_data([1, 2, 3], [2, 3, 4], M_f='f_new')
    F = FitObject(M)
    F._build_fitobject()

    assert (F.parameters["M_a"].value == Parameter().value)
    assert (F.parameters["f_new"].value == 5)
    assert (F.parameters["M_f"].value == 5)

    # Check whether each parameter is actually unique
    F.parameters["f_new"] = 6
    assert (F.parameters["f_new"].value == 6)
    assert (F.parameters["M_f"].value == 5)


def test_model_build_status():
    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_f", "M_q"]

    M = Model("M", g, d=Parameter(4))
    M.load_data([1, 2, 3], [2, 3, 4], M_c=4)
    assert not M._built

    mock_fit_object = 1
    M._build_model(OrderedDict(zip(all_parameters, np.arange(len(all_parameters)))), mock_fit_object)
    assert M._built

    # Make sure that we detect invalidated builds
    assert M.built_against(mock_fit_object)
    assert not M.built_against(2)

    # Loading new data should invalidate the build
    M.load_data([1, 2, 3], [2, 3, 4], M_c=5, M_f='f_new')
    assert not M._built


def test_model_fit_object_linking():
    def fetch_parameters(keys, indices):
        p_list = list(F.parameters.keys)
        return [p_list[x] if x is not None else None for x in indices]

    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    def h(data, mu, e, q, c, r):
        return (data - mu) * 2

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_f", "M_q"]
    M = Model("M", g, d=Parameter(4))
    M2 = Model("M", h)
    M.load_data([1, 2, 3], [2, 3, 4], M_c=4)

    # Model should not be built
    F = FitObject(M, M2)
    assert F.dirty

    # Asking for the parameters should have triggered a build
    F.parameters
    assert not F.dirty
    assert set(F.parameters.keys) == set(all_parameters)

    # Check the parameters included in the model
    assert np.allclose(F.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(F.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(F.parameters, F.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    # Loading data should make it dirty again
    M.load_data([1, 2, 3], [2, 3, 4], M_c=4, M_e="M_e_new")
    assert F.dirty

    # Check the parameters included in the model
    F._rebuild()
    assert np.allclose(F.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(F.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(F.parameters, F.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    assert np.allclose(F.models[0]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(F.models[0]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(F.parameters, F.models[0]._conditions[1]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e_new", "M_f", "M_q"]

    # Load data into model 2
    M2.load_data([1, 2, 3], [2, 3, 4], M_c=4, M_r=6)
    assert F.dirty

    # Since M_r is set fixed in that model, it should not appear as a parameter
    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_e_new", "M_f", "M_q"]
    assert set(F.parameters.keys) == set(all_parameters)

    all_parameters = ["M_mu", "M_sig", "M_a", "M_b", "M_d", "M_e", "M_e_new", "M_f", "M_q", "M_r"]
    M2.load_data([1, 2, 3], [2, 3, 4], M_c=4, M_e=5)
    assert set(F.parameters.keys) == set(all_parameters)
    assert np.allclose(F.models[0]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(F.models[0]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(F.parameters, F.models[0]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e", "M_f", "M_q"]

    assert np.allclose(F.models[0]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(F.models[0]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_parameters(F.parameters, F.models[0]._conditions[1]._p_global_indices) == \
           ["M_mu", "M_sig", "M_a", "M_b", None, "M_d", "M_e_new", "M_f", "M_q"]

    assert np.allclose(F.models[1]._conditions[0].p_external, [0, 1, 2])
    assert np.all(F.models[1]._conditions[0].p_local == [None, None, None, 4, 6])
    assert fetch_parameters(F.parameters, F.models[1]._conditions[0]._p_global_indices) == \
           ["M_mu", "M_e", "M_q", None, None]

    assert fetch_parameters(F.parameters, F.models[1]._conditions[1]._p_global_indices) == \
           ["M_mu", None, "M_q", None, "M_r"]


def test_jacobian_test_fitobject():
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
    model = Model("f", f, f_jac, f_der)
    model.load_data(x, f_data)
    fit_object = FitObject(model)
    fit_object.parameters["f_a"].value = a_true
    fit_object.parameters["f_b"].value = b_true
    assert fit_object.verify_jacobian(fit_object.parameters.values)

    model_bad = Model("f", f, f_jac_wrong, f_der)
    model_bad.load_data(x, f_data)
    fit_object_bad = FitObject(model_bad)
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

    assert Model("M", linear, linear_jac).has_jacobian
    assert not Model("M", linear).has_jacobian

    model = Model("M", linear, linear_jac_wrong)
    assert not model.verify_jacobian([1, 2, 3], [1, 1])

    model = Model("M", linear, linear_jac)
    x = np.arange(3)
    for i in np.arange(3):
        y = 4.0*x*i + 5.0
        model.load_data(x, y, M_a=f"slope_{i}")
    fit = FitObject(model)

    y = 4.0*x + 10.0
    model.load_data(x, y, M_a="slope_1", M_b="M_b_2")
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
    model = Model("M", linear, linear_jac)
    model.load_data(x, 4.0*x + 5.0, M_a=4)
    model.load_data(x, 8.0*x + 10.0, M_b=10)
    fit = FitObject(model)
    fit.fit()
    assert (np.isclose(fit.parameters["M_b"].value, 5))
    assert (np.isclose(fit.parameters["M_a"].value, 8))


def test_models():
    independent = np.arange(0.15, 2, .5)
    parameters = [38.18281266, 0.37704827, 278.50103452, 4.11]
    assert(Model("M", WLC, WLC_jac).verify_jacobian(independent, parameters))
    assert(Model("M", invWLC, invWLC_jac).verify_jacobian(independent, parameters, atol=1e-5))
    assert(Model("M", FJC, FJC_jac).verify_jacobian(independent, parameters, dx=1e-4, atol=1e-6))
    assert (Model("M", Marko_Siggia, Marko_Siggia_jac).verify_jacobian(independent, [5, 5, 4.11], atol=1e-6))
    assert (Model("M", invWLC, invWLC_jac).verify_jacobian(independent, parameters, atol=1e-5))

    assert(force_model('M', 'WLC').verify_derivative(independent, parameters))
    assert(force_model('M', 'invWLC').verify_derivative(independent, parameters))
    assert(force_model('M', 'FJC').verify_derivative(independent, parameters, atol=1e-6))
    assert(force_model('M', 'Marko_Siggia_simplified').verify_derivative(independent, [5, 5, 4.11], atol=1e-6))

    assert(force_model('M', 'Marko_Siggia_eWLC_force').verify_jacobian(independent, parameters, dx=1e-4, rtol=1e-4))
    assert(force_model('M', 'Marko_Siggia_eWLC_distance').verify_jacobian(independent, parameters, dx=1e-4))
    assert(force_model('M', 'Marko_Siggia_eWLC_force').verify_derivative(independent, parameters, dx=1e-4))
    assert(force_model('M', 'Marko_Siggia_eWLC_distance').verify_derivative(independent, parameters, dx=1e-4))

    # The finite differencing version of the FJC performs very poorly numerically, hence the less stringent
    # tolerances and larger dx values.
    assert (force_model('M', 'invFJC').verify_derivative(independent, parameters, dx=1e-3, rtol=1e-2, atol=1e-6))
    assert(Model("M", invFJC, invFJC_jac).verify_jacobian(independent, parameters, dx=1e-3, atol=1e-5, rtol=1e-2))

    # Check the tWLC and inverted tWLC model
    parameters = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert(Model("M", tWLC, tWLC_jac).verify_jacobian(independent, parameters))
    assert (Model("M", invtWLC, invtWLC_jac).verify_jacobian(independent, parameters))

    # Check whether the inverted models invert correctly
    d = np.array([3.0, 4.0])
    parameters = [5.0, 5.0, 5.0]
    assert (np.allclose(WLC(invWLC(d, *parameters), *parameters), d))
    parameters = [5.0, 15.0, 1.0, 4.11]
    assert (np.allclose(FJC(invFJC(independent, *parameters), *parameters), independent))
    parameters = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    assert(np.allclose(tWLC(invtWLC(independent, *parameters), *parameters), independent))

    independent = np.arange(0.15, 2, .5)
    parameters = [38.18281266, 0.37704827, 278.50103452, 4.11]
    M_fwd = force_model('M', 'Marko_Siggia_eWLC_force')
    M_bwd = force_model('M', 'Marko_Siggia_eWLC_distance')
    assert np.allclose(M_bwd._raw_call(M_fwd._raw_call(independent, parameters), parameters), independent)

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

    M1 = Model("M", f, f_jac, derivative=f_der)
    M2 = Model("M", g, g_jac, derivative=g_der)
    t = np.arange(0, 2, .5)

    # Check actual composition
    # (a + b * x) + a - b * x + d * x * x = 2 * a + d * x * x
    assert np.allclose((M1 + M2)._raw_call(t, np.array([1.0, 2.0, 3.0])), 2.0 + 3.0 * t * t), \
        "Model composition returns invalid function evaluation (parameter order issue?)"

    # Check correctness of the Jacobians and derivatives
    assert (M1 + M2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (M1 + M2).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (M2 + M1).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (M2 + M1).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (M2 + M1 + M2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (M2 + M1 + M2).verify_derivative(t, [1.0, 2.0, 3.0])

    M1_wrong_jacobian = Model("M", f, f_jac_wrong, derivative=f_der)
    assert not (M1_wrong_jacobian + M2).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)
    assert not (M2 + M1_wrong_jacobian).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(M1) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(M1 + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(M1 + M2 + M1).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(M1) + M2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(M1 + M2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(M1 + M2 + M1).verify_derivative(t, [-1.0, 2.0, 3.0])

    M1_wrong_derivative = Model("M", f, f_jac, derivative=f_der_wrong)
    assert not (InverseModel(M1_wrong_derivative) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert not (InverseModel(M1_wrong_jacobian) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert not (InverseModel(M1_wrong_derivative) + M2).verify_derivative(t, [-1.0, 2.0, 3.0])

    assert M1.subtract_independent_offset("d_offset").verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert M1.subtract_independent_offset("d_offset").verify_derivative(t, [-1.0, 2.0, 3.0])

    M1 = force_model("DNA", "invWLC").subtract_independent_offset("d_offset") + force_model("f", "offset")
    M2 = InverseModel(force_model("DNA", "WLC") + force_model("DNA_d", "offset")) + force_model("f", "offset")
    t = np.array([.19, .2, .3])
    p1 = np.array([.1, 4.9e1, 3.8e-1, 2.1e2, 4.11, 1.5])
    p2 = np.array([4.9e1, 3.8e-1, 2.1e2, 4.11, .1, 1.5])
    assert np.allclose(M1._raw_call(t, p1), M2._raw_call(t, p2))


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
    model = Model("f", f, f_jac, f_der)
    model.load_data(x, f_data)
    fit_object = FitObject(model)
    fit_object.parameters["f_a"].value = a_true
    fit_object.parameters["f_b"].value = 1.0
    assert np.allclose(parameter_trace(model, fit_object.parameters, 'f_b', x, f_data), b_true)

    a_true = 5.0
    b_true = 3.0
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_plus_g_data = f(x, a_true, b_true) + g(x, a_true, d_true, b_true)
    model = Model("f", f, f_jac, f_der) + Model("f", g, g_jac, g_der)
    model.load_data(x, f_data)
    fit_object = FitObject(model)
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

    linear_model = Model("linear", linear, linear_jac)
    linear_model.load_data(x, y)
    linear_fit = FitObject(linear_model).fit()
    model_quad = Model("quad", quad, quad_jac)
    model_quad.load_data(x, y)
    quad_fit = FitObject(model_quad).fit()

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

    linear_model = Model("linear", linear, linear_jac)
    linear_fit = FitObject(linear_model)

    with pytest.raises(IndexError):
        linear_fit.parameters["linear_a"]

    linear_model.load_data(x, y, linear_a=5)
    linear_fit = FitObject(linear_model)

    # Parameter linear_a is not actually a parameter in the fit object at this point (it was set to 5)
    with pytest.raises(IndexError):
        linear_fit.parameters["linear_a"]

    linear_model.load_data(x, y)
    assert linear_fit.parameters["linear_a"]
