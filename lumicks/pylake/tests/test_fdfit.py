from lumicks.pylake.fdfit import FitObject, Parameter, Parameters, Condition, Data, Model, parse_transformation, \
    _generate_conditions, SubtractIndependentOffset
from lumicks.pylake.fdmodels import *
from collections import OrderedDict
import pytest
import numpy as np


def tests_fit_object():
    pars = ['blip', 'foo']
    parse_transformation(pars, foo='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))
    parse_transformation(pars, foo = 5) == OrderedDict((('blip', 'blip'), ('foo', 5)))

    with pytest.raises(KeyError):
        parse_transformation(pars, blap='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))


def test_parameters():
    params = Parameters()
    params.set_parameters(['alpha', 'beta', 'gamma'], [None]*3)
    assert (params['beta'].value == 0.0)

    params['beta'].value = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 0.0]))

    params.set_parameters(['alpha', 'beta', 'gamma', 'delta'], [None]*4)
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
    params.set_parameters(['alpha', 'beta', 'delta'], [None]*3)
    assert (np.allclose(params.values, [0.0, 5.0, 7.0]))
    assert ([p for p in params] == ['alpha', 'beta', 'delta'])
    assert(len(params) == 3.0)

    for i, p in params.items():
        p.value = 1.0

    assert (np.allclose(params.values, [1.0, 1.0, 1.0]))


def test_transformation_parser():
    parameter_names = ['gamma', 'alpha', 'beta', 'delta']
    parameters = OrderedDict(zip(parameter_names, parameter_names))
    post_parameters = parse_transformation(parameters, gamma='gamma_specific', beta='beta_specific')
    assert (post_parameters['gamma'] == 'gamma_specific')
    assert (post_parameters['alpha'] == 'alpha')
    assert (post_parameters['beta'] == 'beta_specific')
    assert (post_parameters['delta'] == 'delta')

    with pytest.raises(KeyError):
        parse_transformation(parameters, doesnotexist='yep')


def test_build_conditions():
    parameter_names = ['a', 'b', 'c']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    d1 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d2 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d3 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))

    assert _generate_conditions([d1, d2, d3], parameter_lookup, parameter_names)

    # Tests whether we pick up when a parameter that's generated in a transformation doesn't actually exist in the
    # combined model
    d4 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c='i_should_not_exist'))
    with pytest.raises(AssertionError):
        assert _generate_conditions([d1, d2, d4], parameter_lookup, parameter_names)

    # Tests whether we pick up on when a parameter exists in the model, but there's no transformation for it.
    d5 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    parameter_names = ['a', 'b', 'c', 'i_am_new']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    with pytest.raises(AssertionError):
        assert _generate_conditions([d1, d2, d5], parameter_lookup, parameter_names)

    # Verify that the data gets linked up to the correct conditions
    d1 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d2 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names))
    d6 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c='i_am_new'))
    conditions, data_link = _generate_conditions([d1, d2, d6], parameter_lookup, parameter_names)
    assert np.all(data_link[0] == [0, 1])
    assert np.all(data_link[1] == [2])

    # Test whether a parameter transformation to a value doesn't lead to an error
    d4 = Data("name", [1, 2, 3], [1, 2, 3], parse_transformation(parameter_names, c=5))
    assert _generate_conditions([d1, d2, d4], parameter_lookup, parameter_names)


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


def test_models():
    independent = np.arange(0.05, 2, .01)
    parameters = [5, 5, 5, 4.11]
    assert(Model("M", WLC, WLC_jac).verify_jacobian(independent, parameters))
    assert(Model("M", invWLC, invWLC_jac).verify_jacobian(independent, parameters))
    assert(Model("M", FJC, FJC_jac).verify_jacobian(independent, parameters, atol=1e-6))
    assert(Model("M", invFJC, invFJC_jac).verify_jacobian(independent, parameters, atol=1e-6))

    # Check the tWLC and inverted tWLC model
    parameters = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert(Model("M", tWLC, tWLC_jac).verify_jacobian(independent, parameters))
    assert (Model("M", invtWLC, invtWLC_jac).verify_jacobian(independent, parameters))

    # Check whether the inverted models invert correctly
    parameters = [5.0, 5.0, 5.0]
    assert (np.allclose(WLC(invWLC(3, *parameters), *parameters), 3))
    parameters = [5.0, 15.0, 1.0, 4.11]
    assert (np.allclose(FJC(invFJC(independent, *parameters), *parameters), independent))
    parameters = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    assert(np.allclose(tWLC(invtWLC(independent, *parameters), *parameters), independent))


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

def test_model_defaults():
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

    t = np.arange(0, 2, .1)

    # Check actual composition
    # (a + b * x) + a - b * x + d * x * x = 2 * a + d * x * x
    assert np.allclose((M1 + M2)(t, np.array([1.0, 2.0, 3.0])), 2.0 + 3.0 * t * t), \
        "Model composition returns invalid function evaluation (parameter order issue?)"

    # Check self-consistency of the Jacobians
    assert (M1 + M2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (M2 + M1).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (M2 + M1 + M2).verify_jacobian(t, [1.0, 2.0, 3.0])

    assert not (Model("M", f, f_jac_wrong, derivative=f_der) + M2).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)
    assert not (M2 + Model("M", f, f_jac_wrong, derivative=f_der)).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(Model("M", f, f_jac, derivative=f_der)) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(Model("M", f, f_jac, derivative=f_der) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(Model("M", f, f_jac, derivative=f_der) + M2 + M1).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)

    assert not (InverseModel(Model("M", f, f_jac, derivative=f_der_wrong)) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0],
                                                                                                 verbose=False)
    assert not (InverseModel(Model("M", f, f_jac_wrong, derivative=f_der)) + M2).verify_jacobian(t, [-1.0, 2.0, 3.0],
                                                                                                 verbose=False)

    M1 = SubtractIndependentOffset(force_model("DNA", "invWLC"), "d_offset") + force_model("f", "offset")
    M2 = InverseModel(force_model("DNA", "WLC") + force_model("DNA_d", "offset")) + force_model("f", "offset")
    t = np.array([.19, .2, .3])
    p1 = np.array([.1, 4.9e1, 3.8e-1, 2.1e2, 4.11, 1.5])
    p2 = np.array([4.9e1, 3.8e-1, 2.1e2, 4.11, .1, 1.5])
    assert np.allclose(M1(t, p1), M2(t, p2))