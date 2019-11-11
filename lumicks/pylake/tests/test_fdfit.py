from lumicks.pylake.fdfit import FitObject, Parameters, Condition, Data, Model
from lumicks.pylake.fdmodels import *
from collections import OrderedDict
import pytest
import numpy as np


def tests_fit_object():
    pars = ['blip', 'foo']
    FitObject.parse_transformation(pars, foo='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))
    FitObject.parse_transformation(pars, foo = 5) == OrderedDict((('blip', 'blip'), ('foo', 5)))

    with pytest.raises(KeyError):
        FitObject.parse_transformation(pars, blap='new_foo') == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))


def test_parameters():
    params = Parameters()
    params.set_parameters(['alpha', 'beta', 'gamma'])
    assert (params['beta'].value == 0.0)

    params['beta'].value = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 0.0]))

    params.set_parameters(['alpha', 'beta', 'gamma', 'delta'])
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
    params.set_parameters(['alpha', 'beta', 'delta'])
    assert (np.allclose(params.values, [0.0, 5.0, 7.0]))
    assert ([p for p in params] == ['alpha', 'beta', 'delta'])
    assert(len(params) == 3.0)

    for i, p in params.items():
        p.value = 1.0

    assert (np.allclose(params.values, [1.0, 1.0, 1.0]))

def test_transformation_parser():
    parameter_names = ['gamma', 'alpha', 'beta', 'delta']
    parameters = OrderedDict(zip(parameter_names, parameter_names))
    post_parameters = FitObject.parse_transformation(parameters, gamma='gamma_specific', beta='beta_specific')
    assert (post_parameters['gamma'] == 'gamma_specific')
    assert (post_parameters['alpha'] == 'alpha')
    assert (post_parameters['beta'] == 'beta_specific')
    assert (post_parameters['delta'] == 'delta')

    with pytest.raises(KeyError):
        FitObject.parse_transformation(parameters, doesnotexist='yep')


def test_condition_struct():
    parameter_names = ['gamma', 'alpha', 'beta', 'delta', 'gamma_specific', 'beta_specific', 'zeta']
    parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))
    parameter_trafos = FitObject.parse_transformation(['gamma', 'alpha', 'beta', 'delta', 'zeta'],
                                                      gamma='gamma_specific', delta=5, beta='beta_specific')
    parameter_vector = np.array([2, 4, 6, 8, 10, 12, 14])

    c = Condition(parameter_trafos, parameter_lookup)
    assert (np.allclose(c.p_local, [0, 0, 0, 5, 0]))
    assert (np.allclose(parameter_vector[c.p_indices], [10, 4, 12, 14]))
    assert (np.all(c.p_external == np.array([True, True, True, False, True])))
    assert (list(c.transformed) == ['gamma_specific', 'alpha', 'beta_specific', 5, 'zeta'])
    assert (np.allclose(c.get_local_parameters(parameter_vector), [10, 4, 12, 5, 14]))


def test_models():
    independent = np.arange(.2, 1, .01)
    parameters = [5, 5, 5, 4.11]
    assert(Model(WLC, WLC_jac).verify_jacobian(independent, parameters))
    assert(Model(invWLC, invWLC_jac).verify_jacobian(independent, parameters))
    assert(Model(FJC, FJC_jac).verify_jacobian(independent, parameters))

    parameters = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert(Model(tWLC, tWLC_jac).verify_jacobian(independent, parameters))
    assert(np.allclose(WLC(invWLC(3, 5, 5, 5), 5, 5, 5), 3))


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

    assert Model(linear, linear_jac).has_jacobian
    assert not Model(linear).has_jacobian

    with pytest.raises(RuntimeError):
        model = Model(linear, linear_jac_wrong)
        model.verify_jacobian([1, 2, 3], [1, 1])

    model = Model(linear, linear_jac)
    fit = FitObject(model)
    x = np.arange(3)
    for i in np.arange(3):
        y = 4.0*x*i + 5.0
        fit.load_data(x, y, a=f"slope_{i}")

    y = 4.0*x + 10.0
    fit.load_data(x, y, a="slope_1", b="b_2")
    fit.fit()

    assert(len(fit.parameters.values) == 5)
    assert(len(fit.parameters) == 5)
    assert(fit.n_residuals == 12)
    assert(fit.n_parameters == 5)

    assert(np.isclose(fit.parameters["slope_0"].value, 0))
    assert(np.isclose(fit.parameters["slope_1"].value, 4))
    assert(np.isclose(fit.parameters["slope_2"].value, 8))
    assert(np.isclose(fit.parameters["b"].value, 5))
    assert(np.isclose(fit.parameters["b_2"].value, 10))
