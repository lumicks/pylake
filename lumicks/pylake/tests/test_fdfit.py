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


#def test_fit_object():
#    assert(FitObject.parse_transformation(['blip', 'foo', 'blip'], blip=5) == [5, 'foo', 5])
#    assert(FitObject.parse_transformation(['blip', 'foo', 'blip'], foo='new_foo') == ['blip', 'new_foo', 'blip'])
#    assert(FitObject.parse_transformation(['blip', 'foo', 'foo', 'blip'], foo='new_foo') ==
#           ['blip', 'new_foo', 'new_foo', 'blip'])


def test_link_generation():
    #data_sets = Data()

    #FitObject.build_conditions()
    def _build_conditions(data_sets):
        pass;


def test_parameters():
    params = Parameters()
    params.set_parameters(['alpha', 'beta', 'gamma'])
    assert (params['beta'].value == 0)

    params['beta'].value = 5
    assert (np.allclose(params.values, [0, 5, 0]))

    params.set_parameters(['alpha', 'beta', 'gamma', 'delta'])
    assert (params['beta'].value == 5)
    assert (np.allclose(params.values, [0, 5, 0, 0]))

    params['gamma'].value = 6
    params['delta'].value = 7
    params['gamma'].lb = -4
    params['gamma'].ub = 5
    assert (np.allclose(params.values, [0, 5, 6, 7]))
    assert (np.allclose(params.lb, [-np.inf, -np.inf, -4, -np.inf]))
    assert (np.allclose(params.ub, [np.inf, np.inf, 5, np.inf]))
    assert(len(params) == 4)

    params.set_parameters(['alpha', 'beta', 'delta'])
    assert (np.allclose(params.values, [0, 5, 7]))
    assert(len(params) == 3)


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
