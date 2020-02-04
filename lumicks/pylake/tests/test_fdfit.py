from ..fitting.parameters import Parameters, Parameter
from ..fitting.detail.utilities import parse_transformation
from ..fitting.fitdata import Condition, FitData
from ..fitting.detail.link_functions import generate_conditions

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
