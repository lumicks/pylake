from ..fitting.parameters import Parameters, Parameter
from ..fitting.detail.utilities import parse_transformation

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
