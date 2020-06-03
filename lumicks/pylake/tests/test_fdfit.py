import pytest
import numpy as np

from collections import OrderedDict
from ..fitting.parameters import Params, Parameter
from ..fitting.detail.utilities import parse_transformation, unique_idx
from ..fitting.detail.link_functions import generate_conditions
from ..fitting.fitdata import Condition, FitData
from ..fitting.model import Model


def test_transformation_parser():
    pars = ['blip', 'foo']
    assert parse_transformation(pars, {'foo': 'new_foo'}) == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))
    assert parse_transformation(pars, {'foo': 5}) == OrderedDict((('blip', 'blip'), ('foo', 5)))

    with pytest.raises(KeyError):
        parse_transformation(pars, {'blap': 'new_foo'}) == OrderedDict((('blip', 'blip'), ('foo', 'new_foo')))

    param_names = ['gamma', 'alpha', 'beta', 'delta']
    params = OrderedDict(zip(param_names, param_names))
    post_params = parse_transformation(params, {'gamma': 'gamma_specific', 'beta': 'beta_specific'})
    assert (post_params['gamma'] == 'gamma_specific')
    assert (post_params['alpha'] == 'alpha')
    assert (post_params['beta'] == 'beta_specific')
    assert (post_params['delta'] == 'delta')

    with pytest.raises(KeyError):
        parse_transformation(params, {'doesnotexist': 'yep'})


def test_params():
    assert Parameter(5.0) == Parameter(5.0)
    assert not Parameter(5.0) == Parameter(4.0)
    assert Parameter(5.0, 0.0) == Parameter(5.0, 0.0)
    assert not Parameter(5.0, 0.0) == Parameter(5.0, 1.0)
    assert Parameter(5.0, 0.0, 1.0) == Parameter(5.0, 0.0, 1.0)
    assert not Parameter(5.0, 0.0, 1.0) == Parameter(5.0, 0.0, 2.0)
    assert Parameter(5.0, 0.0, 1.0, fixed=True) == Parameter(5.0, 0.0, 1.0, fixed=True)
    assert not Parameter(5.0, 0.0, 1.0, fixed=True) == Parameter(5.0, 0.0, 1.0, fixed=False)
    assert Parameter(5.0, 0.0, 1.0, unit="pN") == Parameter(5.0, 0.0, 1.0, unit="pN")
    assert not Parameter(5.0, 0.0, 1.0, unit="pN") == Parameter(5.0, 0.0, 1.0, unit="potatoes")

    assert Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}) == \
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})

    assert not Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}) == \
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(6.0)})

    with pytest.raises(RuntimeError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}).update_params(5)

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/a":"M/b"]

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/c"].value = 5

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/c"] = 5

    assert str(Params()) == "No parameters"
    assert str(Parameter(5.0, 0.0, 1.0, fixed=True)) == \
        f"lumicks.pylake.fdfit.Parameter(value: {5.0}, lower bound: {0.0}, upper bound: {1.0}, fixed: {True})"

    params = Params()
    params._set_params(['alpha', 'beta', 'gamma'], [None]*3)
    assert (params['beta'].value == 0.0)

    params['beta'].value = 5.0
    assert (np.allclose(params.values, [0.0, 5.0, 0.0]))

    params._set_params(['alpha', 'beta', 'gamma', 'delta'], [None]*4)
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
    params._set_params(['alpha', 'beta', 'delta'], [None]*3)
    assert (np.allclose(params.values, [0.0, 5.0, 7.0]))
    assert ([p for p in params] == ['alpha', 'beta', 'delta'])
    assert(len(params) == 3.0)

    for i, p in params.items():
        p.value = 1.0

    assert (np.allclose(params.values, [1.0, 1.0, 1.0]))

    params = Params()
    params._set_params(['alpha', 'beta', 'gamma'], [Parameter(2), Parameter(3), Parameter(4)])
    params2 = Params()
    params2._set_params(['gamma', 'potato', 'beta'], [Parameter(10), Parameter(11), Parameter(12)])
    params.update_params(params2)
    assert np.allclose(params.values, [2, 12, 10])

    params2 = Params()
    params2._set_params(['spaghetti'], [Parameter(10), Parameter(12)])
    with pytest.raises(RuntimeError):
        params.update_params(params2)


def test_build_conditions():
    param_names = ['a', 'b', 'c']
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    d1 = FitData("name1", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d2 = FitData("name2", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d3 = FitData("name3", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))

    assert generate_conditions({"name1": d1, "name2": d2, "name3": d3}, parameter_lookup, param_names)

    # Tests whether we pick up when a parameter that's generated in a transformation doesn't actually exist in the
    # combined model
    d4 = FitData("name4", [1, 2, 3], [1, 2, 3], parse_transformation(param_names, {'c': 'i_should_not_exist'}))
    with pytest.raises(AssertionError):
        assert generate_conditions({"name1": d1, "name2": d2, "name4": d4}, parameter_lookup, param_names)

    # Tests whether we pick up on when a parameter exists in the model, but there's no transformation for it.
    d5 = FitData("name5", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    param_names = ['a', 'b', 'c', 'i_am_new']
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    with pytest.raises(AssertionError):
        assert generate_conditions({"name1": d1, "name2": d2, "name5": d5}, parameter_lookup, param_names)

    # Verify that the data gets linked up to the correct conditions
    d1 = FitData("name1", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d2 = FitData("name2", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d6 = FitData("name6", [1, 2, 3], [1, 2, 3], parse_transformation(param_names, {'c': 'i_am_new'}))
    conditions, data_link = generate_conditions({"name1": d1, "name2": d2, "name3": d6}, parameter_lookup, param_names)
    assert np.all(data_link[0] == [d1, d2])
    assert np.all(data_link[1] == [d6])

    # Test whether a parameter transformation to a value doesn't lead to an error
    d4 = FitData("name4", [1, 2, 3], [1, 2, 3], parse_transformation(param_names, {'c': 5}))
    assert generate_conditions({"name1": d1, "name2": d2, "name3": d4}, parameter_lookup, param_names)


def test_condition_struct():
    param_names = ['gamma', 'alpha', 'beta', 'delta', 'gamma_specific', 'beta_specific', 'zeta']
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    param_trafos = parse_transformation(['gamma', 'alpha', 'beta', 'delta', 'zeta'],
                                            {'gamma': 'gamma_specific', 'delta': 5, 'beta': 'beta_specific'})
    param_vector = np.array([2, 4, 6, 8, 10, 12, 14])

    c = Condition(param_trafos, parameter_lookup)
    assert (np.all(c.p_local == [None, None, None, 5, None]))
    assert (np.allclose(param_vector[c.p_indices], [10, 4, 12, 14]))
    assert (np.all(c.p_external == np.array([0, 1, 2, 4])))
    assert (list(c.transformed) == ['gamma_specific', 'alpha', 'beta_specific', 5, 'zeta'])
    assert (np.allclose(c.get_local_params(param_vector), [10, 4, 12, 5, 14]))


def test_model_calls():
    def model_function(x, b, c, d):
        return b + c * x + d * x * x

    t = np.array([1.0, 2.0, 3.0])
    model = Model("m", model_function)
    y_ref = model._raw_call(t, [2.0, 3.0, 4.0])

    assert np.allclose(model(t, Params(**{"m/a": Parameter(1), "m/b": Parameter(2), "m/c": Parameter(3),
                                          "m/d": Parameter(4)})), y_ref)

    assert np.allclose(model(t, Params(**{"m/d": Parameter(4), "m/c": Parameter(3), "m/b": Parameter(2)})), y_ref)

    with pytest.raises(IndexError):
        assert np.allclose(model(t, Params(**{"m/a": Parameter(1), "m/b": Parameter(2), "m/d": Parameter(4)})), y_ref)


def test_unique_idx():
    uiq, inv = unique_idx(['str', 'str', 'hmm', 'potato', 'hmm', 'str'])
    assert(uiq == ['str', 'hmm', 'potato'])
    assert(inv == [0, 0, 1, 2, 1, 0])


def test_model_defaults():
    """Test whether model defaults propagate to the fit object correctly"""
    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    # Test whether providing a default for a parameter that doesn't exist throws
    with pytest.raises(AssertionError):
        Model("M", g, z=Parameter(5))

    # Verify that the defaults are in fact copies
    default = Parameter(5)
    m = Model("M", g, f=default)
    m._params["M/f"].value = 6
    assert default.value == 5


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
    with pytest.raises(RuntimeError):
        Model("M", linear).jacobian([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        Model("M", linear).derivative([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    model = Model("M", linear, jacobian=linear_jac_wrong)
    assert not model.verify_jacobian([1, 2, 3], [1, 1])
