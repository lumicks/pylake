import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup

from collections import OrderedDict
from lumicks.pylake.fitting.parameters import Params
from lumicks.pylake.fitting.detail.utilities import parse_transformation, unique_idx, escape_tex, latex_sqrt
from lumicks.pylake.fitting.detail.link_functions import generate_conditions
from lumicks.pylake.fitting.fitdata import Condition, FitData
from lumicks.pylake.fitting.model import Model, InverseModel
from lumicks.pylake.fitting.datasets import Datasets
from lumicks.pylake.fitting.fit import Fit, FdFit
from lumicks.pylake.fitting.detail.model_implementation import solve_cubic_wlc, invwlc_root_derivatives
from lumicks.pylake.fitting.models import *
from lumicks.pylake.fitting.parameter_trace import parameter_trace


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

    m = Model("M", g, f=Parameter(5))
    f = Fit(m)
    f._add_data("test", [1, 2, 3], [2, 3, 4])
    f._add_data("test2", [1, 2, 3], [2, 3, 4], params={'M/f': 'f/new'})
    f._build_fit()

    assert (f["M/a"].value == Parameter().value)
    assert (f.params["M/a"].value == Parameter().value)
    assert (f.params["f/new"].value == 5)
    assert (f.params["M/f"].value == 5)

    # Check whether each parameter is actually unique
    f.params["f/new"] = 6
    assert (f.params["f/new"].value == 6)
    assert (f.params["M/f"].value == 5)

    # Test whether providing a default for a parameter that doesn't exist throws
    with pytest.raises(AssertionError):
        Model("M", g, z=Parameter(5))

    # Verify that the defaults are in fact copies
    default = Parameter(5)
    m = Model("M", g, f=default)
    m._params["M/f"].value = 6
    assert default.value == 5


def test_datasets_build_status():
    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/f", "M/q"]

    m = Model("M", g)
    d = Datasets(m, 0)

    d._add_data("test", [1, 2, 3], [2, 3, 4], {"M/c": 4})
    assert not d.built

    d._link_data(OrderedDict(zip(all_params, np.arange(len(all_params)))))
    assert d.built

    # Loading new data should invalidate the build
    d._add_data("test2", [1, 2, 3], [2, 3, 4], {'M/c': 5, 'M/f': 'f/new'})
    assert not d.built


def test_model_fit_object_linking():
    def fetch_params(keys, indices):
        p_list = list(f.params.keys)
        return [p_list[x] if x is not None else None for x in indices]

    def g(data, mu, sig, a, b, c, d, e, f, q):
        return (data - mu) * 2

    def h(data, mu, e, q, c, r):
        return (data - mu) * 2

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/f", "M/q"]
    m = Model("M", g, d=Parameter(4))
    m2 = Model("M", h)

    # Model should not be built
    f = Fit(m, m2)
    f[m]._add_data("test", [1, 2, 3], [2, 3, 4], {'M/c': 4})
    assert f.dirty

    # Asking for the parameters should have triggered a build
    f.params
    assert not f.dirty
    assert set(f.params.keys) == set(all_params)

    # Check the parameters included in the model
    assert np.allclose(f.datasets[id(m)]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.datasets[id(m)]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_params(f.params, f.datasets[id(m)]._conditions[0]._p_global_indices) == \
           ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]

    # Loading data should make it dirty again
    f[m]._add_data("test2", [1, 2, 3], [2, 3, 4], {'M/c': 4, 'M/e': 'M/e_new'})
    assert f.dirty

    # Check the parameters included in the model
    f._rebuild()
    assert np.allclose(f.datasets[id(m)]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.datasets[id(m)]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_params(f.params, f.datasets[id(m)]._conditions[0]._p_global_indices) == \
           ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]

    assert np.allclose(f.datasets[id(m)]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.datasets[id(m)]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_params(f.params, f.datasets[id(m)]._conditions[1]._p_global_indices) == \
           ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e_new", "M/f", "M/q"]

    # Load data into model 2
    f[m2]._add_data("test", [1, 2, 3], [2, 3, 4], {'M/c': 4, 'M/r': 6})
    assert f.dirty

    # Since M/r is set fixed in that model, it should not appear as a parameter
    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/e_new", "M/f", "M/q"]
    assert set(f.params.keys) == set(all_params)

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/e_new", "M/f", "M/q", "M/r"]
    f[m2]._add_data("test2", [1, 2, 3], [2, 3, 4], {'M/c': 4, 'M/e': 5})
    assert set(f.params.keys) == set(all_params)
    assert np.allclose(f.datasets[id(m)]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.datasets[id(m)]._conditions[0].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_params(f.params, f.datasets[id(m)]._conditions[0]._p_global_indices) == \
           ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]

    assert np.allclose(f.datasets[id(m)]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(f.datasets[id(m)]._conditions[1].p_local == [None, None, None, None, 4, None, None, None, None])
    assert fetch_params(f.params, f.datasets[id(m)]._conditions[1]._p_global_indices) == \
           ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e_new", "M/f", "M/q"]

    assert np.allclose(f.datasets[id(m2)]._conditions[0].p_external, [0, 1, 2])
    assert np.all(f.datasets[id(m2)]._conditions[0].p_local == [None, None, None, 4, 6])
    assert fetch_params(f.params, f.datasets[id(m2)]._conditions[0]._p_global_indices) == \
           ["M/mu", "M/e", "M/q", None, None]

    assert fetch_params(f.params, f.datasets[id(m2)]._conditions[1]._p_global_indices) == \
           ["M/mu", None, "M/q", None, "M/r"]

    f.update_params(Params(**{"M/mu": 4, "M/sig": 6}))
    assert f["M/mu"].value == 4
    assert f["M/sig"].value == 6

    f2 = Fit(m)
    f2._add_data("test", [1, 2, 3], [2, 3, 4])
    f2["M/mu"].value = 12

    f.update_params(f2)
    assert f["M/mu"].value == 12

    with pytest.raises(RuntimeError):
        f.update_params(5)


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
    fit = Fit(model)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = b_true
    assert fit.verify_jacobian(fit.params.values)

    model_bad = Model("f", f, jacobian=f_jac_wrong, derivative=f_der)
    fit = Fit(model_bad)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = b_true
    assert not fit.verify_jacobian(fit.params.values)

    with pytest.raises(ValueError):
        assert (odijk("WLC").verify_jacobian([1.0, 2.0, 3.0], [1.0, 2.0]))

    with pytest.raises(ValueError):
        odijk("WLC").verify_derivative([1, 2, 3], [1, 2, 3])


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

    model = Model("M", linear, jacobian=linear_jac)
    fit = Fit(model)
    x = np.arange(3)
    for i in np.arange(3):
        y = 4.0*x*i + 5.0
        fit._add_data(f"test {i}", x, y, params={'M/a': f'slope_{i}'})

    y = 4.0*x + 10.0
    fit._add_data("test x", x, y, params={'M/a': 'slope_1', 'M/b': 'M/b_2'})

    # Test whether fixed parameters are not fitted
    fit["slope_2"].fixed = True
    fit.fit()
    assert (np.isclose(fit["slope_2"].value, 0))

    fit["slope_2"].fixed = False
    fit.fit()
    assert(len(fit.params.values) == 5)
    assert(len(fit.params) == 5)
    assert(fit.n_residuals == 12)
    assert(fit.n_params == 5)

    assert(np.isclose(fit.params["slope_0"].value, 0))
    assert(np.isclose(fit.params["slope_1"].value, 4))
    assert(np.isclose(fit.params["slope_2"].value, 8))
    assert(np.isclose(fit.params["M/b"].value, 5))
    assert(np.isclose(fit.params["M/b_2"].value, 10))

    # Verify that fixed parameters are correctly removed from sub-models
    model = Model("M", linear, jacobian=linear_jac)
    fit = Fit(model)
    fit._add_data("test1", x, 4.0*x + 5.0, {'M/a': 4})
    fit._add_data("test2", x, 8.0*x + 10.0, {'M/b': 10})
    fit.fit()
    assert (np.isclose(fit.params["M/b"].value, 5))
    assert (np.isclose(fit.params["M/a"].value, 8))

    fit["M/a"].upper_bound = 4
    fit["M/a"].value = 5
    with pytest.raises(ValueError):
        fit.fit()


def integration_test_parameter_linkage():
    """Verify that we estimate correctly across models."""

    def const(x, b):
        f = b * np.ones(x.shape)
        return f

    def const_jac(x, b):
        return np.ones((1, len(x)))

    x = np.arange(3)
    y1 = np.ones(3) * 2
    y2 = np.ones(3) * 4

    # No difference between the offsets for the two datasets (results in average of the two data sets)
    fit = FdFit(Model("M", const, jacobian=const_jac))
    fit.add_data("a", y1, x)
    fit.add_data("b", y2, x)
    fit.fit()
    assert fit["M/b"].value == 3

    # Both models have their own offset (correct estimates)
    m1 = Model("M1", const, jacobian=const_jac)
    m2 = Model("M2", const, jacobian=const_jac)
    fit = FdFit(m1, m2)
    fit[m1].add_data("a", y1, x)
    fit[m2].add_data("b", y2, x)
    fit.fit()
    assert fit["M1/b"].value == 2
    assert fit["M2/b"].value == 4

    # No difference between the offsets for the two datasets because we explicitly say so
    # (results in average of the two data sets)
    m1 = Model("M1", const, jacobian=const_jac)
    m2 = Model("M2", const, jacobian=const_jac)
    fit = FdFit(m1, m2)
    fit[m1].add_data("a", y1, x)
    fit[m2].add_data("b", y2, x, params={"M2/b": "M1/b"})
    fit.fit()
    assert fit["M1/b"].value == 3

    # Both models have their own offset (correct estimates)
    fit = FdFit(Model("M", const, jacobian=const_jac))
    fit.add_data("a", y1, x)
    fit.add_data("b", y2, x, params={"M/b": "M/b2"})
    fit.fit()
    assert fit["M/b"].value == 2
    assert fit["M/b2"].value == 4


def test_models():
    independent = np.arange(0.15, 2, .25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    assert(odijk("WLC").verify_jacobian(independent, params))
    assert(inverted_odijk("iWLC").verify_jacobian(independent, params, atol=1e-5))
    assert(freely_jointed_chain("FJC").verify_jacobian(independent, params, dx=1e-4, atol=1e-6))
    assert(marko_siggia_simplified("MS").verify_jacobian(independent, [5, 5, 4.11], atol=1e-6))
    assert(inverted_marko_siggia_simplified("iMS").verify_jacobian(independent, [38.18281266, 0.37704827, 4.11],
                                                                   atol=1e-5))

    assert(odijk("WLC").verify_derivative(independent, params))
    assert(inverted_odijk("iWLC").verify_derivative(independent, params))
    assert(freely_jointed_chain("FJC").verify_derivative(independent, params, atol=1e-6))
    assert(marko_siggia_simplified("MS").verify_derivative(independent, [5, 5, 4.11], atol=1e-6))

    assert(marko_siggia_ewlc_force("MSF").verify_jacobian(independent, params, dx=1e-4, rtol=1e-4))
    assert(marko_siggia_ewlc_distance("MSD").verify_jacobian(independent, params, dx=1e-4))
    assert(marko_siggia_ewlc_force("MSF").verify_derivative(independent, params, dx=1e-4))
    assert(marko_siggia_ewlc_distance("MSD").verify_derivative(independent, params, dx=1e-4))
    assert(inverted_marko_siggia_simplified("iMS").verify_derivative(independent, [38.18281266, 0.37704827, 4.11],
                                                                   atol=1e-5))

    # The finite differencing version of the FJC performs very poorly numerically, hence the less stringent
    # tolerances and larger dx values.
    assert(inverted_freely_jointed_chain("iFJC").verify_derivative(independent, params, dx=1e-3, rtol=1e-2, atol=1e-6))
    assert(inverted_freely_jointed_chain("iFJC").verify_jacobian(independent, params, dx=1e-3, atol=1e-2, rtol=1e-2))

    # Check the tWLC and inverted tWLC model
    params = [5, 5, 5, 3, 2, 1, 6, 4.11]
    assert(twistable_wlc("tWLC").verify_jacobian(independent, params))
    assert(inverted_twistable_wlc("itWLC").verify_jacobian(independent, params))

    # Check whether the twistable wlc model manipulates the data order
    assert np.allclose(twistable_wlc("tWLC")._raw_call(independent, params),
                       np.flip(twistable_wlc("tWLC")._raw_call(np.flip(independent), params)))

    # Check whether the inverse twistable wlc model manipulates the data order
    assert np.allclose(inverted_twistable_wlc("itWLC")._raw_call(independent, params),
                       np.flip(inverted_twistable_wlc("itWLC")._raw_call(np.flip(independent), params)))

    # Check whether the inverted models invert correctly
    from ..fitting.detail.model_implementation import WLC, invWLC, FJC, invFJC, tWLC, invtWLC

    d = np.array([3.0, 4.0])
    params = [5.0, 5.0, 5.0]
    assert (np.allclose(WLC(invWLC(d, *params), *params), d))
    params = [5.0, 15.0, 1.0, 4.11]
    assert (np.allclose(FJC(invFJC(independent, *params), *params), independent))
    params = [40.0, 16.0, 750.0, 440.0, -637.0, 17.0, 30.6, 4.11]
    assert(np.allclose(tWLC(invtWLC(independent, *params), *params), independent))

    d = np.arange(0.15, 2, .5)
    (Lp, Lc, St, kT) = (38.18281266, 0.37704827, 278.50103452, 4.11)
    params = [Lp, Lc, St, kT]
    m_fwd = marko_siggia_ewlc_force("fwd")
    m_bwd = marko_siggia_ewlc_distance("bwd")
    force = m_fwd._raw_call(d, params)
    assert np.allclose(m_bwd._raw_call(force, params), d)

    # Determine whether they actually fulfill the model
    lhs = (force*Lp/kT)
    rhs = 0.25 * (1.0 - (d/Lc) + (force/St))**(-2) - 0.25 + (d/Lc) - (force/St)
    assert np.allclose(lhs, rhs)

    # Test inverted simplified model
    d = np.arange(0.15, .377, .05)
    (Lp, Lc, kT) = (38.18281266, 0.37704827, 4.11)
    params = [Lp, Lc, kT]
    m_fwd = marko_siggia_simplified("fwd")
    m_bwd = inverted_marko_siggia_simplified("bwd")
    force = m_fwd._raw_call(d, params)
    assert np.allclose(m_bwd._raw_call(force, params), d)

    # This model is nonsense about the contour length, so warn the user about this.
    with pytest.warns(RuntimeWarning):
        m_fwd._raw_call(np.array([Lc+.1]), params)


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

    assert m1.subtract_independent_offset().verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert m1.subtract_independent_offset().verify_derivative(t, [-1.0, 2.0, 3.0])

    m1 = inverted_odijk("DNA").subtract_independent_offset() + force_offset("f")
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
    fit = Fit(model)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = 1.0
    assert np.allclose(parameter_trace(model, fit.params, 'f/b', x, f_data), b_true)

    a_true = 5.0
    b_true = 3.0
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_plus_g_data = f(x, a_true, b_true) + g(x, a_true, d_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der) + Model("f", g, jacobian=g_jac, derivative=g_der)
    fit = Fit(model)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = b_true
    fit.params["f/d"].value = 1.0
    assert np.allclose(parameter_trace(model, fit.params, 'f/d', x, f_plus_g_data), d_true)


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
    linear_fit = Fit(linear_model)
    linear_fit._add_data("test", x, y)
    linear_fit.fit()
    model_quad = Model("quad", quad, jacobian=quad_jac)
    quad_fit = Fit(model_quad)
    quad_fit._add_data("test", x, y)
    quad_fit.fit()

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
        linear_fit.params["linear/a"]

    linear_fit._add_data("test", x, y, {'linear/a': 5})
    linear_fit = Fit(linear_model)

    # Parameter linear_a is not actually a parameter in the fit object at this point (it was set to 5)
    with pytest.raises(IndexError):
        linear_fit.params["linear/a"]

    linear_fit._add_data("test", x, y)
    assert linear_fit.params["linear/a"]


def test_data_loading():
    m = Model("M", lambda x, a: a*x)
    fit = Fit(m)
    fit._add_data("test", [1, np.nan, 3], [2, np.nan, 4])
    assert np.allclose(fit[m].data["test"].x, [1, 3])
    assert np.allclose(fit[m].data["test"].y, [2, 4])
    assert np.allclose(fit[m].data["test"].independent, [1, 3])
    assert np.allclose(fit[m].data["test"].dependent, [2, 4])

    # Name must be unique
    with pytest.raises(KeyError):
        fit._add_data("test", [1, 3, 5], [2, 4, 5])

    with pytest.raises(AssertionError):
        fit._add_data("test2", [1, 3], [2, 4, 5])

    with pytest.raises(AssertionError):
        fit._add_data("test3", [1, 3, 5], [2, 4])

    with pytest.raises(AssertionError):
        fit._add_data("test4", [[1, 3, 5]], [[2, 4, 5]])


def test_parameter_access():
    m = inverted_odijk("DNA")
    fit = FdFit(m)
    data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    data2 = fit._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    assert fit[m]["RecA"] == fit.params[data1]
    assert fit[m]["RecA2"] == fit.params[data2]

    # Test the convenience accessor
    assert fit["RecA"] == fit.params[data1]
    assert fit["RecA2"] == fit.params[data2]

    m1 = inverted_odijk("DNA")
    m2 = inverted_odijk("Protein")
    fit = FdFit(m1, m2)

    # This should throw since we have multiple models.
    with pytest.raises(RuntimeError):
        data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    data1 = fit[m1]._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})
    data2 = fit[m2]._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"Protein/Lc": "Protein/Lc2"})

    with pytest.raises(KeyError):
        data3 = fit[m2]._add_data("RecA3", [1, 2, 3], [1, 2, 3], params={"Protein2/Lc": "Protein2/Lc2"})

    assert fit[m1]["RecA"] == fit.params[data1]
    assert fit[m2]["RecA2"] == fit.params[data2]

    # We can no longer use the convenience accessor.
    with pytest.raises(IndexError):
        assert fit["RecA"] == fit.params[data1]


def test_data_access():
    m = inverted_odijk("DNA")
    fit = FdFit(m)
    data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    data2 = fit._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    assert fit[m].data["RecA"] == data1
    assert fit[m].data["RecA2"] == data2

    # Test the convenience accessor
    assert fit.data["RecA"] == data1
    assert fit.data["RecA2"] == data2

    m1 = inverted_odijk("DNA")
    m2 = inverted_odijk("Protein")
    fit = FdFit(m1, m2)

    # This should throw since we have multiple models.
    with pytest.raises(RuntimeError):
        data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    data1 = fit[m1]._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})
    data2 = fit[m2]._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"Protein/Lc": "Protein/Lc2"})

    with pytest.raises(KeyError):
        data3 = fit[m2]._add_data("RecA3", [1, 2, 3], [1, 2, 3], params={"Protein2/Lc": "Protein2/Lc2"})

    assert fit[m1].data["RecA"] == data1
    assert fit[m2].data["RecA2"] == data2

    # We can no longer use the convenience accessor.
    with pytest.raises(RuntimeError):
        assert fit.data["RecA"] == data1


def test_parameter_slicing():
    # Tests whether parameters coming from a Fit can be sliced by a data handle,
    # i.e. fit.params[data_handle]

    def dummy(t, p1, p2, p3):
        return t * p1 + t * p2 * p2 + t * p3 * p3 * p3

    model = Model("dummy", dummy, p2=Parameter(2), p3=Parameter(3), p1=Parameter(1))
    fit = Fit(model)
    data_set = fit._add_data("data1", [1, 1, 1], [1, 2, 3], {'dummy/p2': 'dummy/p2_b'})
    parameter_slice = fit.params[data_set]
    assert (parameter_slice["dummy/p1"].value == 1)
    assert (parameter_slice["dummy/p2"].value == 2)
    assert (parameter_slice["dummy/p3"].value == 3)

    data_set2 = fit._add_data("data2", [1, 1, 1], [1, 2, 3], {'dummy/p2': 'dummy/p2_c'})
    fit.params["dummy/p2_c"] = 5
    parameter_slice = fit.params[data_set]
    assert (parameter_slice["dummy/p2"].value == 2)
    parameter_slice = fit.params[data_set2]
    assert (parameter_slice["dummy/p2"].value == 5)


def test_analytic_roots():
    a = np.array([0])
    b = np.array([-3])
    c = np.array([1])

    assert np.allclose(
        np.sort(np.roots(np.array([np.array(1.0), a, b, c], dtype=np.float64))),
        np.sort(
            np.array(
                [
                    solve_cubic_wlc(a, b, c, 0)[0],
                    solve_cubic_wlc(a, b, c, 1)[0],
                    solve_cubic_wlc(a, b, c, 2)[0],
                ],
            )
        ),
    )

    with pytest.raises(RuntimeError):
        solve_cubic_wlc(a, b, c, 3)

    def test_root_derivatives(root):
        dx = 1e-5
        ref_root = solve_cubic_wlc(a, b, c, root)
        da = (solve_cubic_wlc(a + dx, b, c, root) - ref_root) / dx
        db = (solve_cubic_wlc(a, b + dx, c, root) - ref_root) / dx
        dc = (solve_cubic_wlc(a, b, c + dx, root) - ref_root) / dx

        assert np.allclose(np.array(invwlc_root_derivatives(a, b, c, root)), np.array([da, db, dc]), atol=1e-5,
                           rtol=1e-5)

    test_root_derivatives(0)
    test_root_derivatives(1)
    test_root_derivatives(2)


def test_reprs():
    assert odijk('test').__repr__()
    assert inverted_odijk('test').__repr__()
    assert freely_jointed_chain('test').__repr__()
    assert marko_siggia_simplified('test').__repr__()
    assert marko_siggia_ewlc_force('test').__repr__()
    assert marko_siggia_ewlc_distance('test').__repr__()
    assert inverted_freely_jointed_chain('test').__repr__()
    assert twistable_wlc('test').__repr__()
    assert inverted_twistable_wlc('test').__repr__()
    assert (odijk('test') + distance_offset('test')).__repr__()
    assert (odijk('test') + distance_offset('test')).invert().__repr__()
    assert (odijk('test') + distance_offset('test')).subtract_independent_offset().__repr__()

    assert odijk('test')._repr_html_()
    assert inverted_odijk('test')._repr_html_()
    assert freely_jointed_chain('test')._repr_html_()
    assert marko_siggia_simplified('test')._repr_html_()
    assert marko_siggia_ewlc_force('test')._repr_html_()
    assert marko_siggia_ewlc_distance('test')._repr_html_()
    assert inverted_freely_jointed_chain('test')._repr_html_()
    assert twistable_wlc('test')._repr_html_()
    assert inverted_twistable_wlc('test')._repr_html_()
    assert (odijk('test') + distance_offset('test'))._repr_html_()
    assert (odijk('test') + distance_offset('test')).invert()._repr_html_()
    assert (odijk('test') + distance_offset('test')).subtract_independent_offset()._repr_html_()
    assert (force_offset("a_b_c") + force_offset("b_c_d")).invert()._repr_html_().find("offset_{b\\_c\\_d}") > 0

    m = odijk('DNA')
    fit = Fit(m)
    d1 = fit._add_data("data_1", [1, 2, 3], [2, 3, 4])
    assert d1.__repr__() == 'FitData(data_1, N=3)'

    d2 = fit._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {'DNA/Lc': 'DNA/Lc_2'})
    assert d2.__repr__() == 'FitData(dataset_2, N=3, Transformations: DNA/Lc → DNA/Lc_2)'

    f = Fit(m)
    assert f.__repr__()
    assert f._repr_html_()

    fit._add_data("data_3", [1, 2, 3], [2, 3, 4], {'DNA/Lc': 5})
    assert fit.__repr__()
    assert fit._repr_html_()

    m = inverted_odijk("DNA")
    fit = FdFit(m)
    fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    fit._add_data("RecA2", [1, 2, 3], [1, 2, 3])
    fit._add_data("RecA3", [1, 2, 3], [1, 2, 3])

    assert fit[m].__repr__() == "lumicks.pylake.FdDatasets(datasets={RecA, RecA2, RecA3}, N=9)"
    assert fit[m].__str__() == 'Data sets:\n- FitData(RecA, N=3)\n- FitData(RecA2, N=3)\n- FitData(RecA3, N=3)\n'
    assert fit[m]._repr_html_() == ('&ensp;&ensp;FitData(RecA, N=3)<br>\n'
                                    '&ensp;&ensp;FitData(RecA2, N=3)<br>\n' 
                                    '&ensp;&ensp;FitData(RecA3, N=3)<br>\n')

    assert fit.__repr__() == 'lumicks.pylake.FdFit(models={DNA}, N=9)'
    assert fit.__str__() == ('Fit\n  - Model: DNA\n  - Equation:\n'
                             '      f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-d))\n\n'
                             '  - Data sets:\n    - FitData(RecA, N=3)\n    - FitData(RecA2, N=3)\n    '
                             '- FitData(RecA3, N=3)\n\n  '
                             '- Fitted parameters:\n'
                             '    Name      Value  Unit      Fitted      Lower bound    Upper bound\n'
                             '    ------  -------  --------  --------  -------------  -------------\n'
                             '    DNA/Lp    40     [nm]      True                  0            100\n'
                             '    DNA/Lc    16     [micron]  True                  0            inf\n'
                             '    DNA/St  1500     [pN]      True                  0            inf\n'
                             '    kT         4.11  [pN*nm]   False                 0              8')


def test_fd_variable_order():
    # Fit takes dependent, then independent
    m = odijk("M")
    fit = Fit(m)
    fit._add_data("test", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test"].x, [1, 2, 3])
    assert np.allclose(fit[m].data["test"].y, [2, 3, 4])

    fit[m]._add_data("test2", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test2"].x, [1, 2, 3])
    assert np.allclose(fit[m].data["test2"].y, [2, 3, 4])

    m = inverted_odijk("M")
    fit = Fit(m)
    fit._add_data("test", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test"].x, [1, 2, 3])
    assert np.allclose(fit[m].data["test"].y, [2, 3, 4])

    fit[m]._add_data("test2", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test2"].x, [1, 2, 3])
    assert np.allclose(fit[m].data["test2"].y, [2, 3, 4])

    # FdFit always takes f, d and maps it to the correct values
    m = odijk("M")
    fit = FdFit(m)

    # Test the FdFit interface
    fit.add_data("test", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test"].x, [1, 2, 3])
    assert np.allclose(fit[m].data["test"].y, [2, 3, 4])

    # Test the FdDatasets interface
    fit[m].add_data("test2", [3, 4, 5], [4, 5, 6])
    assert np.allclose(fit[m].data["test2"].x, [3, 4, 5])
    assert np.allclose(fit[m].data["test2"].y, [4, 5, 6])

    m = inverted_odijk("M")
    fit = FdFit(m)
    fit.add_data("test", [1, 2, 3], [2, 3, 4])
    assert np.allclose(fit[m].data["test"].x, [2, 3, 4])
    assert np.allclose(fit[m].data["test"].y, [1, 2, 3])

    # Test the FdDatasets interface
    fit[m].add_data("test2", [3, 4, 5], [4, 5, 6])
    assert np.allclose(fit[m].data["test2"].x, [4, 5, 6])
    assert np.allclose(fit[m].data["test2"].y, [3, 4, 5])


def test_tex_replacement():
    assert escape_tex("DNA/Hi") == "Hi_{DNA}"
    assert escape_tex("DNA/Hi_There") == "Hi\\_There_{DNA}"
    assert escape_tex("DNA_model/Hi_There") == "Hi\\_There_{DNA\\_model}"
    assert escape_tex("Hi_There") == "Hi\\_There"
    assert latex_sqrt("test") == r"\sqrt{test}"


@cleanup
def test_plotting():
    m = odijk('DNA')
    m2 = odijk('protein')

    # Test single model plotting
    fit = Fit(m)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit.plot()
    fit.plot("data_1")
    with pytest.raises(AssertionError):
        fit.plot("non-existent-data")

    fit.plot(overrides={'DNA/Lc': 12})
    with pytest.raises(KeyError):
        fit.plot(overrides={'DNA/c': 12})

    fit.plot(overrides={'DNA/Lc': 12}, independent=np.arange(1.0, 5.0, 1.0))

    with pytest.raises(KeyError):
        fit[m2].plot()

    # Test multi-model plotting
    fit = Fit(m, m2)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {'DNA/Lc': 'DNA/Lc_2'})
    fit[m2]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m2]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {'protein/Lc': 'protein/Lc_2'})
    fit[m2]._add_data("dataset 3", [1, 2, 3], [2, 3, 4], {'protein/Lc': 'protein/Lc_2'})

    with pytest.raises(AssertionError):
        fit.plot()

    fit[m].plot()
    fit[m2].plot()
    fit[m].plot("data_1")

    with pytest.raises(AssertionError):
        fit.plot(m, "non-existent-data")

    fit[m2].plot("dataset 3")
    with pytest.raises(AssertionError):
        fit[m].plot("dataset 3")

    fit[m].plot(overrides={'DNA/Lc': 12})
    with pytest.raises(KeyError):
        fit[m].plot(overrides={'DNA/c': 12})

    independent = np.arange(0.15, 2, .25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    odijk("WLC").verify_jacobian(independent, params, plot=1)

    independent = np.arange(0.15, 2, .25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    fit = FdFit(odijk("WLC"))
    fit.add_data("dataset 3", [1, 2, 3], [2, 3, 4])
    plt.figure()
    fit.verify_jacobian(params, plot=1)

    # Test live fit plotting
    fit = Fit(m, m2)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {'DNA/Lc': 'DNA/Lc_2'})
    fit[m2]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit.fit(show_fit=True, max_nfev=1)


def test_interpolation_inversion():
    # Test interpolation
    m = odijk("Nucleosome").invert(independent_max=120.0, interpolate=True)
    parvec = [5.77336105517341, 7.014180463612673, 1500.0000064812095, 4.11]
    result = np.array([0.17843862, 0.18101283, 0.18364313, 0.18633117, 0.18907864])
    assert np.allclose(m._raw_call(np.arange(10, 250, 50) / 1000, parvec), result)
