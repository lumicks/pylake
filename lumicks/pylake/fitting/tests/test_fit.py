import re
from collections import OrderedDict

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.fitting.fit import Fit, FdFit, Params, Datasets
from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.models import ewlc_odijk_force, ewlc_odijk_distance
from lumicks.pylake.fitting.parameters import Parameter


def test_model_defaults():
    """Test whether model defaults propagate to the fit object correctly"""

    def g(data, mu, sig, a, b, c, d, e, f, q):
        del sig, a, b, c, d, e, f, q
        return (data - mu) * 2

    model = Model("M", g, f=Parameter(5))
    fit = Fit(model)
    fit._add_data("test", [1, 2, 3], [2, 3, 4])
    fit._add_data("test2", [1, 2, 3], [2, 3, 4], params={"M/f": "f/new"})
    fit._build_fit()

    assert fit["M/a"].value == Parameter().value
    assert fit.params["M/a"].value == Parameter().value
    assert fit.params["f/new"].value == 5
    assert fit.params["M/f"].value == 5

    # Check whether each parameter is actually unique
    fit.params["f/new"] = 6
    assert fit.params["f/new"].value == 6
    assert fit.params["M/f"].value == 5

    with pytest.raises(
        KeyError,
        match=re.escape("Attempted to set default for parameter (z) which is not in the model"),
    ):
        Model("M", g, z=Parameter(5))

    # Verify that the defaults are in fact copies
    default = Parameter(5)
    model = Model("M", g, f=default)
    model._params["M/f"].value = 6
    assert default.value == 5


def test_bad_model_construction():
    with pytest.raises(TypeError, match="First argument must be a model name"):
        Model(5, 5)

    with pytest.raises(TypeError, match="Model must be a callable, got <class 'int'>"):
        Model("Ya", 5)

    with pytest.raises(TypeError, match="Jacobian must be a callable, got <class 'int'>"):
        Model("Ya", lambda x: x, jacobian=5)

    with pytest.raises(TypeError, match="Derivative must be a callable, got <class 'int'>"):
        Model("Ya", lambda x: x, derivative=5)


def test_datasets_build_status():
    def g(data, mu, sig, a, b, c, d, e, f, q):
        del sig, a, b, c, d, e, f, q
        return (data - mu) * 2

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/f", "M/q"]

    m = Model("M", g)
    data_set = Datasets(m, 0)

    data_set._add_data("test", [1, 2, 3], [2, 3, 4], {"M/c": 4})
    assert not data_set.built

    data_set._link_data(OrderedDict(zip(all_params, np.arange(len(all_params)))))
    assert data_set.built

    # Loading new data should invalidate the build
    data_set._add_data("test2", [1, 2, 3], [2, 3, 4], {"M/c": 5, "M/f": "f/new"})
    assert not data_set.built


def test_model_fit_object_linking():
    def fetch_params(parameters, indices):
        p_list = list(parameters.keys())
        return [p_list[x] if x is not None else None for x in indices]

    def g(data, mu, sig, a, b, c, d, e, f, q):
        del sig, a, b, c, d, e, f, q
        return (data - mu) * 2

    def h(data, mu, e, q, c, r):
        del e, q, c, r
        return (data - mu) * 2

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/f", "M/q"]
    m = Model("M", g, d=Parameter(4))
    m2 = Model("M", h)

    # Model should not be built
    fit = Fit(m, m2)
    fit[m]._add_data("test", [1, 2, 3], [2, 3, 4], {"M/c": 4})
    assert fit.dirty

    # Asking for the parameters should have triggered a build
    fit.params
    assert not fit.dirty
    assert set(fit.params.keys()) == set(all_params)

    # Check the parameters included in the model
    np.testing.assert_allclose(
        fit.datasets[m.uuid]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8]
    )
    assert np.all(
        fit.datasets[m.uuid]._conditions[0].p_local
        == [None, None, None, None, 4, None, None, None, None]
    )
    params = ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]
    assert fetch_params(fit.params, fit.datasets[m.uuid]._conditions[0]._p_global_indices) == params

    # Loading data should make it dirty again
    fit[m]._add_data("test2", [1, 2, 3], [2, 3, 4], {"M/c": 4, "M/e": "M/e_new"})
    assert fit.dirty

    # Check the parameters included in the model
    fit._rebuild()
    np.testing.assert_allclose(
        fit.datasets[m.uuid]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8]
    )
    assert np.all(
        fit.datasets[m.uuid]._conditions[0].p_local
        == [None, None, None, None, 4, None, None, None, None]
    )
    params = ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]
    assert fetch_params(fit.params, fit.datasets[m.uuid]._conditions[0]._p_global_indices) == params

    np.testing.assert_allclose(
        fit.datasets[m.uuid]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8]
    )
    assert np.all(
        fit.datasets[m.uuid]._conditions[1].p_local
        == [None, None, None, None, 4, None, None, None, None]
    )
    params = ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e_new", "M/f", "M/q"]
    assert fetch_params(fit.params, fit.datasets[m.uuid]._conditions[1]._p_global_indices) == params

    # Load data into model 2
    fit[m2]._add_data("test", [1, 2, 3], [2, 3, 4], {"M/c": 4, "M/r": 6})
    assert fit.dirty

    # Since M/r is set fixed in that model, it should not appear as a parameter
    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/e_new", "M/f", "M/q"]
    assert set(fit.params.keys()) == set(all_params)

    all_params = ["M/mu", "M/sig", "M/a", "M/b", "M/d", "M/e", "M/e_new", "M/f", "M/q", "M/r"]
    fit[m2]._add_data("test2", [1, 2, 3], [2, 3, 4], {"M/c": 4, "M/e": 5})
    assert set(fit.params.keys()) == set(all_params)
    np.testing.assert_allclose(
        fit.datasets[m.uuid]._conditions[0].p_external, [0, 1, 2, 3, 5, 6, 7, 8]
    )
    assert np.all(
        fit.datasets[m.uuid]._conditions[0].p_local
        == [None, None, None, None, 4, None, None, None, None]
    )
    params = ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e", "M/f", "M/q"]
    assert fetch_params(fit.params, fit.datasets[m.uuid]._conditions[0]._p_global_indices) == params

    np.testing.assert_allclose(
        fit.datasets[m.uuid]._conditions[1].p_external, [0, 1, 2, 3, 5, 6, 7, 8]
    )
    assert np.all(
        fit.datasets[m.uuid]._conditions[1].p_local
        == [None, None, None, None, 4, None, None, None, None]
    )
    params = ["M/mu", "M/sig", "M/a", "M/b", None, "M/d", "M/e_new", "M/f", "M/q"]
    assert fetch_params(fit.params, fit.datasets[m.uuid]._conditions[1]._p_global_indices) == params

    np.testing.assert_allclose(fit.datasets[m2.uuid]._conditions[0].p_external, [0, 1, 2])
    assert np.all(fit.datasets[m2.uuid]._conditions[0].p_local == [None, None, None, 4, 6])
    params = ["M/mu", "M/e", "M/q", None, None]
    assert (
        fetch_params(fit.params, fit.datasets[m2.uuid]._conditions[0]._p_global_indices) == params
    )

    params = ["M/mu", None, "M/q", None, "M/r"]
    assert (
        fetch_params(fit.params, fit.datasets[m2.uuid]._conditions[1]._p_global_indices) == params
    )

    fit.update_params(Params(**{"M/mu": 4, "M/sig": 6}))
    assert fit["M/mu"].value == 4
    assert fit["M/sig"].value == 6

    f2 = Fit(m)
    f2._add_data("test", [1, 2, 3], [2, 3, 4])
    f2["M/mu"].value = 12

    fit.update_params(f2)
    assert fit["M/mu"].value == 12

    with pytest.raises(RuntimeError):
        fit.update_params(5)  # noqa


def test_jacobian_test_fit():
    def f(independent, a, b):
        return a + b * independent

    def f_jac(independent, a, b):
        del a, b
        return np.vstack((np.ones((1, len(independent))), independent))

    def f_der(independent, a, b):
        del a
        return b * np.ones((len(independent)))

    def f_jac_wrong(independent, a, b):
        del a, b
        return np.vstack((2.0 * np.ones((1, len(independent))), independent))

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
        assert ewlc_odijk_distance("WLC").verify_jacobian([1.0, 2.0, 3.0], [1.0, 2.0])

    with pytest.raises(ValueError):
        ewlc_odijk_distance("WLC").verify_derivative([1, 2, 3], [1, 2, 3])


def test_integration_test_fitting():
    def linear(independent, a, b):
        f = a * independent + b
        return f

    def linear_jac(independent, a, b):
        del a, b
        jacobian = np.vstack((independent, np.ones(len(independent))))
        return jacobian

    def linear_jac_wrong(independent, a, b):
        del a, b
        jacobian = np.vstack((np.ones(len(independent)), independent))
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
        y = 4.0 * x * i + 5.0
        fit._add_data(f"test {i}", x, y, params={"M/a": f"slope_{i}"})

    y = 4.0 * x + 10.0
    fit._add_data("test x", x, y, params={"M/a": "slope_1", "M/b": "M/b_2"})

    # Test whether fixed parameters are not fitted
    fit["slope_2"].fixed = True
    fit.fit()
    assert np.isclose(fit["slope_2"].value, 0)

    fit["slope_2"].fixed = False
    fit.fit()
    assert len(fit.params.values) == 5
    assert len(fit.params) == 5
    assert fit.n_residuals == 12
    assert fit.n_params == 5

    assert np.isclose(fit.params["slope_0"].value, 0)
    assert np.isclose(fit.params["slope_1"].value, 4)
    assert np.isclose(fit.params["slope_2"].value, 8)
    assert np.isclose(fit.params["M/b"].value, 5)
    assert np.isclose(fit.params["M/b_2"].value, 10)

    # Verify that fixed parameters are correctly removed from sub-models
    model = Model("M", linear, jacobian=linear_jac)
    fit = Fit(model)
    fit._add_data("test1", x, 4.0 * x + 5.0, {"M/a": 4})
    fit._add_data("test2", x, 8.0 * x + 10.0, {"M/b": 10})
    fit.fit()
    assert np.isclose(fit.params["M/b"].value, 5)
    assert np.isclose(fit.params["M/a"].value, 8)

    fit["M/a"].upper_bound = 4
    fit["M/a"].value = 5
    with pytest.raises(ValueError):
        fit.fit()


def test_integration_parameter_linkage():
    """Verify that we estimate correctly across models when models share parameters."""

    def const(independent, b):
        f = b * np.ones(independent.shape)
        return f

    def const_jac(independent, b):
        del b
        return np.ones((1, len(independent)))

    x = np.arange(3)
    y1 = np.ones(3) * 2
    y2 = np.ones(3) * 4

    # No difference between the offsets for the two datasets (results in average of the two data
    # sets)
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


def test_parameter_availability():
    x = np.arange(10)
    y = np.array(
        [
            8.24869073,
            7.77648717,
            11.9436565,
            14.85406276,
            22.73081526,
            20.39692261,
            32.48962353,
            31.4775862,
            37.63807819,
            40.50125925,
        ]
    )

    def linear(independent, a=1, b=1):
        return a * independent + b

    def linear_jac(independent, a, b):
        del a, b
        return np.vstack((independent, np.ones(len(independent))))

    linear_model = Model("linear", linear, jacobian=linear_jac)
    linear_fit = Fit(linear_model)

    with pytest.raises(IndexError):
        linear_fit.params["linear/a"]

    linear_fit._add_data("test", x, y, {"linear/a": 5})
    linear_fit = Fit(linear_model)

    # Parameter linear_a is not actually a parameter in the fit object at this point (it was set
    # to 5)
    with pytest.raises(IndexError):
        linear_fit.params["linear/a"]

    linear_fit._add_data("test", x, y)
    assert "linear/a" in linear_fit.params


def test_data_loading():
    m = Model("M", lambda x, a: a * x)
    fit = Fit(m)

    with pytest.raises(RuntimeError, match="This model has no data associated with it."):
        fit.fit()

    fit._add_data("test", [1, np.nan, 3], [2, np.nan, 4])
    np.testing.assert_allclose(fit[m].data["test"].x, [1, 3])
    np.testing.assert_allclose(fit[m].data["test"].y, [2, 4])
    np.testing.assert_allclose(fit[m].data["test"].independent, [1, 3])
    np.testing.assert_allclose(fit[m].data["test"].dependent, [2, 4])

    # Name must be unique
    with pytest.raises(KeyError):
        fit._add_data("test", [1, 3, 5], [2, 4, 5])

    for x, y in (([1, 3], [2, 4, 5]), ([1, 3, 5], [2, 4])):
        with pytest.raises(
            ValueError,
            match="Every value for the independent variable x should have a corresponding data "
            "point for the dependent variable y",
        ):
            fit._add_data("test2", x, y)

    with pytest.raises(
        ValueError,
        match=re.escape("Independent variable x should be one dimensional, but has shape (1, 3)"),
    ):
        fit._add_data("test4", [[1, 3, 5]], [2, 4, 5])

    with pytest.raises(
        ValueError,
        match=re.escape("Dependent variable y should be one dimensional, but has shape (1, 3)"),
    ):
        fit._add_data("test4", [1, 3, 5], [[2, 4, 5]])


def test_no_free_parameters():
    fit = FdFit(ewlc_odijk_force("DNA"))
    fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    for pars in ("DNA/Lp", "DNA/Lc", "DNA/St"):
        fit[pars].fixed = True

    with pytest.raises(RuntimeError, match="This model has no free parameters"):
        fit.fit()


def test_parameter_access():
    m = ewlc_odijk_force("DNA")
    fit = FdFit(m)
    data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    data2 = fit._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    assert fit[m]["RecA"] == fit.params[data1]
    assert fit[m]["RecA2"] == fit.params[data2]

    # Test the convenience accessor
    assert fit["RecA"] == fit.params[data1]
    assert fit["RecA2"] == fit.params[data2]

    m1 = ewlc_odijk_force("DNA")
    m2 = ewlc_odijk_force("Protein")
    fit = FdFit(m1, m2)

    # This should throw since we have multiple models.
    with pytest.raises(RuntimeError):
        fit._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    data1 = fit[m1]._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})
    data2 = fit[m2]._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"Protein/Lc": "Protein/Lc2"})

    with pytest.raises(KeyError):
        fit[m2]._add_data("RecA3", [1, 2, 3], [1, 2, 3], params={"Protein2/Lc": "Protein2/Lc2"})

    assert fit[m1]["RecA"] == fit.params[data1]
    assert fit[m2]["RecA2"] == fit.params[data2]

    # We can no longer use the convenience accessor.
    with pytest.raises(IndexError):
        assert fit["RecA"] == fit.params[data1]


def test_data_access():
    m = ewlc_odijk_force("DNA")
    fit = FdFit(m)
    data1 = fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    data2 = fit._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    assert fit[m].data["RecA"] == data1
    assert fit[m].data["RecA2"] == data2

    # Test the convenience accessor
    assert fit.data["RecA"] == data1
    assert fit.data["RecA2"] == data2

    m1 = ewlc_odijk_force("DNA")
    m2 = ewlc_odijk_force("Protein")
    fit = FdFit(m1, m2)

    # This should throw since we have multiple models.
    with pytest.raises(RuntimeError):
        fit._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})

    data1 = fit[m1]._add_data("RecA", [1, 2, 3], [1, 2, 3], params={"DNA/Lc": "DNA/Lc2"})
    data2 = fit[m2]._add_data("RecA2", [1, 2, 3], [1, 2, 3], params={"Protein/Lc": "Protein/Lc2"})

    with pytest.raises(KeyError):
        fit[m2]._add_data("RecA3", [1, 2, 3], [1, 2, 3], params={"Protein2/Lc": "Protein2/Lc2"})

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
    data_set = fit._add_data("data1", [1, 1, 1], [1, 2, 3], {"dummy/p2": "dummy/p2_b"})
    parameter_slice = fit.params[data_set]
    assert parameter_slice["dummy/p1"].value == 1
    assert parameter_slice["dummy/p2"].value == 2
    assert parameter_slice["dummy/p3"].value == 3

    data_set2 = fit._add_data("data2", [1, 1, 1], [1, 2, 3], {"dummy/p2": "dummy/p2_c"})
    fit.params["dummy/p2_c"] = 5
    parameter_slice = fit.params[data_set]
    assert parameter_slice["dummy/p2"].value == 2
    parameter_slice = fit.params[data_set2]
    assert parameter_slice["dummy/p2"].value == 5


def test_fd_variable_order():
    # Fit takes dependent, then independent
    m = ewlc_odijk_distance("M")
    fit = Fit(m)
    fit._add_data("test", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test"].x, [1, 2, 3])
    np.testing.assert_allclose(fit[m].data["test"].y, [2, 3, 4])

    fit[m]._add_data("test2", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test2"].x, [1, 2, 3])
    np.testing.assert_allclose(fit[m].data["test2"].y, [2, 3, 4])

    m = ewlc_odijk_force("M")
    fit = Fit(m)
    fit._add_data("test", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test"].x, [1, 2, 3])
    np.testing.assert_allclose(fit[m].data["test"].y, [2, 3, 4])

    fit[m]._add_data("test2", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test2"].x, [1, 2, 3])
    np.testing.assert_allclose(fit[m].data["test2"].y, [2, 3, 4])

    # FdFit always takes f, d and maps it to the correct values
    m = ewlc_odijk_distance("M")
    fit = FdFit(m)

    # Test the FdFit interface
    fit.add_data("test", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test"].x, [1, 2, 3])
    np.testing.assert_allclose(fit[m].data["test"].y, [2, 3, 4])

    # Test the FdDatasets interface
    fit[m].add_data("test2", [3, 4, 5], [4, 5, 6])
    np.testing.assert_allclose(fit[m].data["test2"].x, [3, 4, 5])
    np.testing.assert_allclose(fit[m].data["test2"].y, [4, 5, 6])

    m = ewlc_odijk_force("M")
    fit = FdFit(m)
    fit.add_data("test", [1, 2, 3], [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test"].x, [2, 3, 4])
    np.testing.assert_allclose(fit[m].data["test"].y, [1, 2, 3])

    # Test the FdDatasets interface
    fit[m].add_data("test2", [3, 4, 5], [4, 5, 6])
    np.testing.assert_allclose(fit[m].data["test2"].x, [4, 5, 6])
    np.testing.assert_allclose(fit[m].data["test2"].y, [3, 4, 5])


def test_plotting():
    m = ewlc_odijk_distance("DNA")
    m2 = ewlc_odijk_distance("protein")

    # Test single model plotting
    fit = Fit(m)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit.plot()
    fit.plot("data_1")
    with pytest.raises(KeyError, match="Did not find dataset with name non-existent-data"):
        fit.plot("non-existent-data")

    fit.plot(overrides={"DNA/Lc": 12})
    with pytest.raises(KeyError):
        fit.plot(overrides={"DNA/c": 12})

    fit.plot(overrides={"DNA/Lc": 12}, independent=np.arange(1.0, 5.0, 1.0))

    with pytest.raises(KeyError):
        fit[m2].plot()

    # Test multi-model plotting
    fit = Fit(m, m2)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {"DNA/Lc": "DNA/Lc_2"})
    fit[m2]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m2]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {"protein/Lc": "protein/Lc_2"})
    fit[m2]._add_data("dataset 3", [1, 2, 3], [2, 3, 4], {"protein/Lc": "protein/Lc_2"})

    with pytest.raises(
        RuntimeError, match=re.escape("Please select a model to plot using fit[model].plot(...)")
    ):
        fit.plot()

    fit[m].plot()
    fit[m2].plot()
    fit[m].plot("data_1")

    with pytest.raises(KeyError, match="Did not find dataset with name non-existent-data"):
        fit[m].plot("non-existent-data")

    fit[m2].plot("dataset 3")
    with pytest.raises(KeyError, match="Did not find dataset with name dataset 3"):
        fit[m].plot("dataset 3")

    fit[m].plot(overrides={"DNA/Lc": 12})
    with pytest.raises(KeyError):
        fit[m].plot(overrides={"DNA/c": 12})

    plt.close("all")
    independent = np.arange(0.15, 2, 0.25)
    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    ewlc_odijk_distance("WLC").verify_jacobian(independent, params, plot=1)

    params = [38.18281266, 0.37704827, 278.50103452, 4.11]
    fit = FdFit(ewlc_odijk_distance("WLC"))
    fit.add_data("dataset 3", [1, 2, 3], [2, 3, 4])
    plt.figure()
    fit.verify_jacobian(params, plot=1)

    # Test live fit plotting
    fit = Fit(m, m2)
    fit[m]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit[m]._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {"DNA/Lc": "DNA/Lc_2"})
    fit[m2]._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit.fit(show_fit=True, max_nfev=1)


def test_fit_reprs():
    m = ewlc_odijk_distance("DNA")
    fit = Fit(m)
    d1 = fit._add_data("data_1", [1, 2, 3], [2, 3, 4])
    assert d1.__repr__() == "FitData(data_1, N=3)"

    d2 = fit._add_data("dataset_2", [1, 2, 3], [2, 3, 4], {"DNA/Lc": "DNA/Lc_2"})
    assert d2.__repr__() == "FitData(dataset_2, N=3, Transformations: DNA/Lc → DNA/Lc_2)"

    f = Fit(m)
    assert f.__repr__()
    assert f._repr_html_()

    fit._add_data("data_3", [1, 2, 3], [2, 3, 4], {"DNA/Lc": 5})
    assert fit.__repr__()
    assert fit._repr_html_()

    m = ewlc_odijk_force("DNA")
    fit = FdFit(m)
    fit._add_data("RecA", [1, 2, 3], [1, 2, 3])
    fit._add_data("RecA2", [1, 2, 3], [1, 2, 3])
    fit._add_data("RecA3", [1, 2, 3], [1, 2, 3])

    assert fit[m].__repr__() == "lumicks.pylake.FdDatasets(datasets={RecA, RecA2, RecA3}, N=9)"
    assert (
        fit[m].__str__()
        == "Data sets:\n- FitData(RecA, N=3)\n- FitData(RecA2, N=3)\n- FitData(RecA3, N=3)\n"
    )
    assert fit[m]._repr_html_() == (
        "&ensp;&ensp;FitData(RecA, N=3)<br>\n"
        "&ensp;&ensp;FitData(RecA2, N=3)<br>\n"
        "&ensp;&ensp;FitData(RecA3, N=3)<br>\n"
    )

    assert fit.__repr__() == "lumicks.pylake.FdFit(models={DNA}, N=9)"
    assert fit.__str__() == (
        "Fit\n  - Model: DNA\n  - Equation:\n"
        "      f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-d))\n\n"
        "  - Data sets:\n    - FitData(RecA, N=3)\n    - FitData(RecA2, N=3)\n    "
        "- FitData(RecA3, N=3)\n\n  "
        "- Fitted parameters:\n"
        "    Name      Value  Unit      Fitted      Lower bound    Upper bound\n"
        "    ------  -------  --------  --------  -------------  -------------\n"
        "    DNA/Lp    40     [nm]      True            0.001              100\n"
        "    DNA/Lc    16     [micron]  True            0.00034            inf\n"
        "    DNA/St  1500     [pN]      True            1                  inf\n"
        "    kT         4.11  [pN*nm]   False           3.77                 8"
    )


def test_custom_legend_labels():
    """Test whether users can provide a custom label for plotting"""

    def test_labels(labels):
        for legend_entry, label in zip(plt.gca().get_legend().texts, labels):
            assert label == legend_entry.get_text()

    fit = Fit(ewlc_odijk_distance("m"))
    fit._add_data("data_1", [1, 2, 3], [2, 3, 4])
    fit.plot()
    test_labels(["data_1 (model)", "data_1 (data)"])
    plt.gca().clear()
    fit.plot(label="custom label")
    test_labels(["custom label (model)", "custom label (data)"])


def test_nan():
    def f(independent, a, b):
        return np.asarray([1, np.nan, 1])

    fit = Fit(Model("f", f))
    data = np.asarray([1, 2, 3])
    fit._add_data("data_name", data, data)
    with pytest.raises(
        RuntimeError,
        match="Residual returned NaN. Model cannot be evaluated at these parameter values.",
    ):
        fit.log_likelihood()
