import re

import numpy as np
import pytest

from lumicks.pylake.fitting.fit import Fit
from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.parameters import Parameter
from lumicks.pylake.fitting.parameter_trace import parameter_trace


def test_parameter_inversion():
    def f(independent, a, b):
        return a + b * independent

    def f_jac(independent, a, b):
        del a, b
        return np.vstack((np.ones((1, len(independent))), independent))

    def g(independent, a, d, b):
        return a - b * independent + d * independent * independent

    def g_jac(independent, a, d, b):
        del a, d, b
        return np.vstack((np.ones((1, len(independent))), independent * independent, -independent))

    def f_der(independent, a, b):
        del independent, a
        return b * np.ones((len(x)))

    def g_der(independent, a, d, b):
        del independent, a
        return -b * np.ones((len(independent))) + 2.0 * d * independent

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a_true = 5.0
    b_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_data = f(x, a_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der)
    fit = Fit(model)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = 1.0
    np.testing.assert_allclose(parameter_trace(model, fit.params, "f/b", x, f_data), b_true)

    a_true = 5.0
    b_true = 3.0
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    f_plus_g_data = f(x, a_true, b_true) + g(x, a_true, d_true, b_true)
    model = Model("f", f, jacobian=f_jac, derivative=f_der) + Model(
        "f", g, jacobian=g_jac, derivative=g_der
    )
    fit = Fit(model)
    fit._add_data("test", x, f_data)
    fit.params["f/a"].value = a_true
    fit.params["f/b"].value = b_true
    fit.params["f/d"].value = 1.0
    np.testing.assert_allclose(parameter_trace(model, fit.params, "f/d", x, f_plus_g_data), d_true)


def test_parameter_trace_invalid_args():
    def f(independent, a, b):
        return a + b * independent

    model = Model("f", f)
    data = {"dependent": np.arange(3), "independent": np.arange(3)}

    with pytest.raises(
        ValueError,
        match="Inverted parameter f/c not in model parameters {'f/a': 5, 'f/b': 3}",
    ):
        parameter_trace(model, {"f/a": 5, "f/b": 3}, inverted_parameter="f/c", **data)

    with pytest.raises(ValueError, match="Missing parameter f/b in supplied params"):
        parameter_trace(model, {"f/a": 5}, inverted_parameter="f/a", **data)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The argument params takes a dictionary with `Parameter` values. This can be "
            "obtained from a fit by slicing it by a dataset (i.e. fit[dataset_handle]) or "
            "from a model (i.e. model.defaults). See help(parameter_trace) for more information."
        ),
    ):
        parameter_trace(model, {"f/a": 5, "f/b": 3}, inverted_parameter="f/a", **data)


def test_parameter_inversion_warning():
    def f(independent, a, b):
        return a + b * independent - 1

    model = Model("m", f, a=Parameter(1, 0, 10), b=Parameter(1, 0, 10))
    model["m/b"].value = 2
    vals = parameter_trace(model, model.defaults, "m/b", [1, 2, 3], [1, 4, 9])
    np.testing.assert_allclose(vals, [1, 2, 3])

    model["m/b"].upper_bound = 2.5
    with pytest.warns(
        RuntimeWarning, match=re.escape("Some values for m/b hit the upper bound (2.5)")
    ):
        _ = parameter_trace(model, model.defaults, "m/b", [1, 2, 3], [1, 4, 9])

    model["m/b"].lower_bound = 1.5
    with pytest.warns(
        RuntimeWarning,
        match=re.escape(
            "Some values for m/b hit the lower bound (1.5), while others hit the upper bound (2.5)"
        ),
    ):
        _ = parameter_trace(model, model.defaults, "m/b", [1, 2, 3], [1, 4, 9])

    model["m/b"].upper_bound = np.inf
    with pytest.warns(
        RuntimeWarning, match=re.escape("Some values for m/b hit the lower bound (1.5)")
    ):
        _ = parameter_trace(model, model.defaults, "m/b", [1, 2, 3], [1, 4, 9])
