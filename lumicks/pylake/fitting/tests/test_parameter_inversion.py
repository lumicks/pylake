from lumicks.pylake.fitting.fit import Fit
from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.parameter_trace import parameter_trace
import numpy as np


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
    np.testing.assert_allclose(parameter_trace(model, fit.params, 'f/b', x, f_data), b_true)

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
    np.testing.assert_allclose(parameter_trace(model, fit.params, 'f/d', x, f_plus_g_data), d_true)
