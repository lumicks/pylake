from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.fit import Fit
import pytest
import numpy as np


def test_no_jac():
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

    linear_model = Model("linear", lambda x, a, b: a * x + b)
    linear_fit = Fit(linear_model)
    linear_fit._add_data("test", x, y, {"linear/a": 5})
    linear_fit.fit()

    with pytest.raises(NotImplementedError):
        linear_fit.cov


def test_uncertainty_analysis():
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

    def quad(x, a=1, b=1, c=1):
        return a * x * x + b * x + c

    def quad_jac(x, a=1, b=1, c=1):
        return np.vstack((x * x, x, np.ones(len(x))))

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

    np.testing.assert_allclose(
        linear_fit.cov, np.array([[0.08524185, -0.38358832], [-0.38358832, 2.42939269]])
    )
    np.testing.assert_allclose(
        quad_fit.cov,
        np.array(
            [
                [0.01390294, -0.12512650, 0.16683533],
                [-0.1251265, 1.21511735, -1.90192281],
                [0.16683533, -1.90192281, 4.53792109],
            ]
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(linear_fit.sigma[0], 2.65187717)
    np.testing.assert_allclose(linear_fit.aic, 49.88412577726061)
    np.testing.assert_allclose(linear_fit.aicc, 51.59841149154632)
    np.testing.assert_allclose(linear_fit.bic, 50.4892959632487)

    np.testing.assert_allclose(quad_fit.sigma[0], 2.70938272)
    np.testing.assert_allclose(quad_fit.aic, 51.31318724618379)
    np.testing.assert_allclose(quad_fit.aicc, 55.31318724618379)
    np.testing.assert_allclose(quad_fit.bic, 52.220942525165924)
