import numpy as np
import pytest

from lumicks.pylake.fitting.fit import Fit
from lumicks.pylake.fitting.model import Model


def data_for_testing():
    x = np.arange(10)
    y = np.array(
        [
            23.250234526945302,
            28.342680403327503,
            34.15303580256364,
            39.74756396347861,
            49.98132078695123,
            60.51421884139438,
            73.2211796692214,
            86.9299566694317,
            104.81050416917682,
            124.25500144427338,
        ]
    )
    return x, y


def linear_func():
    def linear(x, a, b):
        return a * x + b

    def linear_jac(x, a, b):
        del a, b
        return np.vstack((x, np.ones(len(x))))

    return {"model_function": linear, "jacobian": linear_jac}


def quadratic_func():
    def quadratic(x, a, b, c):
        return a * x * x + b * x + c

    def quadratic_jac(x, a, b, c):
        del a, b, c
        return np.vstack((x * x, x, np.ones(len(x))))

    return {"model_function": quadratic, "jacobian": quadratic_jac}


def test_no_jac():
    x, y = data_for_testing()

    linear_model = Model("linear", linear_func()["model_function"])
    linear_fit = Fit(linear_model)
    linear_fit._add_data("test", x, y, {"linear/a": 5})
    linear_fit.fit()

    with pytest.raises(NotImplementedError):
        linear_fit.cov


@pytest.mark.parametrize(
    "model_funcs, sigma, aic, aicc, bic",
    [
        (linear_func, 7.717703868870184, 71.24910853378455, 72.96339424807026, 71.85427871977264),
        (
            quadratic_func,
            0.8795877788756955,
            28.8127323769986,
            32.812732376998596,
            29.720487655980737,
        ),
    ],
)
def test_fit_metrics(model_funcs, sigma, aic, aicc, bic):
    x, y = data_for_testing()

    fit = Fit(Model("linear", **model_funcs()))
    fit._add_data("test", x, y)
    fit.fit()

    np.testing.assert_allclose(fit.sigma[0], sigma)
    np.testing.assert_allclose(fit.aic, aic)
    np.testing.assert_allclose(fit.aicc, aicc)
    np.testing.assert_allclose(fit.bic, bic)


def test_asymptotic_errs_all_parameters():
    """Tests whether the covariance matrix is computed correctly"""
    x, y = data_for_testing()

    quadratic_fit = Fit(Model("quadratic", **quadratic_func()))
    quadratic_fit._add_data("test", x, y)
    quadratic_fit.fit()

    np.testing.assert_allclose(
        quadratic_fit.cov,
        np.array(
            [
                [0.001465292918082, -0.013187636262741, 0.017583515016988],
                [-0.013187636262741, 0.128066601040397, -0.200452071193665],
                [0.017583515016988, -0.200452071193665, 0.478271608462078],
            ]
        ),
    )

    np.testing.assert_allclose(quadratic_fit["quadratic/a"].stderr, 0.038279144688489905)
    np.testing.assert_allclose(quadratic_fit["quadratic/b"].stderr, 0.35786394207910477)
    np.testing.assert_allclose(quadratic_fit["quadratic/c"].stderr, 0.6915718389741429)


def test_asymptotic_errs_subset_parameters():
    """Check whether the asymptotic uncertainty handling is correct by checking a nested model

    Fixing a parameter in quadratic function converts it to a linear one. This should result in
    identical standard errors and covariance matrix"""
    x, y = data_for_testing()

    linear_fit = Fit(Model("m", **linear_func()))
    linear_fit._add_data("test", x, y)
    linear_fit.fit()

    quadratic_fit = Fit(Model("m", **quadratic_func()))
    quadratic_fit._add_data("test", x, y)
    quadratic_fit["m/a"].fixed = True
    quadratic_fit.fit()

    np.testing.assert_allclose(linear_fit.cov, quadratic_fit.cov)
    np.testing.assert_allclose(quadratic_fit["m/b"].stderr, linear_fit["m/a"].stderr)
    np.testing.assert_allclose(quadratic_fit["m/c"].stderr, linear_fit["m/b"].stderr)
    assert quadratic_fit["m/a"].stderr is None
