import warnings
import pytest
import numpy as np
from textwrap import dedent
from lumicks.pylake.fitting.fit import Fit
from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.parameters import Parameter
from matplotlib.testing.decorators import cleanup


def linear(x, a=1, b=1):
    return a * x + b


def linear_jac(x, a, b):
    return np.vstack((x, np.ones(len(x))))


def quad(x, a=1, b=1, c=1):
    return a * x * x + b * x + c


def quad_jac(x, a=1, b=1, c=1):
    return np.vstack((x * x, x, np.ones(len(x))))


def exp_charge(x, mag, k):
    return mag * (1.0 - np.exp(-k * x))


def exp_charge_jac(x, mag, k):
    return np.vstack((1.0 - np.exp(-k * x), mag * x * np.exp(-k * x)))


def test_confidence_intervals():
    eps = 1e-5
    x = np.arange(7)

    # The following is generated from y = 4.0 * x + 5.0 + 2.0*np.random.randn(N):
    y = [8.24869073, 7.77648717, 11.9436565, 14.85406276, 22.73081526, 20.39692261, 32.48962353]

    def validate_profile(name, func, jac, x, y, parameter, stderr, lb, ub):
        F = Fit(Model(name, func, jacobian=jac))
        F._add_data(name, x, y)
        F.fit()
        profile = F.profile_likelihood(name + "/" + parameter)

        assert (F[name + "/" + parameter].stderr - stderr) < eps
        assert F[name + "/" + parameter].profile == profile
        assert abs(profile.lower_bound - lb) < eps
        assert abs(profile.upper_bound - ub) < eps
        assert len(profile.chi2) == len(profile.p)

        # Validate asymptotic confidence intervals based on comparing them with the profiles
        assert abs(F[name + "/" + parameter].ci(0.95)[0] - profile.lower_bound) < 1e-2
        assert abs(F[name + "/" + parameter].ci(0.95)[1] - profile.upper_bound) < 1e-2

    validate_profile(
        "linear",
        linear,
        linear_jac,
        x,
        y,
        "a",
        0.61939832586,
        2.6713088215948138,
        5.096607466976615,
    )
    validate_profile(
        "linear",
        linear,
        linear_jac,
        x,
        y,
        "b",
        2.23327242385,
        0.8928141799335386,
        9.643510828637888,
    )
    validate_profile(
        "quad", quad, quad_jac, x, y, "a", 0.32007865509, -0.14811649590123688, 1.1064951859012375
    )
    validate_profile(
        "quad", quad, quad_jac, x, y, "b", 1.99889056039, -2.9084849155213455, 4.9261290640927715
    )

    with pytest.raises(KeyError):
        validate_profile(
            "linear", linear, linear_jac, x, y, "d", 5, 2.6713088215948138, 5.096607466976615
        )


def test_non_identifiability():
    x = np.arange(10.0) / 100.0
    y = [
        1.37272429,
        1.14759176,
        1.2080786,
        1.79293398,
        1.22606946,
        1.55293523,
        1.73564261,
        1.49623027,
        1.81209629,
        1.69464097,
    ]

    F = Fit(
        Model(
            "exponential",
            exp_charge,
            jacobian=exp_charge_jac,
            mag=Parameter(value=1.0),
            k=Parameter(value=1.0),
        )
    )
    F._add_data("exponential", x, y)
    F.fit()
    num_steps = 100
    profile = F.profile_likelihood("exponential/k", num_steps=num_steps)

    # This model does not have an upper bound for its 95% confidence interval (we're fitting a
    # constant with an exponential rise. The exponential rise is non-identifiable since it can be
    # infinitely fast.
    assert profile.lower_bound is not None
    assert profile.upper_bound is None
    assert len(profile.p) < 2 * num_steps
    assert str(profile) == dedent(
        """\
        Profile likelihood for exponential/k (121 points)
          - chi2
          - p
          - lower_bound: 20.36
          - upper_bound: undefined
        """
    )


@cleanup
def test_plotting():
    x = np.arange(10.0) / 100.0
    y = [
        1.37272429,
        1.14759176,
        1.2080786,
        1.79293398,
        1.22606946,
        1.55293523,
        1.73564261,
        1.49623027,
        1.81209629,
        1.69464097,
    ]

    F = Fit(
        Model(
            "exponential",
            exp_charge,
            jacobian=exp_charge_jac,
            mag=Parameter(value=1.0),
            k=Parameter(value=1.0),
        )
    )
    F._add_data("exponential", x, y)
    F.fit()
    profile = F.profile_likelihood("exponential/k")

    profile.plot()
    profile.plot_relations()


def test_bounded_valid_interval():
    """Test whether profiles don't push the model beyond its valid range."""

    def linear_with_exception(x, a=0.01, b=1):
        if a <= 0:
            raise ValueError("Invalid value!")
        return a * x + b

    # Data generated with: y = np.random.rand(7) + np.ones(1) + 0.01 * np.arange(7)
    x = np.arange(7)
    y = np.array(
        [1.56368527, 1.46190795, 1.51428812, 2.02054457, 1.61288094, 1.36798765, 1.86743863]
    )

    fit = Fit(Model("model", linear_with_exception, jacobian=linear_jac))
    fit._add_data("data", x, y)
    fit["model/a"].value = 1
    fit.fit()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Warning: Step size set to minimum step size")
        warnings.filterwarnings("ignore", message="Optimization error encountered")
        profile = fit.profile_likelihood("model/a")

    assert np.all(profile.p > 0)


def test_fit_failure():
    """Test whether profiles don't push the model beyond its valid range."""

    def linear_with_exception(x, a=1, b=0):
        # This will fail once it starts optimizing along the profile, because then b will eventually
        # drop below -0.1 because a + b should be 1. When a is larger than 1.1 this fails.
        if b < -0.1:
            print(f"a: {a}, b: {b}")
            raise ValueError("Invalid value!")
        return (a + b) * x

    fit = Fit(Model("model", linear_with_exception))
    data = np.array([0.01880579, 1.08151718, 2.03018996, 3.00636749, 4.07885534, 5.03468854])
    fit._add_data("data", np.arange(6), data)
    fit["model/a"].value = 1
    fit["model/b"].value = 0

    with pytest.warns(RuntimeWarning, match="Optimization error encountered"):
        fit.profile_likelihood("model/a", num_steps=100, max_step=10, max_chi2_step=10.0)
