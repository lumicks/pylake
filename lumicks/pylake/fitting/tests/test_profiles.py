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
    J = np.vstack((1.0 - np.exp(-k * x), mag * x * np.exp(-k * x)))
    return J


def test_confidence_intervals():
    eps = 1e-5
    x = np.arange(7)

    # The following is generated from y = 4.0 * x + 5.0 + 2.0*np.random.randn(N):
    y = [8.24869073,  7.77648717, 11.9436565, 14.85406276, 22.73081526, 20.39692261, 32.48962353]

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
        assert abs(F[name + "/" + parameter].ci(.95)[0] - profile.lower_bound) < 1e-2
        assert abs(F[name + "/" + parameter].ci(.95)[1] - profile.upper_bound) < 1e-2

    validate_profile("linear", linear, linear_jac, x, y, "a", 0.61939832586, 2.6713088215948138, 5.096607466976615)
    validate_profile("linear", linear, linear_jac, x, y, "b", 2.23327242385, 0.8928141799335386, 9.643510828637888)
    validate_profile("quad", quad, quad_jac, x, y, "a", 0.32007865509, -0.14811649590123688, 1.1064951859012375)
    validate_profile("quad", quad, quad_jac, x, y, "b", 1.99889056039, -2.9084849155213455, 4.9261290640927715)

    with pytest.raises(KeyError):
        validate_profile("linear", linear, linear_jac, x, y, "d", 5, 2.6713088215948138, 5.096607466976615)


def test_non_identifiability():
    x = np.arange(10.0) / 100.0
    y = [1.37272429, 1.14759176, 1.2080786, 1.79293398, 1.22606946, 1.55293523, 1.73564261, 1.49623027, 1.81209629,
         1.69464097]

    F = Fit(Model("exponential", exp_charge, jacobian=exp_charge_jac, mag=Parameter(value=1.0), k=Parameter(value=1.0)))
    F._add_data("exponential", x, y)
    F.fit()
    num_steps = 100
    profile = F.profile_likelihood("exponential/k", num_steps=num_steps)

    # This model does not have an upper bound for its 95% confidence interval (we're fitting a constant with an
    # exponential rise. The exponential rise is non-identifiable since it can be infinitely fast.
    assert profile.lower_bound is not None
    assert profile.upper_bound is None
    assert len(profile.p) < 2 * num_steps
    assert str(profile) == dedent("""\
        Profile likelihood for exponential/k (121 points)
          - chi2
          - p
          - lower_bound: 20.36
          - upper_bound: undefined
        """)


@cleanup
def test_plotting():
    x = np.arange(10.0) / 100.0
    y = [1.37272429, 1.14759176, 1.2080786, 1.79293398, 1.22606946, 1.55293523, 1.73564261, 1.49623027, 1.81209629,
         1.69464097]

    F = Fit(Model("exponential", exp_charge, jacobian=exp_charge_jac, mag=Parameter(value=1.0), k=Parameter(value=1.0)))
    F._add_data("exponential", x, y)
    F.fit()
    profile = F.profile_likelihood("exponential/k")

    profile.plot()
    profile.plot_relations()
