import numpy as np
import pytest

from lumicks.pylake.fitting.models import *
from lumicks.pylake.fitting.parameters import Params, Parameter


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
    assert float(Parameter(5.0, 0.0, 1.0, unit="pN")) == 5.0

    assert Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}) == Params(
        **{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}
    )

    assert not Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}) == Params(
        **{"M/a": Parameter(5.0), "M/b": Parameter(6.0)}
    )

    with pytest.raises(RuntimeError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)}).update_params(5)

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/a":"M/b"]

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/c"].value = 5

    with pytest.raises(IndexError):
        Params(**{"M/a": Parameter(5.0), "M/b": Parameter(5.0)})["M/c"] = 5

    assert str(Params()) == "No parameters"
    assert (
        str(Parameter(5.0, 0.0, 1.0, fixed=True))
        == f"lumicks.pylake.fdfit.Parameter(value: {5.0}, lower bound: {0.0}, upper bound: {1.0}, fixed: {True})"
    )

    params = Params()
    params._set_params(["alpha", "beta", "gamma"], [None] * 3)
    assert params["beta"].value == 0.0

    params["beta"].value = 5.0
    np.testing.assert_allclose(params.values, [0.0, 5.0, 0.0])

    params._set_params(["alpha", "beta", "gamma", "delta"], [None] * 4)
    assert params["beta"].value == 5.0
    np.testing.assert_allclose(params.values, [0.0, 5.0, 0.0, 0.0])

    params["gamma"].value = 6.0
    params["delta"] = 7.0
    params["gamma"].lower_bound = -4.0
    params["gamma"].upper_bound = 5.0
    np.testing.assert_allclose(params.values, [0.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(params.lower_bounds, [-np.inf, -np.inf, -4.0, -np.inf])
    np.testing.assert_allclose(params.upper_bounds, [np.inf, np.inf, 5.0, np.inf])

    assert len(params) == 4.0
    params._set_params(["alpha", "beta", "delta"], [None] * 3)
    np.testing.assert_allclose(params.values, [0.0, 5.0, 7.0])
    assert [p for p in params] == ["alpha", "beta", "delta"]
    assert len(params) == 3.0

    for i, p in params.items():
        p.value = 1.0

    np.testing.assert_allclose(params.values, [1.0, 1.0, 1.0])

    params = Params()
    params._set_params(["alpha", "beta", "gamma"], [Parameter(2), Parameter(3), Parameter(4)])
    params2 = Params()
    params2._set_params(["gamma", "potato", "beta"], [Parameter(10), Parameter(11), Parameter(12)])
    params.update_params(params2)
    np.testing.assert_allclose(params.values, [2, 12, 10])

    params2 = Params()
    params2._set_params(["spaghetti"], [Parameter(10), Parameter(12)])
    with pytest.raises(RuntimeError):
        params.update_params(params2)
