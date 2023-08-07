from collections import OrderedDict

import numpy as np
import pytest

from lumicks.pylake.fitting.detail.utilities import (
    escape_tex,
    latex_sqrt,
    unique_idx,
    parse_transformation,
)
from lumicks.pylake.fitting.detail.model_implementation import (
    calc_cubic_root,
    calc_cubic_root_derivatives,
)


def test_unique_idx():
    uiq, inv = unique_idx(["str", "str", "hmm", "potato", "hmm", "str"])
    assert uiq == ["str", "hmm", "potato"]
    assert inv == [0, 0, 1, 2, 1, 0]


def test_transformation_parser():
    pars = ["blip", "foo"]
    assert parse_transformation(pars, {"foo": "new_foo"}) == OrderedDict(
        (("blip", "blip"), ("foo", "new_foo"))
    )
    assert parse_transformation(pars, {"foo": 5}) == OrderedDict((("blip", "blip"), ("foo", 5)))

    with pytest.raises(KeyError):
        parse_transformation(pars, {"blap": "new_foo"})

    param_names = ["gamma", "alpha", "beta", "delta"]
    params = OrderedDict(zip(param_names, param_names))
    post_params = parse_transformation(params, {"gamma": "gamma_specific", "beta": "beta_specific"})
    assert post_params["gamma"] == "gamma_specific"
    assert post_params["alpha"] == "alpha"
    assert post_params["beta"] == "beta_specific"
    assert post_params["delta"] == "delta"

    with pytest.raises(KeyError):
        parse_transformation(params, {"doesnotexist": "yep"})


def test_tex_replacement():
    assert escape_tex("DNA/Hi") == "Hi_{DNA}"
    assert escape_tex("DNA/Hi_There") == "Hi\\_There_{DNA}"
    assert escape_tex("DNA_model/Hi_There") == "Hi\\_There_{DNA\\_model}"
    assert escape_tex("Hi_There") == "Hi\\_There"
    assert latex_sqrt("test") == r"\sqrt{test}"


def test_analytic_roots():
    a = np.array([0.0])
    b = np.array([-3.0])
    c = np.array([1.0])

    np.testing.assert_allclose(
        np.sort(np.roots(np.hstack((np.array([1.0]), a, b, c)))),
        np.sort(
            np.array(
                [
                    calc_cubic_root(a, b, c, 0)[0],
                    calc_cubic_root(a, b, c, 1)[0],
                    calc_cubic_root(a, b, c, 2)[0],
                ]
            )
        ),
    )

    with pytest.raises(RuntimeError):
        calc_cubic_root(a, b, c, 3)

    def test_root_derivatives(root):
        dx = 1e-5
        ref_root = calc_cubic_root(a, b, c, root)
        da = (calc_cubic_root(a + dx, b, c, root) - ref_root) / dx
        db = (calc_cubic_root(a, b + dx, c, root) - ref_root) / dx
        dc = (calc_cubic_root(a, b, c + dx, root) - ref_root) / dx

        np.testing.assert_allclose(
            np.array(calc_cubic_root_derivatives(a, b, c, root)),
            np.array([da, db, dc]),
            atol=1e-5,
            rtol=1e-5,
        )

    test_root_derivatives(0)
    test_root_derivatives(1)
    test_root_derivatives(2)
