import numpy as np
import pytest

from lumicks.pylake.fitting.model import Model
from lumicks.pylake.fitting.models import ewlc_odijk_distance
from lumicks.pylake.fitting.parameters import Params, Parameter


def test_simulation_api():
    dna = ewlc_odijk_distance("DNA")
    force = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(
        dna(force, {"DNA/Lp": 50.0, "DNA/Lc": 16.0, "DNA/St": 1500.0, "kT": 4.11}),
        [8.74792941, 10.8733908, 11.81559925],
    )

    np.testing.assert_allclose(
        dna(
            [0.1, 0.2, 0.3],
            Params(
                **{
                    "DNA/Lp": Parameter(50.0),
                    "DNA/Lc": Parameter(16.0),
                    "DNA/St": Parameter(1500.0),
                    "kT": Parameter(4.11),
                }
            ),
        ),
        [8.74792941, 10.8733908, 11.81559925],
    )


def test_simulation_api_wrong_par():
    dna = ewlc_odijk_distance("DNA")

    with pytest.raises(KeyError):
        dna([1], {"DNA/Lp": 50.0, "DNA/Lc": 16.0, "DN/St": 1500.0, "kT": 4.11})


def test_model_calls():
    def model_function(x, b, c, d):
        return b + c * x + d * x * x

    def model_derivative(x, b, c, d):
        return c + 2 * d * x

    t = np.array([1.0, 2.0, 3.0])
    model = Model("m", model_function, derivative=model_derivative)
    y_ref = model._raw_call(t, [2.0, 3.0, 4.0])

    np.testing.assert_allclose(
        model(
            t,
            Params(
                **{
                    "m/a": Parameter(1),
                    "m/b": Parameter(2),
                    "m/c": Parameter(3),
                    "m/d": Parameter(4),
                }
            ),
        ),
        y_ref,
    )

    params = Params(**{"m/d": Parameter(4), "m/c": Parameter(3), "m/b": Parameter(2)})
    np.testing.assert_allclose(model(t, params), y_ref)
    np.testing.assert_allclose(model(t[0], params), y_ref[0])
    np.testing.assert_allclose(model.invert()(y_ref, params), t)
    np.testing.assert_allclose(model.invert()(y_ref[0], params), t[0])

    with pytest.raises(
        KeyError,
        match=r"The following missing parameters must be specified to simulate the model: {'m/c'}",
    ):
        np.testing.assert_allclose(
            model(t, Params(**{"m/a": Parameter(1), "m/b": Parameter(2), "m/d": Parameter(4)})),
            y_ref,
        )

    with pytest.raises(
        TypeError,
        match="Only 0- or 1-dimensional arrays are supported for the independent variable, got 2",
    ):
        model(np.array([[1, 2], [3, 4]]), params)
