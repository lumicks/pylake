import numpy as np
import pytest

from lumicks.pylake.fitting.model import (
    Model,
    InverseModel,
    CompositeModel,
    SubtractIndependentOffset,
)
from lumicks.pylake.fitting.models import (
    force_offset,
    distance_offset,
    ewlc_odijk_force,
    ewlc_odijk_distance,
)


def test_model_composition():
    def f(x, a, b):
        return a + b * x

    def f_jac(x, a, b):
        return np.vstack((np.ones((1, len(x))), x))

    def f_jac_wrong(x, a, b):
        return np.vstack((np.zeros((1, len(x))), x))

    def g(x, a, d, b):
        return a - b * x + d * x * x

    def g_jac(x, a, d, b):
        return np.vstack((np.ones((1, len(x))), x * x, -x))

    def f_der(x, a, b):
        return b * np.ones((len(x)))

    def f_der_wrong(x, a, b):
        return np.ones((len(x)))

    def g_der(x, a, d, b):
        return -b * np.ones((len(x))) + 2.0 * d * x

    m1 = Model("M", f, dependent="x", jacobian=f_jac, derivative=f_der)
    m2 = Model("M", g, dependent="x", jacobian=g_jac, derivative=g_der)
    t = np.arange(0, 2, 0.5)

    # Check actual composition
    # (a + b * x) + a - b * x + d * x * x = 2 * a + d * x * x
    np.testing.assert_allclose(
        (m1 + m2)._raw_call(t, np.array([1.0, 2.0, 3.0])), 2.0 + 3.0 * t * t
    ), "Model composition returns invalid function evaluation (parameter order issue?)"

    # Check correctness of the Jacobians and derivatives
    assert (m1 + m2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m1 + m2).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (m2 + m1).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m2 + m1).verify_derivative(t, [1.0, 2.0, 3.0])
    assert (m2 + m1 + m2).verify_jacobian(t, [1.0, 2.0, 3.0])
    assert (m2 + m1 + m2).verify_derivative(t, [1.0, 2.0, 3.0])

    m1_wrong_jacobian = Model("M", f, dependent="x", jacobian=f_jac_wrong, derivative=f_der)
    assert not (m1_wrong_jacobian + m2).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)
    assert not (m2 + m1_wrong_jacobian).verify_jacobian(t, [1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(m1) + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(m1 + m2).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert InverseModel(m1 + m2 + m1).verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)

    assert (InverseModel(m1) + m2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(m1 + m2).verify_derivative(t, [-1.0, 2.0, 3.0])
    assert InverseModel(m1 + m2 + m1).verify_derivative(t, [-1.0, 2.0, 3.0])

    m1_wrong_derivative = Model("M", f, dependent="x", jacobian=f_jac, derivative=f_der_wrong)
    assert not (InverseModel(m1_wrong_derivative) + m2).verify_jacobian(
        t, [-1.0, 2.0, 3.0], verbose=False
    )
    assert not (InverseModel(m1_wrong_jacobian) + m2).verify_jacobian(
        t, [-1.0, 2.0, 3.0], verbose=False
    )
    assert not (InverseModel(m1_wrong_derivative) + m2).verify_derivative(t, [-1.0, 2.0, 3.0])

    assert m1.subtract_independent_offset().verify_jacobian(t, [-1.0, 2.0, 3.0], verbose=False)
    assert m1.subtract_independent_offset().verify_derivative(t, [-1.0, 2.0, 3.0])

    m1 = ewlc_odijk_force("DNA").subtract_independent_offset() + force_offset("f")
    m2 = (ewlc_odijk_distance("DNA") + distance_offset("DNA_d")).invert() + force_offset("f")
    t = np.array([0.19, 0.2, 0.3])
    p1 = np.array([0.1, 4.9e1, 3.8e-1, 2.1e2, 4.11, 1.5])
    p2 = np.array([4.9e1, 3.8e-1, 2.1e2, 4.11, 0.1, 1.5])
    np.testing.assert_allclose(m1._raw_call(t, p1), m2._raw_call(t, p2))

    # Check whether incompatible variables are found
    with pytest.raises(ValueError, match="These models are incompatible"):
        distance_offset("d") + force_offset("f")

    composite = distance_offset("d") + ewlc_odijk_distance("DNA")
    assert composite.dependent == "d"
    assert composite.independent == "f"
    assert composite._dependent_unit == "micron"
    assert composite._independent_unit == "pN"

    inverted = composite.invert()
    assert inverted.dependent == "f"
    assert inverted.independent == "d"
    assert inverted._dependent_unit == "pN"
    assert inverted._independent_unit == "micron"


@pytest.mark.parametrize(
    "model,param,unit",
    [
        (ewlc_odijk_distance("m").subtract_independent_offset(), "m/f_offset", "pN"),
        (ewlc_odijk_force("m").subtract_independent_offset(), "m/d_offset", "micron"),
        (
            (ewlc_odijk_distance("m") + ewlc_odijk_distance("m")).subtract_independent_offset(),
            "m_with_m/f_offset",
            "pN",
        ),
        (
            ewlc_odijk_distance("m").invert().subtract_independent_offset(),
            "inv(m)/d_offset",
            "micron",
        ),
        (Model("m", lambda c, a: c + a).subtract_independent_offset(), "m/c_offset", "au"),
        (
            Model("m", lambda c, a: c + a).invert().subtract_independent_offset(),
            "inv(m)/y_offset",
            "au",
        ),
    ],
)
def test_subtract_independent_offset_unit(model, param, unit):
    """ "Validate that the model units propagate to the subtracted independent offset parameter"""
    assert model.defaults[param].unit == unit


def test_interpolation_inversion():
    m = ewlc_odijk_distance("Nucleosome").invert(independent_max=120.0, interpolate=True)
    parvec = [5.77336105517341, 7.014180463612673, 1500.0000064812095, 4.11]
    result = np.array([0.17843862, 0.18101283, 0.18364313, 0.18633117, 0.18907864])
    np.testing.assert_allclose(m._raw_call(np.arange(10, 250, 50) / 1000, parvec), result)


@pytest.mark.parametrize("param", [{"independent_max": np.inf}, {"independent_min": -np.inf}])
def test_interpolation_invalid_range(param):
    with pytest.raises(
        ValueError, match="Inversion limits have to be finite when using interpolation method"
    ):
        ewlc_odijk_distance("Nucleosome").invert(**param, interpolate=True)


def test_uuids():
    m1, m2 = (Model(name, lambda x: x, dependent="x") for name in ("M1", "M2"))
    m3 = CompositeModel(m1, m2)
    m4 = InverseModel(m1)
    m5 = SubtractIndependentOffset(m1, "x")

    # Verify that all are unique
    assert len(set([m1.uuid, m2.uuid, m3.uuid, m4.uuid, m5.uuid])) == 5
