import math
import operator

import numpy as np
import pytest

from lumicks.pylake.detail.value import ValueMixin


class Parameter(ValueMixin):
    def __init__(self, value, description):
        super().__init__(value)
        self.description = description


class BadParameter(ValueMixin):
    def __init__(self, description):
        self.description = description


operators = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    divmod,
    operator.pow,
    operator.and_,
    operator.or_,
    operator.xor,
}


def test_mixin():
    a = Parameter(2, "two")
    a + 2

    b = BadParameter("two")
    with pytest.raises(AttributeError, match="'BadParameter' object has no attribute 'value'"):
        b + 2


def test_operators():
    a = Parameter(a0 := 5, "five")
    b = Parameter(b0 := 8, "eight")

    for op in operators:
        assert op(a0, 2) == op(a, 2)
        assert op(2, a0) == op(2, a)
        assert op(a0, b0) == op(a, b)
        assert op(b0, a0) == op(b, a)


def test_comparisons():
    a = Parameter(5, "five")

    assert a == 5
    assert 5 == a

    assert a != 4
    assert 4 != a

    assert a < 10
    assert 10 > a

    assert a > 2
    assert 2 < a

    assert a >= 5
    assert 5 >= a
    assert a <= 5
    assert 5 <= a


def test_operators_str():
    a = Parameter(a0 := "a", "alpha")
    b = Parameter(b0 := "b", "beta")

    assert a0 + "c" == a + "c"
    assert "c" + a0 == "c" + a
    assert a0 + b0 == a + b
    assert a0 * 5 == a * 5
    assert a0 * 5 == a * Parameter(5, "five")
    assert a0 * Parameter(5, "five") == a * 5

    assert a == "a"

    with pytest.raises(TypeError, match="not all arguments converted during string formatting"):
        a % 1

    assert "Hello, venus" == Parameter("Hello, %s", "fmt") % "venus"

    for c in (1, "c", True):
        for op in operators - {operator.add, operator.mul, operator.mod}:
            with pytest.raises(TypeError):
                op(a, c)


def test_operators_bool():
    a = Parameter(a0 := True, "true")

    for op in operators:
        assert op(a0, True) == op(a, True)
        assert op(False, a0) == op(False, a)


def test_not_implemented_operators():
    a = Parameter(a0 := 2, "two")

    for op in (operator.lshift, operator.rshift):
        with pytest.raises(TypeError):
            op(a, 2)
        with pytest.raises(TypeError):
            op(2, a)

    # __matmul__ has not been specifically implemented, but it still works
    A0 = np.array([[1, a0], [3, 4]])
    A = np.array([[1, a], [3, 4]])
    B0 = np.array([[11, 12], [13, 14]])
    np.testing.assert_equal(A @ B0, A0 @ B0)

    with pytest.raises(ValueError, match="matmul: Input operand 1 does not have enough dimensions"):
        A @ 2

    with pytest.raises(ValueError, match="matmul: Input operand 0 does not have enough dimensions"):
        2 @ A

    # not directly implemented but default fallback means it still works
    assert isinstance(a, Parameter)
    a0 += 4
    a += 4
    assert a0 == a
    assert isinstance(a, int)

    a = Parameter(a0 := 2, "two")
    b = a
    a += 4
    assert a0 + 4 == a
    assert a0 == b


def test_casting():
    a = Parameter(a0 := 2, "two")
    assert float(a0) == float(a)

    b = Parameter(b0 := 3.14, "pi")
    assert int(b0) == int(b)

    assert complex(a0) == complex(a)
    assert complex(b0) == complex(b)
    assert complex(a0, 2.1) == complex(a, 2.1)
    assert complex(b0, 2.1) == complex(b, 2.1)


def test_none():
    a = Parameter(None, "none")
    assert not a
    assert a == None

    with pytest.raises(AssertionError):
        assert a is None


def test_math():
    a = Parameter(a0 := 2, "two")
    assert -a0 == -a

    b = Parameter(b0 := -2, "two")
    assert +b0 == +b

    assert abs(b0) == abs(b)

    p = Parameter(p0 := 3.141592, "pi")
    assert round(p0) == round(p)
    assert round(p0, 3) == round(p, 3)

    assert math.trunc(p0) == math.trunc(p)
    assert math.floor(p0) == math.floor(p)
    assert math.ceil(p0) == math.ceil(p)

    assert max(a, b) == max(a0, b0)
    assert min(a, b) == min(a0, b0)


def test_numpy():
    a = Parameter(a0 := 5, "five")
    arr = np.arange(3) + 1

    for op in operators:
        np.testing.assert_equal(op(arr, a), op(arr, a0))
        np.testing.assert_equal(op(a, arr), op(a0, arr))

    arr = np.array([1, 2, a])
    assert all([not isinstance(j, Parameter) for j in arr])
    assert all([not isinstance(j, Parameter) for j in arr * a0])
    assert all([not isinstance(j, Parameter) for j in arr * a])

    np.testing.assert_allclose(a, a0)
    np.testing.assert_almost_equal(a, a0)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(a, a0)

    assert np.sqrt(a) == np.sqrt(a0)
    assert np.exp(a) == np.exp(a0)

    # forward extra arguments to __array__()
    b = np.asarray(a, dtype=float)
    assert isinstance(b, np.ndarray)
