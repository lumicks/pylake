import re

import numpy as np
import pytest

from lumicks.pylake.detail.timeindex import Timeindex, regex, to_seconds, regex_template


def test_regex_template():
    r = re.compile("^" + regex_template.format(suffix="foo") + "$")

    assert r.match("1.2foo").group("foo") == "1.2"
    assert r.match("1 foo").group("foo") == "1"
    assert r.match(".2foo").group("foo") == ".2"
    assert r.match("").group("foo") is None

    assert not r.match("foo")
    assert not r.match("4. foo")


def test_regex():
    groups = regex.match("1.2s 50ms 4.0us 3 ns")
    assert groups["s"] == "1.2"
    assert groups["ms"] == "50"
    assert groups["us"] == "4.0"
    assert groups["ns"] == "3"


def test_timeindex():
    t = Timeindex("1s2ms 3us")
    assert t.sign == 1
    assert t.s == 1
    assert t.ms == 2
    assert t.us == 3
    assert t.ns == 0
    assert t.total_ns == 1_002_003_000

    t = Timeindex("-1d 2h3m   6ns")
    assert t.sign == -1
    assert t.d == 1
    assert t.h == 2
    assert t.m == 3
    assert t.ns == 6
    assert t.total_ns == -93_780_000_000_006

    t = Timeindex("1.51s 3.3ns")
    assert t.sign == 1
    assert t.s == 1.51
    assert t.ns == 3.3
    assert t.total_ns == 1_510_000_003

    with pytest.raises(RuntimeError):
        Timeindex("bad")

    with pytest.raises(RuntimeError):
        Timeindex("ms")

    with pytest.raises(RuntimeError):
        Timeindex("ms 1ns")

    with pytest.raises(RuntimeError):
        Timeindex("1ns 1us")  # wrong order

    with pytest.raises(RuntimeError):
        Timeindex("1")

    with pytest.raises(TypeError):
        Timeindex(1)


@pytest.mark.parametrize(
    "time, ref",
    (
        ("1s", 1.0),
        ("1ms", 0.001),
        ("0.1s", 0.1),
        ("2s", 2.0),
        ("-1s", 9.0),
        (1718968411999124820, 1.0),
        (1718968412499124820, 1.5),
        (1718968412999124820, 2.0),
    ),
)
def test_to_seconds(time, ref):
    np.testing.assert_allclose(to_seconds(time, 1718968410999124820, 1718968420999124820), ref)
