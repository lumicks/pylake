from lumicks.pylake import FdRangeSelector
from lumicks.pylake.nb_widgets.range_selector import (FdTimeRangeSelectorWidget, 
                                                      FdDistanceRangeSelectorWidget, 
                                                      BaseRangeSelectorWidget)
from lumicks.pylake.fdcurve import FdCurve
from lumicks.pylake.channel import TimeSeries, Slice
from matplotlib.testing.decorators import cleanup
import pytest
import numpy as np


def make_mock_fd(force, distance, start=0, dt=600e9):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FdCurve(file=None, start=None, stop=None, name="")
    timestamps = int(dt) * np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


@cleanup
def test_selector_widget(mockevent):
    start_point = int(2500e9)
    dt = int(600e9)
    fd_curve = make_mock_fd([0, 1, 2, 3], [0, 1, 2, 3], start=start_point, dt=dt)

    selector = FdTimeRangeSelectorWidget(fd_curve, show=False)

    assert selector.current_range == []
    selector._add_point(750)
    assert selector.current_range == [750]
    selector._add_point(450)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[450, 750]])
    selector._add_point(850)
    selector._add_point(950)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[450, 750], [850, 950]])

    assert selector.to_seconds(2500e9) == 0
    selector.update_plot()

    selector = FdTimeRangeSelectorWidget(fd_curve)
    lmb = 1
    rmb = 3

    # Remove a segment
    event = mockevent(selector._axes, 650, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []

    # Add a point
    event = mockevent(selector._axes, 650, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Widget is locked, do not add!
    event = mockevent(selector._axes, 650, 1, lmb, True)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Not the axis we own, do not add!
    event = mockevent(5, 650, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Successful add
    event = mockevent(selector._axes, 950, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[start_point + dt, start_point + 2*dt]])

    np.testing.assert_allclose(selector.fdcurves[0].f.data, fd_curve[start_point + dt:start_point + 2*dt].f.data)

    # Add another segment
    selector.handle_button_event(mockevent(selector._axes, 150, 1, lmb, False))
    selector.handle_button_event(mockevent(selector._axes, 1000, 1, lmb, False))

    # Remove a segment (fails since its outside)
    event = mockevent(selector._axes, 1300, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[start_point + dt, start_point + 2*dt], [start_point, start_point + 2*dt]])

    # Remove a segment (inner segment, smallest segment gets removed first)
    event = mockevent(selector._axes, 900, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[start_point, start_point + 2*dt]])

    # Remove a segment (inner segment, smallest segment gets removed first)
    event = mockevent(selector._axes, 900, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [])

    assert selector.fdcurves == []


@cleanup
def test_distance_selector_widget(mockevent):
    start_point = int(2500e9)
    dt = int(600e9)
    fd_curve = make_mock_fd(np.arange(10), np.arange(10), start=start_point, dt=dt)

    selector = fd_curve.distance_range_selector(show=False)

    assert selector.current_range == []
    selector._add_point(2)
    assert selector.current_range == [2]
    selector._add_point(4)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[2, 4]])
    selector._add_point(6)
    selector._add_point(9)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[2, 4], [6, 9]])

    selector.update_plot()

    selector = fd_curve.distance_range_selector(show=True)
    lmb = 1
    rmb = 3

    # Remove a segment
    event = mockevent(selector._axes, 3, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []

    # Add a point
    event = mockevent(selector._axes, 2, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == [2]

    # Widget is locked, do not add!
    event = mockevent(selector._axes, 4, 1, lmb, True)
    selector.handle_button_event(event)
    assert selector.current_range == [2]

    # Not the axis we own, do not add!
    event = mockevent(5, 4, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == [2]

    # Successful add
    event = mockevent(selector._axes, 4, 1, lmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[2, 4]])

    np.testing.assert_allclose(selector.fdcurves[0].f.data, fd_curve[start_point + 2*dt:start_point + 5*dt].f.data)

    # Add another segment
    selector.handle_button_event(mockevent(selector._axes, 6, 1, lmb, False))
    selector.handle_button_event(mockevent(selector._axes, 9, 1, lmb, False))

    # Remove a segment (fails since its outside)
    event = mockevent(selector._axes, 10, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[2, 4], [6, 9]])

    # Remove a segment (inner segment, smallest segment gets removed first)
    event = mockevent(selector._axes, 3, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [[6, 9]])

    # Remove a segment (inner segment, smallest segment gets removed first)
    event = mockevent(selector._axes, 7, 1, rmb, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    np.testing.assert_allclose(selector.ranges, [])

    assert selector.fdcurves == []


@cleanup
def test_multi_selector_widget():
    fd_curve1 = make_mock_fd([0, 1, 2, 3], [0, 1, 2, 3], start=int(2500e9))
    fd_curve2 = make_mock_fd([2, 3, 4, 5], [2, 3, 4, 5], start=int(2500e9))

    with pytest.raises(ValueError):
        FdRangeSelector({})

    with pytest.raises(RuntimeError):
        FdRangeSelector({"fd1": fd_curve1, "fd2": fd_curve2})


@cleanup
def test_multi_distance_selector_widget():
    fd_curve1 = make_mock_fd(np.arange(10), np.arange(10), start=int(2500e9))
    fd_curve2 = make_mock_fd(np.arange(3, 13), np.arange(3, 13), start=int(2500e9))

    with pytest.raises(ValueError):
        FdRangeSelector({})

    with pytest.raises(RuntimeError):
        FdRangeSelector({"fd1": fd_curve1, "fd2": fd_curve2})        


@cleanup
def test_selector_widgets_open():
    channel = Slice(TimeSeries([1, 2, 3, 4], [100, 200, 300, 400]))
    widget = channel.range_selector()
    assert isinstance(widget, BaseRangeSelectorWidget)

    fd_curve = make_mock_fd(np.arange(10), np.arange(10), start=int(2500e9))
    widget = fd_curve.range_selector()
    assert isinstance(widget, BaseRangeSelectorWidget)

    widget = fd_curve.distance_range_selector()
    assert isinstance(widget, BaseRangeSelectorWidget)
