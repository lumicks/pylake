from copy import deepcopy

import numpy as np
import pytest

from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.piezo_tracking.baseline import ForceBaseLine


def test_baseline(poly_baseline_data):
    trap, force = poly_baseline_data
    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2)
    np.testing.assert_allclose(baseline.valid_range(), [12.95, 13.349], atol=1e-3)
    np.testing.assert_allclose(
        baseline.correct_data(force, trap).data, np.zeros(trap.data.shape), atol=1e-6
    )


def test_baseline_downsampled(poly_baseline_data, downsampling_factor):
    trap, force = poly_baseline_data

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2, downsampling_factor=500)
    np.testing.assert_allclose(baseline.valid_range(), [12.95, 13.349], atol=1e-3)
    np.testing.assert_allclose(
        baseline.correct_data(force, trap).data, np.zeros(trap.data.shape), atol=1e-4
    )
    np.testing.assert_allclose(baseline._trap_data, trap.downsampled_by(500))
    np.testing.assert_allclose(baseline._force, force.downsampled_by(500 * downsampling_factor))


def test_baseline_plots(poly_baseline_data):
    trap, force = poly_baseline_data

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2)
    baseline.plot()
    baseline.plot_residual()


@pytest.mark.parametrize("force_start, force_stop", [(0, 1000), (110, 1000), (0, 980), (110, 980)])
def test_baseline_matches_channel_ranges(force_start, force_stop):
    trap = Slice(Continuous(np.arange(100, 1000, 50), start=100, dt=50))
    force = Slice(
        Continuous(2 * np.arange(force_start, force_stop, 10) + 1, start=force_start, dt=10)
    )
    start = max(100, 100 + 50 * int(np.ceil((force_start - 100) / 50)))
    stop = 100 + 50 * ((force_stop - 100) // 50)
    expected_trap = trap[start:stop]
    expected_force = force[start:stop].downsampled_by(5, where="left")

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=1)
    np.testing.assert_array_equal(baseline._trap_data.timestamps, expected_trap.timestamps)
    np.testing.assert_array_equal(baseline._force.timestamps, expected_force.timestamps)
    np.testing.assert_allclose(baseline._force.data, expected_force.data)
    np.testing.assert_allclose(baseline.correct_data(force, trap).data, 0, atol=1e-10)
    direct = ForceBaseLine(baseline._model, trap, force)
    np.testing.assert_array_equal(direct._trap_data.timestamps, expected_trap.timestamps)
    np.testing.assert_array_equal(direct._force.timestamps, expected_force.timestamps)


@pytest.mark.parametrize("force_start, force_length", [(1000, 10), (0, 0), (1, 1)])
def test_baseline_no_complete_overlap(force_start, force_length):
    trap = Slice(Continuous(np.arange(10), start=0, dt=50))
    force = Slice(Continuous(np.arange(force_length), start=force_start, dt=10))
    with pytest.raises(ValueError, match="No complete overlapping samples"):
        ForceBaseLine.polynomial_baseline(trap, force, degree=1)


def test_manually_downsampled_force_error():
    trap = Slice(Continuous(np.arange(20), start=40000, dt=50000))
    force = Slice(Continuous(np.arange(100), start=0, dt=10000))
    averaged_force = force.downsampled_by(5)
    message = "manually downsampled.*native trap-position and high-frequency force channels"

    with pytest.raises(RuntimeError, match=message):
        ForceBaseLine.polynomial_baseline(trap, averaged_force, degree=1)

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=1)
    with pytest.raises(RuntimeError, match=message):
        baseline.correct_data(averaged_force, trap)


@pytest.mark.parametrize("offset_ns", [1, 1000000, 1000001])
def test_timestamp_grid_offset(offset_ns):
    trap = Slice(Continuous(np.arange(200), start=0, dt=50000))
    force = Slice(Continuous(np.arange(200), start=offset_ns, dt=50000))
    if offset_ns % 50000:
        with pytest.raises(RuntimeError, match="timestamps should match"):
            ForceBaseLine.polynomial_baseline(trap, force, degree=1)
    else:
        baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=1)
        np.testing.assert_array_equal(baseline._trap_data.timestamps, baseline._force.timestamps)
        assert baseline._trap_data.start == offset_ns
        assert len(baseline._trap_data) == 180


def test_same_timestamps(poly_baseline_data):
    """Verify that the implementation checks that the provided data is time aligned."""
    trap, force = poly_baseline_data

    misaligned = deepcopy(trap)
    misaligned._src.start += 1

    with pytest.raises(RuntimeError, match="force and trap position timestamps should match"):
        ForceBaseLine.polynomial_baseline(misaligned, force, degree=2, downsampling_factor=500)

    with pytest.raises(RuntimeError, match="force and trap position timestamps should match"):
        baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2, downsampling_factor=500)
        baseline.correct_data(force, misaligned)
