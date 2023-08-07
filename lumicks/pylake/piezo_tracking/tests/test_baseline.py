from copy import deepcopy

import numpy as np
import pytest

from lumicks.pylake.piezo_tracking.baseline import ForceBaseLine


def test_baseline(poly_baseline_data):
    trap, force = poly_baseline_data
    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2)
    np.testing.assert_allclose(baseline.valid_range(), [12.95, 13.35])
    np.testing.assert_allclose(
        baseline.correct_data(force, trap).data, np.zeros(force.data.shape), atol=1e-6
    )


def test_baseline_downsampled(poly_baseline_data):
    trap, force = poly_baseline_data

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2, downsampling_factor=500)
    np.testing.assert_allclose(baseline.valid_range(), [12.95, 13.349626])
    np.testing.assert_allclose(
        baseline.correct_data(force, trap).data, np.zeros(force.data.shape), atol=1e-4
    )
    np.testing.assert_allclose(baseline._trap_data, trap.downsampled_by(500))
    np.testing.assert_allclose(baseline._force, force.downsampled_by(500))


def test_baseline_plots(poly_baseline_data):
    trap, force = poly_baseline_data

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2)
    baseline.plot()
    baseline.plot_residual()


def test_same_timestamps(poly_baseline_data):
    """Verify that the implementation checks that the provided data is time aligned."""
    trap, force = poly_baseline_data

    misaligned = deepcopy(trap)
    misaligned._src.start += 1000000

    with pytest.raises(RuntimeError, match="force and trap position timestamps should match"):
        ForceBaseLine.polynomial_baseline(misaligned, force, degree=2, downsampling_factor=500)

    with pytest.raises(RuntimeError, match="force and trap position timestamps should match"):
        baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2, downsampling_factor=500)
        baseline.correct_data(force, misaligned)
