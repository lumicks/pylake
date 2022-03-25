import numpy as np
from matplotlib.testing.decorators import cleanup
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


@cleanup
def test_baseline_plots(poly_baseline_data):
    trap, force = poly_baseline_data

    baseline = ForceBaseLine.polynomial_baseline(trap, force, degree=2)
    baseline.plot()
    baseline.plot_residual()
