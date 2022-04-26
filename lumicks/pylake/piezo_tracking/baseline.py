import matplotlib.pyplot as plt
import numpy as np
from lumicks.pylake.channel import Slice, Continuous
from csaps import csaps
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error


def unique_sorted(trap_position, force):
    """Sort and remove duplicates trap_position data to prepare for fit smoothing spline.
    Parameters
    ----------
    trap_position : lumicks.pylake.Slice
        Trap mirror position
    force : lumicks.pylake.Slice
        Force data
    """

    x = trap_position.data
    u, c = np.unique(x, return_counts=True)
    m = np.isin(x, [u[c < 2]])
    ind = np.argsort(x[m])

    return x[m][ind], force.data[m][ind]


def optimize_smoothing_factor(
    trap_position,
    force,
    smoothing_factors,
    n_repeats,
    plot_smoothingfactor_mse,
):
    """Find optimal smoothing factor by choosing smoothing factor with lowest mse on test data
    Parameters
    ----------
    trap_position : lumicks.pylake.Slice
        Trap mirror position data
    force : lumicks.pylake.Slice
        Force data
    smoothing_factors : np.array float
        Array of smoothing factors used for optimization fit, 0 <= smoothing_factor <= 1
    n_repeats: int
        number of times to repeat cross validation
    plot_smoothingfactor_mse: bool
        plot mse on test data vs smoothing factors used for optimization
    """

    mse_test_vals = np.zeros(len(smoothing_factors))
    x_sorted, y_sorted = unique_sorted(trap_position, force)
    for i, smooth in enumerate(smoothing_factors):
        mse_test_array = np.zeros(n_repeats * 2)

        rkf = RepeatedKFold(n_splits=2, n_repeats=n_repeats)
        for k, (train_index, test_index) in enumerate(rkf.split(x_sorted)):
            x_train, x_test = x_sorted[train_index], x_sorted[test_index]
            y_train, y_test = y_sorted[train_index], y_sorted[test_index]

            smoothing_result_train = csaps(x_train, y_train, smooth=smooth)
            f_test = smoothing_result_train(x_test)
            mse_test_array[k] = mean_squared_error(y_test, f_test)

        mse_test_vals[i] = np.mean(mse_test_array)
    if plot_smoothingfactor_mse:
        plot_mse_smoothing_factors(smoothing_factors, mse_test_vals)

    return smoothing_factors[np.argmin(mse_test_vals)]


def plot_mse_smoothing_factors(smoothing_factors, mse_test_vals):
    plt.figure()
    plt.plot(
        np.log(1 - smoothing_factors),
        mse_test_vals,
        label=f"optimal s= {smoothing_factors[np.argmin(mse_test_vals)]:0.6f}",
    )
    plt.ylabel("mse test")
    plt.xticks(np.log(1 - smoothing_factors), smoothing_factors)
    plt.xlabel("smoothing factor")
    plt.legend()
    plt.show()


class ForceBaseLine:
    def __init__(self, model, trap_data, force):
        """Force baseline

        Parameters
        ----------
        model : callable
            Model which returns the baseline at specified points.
        trap_data : lumicks.pylake.Slice
            Trap mirror position data
        force : lumicks.pylake.Slice
            Force data
        """
        self._model = model
        self._trap_data = trap_data
        self._force = force

    def valid_range(self):
        return (np.min(self._trap_data.data), np.max(self._trap_data.data))

    def correct_data(self, force, trap_position):
        if not np.array_equal(force.timestamps, trap_position.timestamps):
            raise RuntimeError("Provided force and trap position timestamps should match")

        return Slice(
            Continuous(
                force.data - self._model(trap_position.data),
                force._src.start,
                force._src.dt,
            ),
            labels={
                "title": force.labels.get("title", "Baseline Corrected Force"),
                "y": "Baseline Corrected Force (pN)",
            },
            calibration=force._calibration,
        )

    def plot(self):
        plt.scatter(self._trap_data.data, self._force.data, s=2)
        plt.plot(self._trap_data.data, self._model(self._trap_data.data), "k")
        plt.xlabel("Mirror position")
        plt.ylabel(self._force.labels["y"])
        plt.title("Force baseline")

    def plot_residual(self):
        plt.scatter(self._trap_data.data, self._force.data - self._model(self._trap_data.data), s=2)
        plt.xlabel("Mirror position")
        plt.ylabel(f"Residual {self._force.labels['y']}")
        plt.title("Fit residual")

    @classmethod
    def polynomial_baseline(cls, trap_position, force, degree=7, downsampling_factor=None):
        """Generate a polynomial baseline from data

        Parameters
        ----------
        trap_position : lumicks.pylake.Slice
            Trap mirror position data
        force : lumicks.pylake.Slice
            Force data
        degree : int
            Polynomial degree
        downsampling_factor : int
            Factor by which to downsample before baseline determination
        """
        if not np.array_equal(force.timestamps, trap_position.timestamps):
            raise RuntimeError("Provided force and trap position timestamps should match")

        if downsampling_factor:
            trap_position, force = (
                ch.downsampled_by(downsampling_factor) for ch in (trap_position, force)
            )

        model = np.poly1d(np.polyfit(trap_position.data, force.data, deg=degree))
        return cls(model, trap_position, force)\


    @classmethod
    def smoothingspline_baseline(
            cls,
            trap_position,
            force,
            smoothing_factor=None,
            downsampling_factor=None,
            smoothing_factors=np.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]),
            n_repeats=10,
            plot_smoothingfactor_mse=False,
    ):
        """Generate a smoothing spline baseline from data.
        Items of xdata in smoothing spline must satisfy: x1 < x2 < ... < xN,
        therefore the trap_position data is sorted and duplicates are removed
        Parameters
        ----------
        trap_position : lumicks.pylake.Slice
            Trap mirror position data
        force : lumicks.pylake.Slice
            Force data
        smoothing_factor : float
            Smoothing factor for smoothing spline, 0 <= smoothing_factor <= 1
        downsampling_factor : int
            Factor by which to downsample before baseline determination
        smoothing_factors : np.array float
            Array of smoothing factors used for optimization fit, 0 <= smoothing_factor <= 1
        n_repeats: int
            number of times to repeat cross validation
        plot_smoothingfactor_mse: bool
            plot mse on test data vs smoothing factors used for optimization
        """
        if not np.array_equal(force.timestamps, trap_position.timestamps):
            raise RuntimeError(
                "Provided force and trap position timestamps should match"
            )

        if downsampling_factor:
            trap_position, force = (
                ch.downsampled_by(downsampling_factor) for ch in (trap_position, force)
            )

        x_sorted, y_sorted = unique_sorted(trap_position, force)

        if smoothing_factor:
            model = csaps(x_sorted, y_sorted, smooth=smoothing_factor)
        else:
            smoothing_factor = optimize_smoothing_factor(
                trap_position,
                force,
                smoothing_factors=smoothing_factors,
                n_repeats=n_repeats,
                plot_smoothingfactor_mse=plot_smoothingfactor_mse,
            )
            model = csaps(x_sorted, y_sorted, smooth=smoothing_factor)

        return cls(model, trap_position, force)
