import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
import matplotlib.pyplot as plt


def mack_model(
    wavelength_nm,
    refractive_index_medium,
    nanostage_z_position,
    surface_position,
    displacement_sensitivity,
    intensity_amplitude,
    intensity_phase_shift,
    intensity_decay_length,
    scattering_polynomial_coeffs,
    focal_shift,
    nonlinear_shift,
):
    """Phenomenological model describing axial force near the surface.

    [1] Mack, A. H., Schlingman, D. J., Regan, L., & Mochrie, S. G. J. (2012). Practical axial
    optical trapping. Review of Scientific Instruments, 83(10), 103106.

    Parameters
    ----------
    wavelength_nm : float
        Trapping laser wavelength
    refractive_index_medium : float
        Refractive index of the medium
    nanostage_z_position : np.ndarray
        Nanostage z-position
    surface_position : float
        Position of the surface.
    displacement_sensitivity : float
        Displacement sensitivity in Z.
    intensity_amplitude : float
        Amplitude of the interference pattern.
    intensity_phase_shift : float
        Phase of the interference pattern.
    intensity_decay_length:
        Intensity decay length.
    scattering_polynomial_coeffs : np.ndarray
        Background polynomial coefficients.
    focal_shift : float
        Focal shift.
    nonlinear_shift : float
        Nonlinear focal shift.
    """

    def interference_pattern(h):
        background = np.polyval(scattering_polynomial_coeffs, h)
        bead_center_to_stage = np.polyval([nonlinear_shift, focal_shift, 0], h)
        laser_wavelength = wavelength_nm * 1e-3 / (2 * refractive_index_medium)
        k = 2.0 * np.pi / laser_wavelength

        return background + intensity_amplitude * np.exp(
            -intensity_decay_length * bead_center_to_stage
        ) * np.sin(k * bead_center_to_stage + intensity_phase_shift)

    h = surface_position - nanostage_z_position
    axial_force = np.zeros(h.shape)
    axial_force[h >= 0] = interference_pattern(h[h >= 0])
    axial_force[h < 0] = displacement_sensitivity * np.abs(h[h < 0]) + interference_pattern(0)

    return axial_force


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0])


def fit_piecewise_linear(x, y):
    """Fits a two-segment piecewise linear function and returns the parameters.

    Parameters:
        x: array-like
            independent variable
        y: array-like
            dependent variable
    """
    # We should be able to handle cases where x moves in positive or negative direction
    difference = x[-1] - x[0]
    center = x[0] + difference / 2
    mid_point = np.nonzero(x > center)[0][0] if difference > 0 else np.nonzero(x < center)[0][0]
    x_start, y_start, x_mid, y_mid, x_end, y_end = (
        (x[0], y[0], x[mid_point], y[mid_point], x[-1], y[-1])
        if difference > 0
        else (x[-1], y[-1], x[mid_point], y[mid_point], x[0], y[0])
    )

    slope1_est = (y_mid - y_start) / (x_mid - x_start)
    slope2_est = (y_end - y_mid) / (x_end - x_mid)
    initial_guess = [x_mid, y_mid, slope1_est, slope2_est]
    pars, _ = curve_fit(piecewise_linear, x, y, initial_guess)

    return pars


def fit_sine_with_polynomial(independent, dependent, freq_guess, freq_bounds, background_degree):
    """Fit a sine wave plus polynomial background.

    We wish to fit a sine wave (with phase shift) plus a polynomial. By using a trick we can rewrite
    this equation such that we only have to optimize over 1 variable:

        amp * sin(a * x + phase) + other

    can we written as:

        amp * sin(phase) * cos(a * x) + amp * cos(phase) * sin(a * x) + other

    By absorbing the sin and cosine into the amplitude, we change variables from amplitude and
    phase, to sine amplitude and cosine amplitude. This means the problem of fitting a sine plus
    polynomial is essentially a linear one in all variables except the frequency (our parameter of
    interest). Doing this allows us to estimate this using only one estimated variable.

    Parameters
    ----------
    independent : np.ndarray
        Values for the independent variable
    dependent : np.ndarray
        Values for the dependent variable
    freq_guess : float
        Initial guess for the frequency
    freq_bounds : 2-tuple of array_like
        Bounds for the frequency guess
    background_degree : int
        Polynomial degree to use to fit the background
    """

    def sine_with_polynomial(x, frequency):
        design_matrix = np.vstack(
            (
                np.sin(2.0 * np.pi * frequency * x),
                np.cos(2.0 * np.pi * frequency * x),
                x[np.newaxis, :] ** np.arange(0, background_degree + 1)[:, np.newaxis],
            ),
        )
        ests, _, _, _ = np.linalg.lstsq(design_matrix.T, dependent, rcond=None)
        return np.sum(design_matrix.T * ests, axis=1)

    par, _ = curve_fit(sine_with_polynomial, independent, dependent, freq_guess, bounds=freq_bounds)
    return par[0], sine_with_polynomial(independent, par[0])


@dataclass
class TouchdownResult:
    surface_position: float
    focal_shift: float
    nanostage_position: np.ndarray
    axial_force: np.ndarray
    surface_fit: np.ndarray
    interference_nanostage: np.ndarray
    interference_force: np.ndarray

    def plot(self, legend=True):
        plt.plot(self.nanostage_position, self.axial_force, label="Axial force")
        plt.plot(self.interference_nanostage, self.interference_force, label="Interference fit")
        plt.plot(self.nanostage_position, self.surface_fit, label="Piecewise linear fit")
        plt.axvline(self.surface_position, label="Determined surface position")
        if legend:
            plt.legend()


def touchdown(
    nanostage,
    axial_force,
    wavelength_nm=1064,
    refractive_index_medium=1.333,
    omit_microns=0.5,
    background_degree=3,
):
    """This function determines the surface and focal shift from an approach curve.

    We use a piecewise linear function to find the surface, and a sine fit (with a polynomial
    background to infer the focal shift from the interference pattern).

    Parameters
    ----------
    nanostage : np.ndarray
        Nanostage Z position.
    axial_force : np.ndarray
        Axial force.
    wavelength_nm : float
        Wavelength of the trapping laser in nanometers.
    refractive_index_medium : float
        Refractive index of the medium.
    omit_microns : float
        This parameter sets the gap between the surface and where we begin to fit the interference
        pattern.
    background_degree : int
        Degree of the polynomial to use for the background intensity when fitting the intensity
        pattern.
    """
    if len(nanostage) != len(axial_force):
        min_length = min(len(nanostage), len(axial_force))
        nanostage, axial_force = nanostage[:min_length], axial_force[:min_length]

    # Find the surface position
    piecewise_parameters = fit_piecewise_linear(nanostage, axial_force)
    surface_position = piecewise_parameters[0]

    mask = nanostage < (surface_position - omit_microns)
    stage_trimmed, force_trimmed = nanostage[mask], axial_force[mask]
    expected_wavelength = wavelength_nm * 1e-3 / 2 / refractive_index_medium

    # A poor initial estimate of the focal shift can lead to getting stuck in local optima
    # Hence we optimize from a range of starting values.
    bounds = np.array([0.5, 1.0001]) / expected_wavelength
    pars, errs, simulations = [], [], []
    for freq_guess in np.arange(*bounds, np.diff(bounds) / 10):
        par, simulation = fit_sine_with_polynomial(
            surface_position - stage_trimmed,
            force_trimmed,
            freq_guess,
            bounds,
            background_degree=background_degree,
        )
        err = np.sum((simulation - force_trimmed) ** 2)
        pars.append(par)
        errs.append(err)
        simulations.append(simulation)

    best_fit_index = np.argmin(errs)
    focal_shift = pars[best_fit_index] * expected_wavelength
    return TouchdownResult(
        surface_position=surface_position,
        focal_shift=focal_shift,
        nanostage_position=nanostage,
        axial_force=axial_force,
        surface_fit=piecewise_linear(nanostage, *piecewise_parameters),
        interference_nanostage=stage_trimmed,
        interference_force=simulations[best_fit_index],
    )
