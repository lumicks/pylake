import numpy as np

from lumicks.pylake.force_calibration.calibration_models import diode_params_from_voltage
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    _fit_power_spectra,
    fit_power_spectrum,
)


def fit_multi_spectra(model, powers, power_spectra, loss="gaussian"):
    power, num_points_per_block = [
        np.hstack([getattr(s, prop) for s in power_spectra])
        for prop in ("power", "num_points_per_block")
    ]

    if len(np.unique(num_points_per_block)) != 1:
        raise ValueError("All provided spectra should be averaged the same amount")

    # Use the lowest power spectra for initial guesses
    n_spectra = len(power_spectra)
    fc_min = fit_power_spectrum(power_spectra[0], model)["fc"].value
    fc_two = fit_power_spectrum(power_spectra[1], model)["fc"].value
    fc_per_power_guess = (fc_two - fc_min) / (powers[1] - powers[0])
    diffusion_constant = fit_power_spectrum(power_spectra[0], model)["D"].value
    diffusion_constants = np.full(fill_value=diffusion_constant, shape=(n_spectra,))

    filter_params = np.hstack([14000, 0.1] * n_spectra)
    filter_lb = np.hstack([1, 0] * n_spectra)
    filter_ub = np.hstack([1e6, 1] * n_spectra)

    initial_params = [fc_min, fc_per_power_guess, *diffusion_constants, *filter_params]
    lower_bounds = [1, 0, *np.zeros(diffusion_constants.shape), *filter_lb]
    upper_bounds = [np.inf, np.inf, *np.full(diffusion_constants.shape, np.inf), *filter_ub]

    # The drag coefficient is linearly proportional to the viscosity(T)
    #
    #  fc = stiffness / (2 pi drag)
    #
    #  D_physical = scipy.constants.k * (temperature + 273.15) / drag
    #
    #  Sensitivity might still depend on the laser power.
    def multi_model(_, fc_min, d_fc_per_power, *other_params):
        diffusion_constants = other_params[:n_spectra]
        corner_frequencies = fc_min + d_fc_per_power * powers

        filter_parameters = other_params[n_spectra:]
        f_diode = filter_parameters[::2]
        alpha = filter_parameters[1::2]

        return np.hstack(
            [
                model(ps.frequency, fc, diff, fd, a)
                for ps, fc, diff, fd, a in zip(
                    power_spectra, corner_frequencies, diffusion_constants, f_diode, alpha
                )
            ]
        )

    solution, err_estimates, chi_squared = _fit_power_spectra(
        multi_model,
        np.array([]),
        power,
        int(np.unique(num_points_per_block)),
        np.asarray(initial_params),
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
        ftol=1e-7,
        max_function_evals=10000,
        loss_function=loss,
    )

    fc_min, d_fc_per_power, *other_params = solution
    fc_min_err, d_fc_per_power_err, *other_params_err = err_estimates
    results = {
        "fc": fc_min + d_fc_per_power * powers,
        "D": other_params[:n_spectra],
        "f_diode": other_params[n_spectra::2],
        "alpha": other_params[n_spectra + 1 :: 2],
        "fc_err": fc_min_err + d_fc_per_power_err * powers,
        "D_err": other_params_err[:n_spectra],
        "f_diode_err": other_params_err[n_spectra::2],
        "alpha_err": other_params_err[n_spectra + 1 :: 2],
        "frequency": [],
        "ps": [],
        "ps_model": [],
        "ps_residual": [],
    }

    for ps, fc, diff, fd, a in zip(
        power_spectra, results["fc"], results["D"], results["f_diode"], results["alpha"]
    ):
        ps_model = model(ps.frequency, fc, diff, fd, a)
        results["frequency"].append(ps.frequency)
        results["ps"].append(ps.power)
        results["ps_model"].append(ps_model)
        results["ps_residual"].append(ps.power / ps_model)

    return results


def fit_diode_model(model, powers, power_spectra, loss="gaussian"):
    power, num_points_per_block = [
        np.hstack([getattr(s, prop) for s in power_spectra])
        for prop in ("power", "num_points_per_block")
    ]

    if len(np.unique(num_points_per_block)) != 1:
        raise ValueError("All provided spectra should be averaged the same amount")

    # Use the lowest power spectra for initial guesses
    n_spectra = len(power_spectra)
    fc_min = fit_power_spectrum(power_spectra[0], model)["fc"].value
    fc_two = fit_power_spectrum(power_spectra[1], model)["fc"].value
    fc_per_power_guess = (fc_two - fc_min) / (powers[1] - powers[0])
    diffusion_constant = fit_power_spectrum(power_spectra[0], model)["D"].value
    diffusion_constants = np.full(fill_value=diffusion_constant, shape=(n_spectra,))

    # delta_f_diode, rate_f_diode, max_f_diode, delta_alpha, rate_alpha, max_alpha
    filter_params = [8000, 1, 15000, 0.4, 1, 0.9]
    filter_lb = [0, 0, 0, 0, 0, 0]
    filter_ub = [50000, 1e6, 50000, 1.0, 1e6, 1.0]

    initial_params = [fc_min, fc_per_power_guess, *diffusion_constants, *filter_params]
    lower_bounds = [1, 0, *np.zeros(diffusion_constants.shape), *filter_lb]
    upper_bounds = [np.inf, np.inf, *np.full(diffusion_constants.shape, np.inf), *filter_ub]

    # The drag coefficient is linearly proportional to the viscosity(T)
    #
    #  fc = stiffness / (2 pi drag)
    #
    #  D_physical = scipy.constants.k * (temperature + 273.15) / drag
    #
    #  Sensitivity might still depend on the laser power however.
    def multi_model(_, fc_min, d_fc_per_power, *other_params):
        corner_frequencies = fc_min + d_fc_per_power * powers
        f_diode, alpha, _ = np.asarray(
            [diode_params_from_voltage(p, *other_params[n_spectra:]) for p in powers]
        ).T

        return np.hstack(
            [
                model(ps.frequency, fc, diff, fd, a)
                for ps, fc, diff, fd, a in zip(
                    power_spectra, corner_frequencies, other_params[:n_spectra], f_diode, alpha
                )
            ]
        )

    solution, err_estimates, chi_squared = _fit_power_spectra(
        multi_model,
        np.array([]),
        power,
        int(np.unique(num_points_per_block)),
        np.asarray(initial_params),
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
        ftol=1e-7,
        max_function_evals=10000,
        loss_function=loss,
    )

    fc_min, d_fc_per_power, *other_params = solution
    filter_params = other_params[n_spectra:]
    fc_min_err, d_fc_per_power_err, *other_params_err = err_estimates

    f_diode, alpha, _ = np.asarray([diode_params_from_voltage(p, *filter_params) for p in powers]).T

    results = {
        "fc": fc_min + d_fc_per_power * powers,
        "D": other_params[:n_spectra],
        "f_diode": f_diode,
        "alpha": alpha,
        "fc_err": fc_min_err + d_fc_per_power_err * powers,
        "D_err": other_params_err[:n_spectra],
        "frequency": [],
        "ps": [],
        "ps_model": [],
        "ps_residual": [],
    }

    for ps, fc, diff, fd, a in zip(
        power_spectra, results["fc"], results["D"], results["f_diode"], results["alpha"]
    ):
        ps_model = model(ps.frequency, fc, diff, fd, a)
        results["frequency"].append(ps.frequency)
        results["ps"].append(ps.power)
        results["ps_model"].append(ps_model)
        results["ps_residual"].append(ps.power / ps_model)

    return results
