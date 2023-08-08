import enum
from typing import Tuple, Optional
from warnings import warn
from dataclasses import dataclass

import numpy as np
import scipy


def _validate_in_bound(error_description, params, lower_bounds, upper_bounds, bound_tolerance):
    """Check whether parameters are in bound and raise if they're not.

    Parameters
    ----------
    error_description : str
        Description to be included as part of the error message raised when bounds are violated.
    params : np.ndarray
        Parameter vector
    lower_bounds, upper_bounds : np.ndarray
        Bounds
    bound_tolerance : float
        Tolerance used when checking the bound. This tolerance uses whichever is the most
        permissive of an absolute and relative tolerance threshold.

    Raises
    ------
    RuntimeError
        If params is not within the bounds.
    """
    # Only use tolerance if it is nonzero, otherwise one gets issues with 0 * np.inf being undefined
    if bound_tolerance:
        lower = lower_bounds - bound_tolerance * np.maximum(abs(lower_bounds), 1.0)
        upper = upper_bounds + bound_tolerance * np.maximum(abs(upper_bounds), 1.0)
    else:
        lower, upper = lower_bounds, upper_bounds

    if np.any(params > upper) or np.any(params < lower):
        over_str = " ".join(
            f"Param {idx} was over limit by {params[idx] - upper_bounds[idx]}."
            for idx in np.atleast_1d(np.argwhere(params > upper_bounds).squeeze())
        )
        under_str = " ".join(
            f"Param {idx} was under limit by {lower_bounds[idx] - params[idx]}."
            for idx in np.atleast_1d(np.argwhere(params < lower_bounds).squeeze())
        )

        raise RuntimeError(
            f"{error_description}\nCurrent parameter values: {params}. "
            f"Lower bound: {lower_bounds}, upper bound: {upper_bounds}. {over_str}{under_str}"
        )


def clamp_step(x_origin, x_step, lower_bound, upper_bound):
    """Shortens a step to stay within some box constraints.

    Parameters
    ----------
    x_origin : np.ndarray
        Parameter position we are stepping from.
    x_step : np.ndarray
        Step direction.
    lower_bound, upper_bound : np.ndarray
        Lower and upper bound for the parameter vector.

    Returns
    -------
    new_position : np.ndarray
        New position
    scaling : bool
        Have we shrunk the step?
    """
    alpha_ub = np.inf * np.ones(x_step.shape)
    alpha_lb = np.inf * np.ones(x_step.shape)

    # Fetch distance to the boundary in multiples of the step size. Steps towards the boundary are
    # positive.
    mask = x_step != 0
    alpha_ub[mask] = (upper_bound[mask] - x_origin[mask]) / x_step[mask]
    alpha_lb[mask] = (lower_bound[mask] - x_origin[mask]) / x_step[mask]

    # 1. Grab the distances that are moving towards the boundary (np.maximum). One will typically
    #    be negative, moving away from the boundary, the other will be positive.
    # 2. Take the one that is closest to the boundary (np.min). If it's smaller than one, we need
    #    to shrink the step.
    scaling = np.min(np.maximum(alpha_ub, alpha_lb))
    if scaling > 1.0:
        scaling = 1.0

    return x_origin + scaling * x_step, scaling != 1.0


@dataclass
class StepConfig:
    """Profile likelihood stepsize control configuration

    min_abs_step: float
        minimal step size in parameter space
    max_abs_step: float
        maximal step size in parameter space
    step_factor : float
        factor by which to change the step size
    min_chi2_step_size: float
        minimal step size in chi-squared space
    max_chi2_step_size: float
        maximal step size in chi-squared space
    lower_bound: array_like
        minimal values the parameters can take
    upper_bound: array_like
        maximal value the parameters can take
    """

    min_abs_step: float
    max_abs_step: float
    step_factor: float
    min_chi2_step_size: float
    max_chi2_step_size: float
    lower_bounds: np.array
    upper_bounds: np.array


@dataclass
class ScanConfig:
    """Profile likelihood optimization configuration

    lower_bounds: np.array
        optimization lower bounds
    upper_bounds: np.array
        optimization upper bounds
    fitted: np.array
        which parameters are fitted?
    step_function: callable
        function which performs 1D line scans
    termination_level: float
        chi squared value at which the optimization terminates
    bound_tolerance : float
        tolerance to use when verifying whether solution is inside the bounds
    """

    lower_bounds: np.array
    upper_bounds: np.array
    fitted: np.array
    step_function: callable
    termination_level: float
    bound_tolerance: float


@dataclass
class ProfileInfo:
    minimum_chi2: float
    profiled_parameter_index: int
    delta_chi2: float
    confidence_level: float
    parameter_names: list


class StepCode(enum.IntEnum):
    nochange = 0
    grow = 1
    shrink = 2

    @staticmethod
    def identify_step(new_step_size, current_step_size):
        if new_step_size > current_step_size:
            return StepCode.grow
        elif new_step_size < current_step_size:
            return StepCode.shrink
        else:
            return StepCode.nochange


def do_step(
    chi2_function,
    step_direction_function,
    chi2_last,
    parameter_vector,
    current_step_size,
    step_sign,
    step_config,
):
    """
    Parameters
    ----------
    chi2_function: callable
        function which returns the current chi2 value
    step_direction_function: callable
        function which returns normalized direction in parameter space in which steps are taken
    chi2_last: float
        previous chi squared value
    parameter_vector: array_like
        current parameter vector
    current_step_size: float
        current step size
    step_sign: float
        direction in which to step (-1.0 or 1.0)
    step_config: StepConfig
        configuration for the stepping algorithm
    """
    # Determine an appropriate step size based on chi2 increase
    last_step = StepCode.grow
    new_step_size = np.min([current_step_size, step_config.max_abs_step])

    step_direction = step_direction_function()
    while last_step != StepCode.nochange:
        # Make sure the parameter step doesn't exceed the bounds. If it goes over the edge, scale
        # it back.
        current_step_size = new_step_size
        p_trial, clamped = clamp_step(
            parameter_vector,
            step_sign * current_step_size * step_direction,
            step_config.lower_bounds,
            step_config.upper_bounds,
        )

        if clamped:
            last_step = StepCode.shrink

        try:
            chi2_trial = chi2_function(p_trial)
        except (RuntimeError, ValueError):
            # During stepping we can encounter places where the model fails, this is fine. Just make
            # sure we flag these as non-acceptable steps and the algorithm will shrink the step and
            # continue.
            chi2_trial = np.inf

        chi2_change = chi2_trial - chi2_last

        if chi2_change < step_config.min_chi2_step_size:
            # Do not increase the step-size if we just shrunk. Otherwise we loop forever.
            if last_step == StepCode.shrink:
                break

            new_step_size = min(new_step_size * step_config.step_factor, step_config.max_abs_step)
        elif chi2_change > step_config.max_chi2_step_size:
            new_step_size = max(new_step_size / step_config.step_factor, step_config.min_abs_step)

        last_step = StepCode.identify_step(new_step_size, current_step_size)

    if current_step_size == step_config.min_abs_step:
        warn("Warning: Step size set to minimum step size.", RuntimeWarning)

    return current_step_size, p_trial


def scan_dir_optimisation(
    chi2_function,
    fit_function,
    chi2_last,
    parameter_vector,
    num_steps,
    step_sign,
    scan_config,
    verbose,
):
    """Makes a 1D scan, optimizing parameters at every level.

    Parameters
    ----------
    chi2_function: callable
        function which returns the current chi2 value. These values will be returned as fitting 'scores'
    fit_function: callable
        function which when called fits the model and returns the fitted values
    chi2_last: float
        previous chi squared value
    parameter_vector: array_like
        current parameter vector
    num_steps: float
        number of steps to evaluate
    step_sign: float
        direction in which to step (-1 or 1)
    scan_config: ScanConfig
        configuration for a line scan
    verbose: bool
        produce verbose output
    """
    current_step_size = 1
    p_next = parameter_vector
    chi2_list = []
    parameter_vectors = []

    for step in range(num_steps):
        current_step_size, p_next = scan_config.step_function(
            chi2_last, p_next, current_step_size, step_sign
        )
        try:
            p_next = fit_function(
                p_next, scan_config.lower_bounds, scan_config.upper_bounds, scan_config.fitted
            )

            _validate_in_bound(
                "Optimization failed to stay in bound.",
                p_next,
                scan_config.lower_bounds,
                scan_config.upper_bounds,
                scan_config.bound_tolerance,
            )

        except (ValueError, RuntimeError) as exception:
            warn(
                f"Optimization error encountered at iteration {step}, while attempting "
                f"parameter values: {p_next}:\n {repr(exception)}.\n\nTerminating profile. "
                f"Parameter bound not found.",
                RuntimeWarning,
            )
            break

        chi2_last = chi2_function(p_next)
        chi2_list.append(chi2_last)
        parameter_vectors.append(p_next)

        if verbose:
            print(f"Iteration {step}: Step size: {current_step_size}, p_next: {p_next}")

        step += 1
        if chi2_last > scan_config.termination_level:
            break

    return np.array(chi2_list), (
        np.vstack(parameter_vectors) if parameter_vectors else np.empty((0, parameter_vector.size))
    )


def find_crossing(x, y, crossing):
    (indices,) = np.where(y >= crossing)

    if np.any(indices):
        index = indices[0]
        if index > 0:
            x1, x2, y1, y2 = x[index - 1], x[index], y[index - 1], y[index]

        dydx = (y2 - y1) / (x2 - x1)
        return x1 + (crossing - y1) / dydx


class ProfileLikelihood1D:
    def __init__(
        self,
        parameter_name,
        min_step=1e-4,
        max_step=1.0,
        step_factor=2.0,
        min_chi2_step=0.05,
        max_chi2_step=0.5,
        termination_significance=0.99,
        confidence_level=0.95,
        num_dof=1,
        bound_tolerance=0.0,
    ):
        """Profile likelihood

        This method traces an optimal path through parameter space in order to estimate parameter
        confidence intervals. It iteratively performs a step for the profiled parameter, then
        fixes that parameter and re-optimizes all the other parameters [1]_ [2]_.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to profile
        min_step : float
            Minimum step size, default: 1e-4
        max_step : float
            Maximum step size, default: 1.0
        step_factor : float
            Step factor. This is by what ratio the stepsize is increased when stepping too slow (low
            increase in likelihood) or decreased when stepping too fast. Default: 2.0.
        min_chi2_step : float
            Minimum increase in chi-squared that we aim for in each step. Going below this limit
            results in the step size being increased. Default: 0.05.
        max_chi2_step : float
            Maximum increase in chi-squared that we aim for in each step. Going above this limit
            results in the step being rejected and the step size being decreased. Default: 0.5
        termination_significance : float
            At what significance level should the profiling terminate. Default: 0.99.
        confidence_level : float
            At which confidence level should the confidence interval be determined. Default: 0.95
        num_dof : int
            Number of degrees of freedom. Default: 1.
        bound_tolerance : float
            Bound tolerance. By default, the profiling procedure checks whether the solver stayed
            within the user specified parameter bounds. The valid range specified here is given as
            [lower_bnd - tol * max(1.0, abs(lower_bnd)), upper_bnd + tol * max(1.0, abs(upper_bnd))]
            Default: 0.

        References
        ----------
        .. [1] Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U.,
               & Timmer, J. (2009). Structural and practical identifiability analysis of partially
               observed dynamical models by exploiting the profile likelihood. Bioinformatics,
               25(15), 1923-1929.
        .. [2] Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew,
               F., Bock, H.H., Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to
               its limit: profile likelihood based model reduction. PloS one, 11(9).
        """
        self.parameter_name = parameter_name

        # These are the user exposed options. They can be modified by the user in the struct if
        # desired. They are parsed into actual algorithm parameters once the algorithm starts.
        self.options = {
            "min_step": min_step,
            "max_step": max_step,
            "step_factor": step_factor,
            "min_chi2_step": min_chi2_step,
            "max_chi2_step": max_chi2_step,
            "termination_significance": termination_significance,
            "confidence_level": confidence_level,
            "num_dof": num_dof,
            "bound_tolerance": bound_tolerance,
        }

        self.profile_info = None
        self._chi2 = {}
        self._parameters = {}

    def __str__(self):
        def bound_string(value):
            return f"{value:.2f}" if value else "undefined"

        return (
            f"Profile likelihood for {self.parameter_name} ({len(self.chi2)} points)\n"
            f"  - chi2\n"
            f"  - p\n"
            f"  - lower_bound: {bound_string(self.lower_bound)}\n"
            f"  - upper_bound: {bound_string(self.upper_bound)}\n"
        )

    def prepare_profile(self, chi2_function, fit_function, parameters, parameter_name):
        """Sets up internal data structure for performing a profile likelihood.

        Parameters
        ----------
        chi2_function : callable
            Function which takes Parameters and returns a chi-squared value.
        fit_function : callable
            Function which takes a parameter vector and bounds and performs parameter optimization.
        parameters : pylake.fitting.parameters.Params
            Initial model parameters (best fit values).
        parameter_name : str
            Parameter to profile.

        Returns
        -------
        scan_direction : callable
            Function which makes a 1D scan, optimizing parameters at every level.

        Raises
        ------
        RuntimeError
            If `options["max_step"]` < `options["min_step"]` or
            `options["max_chi2_step"]` < `options["min_chi2_step"]`.
        """
        options = self.options
        if options["max_step"] <= options["min_step"]:
            raise RuntimeError(
                f"max_step must be larger than min_step, got max_step={options['max_step']} and "
                f"min_step={options['min_step']}."
            )

        if options["max_chi2_step"] <= options["min_chi2_step"]:
            raise RuntimeError(
                "max_chi2_step must be larger than min_chi2_step, got max_chi2_step="
                f"{options['max_chi2_step']} and min_chi2_step={options['min_chi2_step']}."
            )

        _validate_in_bound(
            "Initial position was not in box constraints.",
            parameters.values,
            parameters.lower_bounds,
            parameters.upper_bounds,
            options["bound_tolerance"],
        )

        self.profile_info = ProfileInfo(
            minimum_chi2=chi2_function(parameters.values),
            profiled_parameter_index=list(parameters.keys()).index(parameter_name),
            delta_chi2=scipy.stats.chi2.ppf(options["confidence_level"], options["num_dof"]),
            confidence_level=options["confidence_level"],
            parameter_names=list(parameters.keys()),
        )

        fitted = parameters.fitted
        fitted[self.profile_info.profiled_parameter_index] = 0

        # TODO: Allow more complex step direction functions based on approximate Hessian information
        step_direction = np.zeros(len(parameters))
        step_direction[self.profile_info.profiled_parameter_index] = 1.0

        def step_direction_function():
            return step_direction

        step_config = StepConfig(
            min_abs_step=options["min_step"] * np.abs(parameters[parameter_name].value),
            max_abs_step=options["max_step"] * np.abs(parameters[parameter_name].value),
            step_factor=options["step_factor"],
            min_chi2_step_size=options["min_chi2_step"] * self.profile_info.delta_chi2,
            max_chi2_step_size=options["max_chi2_step"] * self.profile_info.delta_chi2,
            lower_bounds=parameters.lower_bounds,
            upper_bounds=parameters.upper_bounds,
        )

        def step_function(chi2_last, parameter_vector, current_step_size, step_sign):
            return do_step(
                chi2_function,
                step_direction_function,
                chi2_last,
                parameter_vector,
                current_step_size,
                step_sign,
                step_config,
            )

        scan_config = ScanConfig(
            lower_bounds=parameters.lower_bounds,
            upper_bounds=parameters.upper_bounds,
            fitted=fitted,
            step_function=step_function,
            termination_level=self.profile_info.minimum_chi2
            + scipy.stats.chi2.ppf(options["termination_significance"], options["num_dof"]),
            bound_tolerance=options["bound_tolerance"],
        )

        def scan_direction(chi2_last, parameter_vector, step_sign, num_steps, verbose):
            return scan_dir_optimisation(
                chi2_function,
                fit_function,
                chi2_last,
                parameter_vector,
                num_steps,
                step_sign,
                scan_config,
                verbose,
            )

        return scan_direction

    # TODO: Add mechanism which stores a hash coming from the FitObject and validates it against what's already done
    def _extend_profile(self, chi2_function, fit_function, parameters, num_steps, forward, verbose):
        scan_direction = self.prepare_profile(
            chi2_function, fit_function, parameters, self.parameter_name
        )

        field = "fwd" if forward else "bwd"
        parameter_vectors = (
            self._parameters[field]
            if field in self._parameters
            else np.expand_dims(parameters.values, 0)
        )
        initial_parameters = parameter_vectors[-1, :]
        chi2_list = self._chi2[field] if field in self._chi2 else [self.profile_info.minimum_chi2]
        chi2_last = chi2_list[-1]

        chi2_new, parameters_new = scan_direction(
            chi2_last, initial_parameters, 1 if forward else -1, num_steps, verbose
        )

        self._parameters[field] = np.vstack((parameter_vectors, parameters_new))
        self._chi2[field] = np.hstack((chi2_list, chi2_new))

    @property
    def lower_bound(self):
        cutoff = self.profile_info.minimum_chi2 + self.profile_info.delta_chi2
        p_index = self.profile_info.profiled_parameter_index
        return find_crossing(self._parameters["bwd"][:, p_index], self._chi2["bwd"], cutoff)

    @property
    def upper_bound(self):
        cutoff = self.profile_info.minimum_chi2 + self.profile_info.delta_chi2
        p_index = self.profile_info.profiled_parameter_index
        return find_crossing(self._parameters["fwd"][:, p_index], self._chi2["fwd"], cutoff)

    def get_interval(self, significance_level=None) -> Tuple[Optional[float], Optional[float]]:
        """Calculate confidence interval at a particular significance level

        Parameter
        ---------
        significance level : float
            Desired significance level (resulting in a 100 * (1 - alpha)% confidence interval).
            If omitted, the value specified when creating the profile is used.

        Returns
        -------
        lower_bound : float, optional
            Lower bound of the confidence interval.
        upper_bound : float, optional
            Upper bound of the confidence interval. If a bound cannot be determined (either due to
            an insufficient number of steps or lack of information in the data, the bound is given
            as `None`).

        Raises
        ------
        RuntimeError
            If significance level is chosen higher than the termination level of the profile.
        """
        profiled_level = 1.0 - self.options["termination_significance"]
        significance_level = (
            significance_level
            if significance_level is not None
            else 1.0 - self.options["confidence_level"]
        )

        if significance_level <= profiled_level:
            raise RuntimeError(
                f"Significance level ({significance_level}) cannot be chosen lower or equal than "
                f"the minimum profiled level ({profiled_level:.2f})."
            )

        cutoff = self.profile_info.minimum_chi2 + scipy.stats.chi2.ppf(
            1.0 - significance_level, self.options["num_dof"]
        )

        p_index = self.profile_info.profiled_parameter_index
        return (
            find_crossing(self._parameters["bwd"][:, p_index], self._chi2["bwd"], cutoff),
            find_crossing(self._parameters["fwd"][:, p_index], self._chi2["fwd"], cutoff),
        )

    @property
    def parameters(self):
        return np.vstack((np.flipud(self._parameters["bwd"]), self._parameters["fwd"]))

    @property
    def chi2(self):
        return np.hstack((np.flipud(self._chi2["bwd"]), self._chi2["fwd"]))

    @property
    def p(self):
        return self.parameters[:, self.profile_info.profiled_parameter_index]

    def plot(self, *, significance_level=None, **kwargs):
        """Plot profile likelihood

        Parameters
        ----------
        significance_level : float, optional
            Desired significance level  (resulting in a 100 * (1 - alpha)% confidence interval) to
            plot. Default is the significance level specified when the profile was generated.
        """
        import matplotlib.pyplot as plt

        dash_length = 5
        plt.plot(self.p, self.chi2, **kwargs)

        confidence_coefficient = (
            1.0 - significance_level
            if significance_level is not None
            else self.profile_info.confidence_level
        )
        delta_chi2 = scipy.stats.chi2.ppf(confidence_coefficient, self.options["num_dof"])
        confidence_chi2 = self.profile_info.minimum_chi2 + delta_chi2
        plt.axhline(y=confidence_chi2, linewidth=1, color="k", dashes=[dash_length, dash_length])

        ci_min, ci_max = self.get_interval(significance_level)
        if ci_min:
            plt.axvline(x=ci_min, linewidth=1, color="k", dashes=[dash_length, dash_length])
        if ci_max:
            plt.axvline(x=ci_max, linewidth=1, color="k", dashes=[dash_length, dash_length])

        plt.text(min(self.p), confidence_chi2, f"{confidence_coefficient * 100}%")
        plt.ylabel("$\\chi^2$")
        plt.xlabel(self.parameter_name)
        plt.ylim(
            [
                self.profile_info.minimum_chi2 - 0.1 * delta_chi2,
                self.profile_info.minimum_chi2 + 1.1 * delta_chi2,
            ]
        )

    def plot_relations(self, params=None, **kwargs):
        """Plot the relations between the different parameters.

        Parameters
        ----------
        params : Optional[Set[str]]
            List of parameter names to plot (optional, omission plots all)
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`."""
        import matplotlib.pyplot as plt

        parameters = self.parameters

        if not params:
            other = [
                x
                for x in range(parameters.shape[1])
                if x != self.profile_info.profiled_parameter_index
            ]
        else:
            other = [
                x
                for x in range(parameters.shape[1])
                if x != self.profile_info.profiled_parameter_index
                and self.profile_info.parameter_names[x] in params
            ]

        line_handles = plt.plot(
            parameters[:, self.profile_info.profiled_parameter_index],
            self.parameters[:, other],
            **kwargs,
        )
        plt.ylabel("Other parameter value")
        plt.xlabel(self.parameter_name)
        plt.legend(line_handles, [self.profile_info.parameter_names[idx] for idx in other])
