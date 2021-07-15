from scipy.stats import chi2
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt
from warnings import warn
import enum


def clamp_step(x_origin, x_step, lb, ub):
    """Shortens a step to stay within some box constraints."""
    assert np.all(
        np.logical_and(x_origin >= lb, x_origin <= ub)
    ), "Initial position was not in box constraints."

    alpha_ub = np.inf * np.ones(x_step.shape)
    alpha_lb = np.inf * np.ones(x_step.shape)

    # Fetch distance to the boundary in multiples of the step size. Steps towards the boundary are positive.
    mask = x_step != 0
    alpha_ub[mask] = (ub[mask] - x_origin[mask]) / x_step[mask]
    alpha_lb[mask] = (lb[mask] - x_origin[mask]) / x_step[mask]

    # 1. Grab the distances that are moving towards the boundary (np.maximum). One will typically be negative, moving
    # away from the boundary, the other will be positive.
    # 2. Take the one that is closest to the boundary (np.min). If it's smaller than one, we need to shrink the step.
    scaling = np.min(np.maximum(alpha_ub, alpha_lb))
    if scaling > 1.0:
        scaling = 1.0

    return x_origin + scaling * x_step, scaling != 1.0


class StepConfig(NamedTuple):
    """
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


class ScanConfig(NamedTuple):
    """
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
    """

    lower_bounds: np.array
    upper_bounds: np.array
    fitted: np.array
    step_function: callable
    termination_level: float


class ProfileInfo(NamedTuple):
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
        # Make sure the parameter step doesn't exceed the bounds. If it goes over the edge, scale it back.
        current_step_size = new_step_size
        p_trial, clamped = clamp_step(
            parameter_vector,
            step_sign * current_step_size * step_direction,
            step_config.lower_bounds,
            step_config.upper_bounds,
        )

        if clamped:
            last_step = StepCode.shrink

        chi2_trial = chi2_function(p_trial)
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
    step = 0
    p_next = parameter_vector
    chi2_list = []
    parameter_vectors = []

    for step in range(num_steps):
        current_step_size, p_next = scan_config.step_function(
            chi2_last, p_next, current_step_size, step_sign
        )
        p_next = fit_function(
            p_next, scan_config.lower_bounds, scan_config.upper_bounds, scan_config.fitted
        )
        chi2_last = chi2_function(p_next)
        chi2_list.append(chi2_last)
        parameter_vectors.append(p_next)

        if verbose:
            print(f"Iteration {step}: Step size: {current_step_size}, p_next: {p_next}")

        step += 1
        if chi2_last > scan_config.termination_level:
            break

    return np.array(chi2_list), np.vstack(parameter_vectors)


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
    ):
        self.parameter_name = parameter_name

        # These are the user exposed options. They can be modified by the user in the struct if desired. THey are parsed
        # into actual algorithm parameters once the algorithm starts.
        self.options = {
            "min_step": min_step,
            "max_step": max_step,
            "step_factor": step_factor,
            "min_chi2_step": min_chi2_step,
            "max_chi2_step": max_chi2_step,
            "termination_significance": termination_significance,
            "confidence_level": confidence_level,
            "num_dof": num_dof,
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
        options = self.options

        assert options["max_step"] > options["min_step"]
        assert options["max_chi2_step"] > options["min_chi2_step"]

        self.profile_info = ProfileInfo(
            minimum_chi2=chi2_function(parameters.values),
            profiled_parameter_index=list(parameters.keys()).index(parameter_name),
            delta_chi2=chi2.ppf(options["confidence_level"], options["num_dof"]),
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
            + chi2.ppf(options["termination_significance"], options["num_dof"]),
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

    @property
    def parameters(self):
        return np.vstack((np.flipud(self._parameters["bwd"]), self._parameters["fwd"]))

    @property
    def chi2(self):
        return np.hstack((np.flipud(self._chi2["bwd"]), self._chi2["fwd"]))

    @property
    def p(self):
        return self.parameters[:, self.profile_info.profiled_parameter_index]

    def plot(self, **kwargs):
        dash_length = 5
        plt.plot(self.p, self.chi2, **kwargs)
        confidence_chi2 = self.profile_info.minimum_chi2 + self.profile_info.delta_chi2
        plt.axhline(y=confidence_chi2, linewidth=1, color="k", dashes=[dash_length, dash_length])

        ci_min, ci_max = self.lower_bound, self.upper_bound
        if ci_min:
            plt.axvline(x=ci_min, linewidth=1, color="k", dashes=[dash_length, dash_length])
        if ci_max:
            plt.axvline(x=ci_max, linewidth=1, color="k", dashes=[dash_length, dash_length])

        plt.text(min(self.p), confidence_chi2, f"{self.profile_info.confidence_level * 100}%")
        plt.ylabel("$\\chi^2$")
        plt.xlabel(self.parameter_name)
        plt.ylim(
            [
                self.profile_info.minimum_chi2 - 0.1 * self.profile_info.delta_chi2,
                self.profile_info.minimum_chi2 + 1.1 * self.profile_info.delta_chi2,
            ]
        )

    def plot_relations(self, params={}, **kwargs):
        """Plot the relations between the different parameters.

        Parameters
        ----------
        params : Set[str]
            List of parameter names to plot (optional, omission plots all)
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`."""
        parameters = self.parameters

        if len(params) == 0:
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
