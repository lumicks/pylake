from scipy.stats import chi2
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class StepConfig:
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
    """
    min_abs_step: float
    max_abs_step: float
    step_factor: float
    min_chi2_step_size: float
    max_chi2_step_size: float


@dataclass
class ScanConfig:
    """
    lb: np.array
        optimization lower bound
    ub: np.array
        optimization upper bound
    fitted: np.array
        which parameters are fitted?
    step_function: callable
        function which performs 1D line scans
    termination_level: float
        chi squared value at which the optimization terminates
    """
    lb: np.array
    ub: np.array
    fitted: np.array
    step_function: callable
    termination_level: float


@dataclass
class ProfileInfo:
    minimum_chi2: float
    profiled_parameter_index: int
    delta_chi2: float
    confidence_level: float
    parameter_names: list


def do_step(chi2_function, step_direction_function, chi2_last, parameter_vector, current_step_size, step_sign,
            step_config):
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
    adjust_trial = True
    just_shrunk = False
    step_direction = step_direction_function()
    while adjust_trial:
        p_trial = parameter_vector + step_sign * current_step_size * step_direction
        chi2_trial = chi2_function(p_trial)

        chi2_change = chi2_trial - chi2_last
        if chi2_change < step_config.min_chi2_step_size:
            # Do not increase the step-size if we just shrunk. We already know it's going to be bad and we'd
            # just be looping forever.
            if not just_shrunk:
                adjust_trial = True
                current_step_size = current_step_size * step_config.step_factor
                if current_step_size > step_config.max_abs_step:
                    current_step_size = step_config.max_abs_step
                    adjust_trial = False
            else:
                adjust_trial = False
        elif chi2_change > step_config.max_chi2_step_size:
            adjust_trial = True
            just_shrunk = True
            current_step_size = current_step_size / step_config.step_factor
            if current_step_size < step_config.min_abs_step:
                print("Warning: Step size set to minimum step size.")
                current_step_size = step_config.min_abs_step
                adjust_trial = False
        else:
            adjust_trial = False
            just_shrunk = False

    return current_step_size, parameter_vector + step_sign * current_step_size * step_direction


def scan_dir_optimisation(chi2_function, fit_function, chi2_last, parameter_vector, num_steps, step_sign, scan_config):
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
    """
    current_step_size = 1
    step = 0
    p_next = parameter_vector
    chi2_list = []
    parameter_vectors = []

    while step < num_steps:
        current_step_size, p_next = scan_config.step_function(chi2_last, p_next, current_step_size, step_sign)
        p_next = fit_function(p_next, scan_config.lb, scan_config.ub, scan_config.fitted)
        chi2_last = chi2_function(p_next)
        chi2_list.append(chi2_last)
        parameter_vectors.append(p_next)
        step += 1
        if chi2_last > scan_config.termination_level:
            break

    return np.array(chi2_list), np.vstack(parameter_vectors)


class ProfileLikelihood1D:
    def __init__(self, parameter_name, min_step=1e-4, max_step=1.0, step_factor=2.0, min_chi2_step=0.01,
                 max_chi2_step=0.2, termination_significance=.99, confidence_level=.95, num_dof=1):
        self.parameter_name = parameter_name

        # These are the user exposed options. They can be modified by the user in the struct if desired. THey are parsed
        # into actual algorithm parameters once the algorithm starts.
        self.options = {"min_step": min_step, "max_step": max_step, "step_factor": step_factor,
                        "min_chi2_step": min_chi2_step, "max_chi2_step": max_chi2_step,
                        "termination_significance": termination_significance, "confidence_level": confidence_level,
                        "num_dof": num_dof}

        self.profile_info = None
        self._chi2 = {}
        self._parameters = {}

    def prepare_profile(self, chi2_function, fit_function, parameters, parameter_name):
        options = self.options

        assert options["max_step"] > options["min_step"]
        assert options["max_chi2_step"] > options["min_chi2_step"]

        self.profile_info = ProfileInfo(minimum_chi2=chi2_function(parameters.values),
                                        profiled_parameter_index=list(parameters.keys).index(parameter_name),
                                        delta_chi2=chi2.ppf(options["confidence_level"], options["num_dof"]),
                                        confidence_level=options["confidence_level"],
                                        parameter_names=list(parameters.keys))

        fitted = parameters.fitted
        fitted[self.profile_info.profiled_parameter_index] = 0

        # TODO: Allow more complex step direction functions based on approximate Hessian information
        step_direction = np.zeros(len(parameters))
        step_direction[self.profile_info.profiled_parameter_index] = 1.0

        def step_direction_function():
            return step_direction

        step_config = StepConfig(min_abs_step=options["min_step"] * parameters[parameter_name].value,
                                 max_abs_step=options["max_step"] * parameters[parameter_name].value,
                                 step_factor=options["step_factor"],
                                 min_chi2_step_size=options["min_chi2_step"] * self.profile_info.delta_chi2,
                                 max_chi2_step_size=options["max_chi2_step"] * self.profile_info.delta_chi2)

        def step_function(chi2_last, parameter_vector, current_step_size, step_sign):
            return do_step(chi2_function, step_direction_function, chi2_last, parameter_vector, current_step_size,
                           step_sign, step_config)

        scan_config = ScanConfig(lb=parameters.lb,
                                 ub=parameters.ub,
                                 fitted=fitted,
                                 step_function=step_function,
                                 termination_level=self.profile_info.minimum_chi2 +
                                                   chi2.ppf(options["termination_significance"], options["num_dof"]))

        def scan_direction(chi2_last, parameter_vector, step_sign, num_steps):
            return scan_dir_optimisation(chi2_function, fit_function, chi2_last, parameter_vector, num_steps, step_sign,
                                         scan_config)

        return scan_direction

    # TODO: Add mechanism which stores a hash coming from the FitObject and validates it against what's already done
    def extend_profile(self, chi2_function, fit_function, parameters, num_steps, forward):
        scan_direction = self.prepare_profile(chi2_function, fit_function, parameters, self.parameter_name)

        field = "fwd" if forward else "bwd"
        parameter_vectors = self._parameters[field] if field in self._parameters else np.expand_dims(parameters.values,
                                                                                                     0)
        initial_parameters = parameter_vectors[-1, :]
        chi2_list = self._chi2[field] if field in self._chi2 else [self.profile_info.minimum_chi2]
        chi2_last = chi2_list[-1]

        chi2_new, parameters_new = scan_direction(chi2_last, initial_parameters, 1 if forward else -1, num_steps)

        self._parameters[field] = np.vstack((parameter_vectors, parameters_new))
        self._chi2[field] = np.hstack((chi2_list, chi2_new))

    def _calc_ci(self):
        def find_crossing(x, y, crossing):
            print(y)
            indices, = np.where(y >= crossing)

            if np.any(indices):
                index = indices[0]
                if index > 0:
                    x1, x2, y1, y2 = x[index-1], x[index], y[index-1], y[index]

                dydx = (y2 - y1) / (x2 - x1)
                return x1 + (crossing - y1) / dydx

        cutoff = self.profile_info.minimum_chi2 + self.profile_info.delta_chi2
        p_index = self.profile_info.profiled_parameter_index
        x_min = find_crossing(self._parameters["bwd"][:, p_index], self._chi2["bwd"], cutoff)
        x_max = find_crossing(self._parameters["fwd"][:, p_index], self._chi2["fwd"], cutoff)
        return x_min, x_max

    def plot(self):
        dash_length = 5
        chi2_list = np.hstack((np.flipud(self._chi2["bwd"]), self._chi2["fwd"]))
        parameters = np.vstack((np.flipud(self._parameters["bwd"]), self._parameters["fwd"]))

        ci_min, ci_max = self._calc_ci()

        plt.figure()
        plt.subplot(2, 1, 1)
        independent = parameters[:, self.profile_info.profiled_parameter_index]
        plt.plot(independent, chi2_list)
        confidence_chi2 = self.profile_info.minimum_chi2 + self.profile_info.delta_chi2
        plt.axhline(y=confidence_chi2, linewidth=1, color='k', dashes=[dash_length, dash_length])
        plt.axvline(x=ci_min, linewidth=1, color='k', dashes=[dash_length, dash_length])
        plt.axvline(x=ci_max, linewidth=1, color='k', dashes=[dash_length, dash_length])

        plt.text(min(independent), confidence_chi2, f"{self.profile_info.confidence_level * 100}%")
        plt.ylabel('$\chi^2$')
        plt.xlabel(self.parameter_name)

        plt.subplot(2, 1, 2)
        other = [x for x in range(parameters.shape[1]) if x != self.profile_info.profiled_parameter_index]
        line_handles = plt.plot(parameters[:, self.profile_info.profiled_parameter_index], parameters[:, other])
        plt.ylabel('Other parameter value')
        plt.xlabel(self.parameter_name)
        plt.legend(line_handles, [self.profile_info.parameter_names[idx] for idx in other])