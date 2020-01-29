import numpy as np
from collections import OrderedDict
from .parameters import Parameters
from ..detail.utilities import unique
from .detail.utilities import print_styled
import scipy.optimize as optim
from .profile_likelihood import ProfileLikelihood1D
from .detail.derivative_manipulation import numerical_jacobian


class FitObject:
    """Object which is used for fitting. It is a collection of a model alongside its data.

    A fit object builds the linkages required to propagate parameters used in sub-models to a global parameter vector
    used by the optimization algorithm.
    """
    def __init__(self, *args):
        self.models = [M for M in args]
        self._data_link = None
        self._parameters = Parameters()
        self._current_new_idx = 0
        self._built = False
        self._invalidate_build()

    def add_model(self, model):
        self.models.append(model)
        self._built = False
        self._invalidate_build()

    def _build_fitobject(self):
        """This function generates the global parameter list from the parameters of the individual submodels.
        It also generates unique conditions from the data specification."""
        all_parameter_names = [p for M in self.models for p in M._transformed_parameters]
        all_defaults = [d for M in self.models for d in M._defaults]
        unique_parameter_names = unique(all_parameter_names)
        parameter_lookup = OrderedDict(zip(unique_parameter_names, np.arange(len(unique_parameter_names))))

        for M in self.models:
            M._build_model(parameter_lookup, self)

        defaults = [all_defaults[all_parameter_names.index(l)] for l in unique_parameter_names]
        self._parameters._set_parameters(unique_parameter_names, defaults)
        self._built = True

    def _rebuild(self):
        """
        Checks whether the model state is up to date. Any user facing methods should ideally check whether the model
        needs to be rebuilt.
        """
        if self.dirty:
            self._build_fitobject()

    def _invalidate_build(self):
        self._built = False

    @property
    def dirty(self):
        dirty = not self._built
        for M in self.models:
            dirty = dirty or not M.built_against(self)

        return dirty

    @property
    def n_residuals(self):
        self._rebuild()
        count = 0
        for M in self.models:
            count += M.n_residuals

        return count

    @property
    def has_jacobian(self):
        has_jacobian = True
        for M in self.models:
            has_jacobian = has_jacobian and M.has_jacobian

        return has_jacobian

    @property
    def parameters(self):
        self._rebuild()
        return self._parameters

    @property
    def n_parameters(self):
        self._rebuild()
        return len(self._parameters)

    def profile_likelihood(self, parameter_name, min_step=1e-4, max_step=1.0, num_steps=100, step_factor=2.0,
                           min_chi2_step=0.05, max_chi2_step=.25, termination_significance=.99, confidence_level=.95,
                           verbose=False):
        """Calculate a profile likelihood. This method traces an optimal path through parameter space in order to
        estimate parameter confidence intervals. It iteratively performs a step for the profiled parameter, then fixes
        that parameter and re-optimizes all the other parameters.

        Parameters
        ----------
        parameter_name: str
            Which parameter to evaluate a profile likelihood for.
        min_step: float
            Minimum step size. This is multiplied by the current parameter value to come to a minimum step size used
            in the step-size estimation procedure (default: 1e-4).
        max_step: float
            Maximum step size (default: 1.0).
        num_steps: integer
            Number of steps to take (default: 100).
        step_factor: float
            Which factor to change the step-size by when step-size is too large or too small (default: 2).
        min_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the step results in a fit
            change smaller than this threshold, the step-size will be increased.
        max_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the step results in a fit
            change bigger than this threshold, the step-size will be reduced.
        termination_significance: float
            Significance level for terminating the parameter scan. When the fit quality exceeds the
            termination_significance confidence level, it stops scanning.
        confidence_level: float
            Significance level for the chi squared test.
        verbose: bool
            Controls the verbosity of the output.
        """

        if parameter_name not in self.parameters:
            raise KeyError(f"Parameter {parameter_name} not present in fitting object.")

        if not self.parameters[parameter_name].vary:
            raise RuntimeError(f"Parameter {parameter_name} is fixed in the fitting object.")

        assert max_step > min_step
        assert max_chi2_step > min_chi2_step

        profile = ProfileLikelihood1D(parameter_name, min_step, max_step, step_factor, min_chi2_step, max_chi2_step,
                                      termination_significance, confidence_level, 1)

        sigma = self.sigma

        def trial(parameters=[]):
            return - 2.0 * self.log_likelihood(parameters, sigma)

        profile.extend_profile(trial, self._fit, self.parameters, num_steps, True, verbose)
        profile.extend_profile(trial, self._fit, self.parameters, num_steps, False, verbose)

        self.parameters[parameter_name].profile = profile

        return profile

    def _fit(self, parameter_vector, lb, ub, fitted, **kwargs):
        """Fit the model

        Parameters
        ----------
        parameter_vector: array_like
            List of parameters
        lb: array_like
            list of lower parameter bounds
        ub: array_like
            list of lower parameter bounds
        fitted: array_like
            list of which parameters are fitted
        """
        def residual(parameters):
            parameter_vector[fitted] = parameters
            return self._calculate_residual(parameter_vector)

        def jacobian(parameters):
            parameter_vector[fitted] = parameters
            return self._calculate_jacobian(parameter_vector)[:, fitted]

        result = optim.least_squares(residual, parameter_vector[fitted],
                                     jac=jacobian if self.has_jacobian else "2-point",
                                     bounds=(lb[fitted], ub[fitted]),
                                     method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, **kwargs)

        parameter_vector[fitted] = result.x

        return parameter_vector

    def _prepare_fit(self):
        """Checks whether the model is ready for fitting and returns the current parameter values, which parameters are
        fitted and the parameter bounds."""
        self._rebuild()
        assert self.n_residuals > 0, "This model has no data associated with it."
        assert self.n_parameters > 0, "This model has no parameters. There is nothing to fit."
        return self.parameters.values, self.parameters.fitted, self.parameters.lb, self.parameters.ub

    def fit(self, **kwargs):
        parameter_vector, fitted, lb, ub = self._prepare_fit()

        out_of_bounds = np.logical_or(parameter_vector[fitted] < lb[fitted], parameter_vector[fitted] > ub[fitted])
        if np.any(out_of_bounds):
            raise ValueError(f"Initial parameters {self.parameters.keys[fitted][out_of_bounds]} are outside the "
                             f"parameter bounds. Please set value, lb or ub for these parameters to consistent values.")

        parameter_vector = self._fit(parameter_vector, lb, ub, fitted, **kwargs)

        parameter_names = self.parameters.keys
        for name, value in zip(parameter_names, parameter_vector):
            self.parameters[name] = value

        return self

    def _calculate_residual(self, parameter_values=[]):
        self._rebuild()
        if len(parameter_values) == 0:
            parameter_values = self.parameters.values

        residual_idx = 0
        residual = np.zeros(self.n_residuals)
        for M in self.models:
            current_residual = M._calculate_residual(parameter_values)
            current_n = len(current_residual)
            residual[residual_idx:residual_idx + current_n] = current_residual
            residual_idx += current_n

        return residual

    def _calculate_jacobian(self, parameter_values=[]):
        self._rebuild()
        if len(parameter_values) == 0:
            parameter_values = self.parameters.values

        residual_idx = 0
        jacobian = np.zeros((self.n_residuals, len(parameter_values)))
        for M in self.models:
            current_jacobian = M._calculate_jacobian(parameter_values)
            current_n = current_jacobian.shape[0]
            jacobian[residual_idx:residual_idx + current_n, :] = current_jacobian
            residual_idx += current_n

        return jacobian

    def verify_jacobian(self, parameters, plot=0, verbose=True, **kwargs):
        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        jacobian = self._calculate_jacobian(parameters).transpose()
        jacobian_fd = numerical_jacobian(self._calculate_residual, parameters)

        if plot:
            import matplotlib.pyplot as plt
            n_x, n_y = optimal_plot_layout(len(self.parameters))
            for i_parameter, parameter in enumerate(self.parameters):
                plt.subplot(n_x, n_y, i_parameter + 1)
                l1 = plt.plot(np.transpose(jacobian[i_parameter, :]))
                l2 = plt.plot(np.transpose(jacobian_fd[i_parameter, :]), '--')
                plt.title(parameter)
                plt.legend({'Analytic', 'FD'})

        is_close = np.allclose(jacobian, jacobian_fd, **kwargs)
        if not is_close:
            parameter_names = list(self.parameters.keys)
            if verbose:
                maxima = np.max(jacobian - jacobian_fd, axis=1)
                for i, v in enumerate(maxima):
                    if np.allclose(jacobian[i, :], jacobian_fd[i, :]):
                        print(f"Parameter {parameter_names[i]}({i}): {v}")
                    else:
                        print_styled('warning', f'Parameter {parameter_names[i]}({i}): {v}')

        return is_close

    @property
    def sigma(self):
        """Error variance of the data points. Ideally, this will eventually depend on the exact error model used. For
        now, we use the a-posteriori variance estimate based on the residual."""
        res = self._calculate_residual()
        return np.sqrt(np.var(res)) * np.ones(len(res))

    def log_likelihood(self, parameters=[], sigma=None):
        """The model residual is given by chi squared = -2 log(L)"""
        self._rebuild()
        res = self._calculate_residual(parameters)
        sigma = sigma if np.any(sigma) else self.sigma
        return - (self.n_residuals/2.0) * np.log(2.0 * np.pi) - np.sum(np.log(sigma)) - sum((res/sigma)**2) / 2.0

    @property
    def aic(self):
        """
        Calculates the Akaike Information Criterion:

            AIC = 2 k - 2 ln(L)

        Where:
        k - Number of parameters
        n - Number of observations / data points
        L - maximized value of the likelihood function

        The emphasis of this criterion is future prediction. It does not lead to consistent model selection and is more
        prone to over-fitting than the Bayesian Information Criterion.

        Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike information criteria.
        Statistics & Probability Letters, 33(2), pp.201-208.
        """
        self._rebuild()
        k = sum(self.parameters.fitted)
        LL = self.log_likelihood()
        return 2.0 * k - 2.0 * LL

    @property
    def aicc(self):
        """
        Calculates the Corrected Akaike Information Criterion:

                          2 k^2 + 2 k
            AICc = AIC + -------------
                           n - k - 1
        Where:
        k - Number of parameters
        n - Number of observations / data points
        L - maximized value of the likelihood function

        The emphasis of this criterion is future prediction. Compared to the AIC it should be less prone to overfitting
        for smaller sample sizes. Analogously to the AIC, it does not lead to a consistent model selection procedure.

        Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike information criteria.
        Statistics & Probability Letters, 33(2), pp.201-208.
        """
        aic = self.aic
        k = sum(self.parameters.fitted)
        return aic + (2.0 * k * k + 2.0 * k)/(self.n_residuals - k - 1.0)

    @property
    def bic(self):
        """
        Calculates the Bayesian Information Criterion:

            BIC = k ln(n) - 2 ln(L)

        Where:
        k - Number of parameters
        n - Number of observations / data points
        L - maximized value of the likelihood function

        The emphasis of the BIC is put on parsimonious models. As such it is less prone to over-fitting. Selection via
        BIC leads to a consistent model selection procedure, meaning that as the number of data points tends to
        infinity, BIC will select the true model assuming the true model is in the set of selected models.
        """

        k = sum(self.parameters.fitted)
        return k * np.log(self.n_residuals) - 2.0 * self.log_likelihood()

    @property
    def cov(self):
        """
        Returns the inverse of the approximate Hessian. This approximation is valid when the model fits well (small
        residuals) and there is sufficient data to assume we're in the asymptotic regime.

        It makes use of the Gauss-Newton approximation of the Hessian, which uses only the first order sensitivity
        information. This is valid for linear problems and problems near the optimum (assuming the model fits).

        Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 1988. Numerical recipes in C.

        Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew, F., Bock, H.H.,
        Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to its limit: profile likelihood
        based model reduction. PloS one, 11(9).
        """
        J = self._calculate_jacobian()
        J = J / np.transpose(np.tile(self.sigma, (J.shape[1], 1)))
        return np.linalg.pinv(np.transpose(J).dot(J))

    def plot(self, **kwargs):
        self.plot_data()
        self.plot_model(**kwargs)

    def plot_data(self):
        self._rebuild()

        for M in self.models:
            M._plot_data()

    def _override_parameters(self, **kwargs):
        parameters = self.parameters
        if kwargs:
            parameters = deepcopy(parameters)
            for key, value in kwargs.items():
                if key in parameters:
                    parameters[key] = value

        return parameters, kwargs

    def plot_model(self, **kwargs):
        self._rebuild()
        parameters, kwargs = self._override_parameters(**kwargs)

        for M in self.models:
            M._plot_model(parameters.values)

    def plot_model_recursive(self, **kwargs):
        self._rebuild()

        parameters, kwargs = self._override_parameters(**kwargs)

        for M in self.models:
            M._plot_model_recursive(parameters.values)
