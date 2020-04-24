from .parameters import Parameters
from ..detail.utilities import unique
from .detail.derivative_manipulation import numerical_jacobian
from .detail.utilities import print_styled, optimal_plot_layout
from collections import OrderedDict
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


class Fit:
    """Object which is used for fitting. It is a collection of models and their data. Once data is loaded, a fit object
    contains ``Parameters``, which can be fitted by invoking fit.

    Parameters
    ----------
    *args
        Variable number of ``pylake.fitting.Model``.

    Examples
    --------
    ::

        from lumicks import pylake

        dna_model = pylake.inverted_odijk("DNA")
        fit = pylake.Fit(dna_model)
        data = dna_model.load_data(d=distance, f=force)

        fit["DNA_Lp"].lower_bound = 35  # Set lower bound for DNA Lp
        fit["DNA_Lp"].upper_bound = 80  # Set upper bound for DNA Lp
        fit.fit()

        dna_model.plot(fit[data], fmt='k--')  # Plot the fitted model
    """
    def __init__(self, *args):
        self.models = [m for m in args]
        self._data_link = None
        self._parameters = Parameters()
        self._built = False
        self._invalidate_build()

    def __getitem__(self, item):
        return self.parameters[item]

    @property
    def has_jacobian(self):
        """
        Returns true if it is possible to evaluate the Jacobian of the fit.
        """
        has_jacobian = True
        for model in self.models:
            has_jacobian = has_jacobian and model.has_jacobian

        return has_jacobian

    @property
    def n_residuals(self):
        """Number of data points."""
        self._rebuild()
        count = 0
        for model in self.models:
            count += model.n_residuals

        return count

    @property
    def n_parameters(self):
        """Number of parameters in the Fit"""
        self._rebuild()
        return len(self._parameters)

    @property
    def parameters(self):
        """Fit parameters. See also ``pylake.fitting.Parameters``"""
        self._rebuild()
        return self._parameters

    @property
    def dirty(self):
        """Validate that all the models that we are about the fit were actually last linked against this fit object."""
        dirty = not self._built
        for model in self.models:
            dirty = dirty or not model.built_against(self)

        return dirty

    def _rebuild(self):
        """
        Checks whether the model state is up to date. Any user facing methods should ideally check whether the model
        needs to be rebuilt.
        """
        if self.dirty:
            self._build_fit()

    def _invalidate_build(self):
        self._built = False

    def _build_fit(self):
        """This function generates the global parameter list from the parameters of the individual sub models.
        It also generates unique conditions from the data specification."""
        all_parameter_names = [p for model in self.models for p in model._transformed_parameters]
        all_defaults = [d for model in self.models for d in model._defaults]
        unique_parameter_names = unique(all_parameter_names)
        parameter_lookup = OrderedDict(zip(unique_parameter_names, np.arange(len(unique_parameter_names))))

        for model in self.models:
            model._build_model(parameter_lookup, self)

        defaults = [all_defaults[all_parameter_names.index(l)] for l in unique_parameter_names]
        self._parameters._set_parameters(unique_parameter_names, defaults)
        self._built = True

    def _prepare_fit(self):
        """Checks whether the model is ready for fitting and returns the current parameter values, which parameters are
        fitted and the parameter bounds."""
        self._rebuild()
        assert self.n_residuals > 0, "This model has no data associated with it."
        assert self.n_parameters > 0, "This model has no parameters. There is nothing to fit."
        return self.parameters.values, self.parameters.fitted, self.parameters.lower_bounds, \
               self.parameters.upper_bounds

    def _fit(self, parameter_vector, lb, ub, fitted, show_fit=False, **kwargs):
        """Fit the model

        Parameters
        ----------
        parameter_vector: array_like
            List of parameters
        lb: array_like
            list of lower parameter bounds
        ub: array_like
            list of lower parameter bounds
        show_fit: bool
            show fitting (slow!)
        fitted: array_like
            list of which parameters are fitted
        """
        if show_fit:
            fig = plt.gcf()

        def residual(parameters):
            parameter_vector[fitted] = parameters

            if show_fit:
                parameter_names = self.parameters.keys
                for name, value in zip(parameter_names, parameter_vector):
                    self.parameters[name] = value
                plt.figure(fig.number)
                self.plot()
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.clf()

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

    def fit(self, show_fit=False, **kwargs):
        """Fit the model

        Parameters
        ----------
        show_fit: bool
            Show the fitting procedure as it is progressing.
        """
        parameter_vector, fitted, lb, ub = self._prepare_fit()

        out_of_bounds = np.logical_or(parameter_vector[fitted] < lb[fitted], parameter_vector[fitted] > ub[fitted])
        if np.any(out_of_bounds):
            raise ValueError(f"Initial parameters {self.parameters.keys[fitted][out_of_bounds]} are outside the "
                             f"parameter bounds. Please set value, lower_bound and upper_bound for these parameters"
                             f"to consistent values.")

        parameter_vector = self._fit(parameter_vector, lb, ub, fitted, show_fit=show_fit, **kwargs)

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
        for model in self.models:
            current_residual = model._calculate_residual(parameter_values)
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
        for model in self.models:
            current_jacobian = model._calculate_jacobian(parameter_values)
            current_n = current_jacobian.shape[0]
            jacobian[residual_idx:residual_idx + current_n, :] = current_jacobian
            residual_idx += current_n

        return jacobian

    def verify_jacobian(self, parameters, plot=0, verbose=True, dx=1e-6, **kwargs):
        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        jacobian = self._calculate_jacobian(parameters).transpose()
        jacobian_fd = numerical_jacobian(self._calculate_residual, parameters, dx)

        if plot:
            n_x, n_y = optimal_plot_layout(len(self.parameters))
            for i_parameter, parameter in enumerate(self.parameters):
                plt.subplot(n_x, n_y, i_parameter + 1)
                l1 = plt.plot(np.transpose(jacobian[i_parameter, :]), linewidth=2)
                l2 = plt.plot(np.transpose(jacobian_fd[i_parameter, :]), '--', linewidth=1)
                plt.title(parameter)
                plt.legend(['Analytic', 'FD'])

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

    def plot(self, fmt='', **kwargs):
        self.plot_data(fmt, **kwargs)
        self.plot_model(fmt, **kwargs)

    def plot_data(self, fmt, **kwargs):
        self._rebuild()

        for model in self.models:
            model._plot_data(fmt, **kwargs)

    def _override_parameters(self, **kwargs):
        from copy import deepcopy
        parameters = self.parameters
        if kwargs:
            parameters = deepcopy(parameters)
            for key, value in kwargs.items():
                if key in parameters:
                    parameters[key] = value

        return parameters, kwargs

    def plot_model(self, fmt='', **kwargs):
        self._rebuild()
        parameters, kwargs = self._override_parameters(**kwargs)

        for model in self.models:
            model._plot_model(parameters, fmt, **kwargs)

    @property
    def sigma(self):
        """Error variance of the data points."""
        # TO DO: Ideally, this will eventually depend on the exact error model used. For now, we use the a-posteriori
        # variance estimate based on the residual.
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
        """Calculates the Akaike Information Criterion:

            AIC = 2 k - 2 ln(L)

        Where k refers to the number of parameters, n to the number of observations (or data points) and L to the
        maximized value of the likelihood function

        The emphasis of this criterion is future prediction. It does not lead to consistent model selection and is more
        prone to over-fitting than the Bayesian Information Criterion.

        References:
            Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike information criteria.
            Statistics & Probability Letters, 33(2), pp.201-208.
        """
        self._rebuild()
        k = sum(self.parameters.fitted)
        LL = self.log_likelihood()
        return 2.0 * k - 2.0 * LL

    @property
    def aicc(self):
        """Calculates the Corrected Akaike Information Criterion:

        .. math::
            AICc = AIC + \\frac{2 k^2 + 2 k}{n - k - 1}

        Where k refers to the number of parameters, n to the number of observations (or data points) and L to the
        maximized value of the likelihood function

        The emphasis of this criterion is future prediction. Compared to the AIC it should be less prone to overfitting
        for smaller sample sizes. Analogously to the AIC, it does not lead to a consistent model selection procedure.

        References:
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

        Where k refers to the number of parameters, n to the number of observations (or data points) and L to the
        maximized value of the likelihood function

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

        References:
            Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 1988. Numerical recipes in C.

            Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew, F., Bock, H.H.,
            Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to its limit: profile likelihood
            based model reduction. PloS one, 11(9).
        """
        J = self._calculate_jacobian()
        J = J / np.transpose(np.tile(self.sigma, (J.shape[1], 1)))
        return np.linalg.pinv(np.transpose(J).dot(J))

    def _repr_html_(self):
        out_string = "<h4>Fit</h4>\n"

        for model in self.models:
            datasets = ''.join([f"&ensp;- {s.__repr__()}<br>\n" for s in model.data])
            out_string += f"<h5>Model: {model.name}</h5>\n" \
                          f"<h5>Equation:</h5>${model.get_formatted_equation_string(tex=True)}$<br>\n" \
                          f"<h5>Data:</h5>\n{datasets}<br>"

        return out_string + f"<h5>Fitted parameters:</h5>\n{self.parameters._repr_html_()}"

    def __repr__(self):
        out_string = "Fit\n"

        for model in self.models:
            datasets = ''.join([f"  - {s.__repr__()}\n" for s in model.data])
            out_string += f"- Model: {model.name}\n  " \
                          f"- Equation:\n      {model.get_formatted_equation_string(tex=False)}\n{datasets}"

        return out_string + f"\n- Fitted parameters:\n    {'    '.join(self.parameters.__str__().splitlines(True))}"
