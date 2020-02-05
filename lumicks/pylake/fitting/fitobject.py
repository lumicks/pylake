from .parameters import Parameters
from ..detail.utilities import unique
from .detail.derivative_manipulation import numerical_jacobian
from .detail.utilities import print_styled, optimal_plot_layout
from collections import OrderedDict
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


class FitObject:
    """Object which is used for fitting. It is a collection of a model alongside its data.

    A fit object builds the linkages required to propagate parameters used in sub-models to a global parameter vector
    used by the optimization algorithm.
    """
    def __init__(self, *args):
        self.models = [M for M in args]
        self._data_link = None
        self._parameters = Parameters()
        self._built = False
        self._invalidate_build()

    @property
    def has_jacobian(self):
        has_jacobian = True
        for M in self.models:
            has_jacobian = has_jacobian and M.has_jacobian

        return has_jacobian

    @property
    def n_residuals(self):
        self._rebuild()
        count = 0
        for M in self.models:
            count += M.n_residuals

        return count

    @property
    def n_parameters(self):
        self._rebuild()
        return len(self._parameters)

    @property
    def parameters(self):
        self._rebuild()
        return self._parameters

    @property
    def dirty(self):
        """Validate that all the models that we are about the fit were actually last linked against this fit object."""
        dirty = not self._built
        for M in self.models:
            dirty = dirty or not M.built_against(self)

        return dirty

    def _rebuild(self):
        """
        Checks whether the model state is up to date. Any user facing methods should ideally check whether the model
        needs to be rebuilt.
        """
        if self.dirty:
            self._build_fitobject()

    def _invalidate_build(self):
        self._built = False

    def _build_fitobject(self):
        """This function generates the global parameter list from the parameters of the individual sub models.
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

    def _prepare_fit(self):
        """Checks whether the model is ready for fitting and returns the current parameter values, which parameters are
        fitted and the parameter bounds."""
        self._rebuild()
        assert self.n_residuals > 0, "This model has no data associated with it."
        assert self.n_parameters > 0, "This model has no parameters. There is nothing to fit."
        return self.parameters.values, self.parameters.fitted, self.parameters.lb, self.parameters.ub

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
                             f"parameter bounds. Please set value, lb or ub for these parameters to consistent values.")

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