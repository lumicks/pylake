import inspect
import numpy as np
from collections import OrderedDict
from .parameters import Parameter
from .fitdata import FitData
from .detail.derivative_manipulation import numerical_jacobian, numerical_diff, invert_jacobian, invert_function, \
    invert_derivative
from .detail.link_functions import generate_conditions
from .detail.utilities import parse_transformation, print_styled, optimal_plot_layout
from copy import deepcopy


class Model:
    def __init__(self, name, model_function, jacobian=None, derivative=None, **kwargs):
        """
        This function creates a model. A Model must be named, and this name will appear in the model parameters.

        Parameters
        ----------
        name: str
            Name for the model. This name will be prefixed to the model parameter names.
        model_function: callable
            Function containing the model function. Must return the model prediction given values for the independent
            variable and parameters.
        jacobian: callable (optional)
            Function which computes the first order derivatives with respect to the parameters for this model.
            When supplied, this function is used to speed up the optimization considerably.
        derivative: callable (optional)
            Function which computes the first order derivative with respect to the independent parameter. When supplied
            this speeds up model inversions considerably.
        **kwargs
            Key pairs containing parameter defaults. For instance, Lc=Parameter(...)
        """
        import types
        assert isinstance(name, str), "First argument must be a model name."
        assert isinstance(model_function, types.FunctionType), "Model must be a callable."

        if jacobian:
            assert isinstance(jacobian, types.FunctionType), "Jacobian must be a callable."

        if derivative:
            assert isinstance(derivative, types.FunctionType), "Derivative must be a callable."

        def formatter(x):
            return f"{name}_{x}"

        self.name = name
        self.model_function = model_function
        parameter_names = inspect.getfullargspec(model_function).args[1:]

        self._parameters = OrderedDict()
        for key in parameter_names:
            if key in kwargs:
                assert isinstance(kwargs[key], Parameter), "Passed a non-parameter as model default."
                if kwargs[key].shared:
                    self._parameters[key] = kwargs[key]
                else:
                    self._parameters[formatter(key)] = kwargs[key]
            else:
                self._parameters[formatter(key)] = None

        self._jacobian = jacobian
        self._derivative = derivative
        self._data = []
        self._conditions = []
        self._built = None

    def __add__(self, other):
        """
        Add two model outputs to form a new model.

        Parameters
        ----------
        other: Model
        """

        return CompositeModel(self, other)

    def subtract_offset(self, parameter_name="independent_offset"):
        """
        Subtract a constant offset from this model.

        Parameters
        ----------
        parameter_name: str
        """
        return SubtractIndependentOffset(self, parameter_name)

    @staticmethod
    def _sanitize_input_types(independent):
        independent = np.array(independent).astype(float)
        return independent

    def __call__(self, independent, parameter_vector):
        independent = self._sanitize_input_types(independent)
        return self.model_function(independent, *parameter_vector)

    @property
    def _defaults(self):
        if self._data:
            return [deepcopy(self._parameters[name]) for data in self._data for name in data.source_parameter_names]
        else:
            return [deepcopy(self._parameters[name]) for name in self.parameter_names]

    def jacobian(self, independent, parameter_vector):
        if self.has_jacobian:
            independent = self._sanitize_input_types(independent)
            return self._jacobian(independent, *parameter_vector)
        else:
            raise RuntimeError(f"Jacobian was requested but not supplied in model {self.name}.")

    def derivative(self, independent, parameter_vector):
        if self.has_derivative:
            independent = self._sanitize_input_types(independent)
            return self._derivative(independent, *parameter_vector)
        else:
            raise RuntimeError(f"Derivative was requested but not supplied in model {self.name}.")

    @property
    def has_jacobian(self):
        if self._jacobian:
            return True

    @property
    def has_derivative(self):
        if self._derivative:
            return True

    @property
    def n_residuals(self):
        count = 0
        for data in self._data:
            count += len(data.independent)

        return count

    @property
    def max_x(self):
        return np.max([np.max(d.x) for d in self._data])

    @property
    def max_y(self):
        return np.max([np.max(d.y) for d in self._data])

    def _calculate_residual(self, global_parameter_values):
        residual_idx = 0
        residual = np.zeros(self.n_residuals)
        for condition, data_sets in zip(self._conditions, self._data_link):
            p_local = condition.get_local_parameters(global_parameter_values)
            for data in data_sets:
                data_set = self._data[data]
                y_model = self(data_set.x, p_local)

                residual[residual_idx:residual_idx + len(y_model)] = data_set.y - y_model
                residual_idx += len(y_model)

        return residual

    def _calculate_jacobian(self, global_parameter_values):
        residual_idx = 0
        jacobian = np.zeros((self.n_residuals, len(global_parameter_values)))
        for condition, data_sets in zip(self._conditions, self._data_link):
            p_local = condition.get_local_parameters(global_parameter_values)
            p_indices = condition.p_indices
            for data in data_sets:
                data_set = self._data[data]
                sensitivities = condition.localize_sensitivities(np.transpose(self.jacobian(data_set.x, p_local)))
                n_res = sensitivities.shape[0]

                jacobian[residual_idx:residual_idx + n_res, p_indices] = \
                    jacobian[residual_idx:residual_idx + n_res, p_indices] - sensitivities

                residual_idx += n_res

        return jacobian

    def verify_derivative(self, independent, parameters, **kwargs):
        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        derivative = self.derivative(independent, parameters)
        derivative_fd = numerical_diff(lambda x: self(x, parameters), independent)

        return np.allclose(derivative, derivative_fd, **kwargs)

    def verify_jacobian(self, independent, parameters, plot=False, verbose=True, **kwargs):
        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        independent = self._sanitize_input_types(independent)

        jacobian = self.jacobian(independent, parameters)
        jacobian_fd = numerical_jacobian(lambda parameter_values: self(independent, parameter_values), parameters)

        if plot:
            import matplotlib.pyplot as plt
            n_x, n_y = optimal_plot_layout(len(self.parameters))
            for i_parameter, parameter in enumerate(self.parameters):
                plt.subplot(n_x, n_y, i_parameter)
                l1 = plt.plot(independent, np.transpose(jacobian[i_parameter, :]))
                l2 = plt.plot(independent, np.transpose(jacobian_fd[i_parameter, :]), '--')
                plt.title(parameter)
                plt.legend({'Analytic', 'FD'})

        is_close = np.allclose(jacobian, jacobian_fd, **kwargs)
        if not is_close:
            if verbose:
                maxima = np.max(jacobian - jacobian_fd, axis=1)
                for i, v in enumerate(maxima):
                    if np.allclose(jacobian[i, :], jacobian_fd[i, :]):
                        print(f"Parameter {self.parameter_names[i]}({i}): {v}")
                    else:
                        print_styled('warning', f'Parameter {self.parameter_names[i]}({i}): {v}')

        return is_close

    @property
    def parameter_names(self):
        return [x for x in self._parameters.keys()]

    def _invalidate_build(self):
        self._built = False

    def built_against(self, fit_object):
        return self._built == fit_object

    def load_data(self, x, y, name="", **kwargs):
        self._invalidate_build()
        parameter_list = parse_transformation(self.parameter_names, **kwargs)
        self._data.append(FitData(name, x, y, parameter_list))
        return self

    @property
    def _transformed_parameters(self):
        """Retrieves the full list of parameters and defaults post-transformation used by this model. This includes the
        parameters for all the data-sets in the model."""
        if self._data:
            return [name for data in self._data for name in data.parameter_names]
        else:
            return self.parameter_names

    def _build_model(self, parameter_lookup, fit_object):
        self._conditions, self._data_link = generate_conditions(self._data, parameter_lookup,
                                                                self.parameter_names)

        self._built = fit_object

    def _plot_data(self, idx=None):
        import matplotlib.pyplot as plt

        names = []
        handles = []
        for data_idx in idx if idx else np.arange(len(self._data)):
            data = self._data[data_idx]
            handle, = plt.plot(data.x, data.y, '.')
            handles.append(handle)
            names.append(data.name)

        plt.legend(handles, names)

    def _plot_model(self, global_parameters, idx=None):
        import matplotlib.pyplot as plt

        def intersection(l1, l2):
            return [value for value in l1 if value in l2]

        if not idx:
            idx = np.arange(len(self._data))

        for condition, data_sets in zip(self._conditions, self._data_link):
            p_local = condition.get_local_parameters(global_parameters)
            [plt.plot(np.sort(self._data[value].x), self(np.sort(self._data[value].x), p_local))
             for value in idx if value in data_sets]


class CompositeModel(Model):
    def __init__(self, lhs, rhs):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        lhs: Model
        rhs: Model
        """
        self.lhs = lhs
        self.rhs = rhs

        self.name = self.lhs.name + "_with_" + self.rhs.name
        self._parameters = OrderedDict()
        for i, v in self.lhs._parameters.items():
            self._parameters[i] = v
        for i, v in self.rhs._parameters.items():
            self._parameters[i] = v

        parameters_lhs = list(self.lhs._parameters.keys())
        parameters_rhs = list(self.rhs._parameters.keys())
        parameters_all = list(self._parameters.keys())

        self.lhs_parameters = [parameters_all.index(par) for par in parameters_lhs]
        self.rhs_parameters = [parameters_all.index(par) for par in parameters_rhs]

        self._data = []
        self._conditions = []
        self._built = False

    def __call__(self, independent, parameter_vector):
        lhs_residual = self.lhs(independent, [parameter_vector[x] for x in self.lhs_parameters])
        rhs_residual = self.rhs(independent, [parameter_vector[x] for x in self.rhs_parameters])

        return lhs_residual + rhs_residual

    @property
    def has_jacobian(self):
        return self.lhs.has_jacobian and self.rhs.has_jacobian

    @property
    def has_derivative(self):
        return self.lhs.has_derivative and self.rhs.has_derivative

    def jacobian(self, independent, parameter_vector):
        if self.has_jacobian:
            jacobian = np.zeros((len(parameter_vector), len(independent)))
            jacobian[self.lhs_parameters, :] += self.lhs.jacobian(independent, [parameter_vector[x] for x in
                                                                                self.lhs_parameters])
            jacobian[self.rhs_parameters, :] += self.rhs.jacobian(independent, [parameter_vector[x] for x in
                                                                                self.rhs_parameters])

            return jacobian

    def derivative(self, independent, parameter_vector):
        if self.has_derivative:
            lhs_derivative = self.lhs.derivative(independent, [parameter_vector[x] for x in self.lhs_parameters])
            rhs_derivative = self.rhs.derivative(independent, [parameter_vector[x] for x in self.rhs_parameters])

            return lhs_derivative + rhs_derivative


class InverseModel(Model):
    def __init__(self, model):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        model: Model
        """
        self.model = model
        self._data = []
        self._conditions = []
        self._built = False
        self.name = "inv(" + model.name + ")"

    def __call__(self, independent, parameter_vector):
        independent_min = 0
        independent_max = np.inf
        initial = np.ones(independent.shape)

        return invert_function(independent, initial, independent_min, independent_max,
                               lambda f_trial: self.model(f_trial, parameter_vector),  # Forward model
                               lambda f_trial: self.model.derivative(f_trial, parameter_vector))

    @property
    def has_jacobian(self):
        """Does the model have sufficient information to determine its inverse numerically?
        This requires a Jacobian and a derivative w.r.t. independent variable."""
        return self.model.has_jacobian and self.model.has_derivative

    @property
    def has_derivative(self):
        return self.model.has_derivative

    def jacobian(self, independent, parameter_vector):
        """Jacobian of the inverted model"""
        return invert_jacobian(independent,
                               lambda f_trial: self(f_trial, parameter_vector),  # Inverse model (me)
                               lambda f_trial: self.model.jacobian(f_trial, parameter_vector),
                               lambda f_trial: self.model.derivative(f_trial, parameter_vector))

    def derivative(self, independent, parameter_vector):
        """Derivative of the inverted model"""
        return invert_derivative(independent,
                                 lambda f_trial: self(f_trial, parameter_vector),  # Inverse model (me)
                                 lambda f_trial: self.model.derivative(f_trial, parameter_vector))

    @property
    def _parameters(self):
        return self.model._parameters


class SubtractIndependentOffset(Model):
    def __init__(self, model, parameter_name='independent_offset'):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        model: Model
        """
        self.model = model
        offset_name = parameter_name

        self.name = self.model.name + "(x-d)"
        self._parameters = OrderedDict()
        self._parameters[offset_name] = None
        for i, v in self.model._parameters.items():
            self._parameters[i] = v

        parameters_parent = list(self.model._parameters.keys())
        parameters_all = list(self._parameters.keys())

        self.model_parameters = [parameters_all.index(par) for par in parameters_parent]
        self.offset_parameter = parameters_all.index(offset_name)

        self._data = []
        self._conditions = []
        self._built = None

    def __call__(self, independent, parameter_vector):
        return self.model(independent - parameter_vector[self.offset_parameter],
                          [parameter_vector[x] for x in self.model_parameters])

    @property
    def has_jacobian(self):
        return self.model.has_jacobian and self.has_derivative

    @property
    def has_derivative(self):
        return self.model.has_derivative

    def jacobian(self, independent, parameter_vector):
        if self.has_jacobian:
            with_offset = independent - parameter_vector[self.offset_parameter]
            jacobian = np.zeros((len(parameter_vector), len(with_offset)))
            jacobian[self.model_parameters, :] += self.model.jacobian(with_offset, [parameter_vector[x] for x in
                                                                                    self.model_parameters])
            jacobian[self.offset_parameter, :] = - self.model.derivative(with_offset, [parameter_vector[x] for x in
                                                                                       self.model_parameters])

            return jacobian

    def derivative(self, independent, parameter_vector):
        if self.has_derivative:
            with_offset = independent - parameter_vector[self.offset_parameter]
            return self.model.derivative(with_offset, parameter_vector[self.model_parameters])

