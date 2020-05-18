from .parameters import Parameter, Parameters
from .detail.utilities import solve_formatter, solve_formatter_tex, escape_tex
from .detail.utilities import print_styled, optimal_plot_layout
from .detail.derivative_manipulation import numerical_jacobian, numerical_diff, invert_function, invert_jacobian, \
    invert_derivative, invert_function_interpolation
from ..detail.utilities import get_color, lighten_color

from collections import OrderedDict
from copy import deepcopy
import inspect
import types
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, name, model_function, dependent=None, independent=None, jacobian=None, derivative=None, eqn=None,
                 eqn_tex=None, **kwargs):
        """
        Model constructor. A Model must be named, and this name will appear in the model parameters.

        Ideally a jacobian and derivative w.r.t. the independent variable are provided with every model. This will
        allow much higher performance when fitting. Jacobians and derivatives are automatically propagated to composite
        models, inversions of models etc. provided that all participating models have jacobians and derivatives
        specified.

        Parameters
        ----------
        name: str
            Name for the model. This name will be prefixed to the model parameter names.
        model_function: callable
            Function containing the model function. Must return the model prediction given values for the independent
            variable and parameters.
        dependent: str (optional)
            Name of the dependent variable
        independent: str (optional)
            Name of the independent variable
        jacobian: callable (optional)
            Function which computes the first order derivatives with respect to the parameters for this model.
            When supplied, this function is used to speed up the optimization considerably.
        derivative: callable (optional)
            Function which computes the first order derivative with respect to the independent parameter. When supplied
            this speeds up model inversions considerably.
        eqn: str (optional)
            Equation that this model is specified by.
        eqn_tex: str (optional)
            Equation that this model is specified by using TeX formatting.
        **kwargs
            Key pairs containing parameter defaults. For instance, Lc=Parameter(...)

        Examples
        --------
        ::

            from lumicks import pylake

            dna_model = pylake.inverted_odijk("DNA")
            fit = pylake.Fit(dna_model)
            fit.add_data("my data", force, distance)

            fit["DNA/Lp"].lower_bound = 35  # Set lower bound for DNA Lp
            fit["DNA/Lp"].upper_bound = 80  # Set upper bound for DNA Lp
            fit.fit()

            dna_model.plot(fit[data], fmt='k--')  # Plot the fitted model
        """
        assert isinstance(name, str), "First argument must be a model name."
        assert isinstance(model_function, types.FunctionType), "Model must be a callable."

        if jacobian:
            assert isinstance(jacobian, types.FunctionType), "Jacobian must be a callable."

        if derivative:
            assert isinstance(derivative, types.FunctionType), "Derivative must be a callable."

        def formatter(x):
            return f"{name}/{x}"

        self.name = name
        (self.eqn, self.eqn_tex) = (eqn, eqn_tex)
        self.model_function = model_function

        args = inspect.getfullargspec(model_function).args
        parameter_names = args[1:]
        self.independent = independent if independent else args[0]
        self.dependent = dependent

        for key in kwargs:
            assert key in parameter_names, "Attempted to set default for parameter which is not present in model."

        self._parameters = OrderedDict()
        for key in parameter_names:
            if key in kwargs:
                assert isinstance(kwargs[key], Parameter), "Passed a non-parameter as model default."
                if kwargs[key].shared:
                    self._parameters[key] = deepcopy(kwargs[key])
                else:
                    self._parameters[formatter(key)] = deepcopy(kwargs[key])
            else:
                self._parameters[formatter(key)] = None

        self._jacobian = jacobian
        self._derivative = derivative

    def __call__(self, independent, parameters):
        """Evaluate the model for specific parameters

        Parameters
        ----------
        independent: array_like
        parameters: ``pylake.fitting.Parameters``
        """
        independent = np.asarray(independent, dtype=np.float64)
        return self._raw_call(independent, np.asarray([parameters[name].value for name in self.parameter_names],
                                                      dtype=np.float64))

    def _raw_call(self, independent, parameter_vector):
        return self.model_function(independent, *parameter_vector)

    def __add__(self, other):
        """
        Add two model outputs to form a new model.

        Parameters
        ----------
        other: pylake.fitting.Model

        Examples
        --------
        ::
            DNA_model = pylake.inverted_odijk("DNA")
            protein_model = pylake.inverted_odijk("protein")
            construct_model = DNA_model + protein_model
        """

        return CompositeModel(self, other)

    def get_formatted_equation_string(self, tex):
        if self.eqn:
            if tex:
                return (f"${self.dependent}\\left({self.independent}\\right) = "
                        f"{self.eqn_tex(self.independent, *[escape_tex(x) for x in self._parameters.keys()])}$")
            else:
                return (f"{self.dependent}({self.independent}) = "
                        f"{self.eqn(self.independent, *[x.replace('/', '.') for x in self._parameters.keys()])}")

    def _repr_html_(self):
        doc_string = ''
        try:
            doc = self.model_function.__doc__.replace('\n', '  <br>\n')
            doc_string = f"{doc}  <br><br>\n"
        except AttributeError:
            # If it is not a top level model, there will be no docstring. This is fine.
            pass

        equation = self.get_formatted_equation_string(tex=True)
        equation = f"<h5>Model equation:</h5>\n{equation}<br><br>\n" if equation else ""

        model_info = (f"<h5>Model: {self.name}</h5>\n"
                      f"{doc_string}{equation}"
                      f"<h5>Parameter defaults:</h5>\n"
                      f"{Parameters(**self._parameters)._repr_html_()}\n")

        return model_info

    def __repr__(self):
        equation = self.get_formatted_equation_string(tex=False)
        equation = f"Model equation:\n\n{equation}\n\n" if equation else ""

        model_info = (f"Model: {self.name}\n\n"
                      f"{equation}Parameter defaults:\n\n"
                      f"{Parameters(**self._parameters)._repr_()}\n")

        return model_info

    def invert(self, independent_min=0.0, independent_max=np.inf, interpolate=False):
        """
        Invert this model (swap dependent and independent parameter).
        """
        return InverseModel(self, independent_min, independent_max, interpolate)

    def subtract_independent_offset(self):
        """
        Subtract a constant offset from independent variable of this model.

        Parameters
        ----------
        parameter_name: str
        """
        parameter_name = f"{self.name}/{self.independent}_offset" if self.independent else f"{self.name}/offset"
        return SubtractIndependentOffset(self, parameter_name)

    @property
    def defaults(self):
        return self._parameters

    @property
    def parameter_names(self):
        return [x for x in self._parameters.keys()]

    def jacobian(self, independent, parameter_vector):
        """
        Return model sensitivities at specific values for the independent variable. Returns None when the model does not
        have an appropriately defined Jacobian.

        Parameters
        ----------
        independent: array_like
            Values for the independent variable at which the Jacobian needs to be returned.
        parameter_vector: array_like
            Parameter vector at which to simulate.
        """
        if self.has_jacobian:
            independent = np.asarray(independent, dtype=np.float64)
            return self._jacobian(independent, *parameter_vector)
        else:
            raise RuntimeError(f"Jacobian was requested but not supplied in model {self.name}.")

    def derivative(self, independent, parameter_vector):
        """
        Return derivative w.r.t. the independent variable at specific values for the independent variable. Returns None
        when the model does not have an appropriately defined derivative.

        Parameters
        ----------
        independent: array_like
            Values for the independent variable at which the derivative needs to be returned.
        parameter_vector: array_like
            Parameter vector at which to simulate.
        """
        if self.has_derivative:
            independent = np.asarray(independent, dtype=np.float64)
            return self._derivative(independent, *parameter_vector)
        else:
            raise RuntimeError(f"Derivative was requested but not supplied in model {self.name}.")

    @property
    def has_jacobian(self):
        """Returns true if the model can return an analytically computed Jacobian."""
        if self._jacobian:
            return True

    @property
    def has_derivative(self):
        """Returns true if the model can return an analytically computed derivative w.r.t. the independent variable."""
        if self._derivative:
            return True

    def _calculate_residual(self, data_sets, global_parameter_values):
        """Calculate the model residual
        """
        residual_idx = 0
        residual = np.zeros(data_sets.n_residuals)
        for condition, data_list in data_sets.conditions():
            p_local = condition.get_local_parameters(global_parameter_values)
            for data in data_list:
                y_model = self._raw_call(data.x, p_local)

                residual[residual_idx:residual_idx + len(y_model)] = data.y - y_model
                residual_idx += len(y_model)

        return residual

    def _calculate_jacobian(self, data_sets, global_parameter_values):
        residual_idx = 0
        jacobian = np.zeros((data_sets.n_residuals, len(global_parameter_values)))
        for condition, data_list in data_sets.conditions():
            p_local = condition.get_local_parameters(global_parameter_values)
            p_indices = condition.p_indices
            for data in data_list:
                sensitivities = condition.localize_sensitivities(np.transpose(self.jacobian(data.x, p_local)))
                n_res = sensitivities.shape[0]

                jacobian[residual_idx:residual_idx + n_res, p_indices] -= sensitivities

                residual_idx += n_res

        return jacobian

    def verify_jacobian(self, independent, parameters, plot=False, verbose=True, dx=1e-6, **kwargs):
        """
        Verify this model's Jacobian with respect to the independent variable by comparing it to the Jacobian
        obtained with finite differencing.

        Parameters
        ----------
        independent: array_like
            Values for the independent variable at which to compare the Jacobian.
        parameters: array_like
            Parameter vector at which to compare the Jacobian.
        plot: bool
            Plot the results (default = False)
        verbose: bool
            Print the result (default = True)
        dx: float
            Finite difference excursion.
        **kwargs:
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        independent = np.asarray(independent, dtype=np.float64)
        jacobian = self.jacobian(independent, parameters)
        jacobian_fd = numerical_jacobian(lambda parameter_values: self._raw_call(independent, parameter_values),
                                         parameters, dx=dx)

        jacobian = np.asarray(jacobian)
        if plot:
            n_x, n_y = optimal_plot_layout(len(self._parameters))
            for i_parameter, parameter in enumerate(self._parameters):
                plt.subplot(n_x, n_y, i_parameter+1)
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

    def verify_derivative(self, independent, parameters, dx=1e-6, **kwargs):
        """
        Verify this model's derivative with respect to the independent variable by comparing it to the derivative
        obtained with finite differencing.

        Parameters
        ----------
        independent: array_like
            Values for the independent variable at which to compare the derivative.
        parameters: array_like
            Parameter vector at which to compare the derivative.
        dx: float
            Finite difference excursion.
        """

        if len(parameters) != len(self._parameters):
            raise ValueError("Parameter vector has invalid length. "
                             f"Expected: {len(self._parameters)}, got: {len(parameters)}.")

        derivative = self.derivative(independent, parameters)
        derivative_fd = numerical_diff(lambda x: self._raw_call(x, parameters), independent, dx=dx)

        return np.allclose(derivative, derivative_fd, **kwargs)

    def _plot_model(self, global_parameters, datasets, fmt='', **kwargs):
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if len(fmt) == 0:
            set_color = kwargs.get("color")
        else:
            set_color = 1

        for i, data in enumerate(datasets.values()):
            if not set_color:
                kwargs["color"] = lighten_color(get_color(i), -.3)
            self.plot(global_parameters[data], data.x, fmt=fmt, **kwargs)

    def plot(self, parameters, independent, fmt='', **kwargs):
        """Plot this model for a specific data set.

        Parameters
        ----------
        parameters: Parameters
            Parameter set, typically obtained from a Fit.
        independent: array_like
            Array of values for the independent variable.
        fmt: str (optional)
            Plot formatting string (see `matplotlib.pyplot.plot` documentation).
        **kwargs:
            Forwarded to `~matplotlib.pyplot.plot`.

        Examples
        --------
        ::

            dna_model = pylake.inverted_odijk("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.Fit(dna_model)
            fit.add_data("data1", force1, distance1)
            fit.add_data("data2", force2, distance2, {"DNA/Lc": "DNA/Lc_RecA"})
            fit.fit()
            dna_model.plot(fit["data1"], distance1, fmt='k--')  # Plot model simulations for data set 1
            dna_model.plot(fit["data2"], distance2, fmt='k--')  # Plot model simulations for data set 2

            # Plot model over a custom time range
            dna_model.plot(fit["data1"], np.arange(1.0, 10.0, .01), fmt='k--')
        """
        # Admittedly not very pythonic, but the errors you get otherwise are confusing.
        if not isinstance(parameters, Parameters):
            raise RuntimeError('Did not pass Parameters')

        plt.plot(independent, self(independent, parameters), fmt, **kwargs)


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

        assert self.lhs.independent == self.rhs.independent, \
            f"Error: Models contain different independent variables {self.lhs.independent} and {self.rhs.independent}"

        assert self.lhs.dependent == self.rhs.dependent, \
            f"Error: Models contain different dependent variables {self.lhs.dependent} and {self.rhs.dependent}"

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

    @property
    def dependent(self):
        return self.lhs.dependent

    @property
    def independent(self):
        return self.lhs.independent

    def eqn(self, independent_name, *parameter_names):
        return self.lhs.eqn(independent_name, *[parameter_names[x] for x in self.lhs_parameters]) + " + " +\
            self.rhs.eqn(independent_name, *[parameter_names[x] for x in self.rhs_parameters])

    def eqn_tex(self, independent, *parameter_names):
        return self.lhs.eqn_tex(independent, *[parameter_names[x] for x in self.lhs_parameters]) + \
               " + " + self.rhs.eqn_tex(independent, *[parameter_names[x] for x in self.rhs_parameters])

    def _raw_call(self, independent, parameter_vector):
        lhs_residual = self.lhs._raw_call(independent, [parameter_vector[x] for x in self.lhs_parameters])
        rhs_residual = self.rhs._raw_call(independent, [parameter_vector[x] for x in self.rhs_parameters])

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
    def __init__(self, model, independent_min=0.0, independent_max=np.inf, interpolate=False):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        model: Model
        independent_min: float
            Minimum value for the independent variable of the forward model. Default: 0.0.
        independent_max: float
            Maximum value for the independent variable of the forward model. Default: np.inf.
            Note that a finite maximum has to be specified if you wish to use the interpolation mode.
        interpolate: bool
            Use interpolation approximation. Default: False.
        """
        self.model = model
        self.name = "inv(" + model.name + ")"
        self.interpolate = interpolate
        self.independent_min = independent_min
        self.independent_max = independent_max
        if self.interpolate:
            assert np.isfinite(independent_min) and np.isfinite(independent_max), \
                "Inversion limits have to be finite when using interpolation method."

    @property
    def dependent(self):
        return self.model.independent

    @property
    def independent(self):
        return self.model.dependent

    def eqn(self, independent_name, *parameter_names):
        return solve_formatter(self.model.eqn(independent_name, *parameter_names), self.dependent, self.independent)

    def eqn_tex(self, independent_name, *parameter_names):
        return solve_formatter_tex(self.model.eqn_tex(independent_name, *parameter_names), self.dependent,
                                   self.independent)

    def _raw_call(self, independent, parameter_vector):
        if self.interpolate:
            return invert_function_interpolation(independent, 1.0, self.independent_min, self.independent_max,
                                                 lambda f_trial: self.model._raw_call(f_trial, parameter_vector),
                                                 lambda f_trial: self.model.derivative(f_trial, parameter_vector))
        else:
            return invert_function(independent, 1.0, self.independent_min, self.independent_max,
                                   lambda f_trial: self.model._raw_call(f_trial, parameter_vector),  # Forward model
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
                               lambda f_trial: self._raw_call(f_trial, parameter_vector),  # Inverse model (me)
                               lambda f_trial: self.model.jacobian(f_trial, parameter_vector),
                               lambda f_trial: self.model.derivative(f_trial, parameter_vector))

    def derivative(self, independent, parameter_vector):
        """Derivative of the inverted model"""
        return invert_derivative(independent,
                                 lambda f_trial: self._raw_call(f_trial, parameter_vector),  # Inverse model (me)
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
        self._parameters[offset_name] = Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="au")
        for i, v in self.model._parameters.items():
            self._parameters[i] = v

        parameters_parent = list(self.model._parameters.keys())
        parameters_all = list(self._parameters.keys())

        self.model_parameters = [parameters_all.index(par) for par in parameters_parent]
        self.offset_parameter = parameters_all.index(offset_name)

    def _raw_call(self, independent, parameter_vector):
        return self.model._raw_call(independent - parameter_vector[self.offset_parameter],
                                    [parameter_vector[x] for x in self.model_parameters])

    def eqn(self, independent_name, *parameter_names):
        return self.model.eqn(f"({independent_name} - {parameter_names[self.offset_parameter]})",
                              *[parameter_names[x] for x in self.model_parameters])

    def eqn_tex(self, independent_name, *parameter_names):
        return self.model.eqn_tex(f"({independent_name} - {parameter_names[self.offset_parameter]})",
                                  *[parameter_names[x] for x in self.model_parameters])

    @property
    def dependent(self):
        return self.model.dependent

    @property
    def independent(self):
        return self.model.independent

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
            return self.model.derivative(with_offset, [parameter_vector[x] for x in self.model_parameters])
