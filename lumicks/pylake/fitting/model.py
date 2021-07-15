from .parameters import Parameter, Params
from .detail.utilities import (
    optimal_plot_layout,
    print_styled,
    solve_formatter_tex,
    escape_tex,
    solve_formatter,
)
from .detail.derivative_manipulation import (
    numerical_jacobian,
    numerical_diff,
    invert_function,
    invert_jacobian,
    invert_derivative,
    invert_function_interpolation,
)
from collections import OrderedDict
from copy import deepcopy

import inspect
import types
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(
        self,
        name,
        model_function,
        dependent=None,
        independent=None,
        jacobian=None,
        derivative=None,
        eqn=None,
        eqn_tex=None,
        **kwargs,
    ):
        """
        Model constructor. A Model must be named, and this name will appear in the model parameters.

        Ideally a jacobian and derivative w.r.t. the independent variable are provided with every model. This will
        allow much higher performance when fitting. Jacobians and derivatives are automatically propagated to composite
        models, inversions of models etc. provided that all participating models have jacobians and derivatives
        specified.

        Parameters
        ----------
        name : str
            Name for the model. This name will be prefixed to the model parameter names.
        model_function : callable
            Function containing the model function. Must return the model prediction given values for the independent
            variable and parameters.
        dependent : str (optional)
            Name of the dependent variable
        independent : str (optional)
            Name of the independent variable
        jacobian : callable (optional)
            Function which computes the first order derivatives with respect to the parameters for this model.
            When supplied, this function is used to speed up the optimization considerably.
        derivative : callable (optional)
            Function which computes the first order derivative with respect to the independent parameter. When supplied
            this speeds up model inversions considerably.
        eqn : str (optional)
            Equation that this model is specified by.
        eqn_tex : str (optional)
            Equation that this model is specified by using TeX formatting.
        **kwargs
            Key pairs containing parameter defaults. For instance, Lc=Parameter(...)

        Examples
        --------
        ::

            from lumicks import pylake

            dna_model = pylake.inverted_odijk("DNA")
            fit = pylake.FdFit(dna_model)
            fit.add_data("my data", force, distance)

            fit["DNA/Lp"].lower_bound = 35  # Set lower bound for DNA Lp
            fit["DNA/Lp"].upper_bound = 80  # Set upper bound for DNA Lp
            fit.fit()

            fit.plot("my data", "k--")  # Plot the fitted model
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
            assert (
                key in parameter_names
            ), "Attempted to set default for parameter which is not present in model."

        self._params = OrderedDict()
        for key in parameter_names:
            if key in kwargs:
                assert isinstance(
                    kwargs[key], Parameter
                ), "Passed a non-parameter as model default."
                if kwargs[key].shared:
                    self._params[key] = deepcopy(kwargs[key])
                else:
                    self._params[formatter(key)] = deepcopy(kwargs[key])
            else:
                self._params[formatter(key)] = None

        self._jacobian = jacobian
        self._derivative = derivative

    def __call__(self, independent, params):
        """Evaluate the model for specific parameters

        Parameters
        ----------
        independent : array_like
        params : ``pylake.fitting.Params``
        """
        independent = np.asarray(independent, dtype=np.float64)
        missing_parameters = set(self.parameter_names) - set(params.keys())
        if missing_parameters:
            raise KeyError(
                f"The following missing parameters must be specified to simulate the "
                f"model: {missing_parameters}."
            )

        return self._raw_call(
            independent,
            np.asarray([float(params[name]) for name in self.parameter_names], dtype=np.float64),
        )

    def _raw_call(self, independent, param_vector):
        return self.model_function(independent, *param_vector)

    def __add__(self, other):
        """
        Add two model outputs to form a new model.

        Parameters
        ----------
        other : pylake.fitting.Model

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
                return (
                    f"${self.dependent}\\left({self.independent}\\right) = "
                    f"{self.eqn_tex(self.independent, *[escape_tex(x) for x in self._params.keys()])}$"
                )
            else:
                return (
                    f"{self.dependent}({self.independent}) = "
                    f"{self.eqn(self.independent, *[x.replace('/', '.') for x in self._params.keys()])}"
                )

    def _repr_html_(self):
        doc_string = ""
        try:
            doc = self.model_function.__doc__.replace("\n", "  <br>\n")
            doc_string = f"{doc}  <br><br>\n"
        except AttributeError:
            # If it is not a top level model, there will be no docstring. This is fine.
            pass

        equation = self.get_formatted_equation_string(tex=True)
        equation = f"<h5>Model equation:</h5>\n{equation}<br><br>\n" if equation else ""

        model_info = (
            f"<h5>Model: {self.name}</h5>\n"
            f"{doc_string}{equation}"
            f"<h5>Parameter defaults:</h5>\n"
            f"{Params(**self._params)._repr_html_()}\n"
        )

        return model_info

    def __repr__(self):
        equation = self.get_formatted_equation_string(tex=False)
        equation = f"Model equation:\n\n{equation}\n\n" if equation else ""

        model_info = (
            f"Model: {self.name}\n\n"
            f"{equation}Parameter defaults:\n\n"
            f"{Params(**self._params)._repr_()}\n"
        )

        return model_info

    def invert(self, independent_min=0.0, independent_max=np.inf, interpolate=False):
        """
        Invert this model (swap dependent and independent parameter).
        """
        return InverseModel(self, independent_min, independent_max, interpolate)

    def subtract_independent_offset(self):
        """
        Subtract a constant offset from independent variable of this model.
        """
        param_name = (
            f"{self.name}/{self.independent}_offset" if self.independent else f"{self.name}/offset"
        )
        return SubtractIndependentOffset(self, param_name)

    @property
    def defaults(self):
        return self._params

    @property
    def parameter_names(self):
        return [x for x in self._params.keys()]

    def jacobian(self, independent, param_vector):
        """
        Return model sensitivities at specific values for the independent variable. Returns None when the model does not
        have an appropriately defined Jacobian.

        Parameters
        ----------
        independent : array_like
            Values for the independent variable at which the Jacobian needs to be returned.
        param_vector : array_like
            Parameter vector at which to simulate.
        """
        if self.has_jacobian:
            independent = np.asarray(independent, dtype=np.float64)
            return self._jacobian(independent, *param_vector)
        else:
            raise RuntimeError(f"Jacobian was requested but not supplied in model {self.name}.")

    def derivative(self, independent, param_vector):
        """
        Return derivative w.r.t. the independent variable at specific values for the independent variable. Returns None
        when the model does not have an appropriately defined derivative.

        Parameters
        ----------
        independent : array_like
            Values for the independent variable at which the derivative needs to be returned.
        param_vector : array_like
            Parameter vector at which to simulate.
        """
        if self.has_derivative:
            independent = np.asarray(independent, dtype=np.float64)
            return self._derivative(independent, *param_vector)
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

    def _calculate_residual(self, data_sets, global_param_values):
        """Calculate the model residual"""
        residual_idx = 0
        residual = np.zeros(data_sets.n_residuals)
        for condition, data_list in data_sets.conditions():
            p_local = condition.get_local_params(global_param_values)
            for data in data_list:
                y_model = self._raw_call(data.x, p_local)

                residual[residual_idx : residual_idx + len(y_model)] = data.y - y_model
                residual_idx += len(y_model)

        return residual

    def _calculate_jacobian(self, data_sets, global_param_values):
        residual_idx = 0
        jacobian = np.zeros((data_sets.n_residuals, len(global_param_values)))
        for condition, data_list in data_sets.conditions():
            p_local = condition.get_local_params(global_param_values)
            p_indices = condition.p_indices
            for data in data_list:
                sensitivities = condition.localize_sensitivities(
                    np.transpose(self.jacobian(data.x, p_local))
                )
                n_res = sensitivities.shape[0]

                jacobian[residual_idx : residual_idx + n_res, p_indices] -= sensitivities

                residual_idx += n_res

        return jacobian

    def verify_jacobian(self, independent, params, plot=False, verbose=True, dx=1e-6, **kwargs):
        """
        Verify this model's Jacobian with respect to the independent variable by comparing it to the Jacobian
        obtained with finite differencing.

        Parameters
        ----------
        independent : array_like
            Values for the independent variable at which to compare the Jacobian.
        params : array_like
            Parameter vector at which to compare the Jacobian.
        plot : bool
            Plot the results (default = False)
        verbose : bool
            Print the result (default = True)
        dx : float
            Finite difference excursion.
        **kwargs :
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        if len(params) != len(self._params):
            raise ValueError(
                "Parameter vector has invalid length. "
                f"Expected: {len(self._params)}, got: {len(params)}."
            )

        independent = np.asarray(independent, dtype=np.float64)
        jacobian = self.jacobian(independent, params)
        jacobian_fd = numerical_jacobian(
            lambda param_values: self._raw_call(independent, param_values), params, dx=dx
        )

        jacobian = np.asarray(jacobian)
        if plot:
            n_x, n_y = optimal_plot_layout(len(self._params))
            for i_param, param in enumerate(self._params):
                plt.subplot(n_x, n_y, i_param + 1)
                l1 = plt.plot(independent, np.transpose(jacobian[i_param, :]))
                l2 = plt.plot(independent, np.transpose(jacobian_fd[i_param, :]), "--")
                plt.title(param)
                plt.legend({"Analytic", "FD"})

        is_close = np.allclose(jacobian, jacobian_fd, **kwargs)
        if not is_close:
            if verbose:
                maxima = np.max(jacobian - jacobian_fd, axis=1)
                for i, v in enumerate(maxima):
                    if np.allclose(jacobian[i, :], jacobian_fd[i, :]):
                        print(f"Parameter {self.parameter_names[i]}({i}): {v}")
                    else:
                        print_styled("warning", f"Parameter {self.parameter_names[i]}({i}): {v}")

        return is_close

    def verify_derivative(self, independent, params, dx=1e-6, **kwargs):
        """
        Verify this model's derivative with respect to the independent variable by comparing it to the derivative
        obtained with finite differencing.

        Parameters
        ----------
        independent : array_like
            Values for the independent variable at which to compare the derivative.
        params : array_like
            Parameter vector at which to compare the derivative.
        dx : float
            Finite difference excursion.
        """

        if len(params) != len(self._params):
            raise ValueError(
                "Parameter vector has invalid length. "
                f"Expected: {len(self._params)}, got: {len(params)}."
            )

        derivative = self.derivative(independent, params)
        derivative_fd = numerical_diff(lambda x: self._raw_call(x, params), independent, dx=dx)

        return np.allclose(derivative, derivative_fd, **kwargs)

    def plot(self, params, independent, fmt="", **kwargs):
        """Plot this model for a specific data set.

        Parameters
        ----------
        params : Params
            Parameter set, typically obtained from a Fit.
        independent : array_like
            Array of values for the independent variable.
        fmt : str (optional)
            Plot formatting string (see `matplotlib.pyplot.plot` documentation).
        **kwargs :
            Forwarded to `~matplotlib.pyplot.plot`.

        Examples
        --------
        ::

            dna_model = pylake.inverted_odijk("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.FdFit(dna_model)
            fit.add_data("data1", force1, distance1)
            fit.add_data("data2", force2, distance2, {"DNA/Lc": "DNA/Lc_RecA"})
            fit.fit()

            # Option 1
            fit.plot("data 1", 'k--', distance1)  # Plot model simulations for data set 1
            fit.plot("data 2", 'k--', distance2)  # Plot model simulations for data set 2

            # Option 2
            dna_model.plot(fit["data1"], distance1, 'k--')  # Plot model simulations for data set 1
            dna_model.plot(fit["data2"], distance2, 'k--')  # Plot model simulations for data set 2
        """
        # Admittedly not very pythonic, but the errors you get otherwise are confusing.
        if not isinstance(params, Params):
            raise RuntimeError("Did not pass Params")

        return plt.plot(independent, self(independent, params), fmt, **kwargs)


class CompositeModel(Model):
    def __init__(self, lhs, rhs):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        lhs : Model
        rhs : Model
        """
        self.lhs = lhs
        self.rhs = rhs

        assert (
            self.lhs.independent == self.rhs.independent
        ), f"Error: Models contain different independent variables {self.lhs.independent} and {self.rhs.independent}"

        assert (
            self.lhs.dependent == self.rhs.dependent
        ), f"Error: Models contain different dependent variables {self.lhs.dependent} and {self.rhs.dependent}"

        self.name = self.lhs.name + "_with_" + self.rhs.name
        self._params = OrderedDict()
        for i, v in self.lhs._params.items():
            self._params[i] = v
        for i, v in self.rhs._params.items():
            self._params[i] = v

        params_lhs = list(self.lhs._params.keys())
        params_rhs = list(self.rhs._params.keys())
        params_all = list(self._params.keys())

        self.lhs_params = [params_all.index(par) for par in params_lhs]
        self.rhs_params = [params_all.index(par) for par in params_rhs]

    @property
    def dependent(self):
        return self.lhs.dependent

    @property
    def independent(self):
        return self.lhs.independent

    def eqn(self, independent_name, *parameter_names):
        return (
            self.lhs.eqn(independent_name, *[parameter_names[x] for x in self.lhs_params])
            + " + "
            + self.rhs.eqn(independent_name, *[parameter_names[x] for x in self.rhs_params])
        )

    def eqn_tex(self, independent, *parameter_names):
        return (
            self.lhs.eqn_tex(independent, *[parameter_names[x] for x in self.lhs_params])
            + " + "
            + self.rhs.eqn_tex(independent, *[parameter_names[x] for x in self.rhs_params])
        )

    def _raw_call(self, independent, param_vector):
        lhs_residual = self.lhs._raw_call(independent, [param_vector[x] for x in self.lhs_params])
        rhs_residual = self.rhs._raw_call(independent, [param_vector[x] for x in self.rhs_params])

        return lhs_residual + rhs_residual

    @property
    def has_jacobian(self):
        return self.lhs.has_jacobian and self.rhs.has_jacobian

    @property
    def has_derivative(self):
        return self.lhs.has_derivative and self.rhs.has_derivative

    def jacobian(self, independent, param_vector):
        if self.has_jacobian:
            jacobian = np.zeros((len(param_vector), len(independent)))
            jacobian[self.lhs_params, :] += self.lhs.jacobian(
                independent, [param_vector[x] for x in self.lhs_params]
            )
            jacobian[self.rhs_params, :] += self.rhs.jacobian(
                independent, [param_vector[x] for x in self.rhs_params]
            )

            return jacobian

    def derivative(self, independent, param_vector):
        if self.has_derivative:
            lhs_derivative = self.lhs.derivative(
                independent, [param_vector[x] for x in self.lhs_params]
            )
            rhs_derivative = self.rhs.derivative(
                independent, [param_vector[x] for x in self.rhs_params]
            )

            return lhs_derivative + rhs_derivative


class InverseModel(Model):
    def __init__(self, model, independent_min=0.0, independent_max=np.inf, interpolate=False):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        model : Model
        independent_min : float
            Minimum value for the independent variable of the forward model. Default: 0.0.
        independent_max : float
            Maximum value for the independent variable of the forward model. Default: np.inf.
            Note that a finite maximum has to be specified if you wish to use the interpolation mode.
        interpolate : bool
            Use interpolation approximation. Default: False.
        """
        self.model = model
        self.name = "inv(" + model.name + ")"
        self.interpolate = interpolate
        self.independent_min = independent_min
        self.independent_max = independent_max
        if self.interpolate:
            assert np.isfinite(independent_min) and np.isfinite(
                independent_max
            ), "Inversion limits have to be finite when using interpolation method."

    @property
    def dependent(self):
        return self.model.independent

    @property
    def independent(self):
        return self.model.dependent

    def eqn(self, independent_name, *parameter_names):
        return solve_formatter(
            self.model.eqn(independent_name, *parameter_names), self.dependent, self.independent
        )

    def eqn_tex(self, independent_name, *parameter_names):
        return solve_formatter_tex(
            self.model.eqn_tex(independent_name, *parameter_names), self.dependent, self.independent
        )

    def _raw_call(self, independent, param_vector):
        if self.interpolate:
            return invert_function_interpolation(
                independent,
                1.0,
                self.independent_min,
                self.independent_max,
                lambda f_trial: self.model._raw_call(f_trial, param_vector),
                lambda f_trial: self.model.derivative(f_trial, param_vector),
            )
        else:
            return invert_function(
                independent,
                1.0,
                self.independent_min,
                self.independent_max,
                lambda f_trial: self.model._raw_call(f_trial, param_vector),  # Forward model
                lambda f_trial: self.model.derivative(f_trial, param_vector),
            )

    @property
    def has_jacobian(self):
        """Does the model have sufficient information to determine its inverse numerically?
        This requires a Jacobian and a derivative w.r.t. independent variable."""
        return self.model.has_jacobian and self.model.has_derivative

    @property
    def has_derivative(self):
        return self.model.has_derivative

    def jacobian(self, independent, param_vector):
        """Jacobian of the inverted model"""
        return invert_jacobian(
            independent,
            lambda f_trial: self._raw_call(f_trial, param_vector),  # Inverse model (me)
            lambda f_trial: self.model.jacobian(f_trial, param_vector),
            lambda f_trial: self.model.derivative(f_trial, param_vector),
        )

    def derivative(self, independent, param_vector):
        """Derivative of the inverted model"""
        return invert_derivative(
            independent,
            lambda f_trial: self._raw_call(f_trial, param_vector),  # Inverse model (me)
            lambda f_trial: self.model.derivative(f_trial, param_vector),
        )

    @property
    def _params(self):
        return self.model._params


class SubtractIndependentOffset(Model):
    def __init__(self, model, parameter_name="independent_offset"):
        """
        Combine two model outputs to form a new model (addition).

        Parameters
        ----------
        model : Model
        """
        self.model = model
        offset_name = parameter_name

        self.name = self.model.name + "(x-d)"
        self._params = OrderedDict()
        self._params[offset_name] = Parameter(
            value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="au"
        )
        for i, v in self.model._params.items():
            self._params[i] = v

        params_parent = list(self.model._params.keys())
        params_all = list(self._params.keys())

        self.model_params = [params_all.index(par) for par in params_parent]
        self.offset_parameter = params_all.index(offset_name)

    def _raw_call(self, independent, param_vector):
        return self.model._raw_call(
            independent - param_vector[self.offset_parameter],
            [param_vector[x] for x in self.model_params],
        )

    def eqn(self, independent_name, *parameter_names):
        return self.model.eqn(
            f"({independent_name} - {parameter_names[self.offset_parameter]})",
            *[parameter_names[x] for x in self.model_params],
        )

    def eqn_tex(self, independent_name, *parameter_names):
        return self.model.eqn_tex(
            f"({independent_name} - {parameter_names[self.offset_parameter]})",
            *[parameter_names[x] for x in self.model_params],
        )

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

    def jacobian(self, independent, param_vector):
        if self.has_jacobian:
            with_offset = independent - param_vector[self.offset_parameter]
            jacobian = np.zeros((len(param_vector), len(with_offset)))
            jacobian[self.model_params, :] += self.model.jacobian(
                with_offset, [param_vector[x] for x in self.model_params]
            )
            jacobian[self.offset_parameter, :] = -self.model.derivative(
                with_offset, [param_vector[x] for x in self.model_params]
            )

            return jacobian

    def derivative(self, independent, param_vector):
        if self.has_derivative:
            with_offset = independent - param_vector[self.offset_parameter]
            return self.model.derivative(with_offset, [param_vector[x] for x in self.model_params])
