import inspect
import numpy as np
import scipy as sp
from .detail.utilities import unique, unique_idx, optimal_plot_layout, print_styled
from collections import OrderedDict
from copy import deepcopy
import scipy.optimize as optim


def parameter_trace(model, parameters, inverted_parameter, independent, dependent, **kwargs):
    """Invert the model with respect to one parameter. This function fits a unique parameter for every data point in
    this data-set while keeping all other parameters fixed. This can be used to for example invert the model with
    respect to the contour length or some other parameter.

    Parameters
    ----------
    model : Model
        Fitting model.
    parameters : Parameters
        Model parameters.
    inverted_parameter : str
        Parameter to invert.
    independent : array_like
        vector of values for the independent variable
    dependent: array_like
        vector of values for the dependent variable
    **kwargs
        parameter renames (e.g. protein_Lc="protein_Lc_1")

    Examples
    --------
    ::
        # Define the model to be fitted
        M_protein = force_model("protein", "invWLC") + force_model("f", "offset")

        # Fit the overall model first
        M_protein.load_data(distances_corrected, forces)
        protein_fit = FitObject(M_protein)
        protein_fit.fit()

        # Calculate a per datapoint contour length
        lcs = parameter_trace(M_protein, protein_fit.parameters, "protein_Lc", distances, forces)
    """
    import scipy as sp

    parameter_names = list(parse_transformation(model.parameter_names, **kwargs).keys())
    assert inverted_parameter in parameters, "Inverted parameter not in model parameter vector."
    for key in parameter_names:
        assert key in parameters, f"Missing parameter {key} in supplied parameter vector."

    # Grab reference parameter vector and index for the parameter list
    parameter_vector = [parameters[key].value for key in parameter_names]
    lb = parameters[inverted_parameter].lb
    ub = parameters[inverted_parameter].ub
    inverted_parameter_index = parameter_names.index(inverted_parameter)

    def residual(inverted_parameter_values):
        parameter_vector[inverted_parameter_index] = inverted_parameter_values
        return dependent - model(independent, parameter_vector)

    def jacobian(inverted_parameter_values):
        parameter_vector[inverted_parameter_index] = inverted_parameter_values
        return -sp.sparse.diags(model.jacobian(independent, parameter_vector)[inverted_parameter_index, :], offsets=0)

    initial_estimate = np.ones(independent.shape) * parameter_vector[inverted_parameter_index]

    jac = jacobian if model.has_jacobian else "2-point"
    result = optim.least_squares(residual, initial_estimate, jac=jac,
                                 jac_sparsity=sp.sparse.identity(len(independent)),
                                 bounds=(lb, ub), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-8, verbose=2)

    return result.x


def parse_transformation(parameters, **kwargs):
    transformed = OrderedDict(zip(parameters, parameters))

    for key, value in kwargs.items():
        if key in transformed:
            transformed[key] = value
        else:
            raise KeyError(f"Parameter {key} to be substituted not found in model. Valid keys for this model are: "
                           f"{[x for x in transformed.keys()]}.")

    return transformed


def _generate_conditions(data_sets, parameter_lookup, model_parameters):
    """
    This function builds a list of unique conditions from a list of data sets and a list of index lists which link back
    the individual data fields to their simulation conditions.

    Parameters
    ----------
    data_sets : list of Data
        References to data
    parameter_lookup: OrderedDict[str, int]
        Lookup table for looking up parameter indices by name
    model_parameters: list of str
        Base model parameter names
    """
    # Quickly concatenate the parameter transformations corresponding to this condition
    str_conditions = []
    for data_set in data_sets:
        str_conditions.append(data_set.condition_string)

        assert set(data_set.transformations.keys()) == set(model_parameters), \
            "Source parameters in data parameter transformations are incompatible with the specified model parameters."

        target_parameters = [x for x in data_set.transformations.values() if isinstance(x, str)]
        assert set(target_parameters).issubset(parameter_lookup.keys()), \
            "Parameter transformations contain transformed parameter names that are not in the combined parameter list."

    # Determine unique parameter conditions and the indices to get the appropriate unique condition from data index.
    unique_condition_strings, indices = unique_idx(str_conditions)
    indices = np.array(indices)

    data_link = []
    for condition_idx in np.arange(len(unique_condition_strings)):
        data_indices, = np.nonzero(np.equal(indices, condition_idx))
        data_link.append(data_indices)

    conditions = []
    for idx in data_link:
        transformations = data_sets[idx[0]].transformations
        conditions.append(Condition(transformations, parameter_lookup))

    return conditions, data_link


def numerical_jacobian(fn, parameter_vector, dx=1e-6):
    finite_difference_jacobian = np.zeros((len(parameter_vector), len(fn(parameter_vector))))
    for i in np.arange(len(parameter_vector)):
        parameters = np.copy(parameter_vector)
        parameters[i] = parameters[i] + dx
        up = fn(parameters)
        parameters[i] = parameters[i] - 2.0 * dx
        down = fn(parameters)
        finite_difference_jacobian[i, :] = (up - down) / (2.0*dx)

    return finite_difference_jacobian


def invert_function(d, initial, f_min, f_max, model_function, derivative_function=None):
    """This function inverts a function using a least squares optimizer. For models where this is required, this is the
    most time consuming step.

    Parameters
    ----------
    d : array_like
        old independent parameter
    initial : array_like
        initial guess for the optimization procedure
    f_min : float
        minimum bound for inverted parameter
    f_max : float
        maximum bound for inverted parameter
    model_function : callable
        non-inverted model function
    derivative_function : callable
        model derivative with respect to the independent variable (returns an element per data point)
    """
    def jacobian(f_trial):
        return sp.sparse.diags(derivative_function(f_trial), offsets=0)

    jac = jacobian if derivative_function else "2-point"

    result = optim.least_squares(lambda f_trial: model_function(f_trial) - d, initial, jac=jac,
                                 jac_sparsity=sp.sparse.identity(len(d)),
                                 bounds=(f_min, f_max), method='trf', ftol=1e-06, xtol=1e-08, gtol=1e-8)

    return result.x


def invert_jacobian(d, inverted_model_function, jacobian_function, derivative_function):
    """This function computes the jacobian of the model when the model has been inverted with respect to the independent
    variable.

    The Jacobian of the function with one variable inverted is related to the original Jacobian
    The transformation Jacobian is structured as follows:

    [  dy/dF   dy/db   dy/dc  ]
    [   0        1       0    ]
    [   0        0       1    ]

    The inverse of this Jacobian provides us with the actual parameters that we are interested in. It is given by:
    [ (dy/da)^-1  -(dy/db)(dy/dF)^-1    -(dy/dc)(dy/dF)^-1 ]
    [    0                1                     0          ]
    [    0                0                     1          ]

    Parameters
    ----------
    d : values for the old independent variable
    inverted_model_function : callable
        inverted model function (model with the dependent and independent variable exchanged)
    jacobian_function : callable
        derivatives of the non-inverted model
    derivative_function : callable
        derivative of the non-inverted model w.r.t. the independent variable
    """
    F = inverted_model_function(d)
    jacobian = jacobian_function(F)
    derivative = derivative_function(F)
    inverse = 1.0/derivative
    inverted_dyda = np.tile(inverse, (jacobian.shape[0], 1))
    jacobian = -jacobian * inverted_dyda

    return jacobian


def invert_derivative(d, inverted_model_function, derivative_function):
    """
    Calculates the derivative of the inverted function.
    Parameters
    ----------
    d : values for the old independent variable
    inverted_model_function : callable
        inverted model function (model with the dependent and independent variable exchanged)
    derivative_function : callable
        derivative of the non-inverted model w.r.t. the independent variable
    """
    return 1.0 / derivative_function(inverted_model_function(d))

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

    @staticmethod
    def _sanitize_input_types(independent):
        independent = np.array(independent).astype(float)
        return independent

    def __call__(self, independent, parameter_vector):
        independent = self._sanitize_input_types(independent)
        return self.model_function(independent, *parameter_vector)

    @property
    def _defaults(self):
        from copy import deepcopy
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
        self._data.append(Data(name, x, y, parameter_list))
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
        self._conditions, self._data_link = _generate_conditions(self._data, parameter_lookup,
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
        return False

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


class Parameter:
    def __init__(self, value=0.0, lb=-np.inf, ub=np.inf, vary=True, init=None, shared=False):
        """Model parameter

        Parameters
        ----------
        value: float
        lb, ub: float
        vary: boolean
        init: float
        shared: boolean
        """
        self.value = value
        self.lb = lb
        self.ub = ub
        self.vary = vary
        self.shared = shared
        if init:
            self.init = init
        else:
            self.init = self.value

    def __repr__(self):
        return f"lumicks.pylake.fdfit.Parameter(value: {self.value}, lb: {self.lb}, ub: {self.ub}, vary: {self.vary})"

    def __str__(self):
        return self.__repr__()


class Parameters:
    def __init__(self):
        self._src = OrderedDict()

    def __iter__(self):
        return self._src.__iter__()

    def __lshift__(self, other):
        """
        Set parameters

        Parameters
        ----------
        other: Parameters
        """
        if isinstance(other, Parameters):
            found = False
            for key, param in other._src.items():
                if key in self._src:
                    self._src[key] = param
                    found = True

            if not found:
                raise RuntimeError("Tried to set parameters which do not exist in the target model.")
        else:
            raise RuntimeError("Attempt to stream non-parameter list to parameter list.")

    def items(self):
        return self._src.items()

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise IndexError("Slicing not supported. Only indexing.")

        if item in self._src:
            return self._src[item]
        else:
            raise IndexError(f"Parameter {item} does not exist.")

    def __setitem__(self, item, value):
        if item in self._src:
            self._src[item].value = value

    def __len__(self):
        return len(self._src)

    def __str__(self):
        return_str = ""
        maxlen = np.max([len(x) for x in self._src.keys()])
        for key, param in self._src.items():
            return_str = return_str + ("{:"+f"{maxlen+1}"+"s} {:1.4e} {:1d}\n").format(key, param.value, param.vary)

        return return_str

    def set_parameters(self, parameters, defaults):
        """Rebuild the parameter vector. Note that this can potentially alter the parameter order if the strings are
        given in a different order.

        Parameters
        ----------
        parameters : list of str
            parameter names
        defaults : Parameter or None
            default parameter objects
        """
        new_parameters = OrderedDict(zip(parameters, [Parameter() if x is None else x for x in defaults]))
        for key, value in self._src.items():
            if key in new_parameters:
                new_parameters[key] = value

        self._src = new_parameters

    @property
    def keys(self):
        return np.array([key for key in self._src.keys()])

    @property
    def values(self):
        return np.array([param.value for param in self._src.values()])

    @property
    def fitted(self):
        return np.array([param.vary for param in self._src.values()])

    @property
    def lb(self):
        return np.array([param.lb for param in self._src.values()])

    @property
    def ub(self):
        return np.array([param.ub for param in self._src.values()])


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
        defaults = [all_defaults[all_parameter_names.index(l)] for l in unique_parameter_names]

        for M in self.models:
            M._build_model(parameter_lookup, self)

        self._parameters.set_parameters(unique_parameter_names, defaults)
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
                           min_chi2_step=0.01, max_chi2_step=0.2, termination_significance=.99, confidence_level=.95):

        from scipy.stats import chi2

        if parameter_name not in self.parameters:
            raise KeyError(f"Parameter {parameter_name} not present in fitting object.")

        if not self.parameters[parameter_name].vary:
            raise RuntimeError(f"Parameter {parameter_name} is fixed in the fitting object.")

        assert max_step > min_step
        assert max_chi2_step > min_chi2_step

        n_dof = 1
        termination_level = chi2.ppf(termination_significance, n_dof)
        confidence_level = chi2.ppf(confidence_level, n_dof)
        max_chi2_step_size = max_chi2_step * confidence_level
        min_chi2_step_size = min_chi2_step * confidence_level
        sigma = self.sigma

        def trial(parameters=[]):
            return - 2.0 * self.log_likelihood(parameters, sigma)

        def do_step(chi2_last, parameter_vector, step_direction, current_step_size, sign):
            """
            Parameters
            ----------
            chi2_last: float
                previous chi squared value
            parameter_vector: array_like
                current parameter vector
            step_direction: array_like
                normalized direction in parameter space in which steps are taken
            current_step_size: float
                current step size
            sign: float
                sign of the stepping mechanism
            """
            # Determine an appropriate step size based on chi2 increase
            adjust_trial = True
            just_shrunk = False
            while adjust_trial:
                p_trial = parameter_vector + sign * current_step_size * step_direction
                chi2_trial = trial(p_trial)

                chi2_change = chi2_trial - chi2_last
                if chi2_change < min_chi2_step_size:
                    # Do not increase the step-size if we just shrunk. We already know it's going to be bad and we'd
                    # just be looping forever.
                    if not just_shrunk:
                        adjust_trial = True
                        current_step_size = current_step_size * step_factor
                        if current_step_size > max_step:
                            current_step_size = max_step
                            adjust_trial = False
                    else:
                        adjust_trial = False
                elif chi2_change > max_chi2_step_size:
                    adjust_trial = True
                    just_shrunk = True
                    current_step_size = current_step_size / step_factor
                    if current_step_size < min_step:
                        print("Warning: Step size set to minimum step size.")
                        current_step_size = min_step
                        adjust_trial = False
                else:
                    adjust_trial = False
                    just_shrunk = False

            return current_step_size, parameter_vector + sign * current_step_size * step_direction

        current_step_size = 1.0  # TODO: Better initial step size, maybe based on local Hessian approximation
        profiled_parameter_index = list(self.parameters.keys).index(parameter_name)
        parameter_vector, fitted, lb, ub = self._prepare_fit()
        fitted[profiled_parameter_index] = 0
        min_step = min_step * parameter_vector[profiled_parameter_index]
        max_step = max_step * parameter_vector[profiled_parameter_index]
        chi2_last = trial()

        step = 0
        step_direction = np.zeros(parameter_vector.shape)
        step_direction[profiled_parameter_index] = 1.0

        chi2s = []
        parameter_values = []
        p_next = parameter_vector
        while step < .5 * num_steps:
            current_step_size, p_next = do_step(chi2_last, p_next, step_direction, current_step_size, -1)
            p_next = self._fit(p_next, lb, ub, fitted, verbose=2)
            chi2s.append(trial(p_next))
            parameter_values.append(p_next[profiled_parameter_index])
            step += 1

        chi2s.reverse()
        parameter_values.reverse()
        p_next = parameter_vector
        while step < num_steps:
            current_step_size, p_next = do_step(chi2_last, p_next, step_direction, current_step_size, 1)
            p_next = self._fit(p_next, lb, ub, fitted)
            chi2s.append(trial(p_next))
            parameter_values.append(p_next[profiled_parameter_index])
            step += 1

        return chi2s, parameter_values

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
        res = self._calculate_residual(parameters)
        sigma = sigma if np.any(sigma) else self.sigma
        return - (self.n_residuals/2.0) * np.log(2.0 * np.pi) - np.sum(np.log(sigma)) - sum((res/sigma)**2) / 2.0

    @property
    def aic(self):
        self._rebuild()
        k = sum(self.parameters.fitted)
        LL = self.log_likelihood()
        return 2.0 * k - 2.0 * LL

    @property
    def aicc(self):
        aic = self.aic
        k = sum(self.parameters.fitted)
        return aic + (2.0 * k * k + 2.0 * k)/(self.n_residuals - k - 1.0)

    @property
    def bic(self):
        k = sum(self.parameters.fitted)
        return k * np.log(self.n_residuals) - 2.0 * self.log_likelihood()

    @property
    def cov(self):
        """
        Returns the inverse of the approximate Hessian. This approximation is valid when the residuals of the fitting
        problem are small.
        """
        J = self._calculate_jacobian()
        J = J / np.transpose(np.tile(self.sigma, (J.shape[1], 1)))
        return np.linalg.inv(np.transpose(J).dot(J))

    @property
    def asymptotic_ci(self, mode="chi2"):
        cov_est = self.cov()

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


class Data:
    def __init__(self, name, x, y, transformations):
        self.x = np.array(x)
        self.y = np.array(y)
        self.name = name
        self.transformations = transformations

    @property
    def independent(self):
        return self.x

    @property
    def dependent(self):
        return self.y

    @property
    def condition_string(self):
        return '|'.join(str(x) for x in self.transformations.values())

    @property
    def parameter_names(self):
        """
        Parameter names for free parameters after transformation
        """
        return [x for x in self.transformations.values() if isinstance(x, str)]

    @property
    def source_parameter_names(self):
        """
        Parameter names for free parameters after transformation
        """
        return [x for x, y in self.transformations.items() if isinstance(y, str)]


class Condition:
    def __init__(self, transformations, global_dictionary):
        self.transformations = deepcopy(transformations)

        # Which sensitivities actually need to be exported?
        self.p_external = np.flatnonzero([True if isinstance(x, str) else False for x in self.transformed])

        # p_global_indices contains a list with indices for each parameter that is mapped to the globals
        self.p_global_indices = np.array([global_dictionary.get(key, None) for key in self.transformed])

        # p_indices map internal sensitivities to the global parameters.
        # Note that they are part of the public interface.
        self.p_indices = [x for x in self.p_global_indices if x is not None]

        # Which sensitivities are local (set to a fixed local value)?
        self.p_local = np.array([None if isinstance(x, str) else x for x in self.transformed])

    @property
    def transformed(self):
        return self.transformations.values()

    def localize_sensitivities(self, sensitivities):
        return sensitivities[:, self.p_external]

    def get_local_parameters(self, par_global):
        return [par_global[a] if a is not None else b for a, b in zip(self.p_global_indices, self.p_local)]
