import inspect
import numpy as np
from .detail.utilities import unique, unique_idx
from collections import OrderedDict
from copy import deepcopy
import scipy.optimize as optim
from itertools import chain


class Model:
    def __init__(self, model_function, jacobian=None):
        self.model_function = model_function
        self._parameter_names = inspect.getfullargspec(model_function).args[1:]
        if jacobian:
            self._jacobian = jacobian

    @staticmethod
    def _sanitize_input_types(data, parameter_vector):
        data = np.array(data).astype(float)
        parameter_vector = np.array(parameter_vector).astype(float)
        return data, parameter_vector

    def __call__(self, data, parameter_vector):
        data, parameter_vector = self._sanitize_input_types(data, parameter_vector)
        return self.model_function(data, *parameter_vector)

    def jacobian(self, data, parameter_vector):
        data, parameter_vector = self._sanitize_input_types(data, parameter_vector)
        return self._jacobian(data, *parameter_vector)

    @property
    def has_jacobian(self):
        return hasattr(self, "_jacobian")

    def numerical_jacobian(self, data, parameter_vector, dx=1e-6):
        data, parameter_vector = self._sanitize_input_types(data, parameter_vector)

        reference_evaluation = self(data, parameter_vector)
        finite_difference_jacobian = np.zeros((len(parameter_vector), len(data)))
        for i in np.arange(len(parameter_vector)):
            parameters = np.copy(parameter_vector)
            parameters[i] = parameters[i] + dx
            finite_difference_jacobian[i, :] = (self(data, parameters) - reference_evaluation) / dx

        return finite_difference_jacobian

    def verify_jacobian(self, data, parameters):
        jacobian = self.jacobian(data, parameters)
        numerical_jacobian = self.numerical_jacobian(data, parameters)

        is_close = np.allclose(jacobian, numerical_jacobian)
        if not is_close:
            maxima = np.max(jacobian - numerical_jacobian, axis=1)
            for i, v in enumerate(maxima):
                print(f"Parameter {self._parameter_names[i]}({i}): {v}")

            raise RuntimeError('Numerical Jacobian did not pass.')

        return is_close

    @property
    def parameter_names(self):
        return np.copy(self._parameter_names)


class Parameter:
    def __init__(self, value=0.0, lb=-np.inf, ub=np.inf):
        self.value = value
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"lumicks.pylake.fdfit.Parameter(value: {self.value}, lb: {self.lb}, ub: {self.ub})"

    def __str__(self):
        return self.__repr__()


class Parameters:
    def __init__(self):
        self._src = OrderedDict()

    def __iter__(self):
        return self._src.__iter__()

    def items(self):
        return self._src.items()

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise IndexError("Slicing not supported. Only indexing.")

        if item in self._src:
            return self._src[item]

    def __setitem__(self, item, value):
        if item in self._src:
            self._src[item].value = value

    def __len__(self):
        return len(self._src)

    def __str__(self):
        return_str = ""
        for key, param in self._src.items():
            return_str = return_str + f"{key}      {param.value}\n"

        return return_str

    def set_parameters(self, parameters):
        """Rebuild the parameter vector. Note that this can potentially alter the parameter order if the strings are given
        in a different order.

        Parameters
        ----------
        parameters : list of str
            parameter names
        """
        new_parameters = OrderedDict(zip(parameters, [Parameter() for x in range(len(parameters))]))
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
    def __init__(self, model):
        self.model = model
        self._data = []
        self._conditions = []
        self._data_link = None
        self._parameters = Parameters()
        self._current_new_idx = 0

        self._invalidate_build()

    @staticmethod
    def parse_transformation(parameters, **kwargs):
        transformed = OrderedDict(zip(parameters, parameters))

        for key, value in kwargs.items():
            if key in transformed:
                transformed[key] = value
            else:
                raise KeyError(f"Parameter {key} to be substituted not found in model.")

        return transformed

    def load_data(self, x, y, **kwargs):
        self._invalidate_build()

        parameter_list = FitObject.parse_transformation(self.model.parameter_names, **kwargs)
        self._data.append(Data(x, y, parameter_list))

    @staticmethod
    def _build_conditions(data_sets, parameter_lookup):
        """
        This function builds a list of unique conditions from a list of data sets and a list of index lists which link
        back the individual data fields to their simulation conditions.

        Parameters
        ----------
        data_sets : list of Data
            References to data
        parameter_lookup: OrderedDict[str, int]
            Lookup table for looking up parameter indices by name
        """
        # Quickly concatenate the parameter transformations corresponding to this condition
        str_conditions = []
        for data_set in data_sets:
            str_conditions.append(data_set.condition_string)

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

    def _rebuild_structure(self):
        """This function generates the global parameter list from the parameters of the individual submodels.
        It also generates unique conditions from the data specification."""
        parameter_names = [name for data in self._data for name in data.parameter_names]
        parameter_names = unique(parameter_names)
        parameter_lookup = OrderedDict(zip(parameter_names, np.arange(len(parameter_names))))

        self._conditions, self._data_link = self._build_conditions(self._data, parameter_lookup)
        self._parameters.set_parameters(parameter_names)

        self._built = True

    def _check_rebuild(self):
        """
        Checks whether the model state is up to date. Any user facing methods should ideally check whether the model
        needs to be rebuilt.
        """
        if not self._built:
            self._rebuild_structure()

    def _invalidate_build(self):
        self._built = False

    @property
    def n_residuals(self):
        self._check_rebuild()
        count = 0
        for data in self._data:
            count += len(data.independent)

        return count

    @property
    def parameters(self):
        self._check_rebuild()
        return self._parameters

    @property
    def n_parameters(self):
        self._check_rebuild()
        return len(self._parameters)

    def fit(self, **kwargs):
        parameter_vector = self.parameters.values
        lb = self.parameters.lb
        ub = self.parameters.ub

        result = optim.least_squares(self._evaluate_model, parameter_vector,
                                     jac=self._evaluate_jacobian if self.model.has_jacobian else "2-point",
                                     bounds=(lb, ub), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-10, **kwargs)

        parameter_names = self.parameters.keys
        parameter_vector = result.x

        for name, value in zip(parameter_names, parameter_vector):
            self.parameters[name] = value

    def _evaluate_model(self, parameter_values):
        self._check_rebuild()
        residual_idx = 0
        residual = np.zeros(self.n_residuals)
        for condition, data_sets in zip(self._conditions, self._data_link):
            p_local = condition.get_local_parameters(parameter_values)
            for data in data_sets:
                 data_set = self._data[data]
                 y_model = self.model(data_set.x, p_local)

                 residual[residual_idx:residual_idx + len(y_model)] = data_set.y - y_model
                 residual_idx += len(y_model)

        return residual

    def _evaluate_jacobian(self, parameter_values):
        self._check_rebuild()
        residual_idx = 0
        jacobian = np.zeros((self.n_residuals, self.n_parameters))
        for condition, data_sets in zip(self._conditions, self._data_link):
            p_local = condition.get_local_parameters(parameter_values)
            p_indices = condition.p_indices
            for data in data_sets:
                data_set = self._data[data]
                sensitivities = np.transpose(self.model.jacobian(data_set.x, p_local))
                n_res = sensitivities.shape[0]

                jacobian[residual_idx:residual_idx + n_res, p_indices] = \
                    jacobian[residual_idx:residual_idx + n_res, p_indices] - sensitivities

                residual_idx += n_res

        return jacobian


class Data:
    def __init__(self, x, y, transformations):
        self.x = x
        self.y = y
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
        return [x for x in self.transformations.values() if isinstance(x, str)]


class Condition:
    def __init__(self, transformations, global_dictionary):
        self.transformations = deepcopy(transformations)
        self.p_external = np.array([True if isinstance(x, str) else False for x in self.transformed])
        self.p_local = np.array([0.0 if isinstance(x, str) else x for x in self.transformed])
        self.p_reference = [x for x in self.transformed if isinstance(x, str)]
        self.p_indices = [global_dictionary[key] for key in self.p_reference]

    @property
    def transformed(self):
        return self.transformations.values()

    def get_local_parameters(self, par_global):
        self.p_local[self.p_external] = par_global[self.p_indices]
        return self.p_local
