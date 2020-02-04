from .fitdata import FitData
from .parameters import Parameter
from .detail.utilities import parse_transformation
from .detail.link_functions import generate_conditions

from collections import OrderedDict
from copy import deepcopy
import inspect, types
import numpy as np


class Model:
    def __init__(self, name, model_function, jacobian=None, derivative=None, **kwargs):
        """
        Model constructor. A Model must be named, and this name will appear in the model parameters.
        A model contains references to data associated with the model by using the member function load_data.

        Prior to fitting the model will automatically generate a list of unique conditions (defined as conditions
        characterized by a unique set of conditions).

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
        jacobian: callable (optional)
            Function which computes the first order derivatives with respect to the parameters for this model.
            When supplied, this function is used to speed up the optimization considerably.
        derivative: callable (optional)
            Function which computes the first order derivative with respect to the independent parameter. When supplied
            this speeds up model inversions considerably.
        **kwargs
            Key pairs containing parameter defaults. For instance, Lc=Parameter(...)
        """
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
        self._data = []  # Stores the data sets with their relevant parameter transformations
        self._conditions = []  # Built from self._data and stores unique conditions and parameter maps

        # Since models are in principle exposed to a user by reference, one model can be bound to multiple FitObjects.
        # Prior to any fitting operation, we have to check whether the parameter mappings in the Conditions for this
        # model actually correspond to the one we have in FitObject. If not, we have to trigger a rebuild.
        self._built = None

    def __call__(self, independent, parameters):
        """Evaluate the model for specific parameters

        Parameters
        ----------
        independent: array_like
        parameters: Parameters
        """
        independent = np.array(independent).astype(float)
        return self._raw_call(independent, np.array([parameters[name].value for name in self.parameter_names]))

    def _raw_call(self, independent, parameter_vector):
        return self.model_function(independent, *parameter_vector)

    @property
    def _defaults(self):
        if self._data:
            return [deepcopy(self._parameters[name]) for data in self._data for name in data.source_parameter_names]
        else:
            return [deepcopy(self._parameters[name]) for name in self.parameter_names]

    @property
    def parameter_names(self):
        return [x for x in self._parameters.keys()]

    @property
    def _transformed_parameters(self):
        """Retrieves the full list of fitted parameters and defaults post-transformation used by this model. Includes
        parameters for all the data-sets in the model."""
        return [name for data in self._data for name in data.parameter_names]

    def jacobian(self, independent, parameter_vector):
        if self.has_jacobian:
            independent = np.array(independent).astype(float)
            return self._jacobian(independent, *parameter_vector)
        else:
            raise RuntimeError(f"Jacobian was requested but not supplied in model {self.name}.")

    def derivative(self, independent, parameter_vector):
        if self.has_derivative:
            independent = np.array(independent).astype(float)
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

    def _invalidate_build(self):
        self._built = False

    def built_against(self, fit_object):
        return self._built == fit_object

    def load_data(self, x, y, name="", **kwargs):
        """
        Loads a data set for this model.

        Parameters
        ----------
        x: array_like
            Independent variable.
        y: array_like
            Dependent variable.
        name: str
            Name of this data set.
        **kwargs:
            List of parameter transformations. These can be used to convert one parameter in the model, to a new
            parameter name or constant for this specific dataset (for more information, see the examples).

        Examples
        --------
        ::
            dna_model = pylake.force_model("DNA", "invWLC")  # Use an inverted Odijk eWLC model.
            dna_model.load_data(x1, y1, name="my first data set")  # Load the first dataset like that
            dna_model.load_data(x2, y2, name="my first data set", DNA_Lc="DNA_Lc_RecA")  # Different contour length Lc

            dna_model = pylake.force_model("DNA", "invWLC")
            dna_model.load_data(x1, y1, name="my second data set", DNA_St=1200)  # Set stretch modulus to 1200 pN
        """
        self._invalidate_build()
        parameter_list = parse_transformation(self.parameter_names, **kwargs)
        data = FitData(name, x, y, parameter_list)
        self._data.append(data)
        return data

    def _build_model(self, parameter_lookup, fit_object):
        self._conditions, self._data_link = generate_conditions(self._data, parameter_lookup,
                                                                self.parameter_names)

        self._built = fit_object
