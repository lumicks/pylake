import numpy as np
from .parameters import Parameters
from collections import OrderedDict


class FitData:
    """
    This class contains data to be fitted, and a set of transformations that correspond to this specific dataset. The
    transformations are parameter mappings from the model this data is part of to the outer parameters (the ones that
    are going to be fitted in the global fit).

    Parameters
    ----------
    name: str
        name of this dataset
    x, y: array_like
        Actual data
    transformations: OrderedDict
        set of transformations from internal model parameters to outer parameters
    """
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

    def get_parameters(self, parameters):
        """
        This function maps parameters from a global fit parameter vector into internal parameters for this model,
        which can be used to simulate this model.

        Parameters
        ----------
        parameters: Parameters
            Fit parameters, typically obtained from a FitObject.
        """
        mapping = OrderedDict((key, parameters[x]) if isinstance(x, str) else (key, float(x)) for
                              key, x in self.transformations.items())
        return Parameters(**mapping)

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
    """
    This class maintains the linkage between the index-based parameter vectors and matrices for a data-set. It is not
    intended to be a user facing class as working with raw indices is error prone.
    """
    def __init__(self, transformations, global_dictionary):
        from copy import deepcopy
        self.transformations = deepcopy(transformations)

        # Which sensitivities actually need to be exported?
        self.p_external = np.flatnonzero([True if isinstance(x, str) else False for x in self.transformed])

        # p_global_indices contains a list with indices for each parameter that is mapped to the globals
        self.p_global_indices = np.array([global_dictionary.get(key, None) for key in self.transformed])

        # p_indices map internal sensitivities to the global parameters.
        # Note that they are part of the "public" interface.
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
