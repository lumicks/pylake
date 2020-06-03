import numpy as np
from .parameters import Params
from collections import OrderedDict
import matplotlib.pyplot as plt


class FitData:
    """
    This class contains data to be fitted, and a set of transformations that correspond to this specific data set. The
    transformations are parameter mappings from the model this data is part of to the outer parameters (the ones that
    are going to be fitted in the global fit).

    Parameters
    ----------
    name : str
        name of this dataset
    x, y : array_like
        Actual data
    transformations : OrderedDict
        set of transformations from internal model parameters to outer parameters
    """
    def __init__(self, name, x, y, transformations):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.name = name
        self.transformations = transformations

    @property
    def independent(self):
        """Values for the independent variable"""
        return self.x

    @property
    def dependent(self):
        """Values for the dependent variable"""
        return self.y

    @property
    def condition_string(self):
        return '|'.join(str(x) for x in self.transformations.values())

    def get_params(self, params):
        """
        This function maps parameters from a global fit parameter vector into internal parameters for this model,
        which can be used to simulate this model.

        Parameters
        ----------
        params : Params
            Fit parameters, typically obtained from a Fit.
        """
        mapping = OrderedDict((key, params[x]) if isinstance(x, str) else (key, float(x)) for
                              key, x in self.transformations.items())
        return Params(**mapping)

    @property
    def parameter_names(self):
        """
        Parameter names for free parameters after transformation
        """
        return [x for x in self.transformations.values() if isinstance(x, str)]

    @property
    def source_parameter_names(self):
        """
        Parameter names for free parameters before transformation
        """
        return [x for x, y in self.transformations.items() if isinstance(y, str)]

    def plot(self, fmt, **kwargs):
        return plt.plot(self.x, self.y, fmt, **kwargs)

    def __repr__(self):
        out_string = f'{self.__class__.__name__}({self.name}, N={len(self.independent)}'

        changes = []
        for i, v in self.transformations.items():
            if i != v:
                changes.append([i, str(v)])

        if len(changes) > 0:
            out_string += ', Transformations: ' + ', '.join([' → '.join(s) for s in changes])

        return out_string + ')'


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
        self._p_global_indices = [global_dictionary.get(key, None) for key in self.transformed]

        # p_indices map internal sensitivities to the global parameters. Note that they are part of the "public"
        # interface. Basically, it is the indices of the exported model variables in the global parameter vector.
        self.p_indices = [x for x in self._p_global_indices if x is not None]

        # Which sensitivities are local (set to a fixed local value)?
        self.p_local = [None if isinstance(x, str) else x for x in self.transformed]

    @property
    def transformed(self):
        return self.transformations.values()

    def localize_sensitivities(self, sensitivities):
        """Convert raw model sensitivities to external sensitivities as used by the Fit."""
        return sensitivities[:, self.p_external]

    def get_local_params(self, par_global):
        """Grab parameters required to simulate the model from the global parameter vector. Merge in the local
        parameters as well."""
        return [par_global[a] if a is not None else b for a, b in zip(self._p_global_indices, self.p_local)]
