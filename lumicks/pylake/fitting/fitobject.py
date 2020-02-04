from .parameters import Parameters
from ..detail.utilities import unique
from collections import OrderedDict
import numpy as np


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
