import numpy as np
from collections import OrderedDict


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
        self.profile = None
        if init:
            self.init = init
        else:
            self.init = self.value

    def __repr__(self):
        return f"lumicks.pylake.fdfit.Parameter(value: {self.value}, lb: {self.lb}, ub: {self.ub}, vary: {self.vary})"

    def __str__(self):
        return self.__repr__()


class Parameters:
    """
    Model parameters.
    """
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
            return_str = return_str + ("{:"+f"{maxlen+1}"+"s} {:+1.4e} {:1d} [{:+1.4e}, {:+1.4e}]\n").format(key, param.value, param.vary, param.lb, param.ub)

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