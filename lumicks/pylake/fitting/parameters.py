import numpy as np
from collections import OrderedDict
from tabulate import tabulate


class Parameter:
    __slots__ = [
        "value",
        "lower_bound",
        "upper_bound",
        "fixed",
        "shared",
        "unit",
        "profile",
        "stderr",
    ]

    def __init__(
        self,
        value=0.0,
        lower_bound=-np.inf,
        upper_bound=np.inf,
        fixed=False,
        shared=False,
        unit=None,
    ):
        """Model parameter

        Parameters
        ----------
        value : float
            Parameter value
        lower_bound, upper_bound : float
            Lower and upper bound used in the fitting process. Parameters are not allowed to go beyond these bounds.
        fixed : bool
            Is this parameter fixed (not estimated from data)?
        shared : bool
            Is this parameter typically model specific or shared between models? An example of a model specific
            parameter is the contour length of a protein, whereas the Boltzmann constant times temperate (kT) is an
            example of a parameter that is typically shared between models.
        unit : str
            Unit of the parameter
        """
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fixed = fixed
        self.shared = shared
        self.unit = unit
        self.profile = None
        self.stderr = None

    def __float__(self):
        return float(self.value)

    def __eq__(self, other):
        return (
            all((getattr(self, x) == getattr(other, x) for x in self.__slots__))
            if isinstance(other, self.__class__)
            else False
        )

    def __repr__(self):
        return (
            f"lumicks.pylake.fdfit.Parameter(value: {self.value}, lower bound: {self.lower_bound}, upper bound: "
            f"{self.upper_bound}, fixed: {self.fixed})"
        )

    def __str__(self):
        return self.__repr__()

    def ci(self, percentile=0.95, dof=1):
        """Calculate confidence intervals

        Parameters
        ----------
        percentile : float
            1 - Significance level
        dof : float
            Degrees of freedom (should be 1 for single parameter CIs)."""
        from scipy import stats

        if self.stderr:
            dp = self.stderr * np.sqrt(stats.chi2.ppf(percentile, dof))
            return [self.value - dp, self.value + dp]
        else:
            raise RuntimeError("These parameters are not associated with a fitted model.")


class Params:
    """
    Model parameters. Internally stored as a list of Parameter.

    Examples
    --------
    ::
        fit = pylake.FdFit(pylake.odijk("my_model"))

        print(fit.params)  # Prints the model parameters
        fit["test_parameter"].value = 5  # Set parameter test_parameter to 5
        fit["fix_me"].fixed = True  # Fix parameter fix_me (do not fit)

        # Copy parameters from another Parameters into this one.
        parameters.update_params(other_parameters)

        # Copy the parameters from an earlier fit into the combined model.
        fit_combined_model.update_params(fit)
    """

    def __init__(self, **kwargs):
        self._src = OrderedDict()
        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self._src[key] = value
            else:
                self._src[key] = Parameter(float(value)) if value else Parameter(0)

    def __iter__(self):
        return self._src.__iter__()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

    def update_params(self, other):
        """
        Sets parameters if they are found in the target parameter list.

        Parameters
        ----------
        other : Params
        """
        if isinstance(other, Params):
            found = False
            for key, param in other._src.items():
                if key in self._src:
                    self._src[key] = param
                    found = True

            if not found:
                raise RuntimeError(
                    "Tried to set parameters which do not exist in the target model."
                )
        else:
            raise RuntimeError("Attempt to stream non-parameter list to parameter list.")

    def items(self):
        return self._src.items()

    def __getitem__(self, item):
        from .fitdata import FitData

        if isinstance(item, slice):
            raise IndexError("Slicing not supported. Only indexing.")

        if isinstance(item, FitData):
            return item.get_params(self)

        if item in self._src:
            return self._src[item]
        else:
            raise IndexError(f"Parameter {item} does not exist.")

    def __setitem__(self, item, value):
        if item in self._src:
            self._src[item].value = value
        else:
            raise IndexError(f"Parameter {item} not found in parameter vector {self.keys}!")

    def __len__(self):
        return len(self._src)

    def _print_data(self):
        table = [
            [
                key,
                par.value,
                f"[{par.unit}]" if par.unit else "NA",
                not par.fixed,
                par.lower_bound,
                par.upper_bound,
            ]
            for key, par in self._src.items()
        ]
        header = ["Name", "Value", "Unit", "Fitted", "Lower bound", "Upper bound"]
        return table, header

    def _repr_html_(self):
        table, header = self._print_data()
        return tabulate(table, header, tablefmt="html")

    def _repr_(self):
        table, header = self._print_data()
        return tabulate(table, header, tablefmt="text")

    def __str__(self):
        if len(self._src) > 0:
            table, header = self._print_data()
            return tabulate(table, header)
        else:
            return "No parameters"

    def _set_params(self, params, defaults):
        """Rebuild the parameter vector. Note that this can potentially alter the parameter order if the strings are
        given in a different order.

        It mutates the parameter vector to contain the elements as specified in "parameters" with the defaults as
        specified in defaults. If the parameter already exists in the vector nothing happens to it. If it doesn't,
        it gets initialized to its default.

        Parameters
        ----------
        params : list of str
            parameter names
        defaults : Parameter or None
            default parameter objects
        """
        new_params = OrderedDict(
            zip(params, [x if isinstance(x, Parameter) else Parameter() for x in defaults])
        )
        for key, value in self._src.items():
            if key in new_params:
                new_params[key] = value

        self._src = new_params

    def keys(self):
        return np.asarray([key for key in self._src.keys()])

    @property
    def values(self):
        return np.asarray([param.value for param in self._src.values()], dtype=np.float64)

    @property
    def fitted(self):
        return np.asarray([not param.fixed for param in self._src.values()], dtype=bool)

    @property
    def lower_bounds(self):
        return np.asarray([param.lower_bound for param in self._src.values()], dtype=np.float64)

    @property
    def upper_bounds(self):
        return np.asarray([param.upper_bound for param in self._src.values()], dtype=np.float64)
