from copy import deepcopy
from collections import OrderedDict

import numpy as np

from .fitdata import FitData
from .detail.utilities import parse_transformation
from .detail.link_functions import generate_conditions


class Datasets:
    def __init__(self, model, fit):
        """A collection of datasets to be fitted for a model.

        Parameters
        ----------
        model : Model
            The model these datasets are for.
        fit : Fit
            Fit this dataset is associated with.
        """
        self._model = model
        self._fit = fit
        self.data = OrderedDict()
        self._conditions = []
        self._data_link = []
        self.built = False

    def __getitem__(self, item):
        return self._fit.params[self.data.__getitem__(item)]

    def _link_data(self, parameter_lookup):
        self._conditions, self._data_link = generate_conditions(
            self.data, parameter_lookup, self._model.parameter_names
        )
        self.built = True

    def conditions(self):
        # We need to return a list of conditions; and a list which links which data is linked to which conditions
        condition_iterator = self._conditions.__iter__()
        data_link = self._data_link.__iter__()

        while True:
            try:
                yield next(condition_iterator), next(data_link)
            except StopIteration:
                return

    @property
    def n_residuals(self):
        """Number of data points loaded into this model."""
        count = 0
        for data in self.data.values():
            count += len(data.independent)

        return count

    def _add_data(self, name, x, y, params=None):
        """Loads a data set.

        Parameters
        ----------
        name : str
            Name of this data set.
        x : array_like
            Independent variable. NaNs are silently dropped.
        y : array_like
            Dependent variable. NaNs are silently dropped.
        params : Optional[dict of {str : str or int}]
            List of parameter transformations. These can be used to convert one parameter in the
            model, to a new parameter name or constant for this specific data set (for more
            information, see the examples).

        Returns
        -------
        dataset : FitData
            Handle to added dataset.

        Raises
        ------
        KeyError
            If the `name` of the dataset is not unique.
        ValueError
            If the independent and dependent variables are not one dimensional or are of unequal
            length.

        Examples
        --------
        ::
            dna_model = pylake.ewlc_odijk_force("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.FdFit(dna_model)

            fit.add_data("Control", f1, d1)  # Load the first dataset like that
            fit.add_data("RecA", f2, d2, params={"DNA/Lc": "DNA/Lc_RecA"})  # Different contour length Lc
        """
        if name in self.data:
            raise KeyError("The name of the data set must be unique.")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(
                f"Independent variable x should be one dimensional, but has shape {x.shape}."
            )
        if y.ndim != 1:
            raise ValueError(
                f"Dependent variable y should be one dimensional, but has shape {y.shape}."
            )
        if len(x) != len(y):
            raise ValueError(
                "Every value for the independent variable x should have a corresponding data point "
                "for the dependent variable y."
            )

        filter_nan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
        y = y[filter_nan]
        x = x[filter_nan]

        self.built = False
        parameter_list = parse_transformation(self._model.parameter_names, params)
        data = FitData(name, x, y, parameter_list)
        self.data[name] = data

        return data

    def plot(
        self,
        data=None,
        fmt="",
        independent=None,
        legend=True,
        plot_data=True,
        overrides=None,
        **kwargs,
    ):
        """Plot model and data

        Parameters
        ----------
        data : str
            Name of the data set to plot (optional, omission plots all for that model).
        fmt : str
            Format string, forwarded to :func:`matplotlib.pyplot.plot`.
        independent : array_like
            Array with values for the independent variable (used when plotting the model).
        legend : bool
            Show legend (default: True).
        plot_data : bool
            Show data (default: True).
        overrides : dict
            Parameter / value pairs which override parameter values in the current fit. Should be a dict of
            {str: float} that provides values for parameters which should be set to particular values in the plot
            (default: None);
        ``**kwargs``
            Forwarded to :func:`matplotlib.pyplot.plot`.

        Examples
        --------
        ::
            from lumicks import pylake

            model = pylake.ewlc_odijk_force("DNA")
            fit = pylake.FdFit(model)
            fit.add_data("Control", force, distance)
            fit.fit()

            # Basic plotting of one data set over a custom range can be done by just invoking plot.
            fit.plot("Control", 'k--', np.arange(2.0, 5.0, 0.01))

            # Have a quick look at what a stiffness of 5 would do to the fit.
            fit.plot("Control", overrides={"DNA/St": 5})

            # When dealing with multiple models in one fit, one has to select the model first when we want to plot.
            model1 = pylake.ewlc_odijk_distance("DNA")
            model2 = pylake.ewlc_odijk_distance("DNA") + pylake.ewlc_odijk_distance("protein")
            fit[model1].add_data("Control", force1, distance2)
            fit[model2].add_data("Control", force1, distance2)
            fit.fit()

            fit = pylake.FdFit(model1, model2)
            fit[model1].plot("Control")  # Plots data set Control for model 1
            fit[model2].plot("Control")  # Plots data set Control for model 2
        """
        self._fit._plot(self._model, data, fmt, overrides, independent, legend, plot_data, **kwargs)

    @property
    def names(self):
        return [data.name for data in self.data.values()]

    @property
    def _transformed_params(self):
        """Retrieves the full list of fitted parameters post-transformation used by these data sets."""
        return [name for data in self.data.values() for name in data.parameter_names]

    @property
    def _defaults(self):
        if self.data:
            return [
                deepcopy(self._model.defaults[name])
                for data in self.data.values()
                for name in data.source_parameter_names
            ]
        else:
            return [deepcopy(self._model.defaults[name]) for name in self._model.parameter_names]

    def _repr_html_(self):
        repr_text = ""
        for d in self.data.values():
            repr_text += f"&ensp;&ensp;{d.__str__()}<br>\n"

        return repr_text

    def __repr__(self):
        return (
            f"lumicks.pylake.{self.__class__.__name__}"
            f"(datasets={{{', '.join([x.name for x in self.data.values()])}}}, "
            f"N={self.n_residuals})"
        )

    def __str__(self):
        repr_text = "Data sets:\n"
        for d in self.data.values():
            repr_text += f"- {d.__repr__()}\n"

        return repr_text


class FdDatasets(Datasets):
    def add_data(self, name, f, d, params=None):
        """Adds a data set to this fit.

        Parameters
        ----------
        name : str
            Name of this data set.
        f : array_like
            An array_like containing force data.
        d : array_like
            An array_like containing distance data.
        params : Optional[dict of {str : str or int}]
            List of parameter transformations. These can be used to convert one parameter in the model, to a new
            parameter name or constant for this specific data set (for more information, see the examples).

        Returns
        -------
        dataset : FitData
            Handle to added dataset.

        Raises
        ------
        KeyError
            If the `name` of the dataset is not unique.
        ValueError
            If the independent and dependent variables are not one dimensional or are of unequal
            length.

        Examples
        --------
        ::

            dna_model = pylake.ewlc_odijk_force("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.FdFit(dna_model)

            fit.add_data("Data1", force1, distance1)  # Load the first data set like that
            fit.add_data("Data2", force2, distance2, params={"DNA/Lc": "DNA/Lc_RecA"})  # Different DNA/Lc
        """
        if self._model.independent == "f":
            return self._add_data(name, f, d, params)
        else:
            return self._add_data(name, d, f, params)

    @staticmethod
    def dataset(model):
        return FdDatasets(model)
