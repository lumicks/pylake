from .detail.link_functions import generate_conditions
from .detail.utilities import parse_transformation
from .fitdata import FitData
from copy import deepcopy
from collections import OrderedDict
from ..detail.utilities import get_color
import numpy as np
import matplotlib.pyplot as plt


class Datasets:
    def __init__(self, model, fit):
        """A collection of datasets to be fitted for a model.

        Parameters
        ----------
        model: Model
            The model these datasets are for.
        fit: Fit
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

    def __iter__(self):
        return self.data.__iter__()

    def _link_data(self, parameter_lookup):
        self._conditions, self._data_link = generate_conditions(self.data, parameter_lookup, self._model.parameter_names)
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

    def _add_data(self, name, x, y, params={}):
        """
        Loads a data set.

        Parameters
        ----------
        name: str
            Name of this data set.
        x: array_like
            Independent variable. NaNs are silently dropped.
        y: array_like
            Dependent variable. NaNs are silently dropped.
        params: dict of {str : str or int}
            List of parameter transformations. These can be used to convert one parameter in the model, to a new
            parameter name or constant for this specific data set (for more information, see the examples).

        Examples
        --------
        ::
            dna_model = pylake.inverted_odijk("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.FdFit(dna_model)

            fit.add_data("Control", f1, d1)  # Load the first dataset like that
            fit.add_data("RecA", f2, d2, params={"DNA/Lc": "DNA/Lc_RecA"})  # Different contour length Lc
        """
        if name in self.data:
            raise KeyError("The name of the data set must be unique.")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        assert x.ndim == 1, "Independent variable should be one dimension"
        assert y.ndim == 1, "Dependent variable should be one dimension"
        assert len(x) == len(y), "Every value for the independent variable should have a corresponding data point"

        filter_nan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
        y = y[filter_nan]
        x = x[filter_nan]

        self.built = False
        parameter_list = parse_transformation(self._model.parameter_names, params)
        data = FitData(name, x, y, parameter_list)
        self.data[name] = data

        return data

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
            return [deepcopy(self._model.defaults[name]) for data in self.data.values() for name in
                    data.source_parameter_names]
        else:
            return [deepcopy(self._model.defaults[name]) for name in self._model.parameter_names]

    def _repr_html_(self):
        repr_text = ''
        for d in self.data.values():
            repr_text += f"&ensp;&ensp;{d.__str__()}<br>\n"

        return repr_text

    def __repr__(self):
        return (f"lumicks.pylake.{self.__class__.__name__}"
                f"(datasets={{{', '.join([x.name for x in self.data.values()])}}}, "
                f"N={self.n_residuals})")

    def __str__(self):
        repr_text = 'Data sets:\n'
        for d in self.data.values():
            repr_text += f"- {d.__repr__()}\n"

        return repr_text

    def _plot_data(self, fmt='', **kwargs):
        names = []
        handles = []

        if len(fmt) == 0:
            kwargs["marker"] = kwargs.get("marker", '.')
            kwargs["markersize"] = kwargs.get("markersize", .5)
            set_color = kwargs.get("color")
        else:
            set_color = 1

        for i, data in enumerate(self.data.values()):
            if not set_color:
                kwargs["color"] = get_color(i)
            handle, = plt.plot(data.x, data.y, fmt, **kwargs)
            handles.append(handle)
            names.append(data.name)

        plt.legend(handles, names)


class FdDatasets(Datasets):
    def add_data(self, name, f, d, params={}):
        """
        Adds a data set to this fit.

        Parameters
        ----------
        name: str
            Name of this data set.
        f: array_like
            An array_like containing force data.
        d: array_like
            An array_like containing distance data.
        params: dict of {str : str or int}
            List of parameter transformations. These can be used to convert one parameter in the model, to a new
            parameter name or constant for this specific data set (for more information, see the examples).
        Examples
        --------
        ::
            dna_model = pylake.inverted_odijk("DNA")  # Use an inverted Odijk eWLC model.
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
