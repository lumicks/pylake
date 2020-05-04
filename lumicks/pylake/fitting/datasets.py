from .detail.link_functions import generate_conditions
from .detail.utilities import parse_transformation
from .fitdata import FitData
from copy import deepcopy
import numpy as np
from collections import OrderedDict


class Datasets:
    def __init__(self, model):
        """A collection of datasets for a model

        Parameters
        ----------
        model: Model
            The model these datasets are for.
        """
        self._model = model
        self._data = OrderedDict()
        self._conditions = []
        self._data_link = []
        self.built = False

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def __iter__(self):
        return self._data.__iter__()

    def _link_data(self, parameter_lookup):
        self._conditions, self._data_link = generate_conditions(self._data, parameter_lookup, self._model.parameter_names)
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
        for data in self._data.values():
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
            dna_model.add_data("Control", f1, d1)  # Load the first dataset like that
            dna_model.add_data("RecA", f2, d2, params={"DNA_Lc": "DNA_Lc_RecA"})  # Different contour length Lc

            dna_model = pylake.inverted_odijk("DNA")
            dna_model.add_data("Unusual", f1, d1, params={"DNA_St": "1200"})  # Set stretch modulus to 1200 pN
        """
        if name in self._data:
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
        self._data[name] = data

        return data

    @property
    def names(self):
        return [data.name for data in self._data.values()]

    @property
    def _transformed_parameters(self):
        """Retrieves the full list of fitted parameters post-transformation used by these data sets."""
        return [name for data in self._data.values() for name in data.parameter_names]

    @property
    def _defaults(self):
        if self._data:
            return [deepcopy(self._model.defaults[name]) for data in self._data.values() for name in
                    data.source_parameter_names]
        else:
            return [deepcopy(self._model.defaults[name]) for name in self._model.parameter_names]

    def _repr_html_(self):
        repr_text = ''
        for d in self._data.values():
            repr_text += f"&ensp;&ensp;{d.__str__()}<br>\n"

        return repr_text

    def __repr__(self):
        return (f"lumicks.pylake.{self.__class__.__name__}"
                f"(datasets={{{', '.join([x.name for x in self._data.values()])}}}, "
                f"N={self.n_residuals})")

    def __str__(self):
        repr_text = 'Data sets:\n'
        for d in self._data.values():
            repr_text += f"- {d.__repr__()}\n"

        return repr_text

class FdDatasets(Datasets):
    def add_data(self, name, f, d, params={}):
        """
        Adds a data set for this model.

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
            dna_model.add_data("Data1", force1, distance1)  # Load the first data set like that
            dna_model.add_data("Data2", force2, distance2, params={"DNA_Lc": "DNA_Lc_RecA"})  # Different DNA_Lc
            dna_model = pylake.inverted_odijk("DNA")
            dna_model.add_data("Data3", force3, distance3, params={"DNA_St": 1200})  # Set DNA_St to 1200
        """
        if self._model.independent == "f":
            return self._add_data(name, f, d, params)
        else:
            return self._add_data(name, d, f, params)

    @staticmethod
    def dataset(model):
        return FdDatasets(model)
