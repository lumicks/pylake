from ..fitdata import Condition
from .utilities import unique_idx
import numpy as np


def generate_conditions(data_sets, parameter_lookup, model_parameters):
    """
    This function builds a list of unique conditions from a list of data sets and a list of index lists which link back
    the individual data fields to their simulation conditions.

    Parameters
    ----------
    data_sets : list of Data
        References to data
    parameter_lookup: OrderedDict[str, int]
        Lookup table for looking up parameter indices by name
    model_parameters: list of str
        Base model parameter names
    """
    # Quickly concatenate the parameter transformations corresponding to this condition
    str_conditions = []
    for data_set in data_sets:
        str_conditions.append(data_set.condition_string)

        assert set(data_set.transformations.keys()) == set(model_parameters), \
            "Source parameters in data parameter transformations are incompatible with the specified model parameters."

        target_parameters = [x for x in data_set.transformations.values() if isinstance(x, str)]
        assert set(target_parameters).issubset(parameter_lookup.keys()), \
            "Parameter transformations contain transformed parameter names that are not in the combined parameter list."

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
