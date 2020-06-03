from ..fitdata import Condition
from .utilities import unique_idx
import numpy as np


def generate_conditions(data_sets, parameter_lookup, model_params):
    """
    This function builds a list of unique conditions from a list of data sets and a list of references pointing to the
    data that belongs to each condition.

    Parameters
    ----------
    data_sets : list of Data
        References to data
    parameter_lookup : OrderedDict[str, int]
        Lookup table for looking up parameter indices by name
    model_params : list of str
        Base model parameter names
    """
    # Quickly concatenate the parameter transformations corresponding to this condition
    str_conditions = []
    for data_set in data_sets.values():
        str_conditions.append(data_set.condition_string)

        assert set(data_set.transformations.keys()) == set(model_params), \
            "Source parameters in data parameter transformations are incompatible with the specified model parameters."

        target_params = [x for x in data_set.transformations.values() if isinstance(x, str)]
        assert set(target_params).issubset(parameter_lookup.keys()), \
            "Parameter transformations contain transformed parameter names that are not in the combined parameter list."

    # Determine unique parameter conditions and the indices to get the appropriate unique condition from data index.
    unique_condition_strings, indices = unique_idx(str_conditions)
    indices = np.asarray(indices)

    data_link = []
    keys = list(data_sets.keys())
    for condition_idx in np.arange(len(unique_condition_strings)):
        data_indices, = np.nonzero(np.equal(indices, condition_idx))
        data_link.append([data_sets[keys[x]] for x in data_indices])

    conditions = []
    for data_sets in data_link:
        transformations = data_sets[0].transformations
        conditions.append(Condition(transformations, parameter_lookup))

    return conditions, data_link
