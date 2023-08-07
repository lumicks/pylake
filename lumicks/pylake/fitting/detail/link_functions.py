import numpy as np

from ..fitdata import Condition
from .utilities import unique_idx


def generate_conditions(data_sets, parameter_lookup, model_params):
    """
    This function builds a list of unique conditions from a list of data sets and a list of references pointing to the
    data that belongs to each condition.

    Parameters
    ----------
    data_sets : dict of FitData
        References to data
    parameter_lookup : OrderedDict[str, int]
        Lookup table for looking up parameter indices by name
    model_params : list of str
        Base model parameter names

    Returns
    -------
    conditions : list of Condition
        Unique simulation conditions
    data_link : list of list of FitData
        Link between conditions and datasets (used to look up which datasets belong to which
        condition).

    Raises
    ------
    RuntimeError
        If the parameters as found in the dataset are incompatible with the model parameters.
    RuntimeError
        If the parameter transformations contain transformed parameter names that are not in the
        combined parameter list.
    """
    # Quickly concatenate the parameter transformations corresponding to this condition
    str_conditions = []
    for data_set in data_sets.values():
        str_conditions.append(data_set.condition_string)

        if set(data_set.transformations.keys()) != set(model_params):
            raise RuntimeError(
                "Source parameters in the data parameter transformations of data_sets are "
                "incompatible with the specified model parameters in model_params."
            )

        target_params = [x for x in data_set.transformations.values() if isinstance(x, str)]
        if not set(target_params).issubset(parameter_lookup.keys()):
            raise RuntimeError(
                "Parameter transformations of data_sets contain transformed parameter names that "
                "are not in the combined parameter list parameter_lookup."
            )

    # Determine unique parameter conditions and the indices to get the appropriate unique condition
    # from data index.
    unique_condition_strings, indices = unique_idx(str_conditions)
    indices = np.asarray(indices)

    data_link = []
    keys = list(data_sets.keys())
    for condition_idx in np.arange(len(unique_condition_strings)):
        (data_indices,) = np.nonzero(np.equal(indices, condition_idx))
        data_link.append([data_sets[keys[x]] for x in data_indices])

    conditions = []
    for data_sets in data_link:
        transformations = data_sets[0].transformations
        conditions.append(Condition(transformations, parameter_lookup))

    return conditions, data_link
