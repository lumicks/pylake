from collections import OrderedDict


def unique_idx(input_list):
    """
    Determine unique elements of a list and return indices which reconstruct the original list from the unique elements.
    """
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    inverse_list = [unique_list.index(l) for l in input_list]
    return unique_list, inverse_list


def parse_transformation(parameters, **kwargs):
    transformed = OrderedDict(zip(parameters, parameters))

    for key, value in kwargs.items():
        if key in transformed:
            transformed[key] = value
        else:
            raise KeyError(f"Parameter {key} to be substituted not found in model. Valid keys for this model are: "
                           f"{[x for x in transformed.keys()]}.")

    return transformed
