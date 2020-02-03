import numpy as np
from collections import OrderedDict


def parse_transformation(parameters, **kwargs):
    transformed = OrderedDict(zip(parameters, parameters))

    for key, value in kwargs.items():
        if key in transformed:
            transformed[key] = value
        else:
            raise KeyError(f"Parameter {key} to be substituted not found in model. Valid keys for this model are: "
                           f"{[x for x in transformed.keys()]}.")

    return transformed


def unique_idx(input_list):
    """
    Determine unique elements of a list and return indices which reconstruct the original list from the unique elements.
    """
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    inverse_list = [unique_list.index(l) for l in input_list]
    return unique_list, inverse_list


def optimal_plot_layout(n_plots):
    n_x = np.ceil(np.sqrt(n_plots))
    n_y = np.ceil(n_plots/n_x)

    return n_x, n_y


def print_styled(style, print_string, **kwargs):
    print_dict = {
        'header': '\033[95m',
        'ok_blue': '\033[94m',
        'ok_green': '\033[92m',
        'warning': '\033[93m',
        'fail': '\033[91m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    if style in print_dict:
        print(print_dict[style] + print_string + '\033[0m')


def clamp_step(x_origin, x_step, lb, ub):
    """Shortens a step to stay within some box constraints."""
    alpha_ub = np.inf * np.ones(x_step.shape)
    alpha_lb = -np.inf * np.ones(x_step.shape)

    mask = x_step != 0
    alpha_ub[mask] = (ub[mask] - x_origin[mask]) / x_step[mask]
    alpha_lb[mask] = (lb[mask] - x_origin[mask]) / x_step[mask]

    scaling = np.min(np.maximum(alpha_ub, alpha_lb))
    if abs(scaling) > 1.0:
        scaling = 1.0

    return x_origin + scaling * x_step, scaling != 1.0
