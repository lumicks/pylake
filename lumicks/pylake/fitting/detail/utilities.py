from collections import OrderedDict
import numpy as np


def unique_idx(input_list):
    """
    Determine unique elements of a list and return indices which reconstruct the original list from the unique elements.
    """
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    inverse_list = [unique_list.index(l) for l in input_list]
    return unique_list, inverse_list


def parse_transformation(params, param_transform={}):
    transformed = OrderedDict(zip(params, params))

    for key, value in param_transform.items():
        if key in transformed:
            transformed[key] = value
        else:
            raise KeyError(f"Parameter {key} to be substituted not found in model. Valid keys for this model are: "
                           f"{[x for x in transformed.keys()]}.")

    return transformed


def optimal_plot_layout(n_plots):
    n_x = int(np.ceil(np.sqrt(n_plots)))
    n_y = int(np.ceil(n_plots/n_x))

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


def latex_sqrt(arg):
    return f"\\sqrt{{{arg}}}"


def latex_frac(a, b):
    return f"\\frac{{{a}}}{{{b}}}"


def solve_formatter(a, solved_for, rhs):
    return f"argmin[{solved_for}](norm({a}-{rhs}))"


def solve_formatter_tex(a, solved_for, rhs):
    norm = f"\\left\\|{a} - {rhs}\\right\\|"
    return f"\\underset{{{solved_for}}}{{arg\\,min}}\\left({norm}_{2}\\right)"


def escape_tex(name):
    def escape_underscores(txt):
        return '\\_'.join(txt.split('_'))

    splits = name.split('/')
    if len(splits) > 1:
        return escape_underscores(splits[1]) + '_{' + escape_underscores(splits[0]) + "}"
    else:
        return escape_underscores(name)
