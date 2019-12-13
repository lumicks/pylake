
def first(iterable, condition=lambda x: True):
    """Return the first item in the `iterable` that satisfies the `condition`.

    If the condition is not given, returns the first item of the iterable.

    Raises `StopIteration` if no item satisfying the condition is found.

    Parameters
    ----------
    iterable : iterable
    condition : callable
        callable which returns true when the element is eligible as return value
    """

    return next(x for x in iterable if condition(x))


def unique(input_list):
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    return unique_list


def unique_idx(input_list):
    """
    Determine unique elements of a list and return indices which reconstruct the original list from the unique elements.
    """
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    inverse_list = [unique_list.index(l) for l in input_list]
    return unique_list, inverse_list


def optimal_plot_layout(n_plots):
    import numpy as np

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
