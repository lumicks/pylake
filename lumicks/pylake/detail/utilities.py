import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def get_color(i):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color_cycle[i % len(color_cycle)]


def lighten_color(c, amount):
    hsv = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c))
    hsv[2] = np.clip(hsv[2]+amount, 0.0, 1.0)
    return mpl.colors.hsv_to_rgb(hsv)

def find_contiguous(mask):
    """Find [start,stop] indices and lengths of contiguous blocks where mask is True."""
    padded = np.hstack((0, mask.astype(np.bool), 0))
    change_points = np.abs(np.diff(padded))
    ranges = np.argwhere(change_points == 1).reshape(-1,2)
    run_lengths = np.diff(ranges, axis=1).squeeze()
    return ranges, np.atleast_1d(run_lengths)
