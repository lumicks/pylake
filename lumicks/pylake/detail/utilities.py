import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


def use_docstring_from(copy_func):
    def wrapper(func):
        func.__doc__ = copy_func.__doc__
        return func

    return wrapper


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
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return color_cycle[i % len(color_cycle)]


def lighten_color(c, amount):
    hsv = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c))
    hsv[2] = np.clip(hsv[2] + amount, 0.0, 1.0)
    return mpl.colors.hsv_to_rgb(hsv)


def find_contiguous(mask):
    """Find [start,stop] indices and lengths of contiguous blocks where mask is True."""
    padded = np.hstack((0, mask.astype(bool), 0))
    change_points = np.abs(np.diff(padded))
    ranges = np.argwhere(change_points == 1).reshape(-1, 2)
    run_lengths = np.diff(ranges, axis=1).squeeze()
    return ranges, np.atleast_1d(run_lengths)


def downsample(data, factor, reduce):
    """This function is used to downsample data in blocks of size `factor`.

    This function downsamples blocks of size `factor` using the function specified in reduce. If
    there are insufficient points to fill a final block, this last block is discarded.

    Parameters
    ----------
    data : array_like
        Input data
    factor : int
        Factor to downsample by
    reduce : callable
        Function to use for reducing the data
    """

    def round_down(size, n):
        """Round down `size` to the nearest multiple of `n`"""
        return int(math.floor(size / n)) * n

    data = data[: round_down(data.size, factor)]
    return reduce(data.reshape(-1, factor), axis=1)


def will_mul_overflow(a, b):
    """Will `a * b` overflow?

    We know that `a * b > int_max` iff `a > int_max // b`. The expression `a * b > int_max` can
    overflow but the `a > int_max // b` cannot. Hence the latter can be used for testing whether
    the former will overflow without actually incurring an overflow. The result is an array if
    the inputs are arrays.
    """
    int_max = np.iinfo(np.result_type(a, b)).max
    return np.logical_and(b > 0, a > int_max // b)


def could_sum_overflow(a, axis=None):
    """Could the sum of `a` overflow?

    For safety, this estimate is conservative. A `True` result indicates that the result could
    overflow, but it may not. This is because it assumes that the largest number in the array
    could appear at every index.

    A `False` result is definitive. The sum will definitely not overflow.
    """
    return np.any(will_mul_overflow(np.max(a, axis), a.size if axis is None else a.shape[axis]))
