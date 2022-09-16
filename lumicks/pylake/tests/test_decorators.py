import functools
import matplotlib.pyplot as plt
import matplotlib as mpl


def mpl_test_cleanup(func):
    """Runs tests in a context manager and closes figures after each test"""

    @functools.wraps(func)
    def wrapped_callable(*args, **kwargs):
        try:
            orig_backend = plt.get_backend()
            with mpl.style.context(["classic", "_classic_test_patch"]):
                # Use a headless backend for testing, note that passing it as a context parameter
                # did not work.
                plt.switch_backend("agg")
                func(*args, **kwargs)
        finally:
            plt.close("all")
            plt.switch_backend(orig_backend)

    return wrapped_callable
