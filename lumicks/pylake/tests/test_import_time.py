import ast
import sys
import subprocess
from textwrap import dedent

import numpy as np
import pytest


@pytest.mark.slow
def test_disabling_capturing(report_line):
    repeats = 3

    code = dedent(
        """\
        import time
        tic = time.time()
        import lumicks.pylake
        print(time.time() - tic)
        """
    )

    times = [
        float(subprocess.check_output([f"{sys.executable}", "-c", code])) for i in range(repeats)
    ]

    report_line(
        f"Module import time: {np.mean(times):.2f} +- {np.std(times):.2f} seconds (N={repeats})"
    )


def test_lazy_imports():
    """Ensure that specific third-party modules are lazily imported so that `pylake` stays fast"""

    expected_lazy_imports = [
        "h5py",
        "sklearn",
        "matplotlib",
        "scipy.stats",
        "scipy.signal",
        "scipy.ndimage",
        "scipy.special",
        "scipy.optimize",
        "scipy.interpolate",
    ]

    # `sys.modules` in this new Python process will only contain what `pylake` imports
    code = dedent(
        """\
        import sys
        import lumicks.pylake
        print(list(sys.modules.keys()))
        """
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)

    imported_modules = ast.literal_eval(output)
    for lazy_import in expected_lazy_imports:
        assert not any(lazy_import in m for m in imported_modules)
