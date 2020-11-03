from textwrap import dedent
import numpy as np
import subprocess
import sys
import pytest


@pytest.mark.slow
def test_disabling_capturing(report_line):
    repeats = 3

    code = dedent("""\
        import time
        tic = time.time()
        import lumicks.pylake
        print(time.time() - tic)
        """)

    times = [float(subprocess.check_output([f'{sys.executable}', '-c', code])) for i in range(repeats)]

    report_line(f"Module import time: {np.mean(times):.2f} +- {np.std(times):.2f} seconds (N={repeats})")
