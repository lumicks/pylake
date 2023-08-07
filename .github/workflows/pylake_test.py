import os
import sys
import pathlib

import lumicks.pylake as lk

os.chdir(str(pathlib.Path(lk.__file__).parent))
sys.exit(lk.pytest(args=["--runslow", "--strict_reference_data", "--color=yes", "-Werror"]))
