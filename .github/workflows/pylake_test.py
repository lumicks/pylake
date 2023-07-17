import lumicks.pylake as lk
import sys
import os
import pathlib

os.chdir(str(pathlib.Path(lk.__file__).parent))
sys.exit(lk.pytest(args=["--runslow", "--strict_reference_data", "--color=yes", "-Werror"]))
