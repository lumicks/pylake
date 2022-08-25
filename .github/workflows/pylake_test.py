import lumicks.pylake as lk
import sys
import os
import pathlib

os.chdir(str(pathlib.Path(lk.__file__).parent))
sys.exit(lk.pytest(args=["--runslow", "--color=yes", "-Werror"]))
