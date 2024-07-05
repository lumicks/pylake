import io
import os
import sys
import pathlib
import warnings
import traceback
from json import load
from contextlib import redirect_stderr, redirect_stdout, contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt


@contextmanager
def with_cwd(directory):
    old_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old_directory)


def execute_notebook(nb_test_dir, filename):
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

    with open(filename) as fp:
        nb = load(fp)

    mpl.use("agg")  # Use a non-interactive backend

    with with_cwd(nb_test_dir):
        notebook_state = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(line for line in cell["source"] if not line.startswith("%"))
                with redirect_stdout(io.StringIO()) as _, redirect_stderr(io.StringIO()) as _:
                    exec(source, notebook_state, notebook_state)
                    plt.close("all")


def run_notebooks(include_list):
    exclude_list = [
        "nbwidgets",  # Exclude the notebook widgets since those require interaction
        "cas9_kymotracking",
        "checkpoints",  # Don't want any checkpoint files in here
    ]
    base_dir = pathlib.Path(__file__).parent.parent.resolve()
    nb_test_dir = base_dir / "nb_test"
    os.makedirs(nb_test_dir, exist_ok=True)
    print(f"Output folder: {nb_test_dir}")
    notebook_folder = base_dir / "build" / "html" / "nbexport"

    if not notebook_folder.exists():
        raise FileNotFoundError(
            f"Could not find notebook folder at {notebook_folder}. Are you on the root pylake "
            f"folder and did you compile the docs? Please see docs/readme.md for instructions on "
            f"how to compile the docs."
        )

    warnings.filterwarnings("error")  # Treat warnings as errors
    warnings.filterwarnings(
        "ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown"
    )  # we know
    warnings.filterwarnings(
        "ignore",
        message="Warning: Step size set to minimum step size.",  # fd-fitting
    )
    warnings.filterwarnings(
        "ignore",
        message="Maximum iterations reached! Reverting to two-point OLS.",  # kymotracking
    )

    for root, _, files in os.walk(notebook_folder):
        root = pathlib.Path(root)
        notebooks = [f for f in files if pathlib.Path(f).suffix == ".ipynb"]

        for nb_file in notebooks:
            notebook = root / pathlib.Path(nb_file)

            excluded = any(ex in str(notebook) for ex in exclude_list)
            included = not include_list or any(inc in str(notebook) for inc in include_list)

            if not excluded and included:
                print(f"Testing notebook: {notebook}")

                try:
                    execute_notebook(nb_test_dir, notebook)
                except Exception as e:
                    print("\nAn exception was raised:\n")
                    print(f"   {str(e)}\n")
                    print(traceback.format_exc())


if __name__ == "__main__":
    run_notebooks(sys.argv[1:])
