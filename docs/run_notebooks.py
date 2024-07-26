import io
import os
import json
import time
import hashlib
import pathlib
import argparse
import warnings
import traceback
from shutil import copyfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout

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


def execute_notebook(nb_test_dir, json_nb):
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

    mpl.use("agg")  # Use a non-interactive backend

    with with_cwd(nb_test_dir):
        notebook_state = {}
        for cell in json_nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(line for line in cell["source"] if not line.startswith("%"))
                with redirect_stdout(io.StringIO()) as _, redirect_stderr(io.StringIO()) as _:
                    exec(source, notebook_state, notebook_state)
                    plt.close("all")


def read_report(filename):
    """Reads a dictionary from a file"""
    if not filename.is_file():
        return {}

    try:
        with open(filename, "r") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        return {}


def process_notebook(notebook, report, nb_test_dir, force_run):
    """Process a single notebook and return a report on it

    Parameters
    ----------
    notebook : pathlib.Path
        Filename of a notebook
    report : dict
        Dictionary with notebook results
    nb_test_dir : pathlib.Path
        Path to the testing directory.
    force_run : bool
        Force this one to re-run.
    """
    finished_success = report.get("result", "") == "success"

    # Notebooks have a bunch of fields that change every render, but are irrelevant to the
    # content. Only hash the source fields to check whether they need to be rerun.
    with open(notebook) as fp:
        nb_json = json.load(fp)
        md5_hash = hashlib.md5()
        for cell in nb_json["cells"]:
            md5_hash.update(str(cell["source"]).encode("utf-8"))

    checksum = md5_hash.hexdigest()

    if report.get("checksum", "") != checksum:
        finished_success = False

    if finished_success and not force_run:
        print(f"Skipping notebook: {notebook} (already successful)")
    else:
        print(f"Testing notebook: {notebook}")

        try:
            tic = time.time()
            execute_notebook(nb_test_dir, nb_json)
            report["time"] = f"{time.time() - tic:.2f}"
            report["checksum"] = checksum
        except Exception as e:
            print("\nAn exception was raised:\n")
            print(f"   {e}\n")
            print(traceback.format_exc())

            report["result"] = str(e)
        else:
            report["result"] = "success"

    return report


def run_notebooks(include_list, reset_cache, only_copy):
    """Run all notebooks

    Parameters
    ----------
    include_list : List[str]
        Override which notebooks to run.
    reset_cache : bool
        Wipe the cache of which notebooks have already been run.
    only_copy : bool
        Only copy the notebooks to the notebook testing folder, but don't run them.
    """
    exclude_list = [
        "nbwidgets",  # Exclude the notebook widgets since those require interaction
        "cas9_kymotracking",
    ]
    base_dir = pathlib.Path(__file__).parent.parent.resolve()
    nb_test_dir = base_dir / "nb_test"
    nb_source_folder_name = "nbexport"
    os.makedirs(nb_test_dir, exist_ok=True)

    report_file = nb_test_dir / "test_report.txt"

    print(f"Output folder: {nb_test_dir}")
    notebook_folder = base_dir / "build" / "html" / nb_source_folder_name

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
    warnings.filterwarnings(
        "ignore",
        message="Polyfit may be poorly conditioned",  # piezotracking
    )

    data = {} if reset_cache else read_report(report_file)

    try:
        for root, _, files in os.walk(notebook_folder):
            notebooks = [
                f for f in files if pathlib.Path(f).suffix == ".ipynb" and "checkpoint" not in f
            ]

            for nb_file in notebooks:
                notebook = pathlib.Path(root) / nb_file
                start_point = notebook.parts.index(nb_source_folder_name)
                notebook_name = "-".join(notebook.parts[start_point + 1 :])

                if only_copy:
                    copyfile(notebook, base_dir / nb_test_dir / notebook_name)
                    continue

                excluded = any(ex in str(notebook) for ex in exclude_list)
                force_included = any(inc in str(notebook) for inc in include_list)
                included = not include_list or force_included

                report = data[notebook_name] if notebook_name in data else {}
                if not excluded and included:
                    process_notebook(notebook, report, nb_test_dir, force_included)

                data[notebook_name] = report
    finally:
        with open(report_file, "w") as f:
            json.dump(data, f, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="run_notebooks",
        description="Runs python notebooks from the docs. Considering these notebooks take a while "
        "to run, this script maintains a list of notebooks already run. To rerun all notebooks, "
        "specify the commandline option --reset.",
    )
    parser.add_argument(
        "-n",
        "--notebooks",
        required=False,
        default=[],
        nargs="+",
        help="Run these notebooks specifically (ignored whether they have already been run)",
    )
    parser.add_argument(
        "-r",
        "--reset",
        default=False,
        required=False,
        nargs="?",
        const=True,
        help="Reset list of processed notebooks before starting",
    )
    parser.add_argument(
        "-c",
        "--copy",
        default=False,
        required=False,
        nargs="?",
        const=True,
        help="Copy the notebooks to the test folder, but do not run them",
    )

    return parser.parse_args()


if __name__ == "__main__":
    options = parse_arguments()
    run_notebooks(options.notebooks, options.reset, options.copy)
