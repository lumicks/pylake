import os
import sys

from setuptools import PEP420PackageFinder, setup
from setuptools.command.egg_info import manifest_maker

if sys.version_info[:2] < (3, 10):
    print("Python >= 3.10 is required.")
    sys.exit(-1)


def about(package):
    ret = {}
    filename = os.path.join(os.path.dirname(__file__), package.replace(".", "/"), "__about__.py")
    with open(filename, "rb") as file:
        exec(compile(file.read(), filename, "exec"), ret)
    return ret


def read(filename):
    if not os.path.exists(filename):
        return ""

    with open(filename) as f:
        return f.read()


info = about("lumicks.pylake")
manifest_maker.template = "setup.manifest"
setup(
    name=info["__title__"],
    version=info["__version__"],
    description=info["__summary__"],
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    url=info["__url__"],
    license=info["__license__"],
    keywords="",
    author=info["__author__"],
    author_email=info["__email__"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    packages=PEP420PackageFinder.find(include=["lumicks.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "pytest>=3.5",
        "h5py>=3.4, <4",
        "numpy>=1.24, <2",  # 1.24 is needed for dtype in vstack/hstack (Dec 18th, 2022)
        "scipy>=1.9, <2",  # 1.9.0 needed for lazy imports (July 29th, 2022)
        "matplotlib>=3.8",
        "tifffile>=2022.7.28",
        "tabulate>=0.8.8, <0.9",
        "cachetools>=3.1",
        "deprecated>=1.2.8",
        "scikit-learn>=0.18.0",
        "scikit-image>=0.17.2",
        "tqdm>=4.27.0",  # 4.27.0 introduced tqdm.auto which auto-selects notebook or console
    ],
    extras_require={
        "notebook": [
            "notebook>=7",
            "ipywidgets>=7.0.0",
            "jupyter_client>=8",
            "ipympl>=0.9.3",  # Needed for mpl compatibility (previous vers are only up to mpl 3.7)
        ],
    },
    zip_safe=False,
)
