import os
import sys
from setuptools import setup, PEP420PackageFinder
from setuptools.command.egg_info import manifest_maker

if sys.version_info[:2] < (3, 7):
    print("Python >= 3.7 is required.")
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    packages=PEP420PackageFinder.find(include=["lumicks.*"]),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "pytest>=3.5",
        "h5py>=3.0, <4",
        "numpy>=1.20, <2",
        "scipy>=1.1, <2",
        "matplotlib>=2.2",
        "tifffile>=2019.7.26",
        "tabulate==0.8.6",
        "opencv-python-headless>=3.0",
        "ipywidgets>=7.0.0",
        "cachetools>=3.1",
        "deprecated>=1.2.8",
        "scikit-learn>=0.18.0, <1.0",
        "scikit-image>=0.17.2",
    ],
    zip_safe=False,
)
