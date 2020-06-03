import os
import sys
from setuptools import setup, PEP420PackageFinder
from setuptools.command.egg_info import manifest_maker

if sys.version_info[:2] < (3, 6):
    print("Python >= 3.6 is required.")
    sys.exit(-1)


def about(package):
    ret = {}
    filename = os.path.join(os.path.dirname(__file__), package.replace(".", "/"), "__about__.py")
    with open(filename, 'rb') as file:
        exec(compile(file.read(), filename, 'exec'), ret)
    return ret


def read(filename):
    if not os.path.exists(filename):
        return ""

    with open(filename) as f:
        return f.read()


info = about("lumicks.pylake")
manifest_maker.template = "setup.manifest"
setup(
    name=info['__title__'],
    version=info['__version__'],
    description=info['__summary__'],
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    url=info['__url__'],
    license=info['__license__'],
    keywords="",

    author=info['__author__'],
    author_email=info['__email__'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    packages=PEP420PackageFinder.find(include=["lumicks.*"]),
    python_requires='>=3.6',
    install_requires=['pytest>=3.5', 'h5py>=2.9, <3.0', 'numpy>=1.14, <2',
                      'scipy>=1.1, <2', 'matplotlib>=2.2', 'tifffile>=2018.11.6',
                      'tabulate==0.8.6'],
    zip_safe=False,
)
