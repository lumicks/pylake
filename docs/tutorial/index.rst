:orphan:

.. _tutorials-index:

Tutorial
========

This section will present the essential features of Pylake with example code to get you started quickly.
Most of the tutorial pages are also available for download as `Jupyter notebooks <http://jupyter.org/>`_.

Code snippets are included directly within the tutorial text to illustrate features, thus they omit some common and repetitive code (like import statements) in order to save space and avoid distractions.
It is assumed that the following lines precede any other code::

    import numpy as np
    import matplotlib.pyplot as plt

    import lumicks.pylake as lk


This uses the common scientific package aliases: `np` and `plt`.
These import conventions are used consistently in the tutorial.


.. toctree::
    :caption: Contents
    :maxdepth: 1

    file
    fdcurves
    scans
    kymographs
    imagestack
    fdfitting
    nbwidgets
    kymotracking
    force_calibration
    population_dynamics
    piezotracking
