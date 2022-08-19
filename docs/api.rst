API Reference
=============

This detailed reference lists all the classes and functions contained in the package.
If you are just looking to get started, read the :doc:`/tutorial/index` first.

.. currentmodule:: lumicks.pylake

.. autosummary::
    :template: class.rst
    :toctree: _api

    File
    channel.Slice
    fdcurve.FdCurve
    kymo.Kymo
    scan.Scan
    point_scan.PointScan
    correlated_stack.CorrelatedStack
    ColorAdjustment


Force calibration
-----------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    PassiveCalibrationModel
    ActiveCalibrationModel

    :template: function.rst

    calibrate_force
    calculate_power_spectrum
    fit_power_spectrum
    force_calibration.power_spectrum.PowerSpectrum
    force_calibration.power_spectrum_calibration.CalibrationResults
    viscosity_of_water


FD Fitting
----------

.. autosummary::
    :toctree: _api
    :template: class.rst

    fitting.model.Model
    FdFit

    :template: function.rst
    parameter_trace

.. _fd_models:
.. rubric:: Available models

.. autosummary::
    :toctree: _api
    :template: function.rst

    force_offset
    distance_offset
    marko_siggia_ewlc_force
    marko_siggia_ewlc_distance
    marko_siggia_simplified
    inverted_marko_siggia_simplified
    odijk
    inverted_odijk
    freely_jointed_chain
    inverted_freely_jointed_chain
    twistable_wlc
    inverted_twistable_wlc


Kymotracking
------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    kymotracker.kymotrack.KymoTrack

    :template: function.rst

    track_greedy
    track_lines
    filter_lines
    refine_lines_centroid
    refine_lines_gaussian

Notebook widgets
----------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    FdRangeSelector
    FdDistanceRangeSelector

Population Dynamics
-------------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    GaussianMixtureModel
    DwelltimeModel
    population.dwelltime.DwelltimeBootstrap

Piezo tracking
--------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    DistanceCalibration
    ForceBaseLine
    PiezoTrackingCalibration
    PiezoForceDistance
