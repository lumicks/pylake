API Reference
=============

This detailed reference lists all the classes and functions contained in the package.
If you are just looking to get started, read the :doc:`/tutorial/index` first.

.. currentmodule:: lumicks.pylake

.. autosummary::
    :toctree: _api

    File
    channel.Slice
    fdcurve.FdCurve
    kymo.Kymo
    scan.Scan
    point_scan.PointScan
    correlated_stack.CorrelatedStack


Force calibration
-----------------

.. autosummary::
    :toctree: _api


    calculate_power_spectrum
    fit_power_spectrum
    PassiveCalibrationModel
    ActiveCalibrationModel
    force_calibration.power_spectrum.PowerSpectrum
    force_calibration.power_spectrum_calibration.CalibrationResults


FD Fitting
----------

.. autosummary::
    :toctree: _api

    fitting.model.Model
    FdFit
    parameter_trace

.. _fd_models:
.. rubric:: Available models

.. autosummary::
    :toctree: _api

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

    kymotracker.kymoline.KymoLine
    track_greedy
    track_lines
    filter_lines
    refine_lines_centroid
    refine_lines_gaussian
    kymotracker.detail.binding_times.BindingDwelltimes
    kymotracker.detail.binding_times.BindingDwelltimesBootstrap

Notebook widgets
----------------

.. autosummary::
    :toctree: _api

    FdRangeSelector
    FdDistanceRangeSelector

Population Dynamics
-------------------

.. autosummary::
    :toctree: _api

    GaussianMixtureModel
