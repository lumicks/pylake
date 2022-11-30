API Reference
=============

This detailed reference lists all the classes and functions contained in the package.
If you are just looking to get started, read the :doc:`/tutorial/index` first.

.. currentmodule:: lumicks.pylake

.. autosummary::
    :toctree: _api
    :template: class.rst

    File
    channel.Slice
    fdcurve.FdCurve
    kymo.Kymo
    scan.Scan
    point_scan.PointScan
    correlated_stack.CorrelatedStack
    ColorAdjustment

    :template: instance.rst

    colormaps


Force calibration
-----------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    PassiveCalibrationModel
    ActiveCalibrationModel
    force_calibration.power_spectrum.PowerSpectrum
    force_calibration.power_spectrum_calibration.CalibrationResults

    :template: function.rst

    calibrate_force
    calculate_power_spectrum
    fit_power_spectrum
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
    wlc_marko_siggia_force
    wlc_marko_siggia_distance
    ewlc_marko_siggia_force
    ewlc_marko_siggia_distance
    ewlc_odijk_force
    ewlc_odijk_distance
    twlc_force
    twlc_distance
    efjc_force
    efjc_distance


Kymotracking
------------

.. autosummary::
    :toctree: _api
    :template: class.rst

    kymotracker.kymotrack.KymoTrack
    kymotracker.kymotrack.KymoTrackGroup
    kymotracker.detail.msd_estimation.DiffusionEstimate
    kymotracker.detail.msd_estimation.EnsembleMSD

    :template: function.rst

    track_greedy
    track_lines
    filter_tracks
    refine_tracks_centroid
    refine_tracks_gaussian

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
