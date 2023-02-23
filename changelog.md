# Changelog

## v1.0.0 | t.b.d.

#### Breaking changes

* When performing particle tracking on kymographs, bias correction is now enabled by default; without this correction, kymographs with high background signal will suffer from biased localization estimates. To disable bias correction, specify `bias_correction=False` to [`lk.track_greedy()`](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.track_greedy.html#) and [`lk.refine_tracks_centroid()`](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.refine_tracks_centroid.html).
* `lk.track_lines()` now performs bias-corrected centroid refinement after tracking to improve localization accuracy. Note that the old behaviour can be recovered by passing `refine=False`.
* Changed several `asserts` to `Exceptions`.
  * Attempting to read `KymoTracks` from a `CSV` file that doesn't have the expected file format will result in an `IOError`.
  * Attempting to extend `KymoTracks` by `KymoTracks` originating from a different `Kymograph` now results in a `ValueError`.
  * Attempting to connect two tracks with the same start and ending time point now raises a `ValueError`. 
  * FdFitter: `FdFit.fit()` now raises a `RuntimeError` when a fit has no data or fittable parameters.
  * FdFitter: `FdFit.plot()` now raises a `KeyError` when trying to plot data that does not exist.
  * FdFitter: `FdFit.plot()` now raises a `RuntimeError` when trying to plot a fit with multiple models without selecting a model using angular brackets `[]` first.
  * FdFitter: `FdFit.profile()` now raises a `ValueError` when `max_step <= min_step` or `max_chi2_step <= min_chi2_step`.
  * FdFitter: when inverting a model with `interpolate=True`, Pylake now raises a `ValueError` if the minimum or maximum is not finite.
  * FdFitter: a `ValueError` is raised when adding incompatible models.
  * FdFitter: When adding data to a fit, adding data with an unequal number of points for the dependent and independent variable will now raise a `ValueError`.
  * FdFitter: When adding data to a fit, adding data with more than one dimension will raise a `ValueError`.
  * FdFitting: Attempting to evaluate a parameter trace with `lk.parameter_trace()` for a parameter that is not part of the model now results in a `ValueError`.
  * FdFitting: Attempting to compute a parameter trace while providing an incomplete set of parameters will now result in a `ValueError`.
  * Attempting to use `Slice.downsampled_over()` or `Slice.downsampled_like()` with timestamps ranges or another channel that doesn't overlap with the channel now produces a `ValueError`.
  * Attempting to construct a `TimeSeries` where the length of the timestamp array is not equal to the length of the data array results in a `ValueError`.
  * `FdEnsemble` alignment now produces a `ValueError` if fewer than 2 datasets are provided.
  * Image reconstruction now raises a `ValueError` if the length of the data and infowave are not equal.
  * Plotting: When creating a plot providing `axes` and an `image_handle`, a `ValueError` is raised when those `axes` do not belong to the `image_handle` provided. 
  * Widefield: Attempting to open multiple `TIFF` as a single ImageStack will now raise a `ValueError` if the alignment matrices of the individual `TIFF` are different.
  * PowerSpectrum: Attempting to replace the power spectral values of a `PowerSpectrum` using `with_spectrum` using a vector of incorrect length will raise a `ValueError`.
* When removing tracks with the kymotracking widget, only tracks that are entirely in the selection rectangle will be removed. Prior to this change, any tracks intersecting with the selection rectangle would be removed.

#### New features

* Added API for notes, i.e. `file.notes` returns a dictionary of notes.
* Added option to add a scale bar to by providing a [`lk.ScaleBar()`](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.ScaleBar.html) to plotting or export functions.
* Implemented bias correction for centroid refinement that shrinks the window to reduce estimation bias. This bias correction can be toggled by passing a `bias_correction` argument to [`lk.track_greedy()`](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.track_greedy.html#) and [`lk.refine_tracks_centroid()`](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.refine_tracks_centroid.html).
* Added `KymoTrackGroup.remove()` to remove a `KymoTrack` from a `KymoTrackGroup`.
* Allow boolean array indexing (e.g. `tracks[[False, False, True]]`) and indexing with arrays of indices (e.g. `tracks[[1, 3]]`) for `KymoTrackGroup`. See the [API documentation](https://lumicks-pylake.readthedocs.io/en/latest/_api/lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup.html#lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup) for more information.

#### Bug fixes

* Fixed incorrect behaviour in `lk.track_lines()` by interpolating back to integer frame times. Prior to this change, `lk.track_lines()` would provide a subpixel accurate position along the time axis of the kymograph as well. However, this position was specified with respect to the coordinate system of the image, rather than actual acquisition times. As such, it would produce incorrect results when performing downstream analysis that rely on the time corresponding to an actual time. Note that `lk.track_greedy()` is not affected.
* Fixed bug in `lk.track_lines()` where one extra line was returned rather than the number requested through the parameter `max_lines`.

## v0.13.3 | 2023-01-26

#### New features

* Added [`KymoTrackGroup.ensemble_diffusion()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup.html#lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup.ensemble_diffusion) for estimating an average diffusion constant for a collection of tracks.
* Added [diffusion theory](https://lumicks-pylake.readthedocs.io/en/v0.13.3/theory/diffusion/diffusion.html) section which details methods used for quantifying diffusive motion. Note that this documentation can be downloaded and run inside a Jupyter notebook.
* Added [`KymoTrack.plot()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.kymotrack.KymoTrack.html#lumicks.pylake.kymotracker.kymotrack.KymoTrack.plot) and [`KymoTrackGroup.plot()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup.html#lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup.plot) methods to conveniently plot the coordinates of tracked lines. See the [kymotracking documentation](https://lumicks-pylake.readthedocs.io/en/v0.13.3/tutorial/kymotracking.html) for more details.
* Added the [`lk.GaussianMixtureModel.extract_dwell_times()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.GaussianMixtureModel.html#lumicks.pylake.GaussianMixtureModel.extract_dwell_times) method to calculate dwell-times for all states from channel data.
* Added [`lk.colormaps`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.colormaps.html) with custom colormaps for plotting single-channel images. Note: this is a `namedtuple` so you can access the attributes using the dot notation (for example `kymo.plot("blue", cmap=lk.colormaps.cyan)`). The available attributes are `red`, `green`, `blue`, `magenta`, `yellow`, `cyan`.
* Added `localization_variance` to [`DiffusionEstimate`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.detail.msd_estimation.DiffusionEstimate.html). This quantity is useful for determining the diffusive SNR.
* Added `variance_of_localization_variance` to [`DiffusionEstimate`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.detail.msd_estimation.DiffusionEstimate.html) when calculating an ensemble CVE. This provides an estimate of the variance of the averaged localization uncertainty.
* Added option to pass a localization variance and its uncertainty to [`KymoTrack.estimate_diffusion()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymotracker.kymotrack.KymoTrack.html#lumicks.pylake.kymotracker.kymotrack.KymoTrack.estimate_diffusion).
* Added option to calculate a diffusion estimate based on the ensemble MSDs using the Ordinary Least Squares (OLS) method.
* Added [`DwelltimeBootstrap.extend()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html#lumicks.pylake.population.dwelltime.DwelltimeBootstrap.extend) to draw additional samples for the distribution.

#### Bug fixes

* Fixed a bug that resulted in an incorrect round-off of the window size to pixels when kymotracking. This bug resulted in using one more pixel on each side than intended for specific `track_widths`. Track width is selected by rounding to the next odd window size. Prior to this change, the number of points used would increase on even window sizes. As a result, requesting a track width of `2.5` pixels, would result in using a window of size `5`. Currently, requested a track width of `3` pixels results in a window of size 3, while `3.1` rounds up to the next odd window size (`5`). This bug affected the kymo tracking widget (tracking, refinement and photon count sampling during saving), [`lk.track_greedy()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.track_greedy.html#) and [`lk.refine_tracks_centroid()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.refine_tracks_centroid.html).
* Updated default slider ranges for the Kymotracker widget to reflect minimum track width required for tracking.
* Fixed issue with model description not being available in Jupyter notebooks for some force-distance models.
* Show validity criterion for Marko Siggia WLC models in terms of model parameters. Prior to this change the limit was simply denoted as `10 pN` where in reality it depends on the model parameters. The `10 pN` was a reasonable value for most DNA constructs.
* Fixed bug which occurred when exporting images to TIFF files of a numerical type with insufficient range for the data with the flag `clip=True`. Prior to this change, values exceeding the range of the numerical type were not clearly defined. After this change values below and above the supported range are clipped to the lowest or highest value of the data type respectively.
* Fixed bug in [`DwelltimeBootstrap.hist()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html#lumicks.pylake.population.dwelltime.DwelltimeBootstrap.hist) (previously named `DwelltimeBootstrap.plot()`, see below). Previously, only up to two components were plotted; now all components are plotted appropriately.
* [`DwelltimeBootstrap.hist()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html#lumicks.pylake.population.dwelltime.DwelltimeBootstrap.hist) now displays the original parameter estimate rather than the mean of the bootstrap distribution; the bootstrap distribution is used solely to calculate the confidence intervals via `DwelltimeBootstrap.get_interval()`.
* Fixed a bug where [`Scan.export_video()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.scan.Scan.html#lumicks.pylake.scan.Scan.export_video) and [`ImageStack.export_video()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.ImageStack.html#lumicks.pylake.ImageStack.export_video) would show elements from a previous plot.
* Fixed a bug that caused a misalignment of half a pixel between the kymograph and its position histogram when using [`Kymo.plot_with_position_histogram()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.kymo.Kymo.html#lumicks.pylake.kymo.Kymo.plot_with_position_histogram).

#### Deprecations

* Renamed `CorrelatedStack` to [`ImageStack`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/tutorial/imagestack.html).
* Deprecated the `DwelltimeModel.bootstrap` attribute; this attribute will be removed in a future release. Instead, `DwelltimeModel.calculate_bootstrap()` now returns a `DwelltimeBootstrap` instance directly. See the [population dynamics](https://lumicks-pylake.readthedocs.io/en/v0.13.3/tutorial/population_dynamics.html#confidence-intervals-from-bootstrapping) documentation for more details.
* Deprecated `DwelltimeBootstrap.plot()` and renamed to [`DwelltimeBootstrap.hist()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html#lumicks.pylake.population.dwelltime.DwelltimeBootstrap.hist) to more closely match the figure that is generated.
* Deprecated `DwelltimeBootstrap.calculate_stats()`. This method is replaced with [`DwelltimeBootstrap.get_interval()`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html#lumicks.pylake.population.dwelltime.DwelltimeBootstrap.get_interval) which returns the `100*(1-alpha)` % confidence interval; unlike `DwelltimeBootstrap.calculate_stats()`, it does not return the mean.

#### Other changes

* [`DwelltimeBootstrap`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.population.dwelltime.DwelltimeBootstrap.html) is now a frozen dataclass.
* Attempting to access [`DwelltimeModel.bootstrap`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.DwelltimeModel.html#lumicks.pylake.DwelltimeModel.bootstrap) before sampling now raises a `RuntimeError`; however, see the deprecation note above for proper API to access bootstrapping distributions.
* Suppress legend entry for outline when invoking `KymoTrack.plot()`.
* Allow pickling force calibration results ([`CalibrationResults`](https://lumicks-pylake.readthedocs.io/en/v0.13.3/_api/lumicks.pylake.force_calibration.power_spectrum_calibration.CalibrationResults.html#calibrationresults)). Prior to this change two functions involved in calculating upper parameter bounds prevented this class from being pickled.

## v0.13.2 | 2022-11-15

#### New features

* Added covariance-based estimator (`cve`) option to `KymoTrack.estimate_diffusion()`. See [kymotracker documentation](https://lumicks-pylake.readthedocs.io/en/v0.13.2/tutorial/kymotracking.html#studying-diffusion-processes) for more details.
* TIFFs exported from `Scan` and `Kymo` now contain metadata. The `DateTime` tag indicates the start/stop timestamp of each frame. The `ImageDescription` tag contains additional information about the confocal acquisition parameters.
* Added the `Kymo.duration` property to provide convenient access to the total scan time in seconds.
* Added addition operator to `KymoTrackGroup`. `KymoTrackGroups` tracked on the same `Kymo` can be concatenated with the `+` operator.
* Added the optional `min_length` parameter to `KymoTrackGroup.estimate_diffusion()` to discard tracks shorter than a specified length from the analysis.
* Added the `DwelltimeModel.rate_constants` property along with additional documentation explaining the assumptions that underlie using the exponential model. See the [dwell time analysis documentation](https://lumicks-pylake.readthedocs.io/en/v0.13.2/tutorial/population_dynamics.html#dwelltime-analysis) for more information.
* Ensure same call signature for `plot()` methods for `Scan`, `Kymo`, `PointScan` and `CorrelatedStack`:
  * `Scan`, `Kymo` and `PointScan`: Made argument `channel` optional.
  * `Scan` and `PointScan`: Added argument `show_title`.
  * `Kymo`: Added arguments `image_handle` and `show_title`.
  * `CorrelatedStack`: See deprecation changelog entry.
* Introduced lazy loading for `TimeSeries` data. This means that the data corresponding to a `TimeSeries` channel is not read from disk until it is used.

#### Bugfixes

* Added a check which verifies that a `Kymo` is not downsampled prior to estimating a diffusion constant. Computing diffusion constants from temporally downsampled kymographs is now explicitly disallowed.
* Fixed a bug in `KymoTrack.estimate_diffusion()` that could lead to biased estimates obtained with the `"ols"` estimator. For the diffusion estimate itself to be affected, specific lags have to be missing from the track (for example every second point in a track). This is a regression that was introduced in `v0.12.1`.
* Added a warning to `KymoTrack.estimate_diffusion()` used with the `"ols"` method when points are missing from a track. In this case the uncertainty estimate is biased. See the section on [diffusive processes](https://lumicks-pylake.readthedocs.io/en/v0.13.2/tutorial/kymotracking.html#studying-diffusion-processes) for more details.
* Added a warning that estimating the optimal number of points to use when using the `"ols"` method can be biased if many points are missing.
* Fixed a bug in `KymoTrack.estimate_diffusion()` that could lead to biased estimates obtained with the `"gls"` estimator when gaps occur in the track. Such cases now produce an exception recommending the user to refine the track prior to diffusion estimation. See the section on [diffusive processes](https://lumicks-pylake.readthedocs.io/en/v0.13.2/tutorial/kymotracking.html#studying-diffusion-processes) for more details.
* Fixed issue where on Jupyter Lab the kymotracker widget would align the Kymograph and track parameters vertically rather than horizontally.
* Functions that use `KymoTrackGroup` now gracefully handle the cases where no tracks are available. The refinement functions `refine_tracks_centroid()` and `refine_tracks_gaussian()` return an empty list, while `KymoTrackGroup.fit_binding_times()` and `KymoTrackGroup.plot_binding_histogram()` raise an exception.
* `lk.track_greedy()` now returns an empty `KymoTrackGroup` instead of an empty list when no coordinates exceed the threshold.
* `lk.track_greedy()` now returns an empty `KymoTrackGroup` instead of an error when an ROI is selected that results in no lines tracked.
* Fixed a bug where the `pixel_threshold` could be set to zero for an empty image. Now the minimum `pixel_threshold` is one.
* Fixed a bug where single pixel detections in a `KymoTrackGroup` would contribute values with a dwell time of zero. These are now dropped, the correct minimally observable time is set appropriately and a warning is issued.
* Fixed slicing of a `Kymo` where slicing from a time point inside the last line to the end (e.g. `kymo["5s":]`) resulted in a `Kymo` which returned errors upon trying to access its contents.
* Fixed a minor bug in force calibration. In rare cases it was possible that the procedure to generate an initial guess for the power spectral fit failed. This seemed to occur when the spectrum supplied is a mostly flat plateau. After the fix, an alternative method to compute an initial guess is applied in cases where the regular method fails. Note that successful calibrations were not at risk for being incorrect due to this bug since they would have resulted in an exception rather than an invalid result.

#### Deprecations

* Deprecated property `CorrelatedStack.src`. In future versions, the contents of `src` will be considered an implementation detail that is not directly accessible. Data should be accessed through the [public API](https://lumicks-pylake.readthedocs.io/en/v0.13.2/_api/lumicks.pylake.correlated_stack.CorrelatedStack.html).
* Reordered the keyword arguments of the method `CorrelatedStack.plot()` and enforced all parameters after `channel` to be keyword arguments. For details see the [docstring](https://lumicks-pylake.readthedocs.io/en/v0.13.2/_api/lumicks.pylake.correlated_stack.CorrelatedStack.html#lumicks.pylake.correlated_stack.CorrelatedStack.plot).
* Enforced the argument `axes` of the method `plot()` for `Scan`, `Kymo` and `PointScan` to be a keyword argument.
* Renamed force distance fitting functions. They are deprecated now and will be removed in the future:
  * `inverted_marko_siggia_simplified` -> `wlc_marko_siggia_distance`
  * `marko_siggia_simplified` -> `wlc_marko_siggia_force`
  * `marko_siggia_ewlc_distance` -> `ewlc_marko_siggia_distance`
  * `marko_siggia_ewlc_force` -> `ewlc_marko_siggia_force`
  * `odijk` -> `ewlc_odijk_distance`
  * `inverted_odijk` -> `ewlc_odijk_force`
  * `freely_jointed_chain` -> `efjc_distance`
  * `inverted_freely_jointed_chain` -> `efjc_force`
  * `twistable_wlc` -> `twlc_distance`
  * `inverted_twistable_wlc` -> `twlc_force`

#### Other changes

* All `KymoTrack` instances must have the same source `Kymo` and color channel in order to be in the same `KymoTrackGroup` instance. While this behavior was required previously for some downstream analyses on the tracks, it is now explicitly enforced upon `KymoTrackGroup` construction.
* When calling `KymoTrackGroup.estimate_diffusion()` without specifying the `min_length` parameter, tracks which are shorter than the required length for the specified method will be discarded from analysis and a warning emitted. Previously, if any tracks were shorter than required, an error would be raised.
* Updated benchmark to not use deprecated functions and arguments. Prior to this change, running the benchmark would produce deprecation warnings.
* `Kymo.plot()` now returns a handle of the plotted image.
* `PointScan.plot()` now returns a list of handles of the plotted lines.

## v0.13.1 | 2022-09-08

#### Bug fixes

* Reverted lazy loading for `TimeSeries` data as this actually caused significantly long wait times for accessing the data.

## v0.13.0 | 2022-09-06

#### New features

* Added possibility to access property `sample_rate` for `TimeSeries` data with constant sample rate.
* Allow reading multiple files with `lk.CorrelatedStack` (e.g. `lk.CorrelatedStack("image1.tiff", "image2.tiff")`).
* Added `CorrelatedStack.export_video()` to export videos to export multi-frame videos to video formats or GIFs.
* Added support for steps when slicing frames from `CorrelatedStack`s.
* Added function `Kymo.line_timestamp_ranges()` to obtain the start and stop timestamp of each scan line in a `Kymo`. Please refer to [Confocal images](https://lumicks-pylake.readthedocs.io/en/stable/tutorial/kymographs.html) for more information.
* Added `Kymo.flip()` to flip a Kymograph along its positional axis.
* Added `KymoTrackGroup.estimate_diffusion()` to estimate diffusion constants for a group of kymograph traces.
* Include unit in `DiffusionEstimate` dataclass.
* Added `shape` property to `Scan` and `Kymo`.
* Allow slicing `CorrelatedStack`s with timestamps and time strings (e.g. `stack["5s":"10s"]` or `stack[f.force1x.start:f.force1x.stop]`).
* Allow slicing `Scan` with timestamps and time strings (e.g. `scan["5s":"10s"]` or `scan[f.force1x.start:f.force1x.stop]`).
* Allow downloading files directly from Zenodo using `lk.download_from_doi()`. See the [example on Cas9 binding](https://lumicks-pylake.readthedocs.io/en/stable/examples/cas9_kymotracking/cas9_kymotracking.html) for an example of its use.
* Made piezo tracking functionality public and added [piezo tracking tutorial](https://lumicks-pylake.readthedocs.io/en/stable/tutorial/piezotracking.html).
* Lazily load `data` and `timestamps` for `TimeSeries` data
* Propagate `Slice` axis labels when performing arithmetic (when possible).
* Added a warning to the Kymotracker widget if the threshold parameter is set too low, which may result in slow tracking and the widget hanging.
* Added header line to exported track coordinate CSV files. The first header line now contains the version of `Pylake` which generated the file and a version number for the CSV file itself (starting with `v2` from this release).
* It is now possible to pickle `FdFit` objects. Prior to this change, unpickling an `FdFit` would fail since model identification relied on a stored `id` for each of the models used. The `id` of a variable changes whenever a new variable is created however. After this change, each model is associated with a universally unique identifier (uuid) that is used for identification instead. This uuid is serialized with the `Model` and used by `FdFit` thereby preserving their relationship when pickled/unpickled.

#### Bug fixes

* Improved `scan.get_image("rgb")` to handle missing channel data. Missing channels are now handled gracefully. Missing channels are zero filled matching the dimensions of the remaining channels.
* Added calls to manually redraw the axes in the kymotracker widget during horizontal pan and line connection callbacks. Without this, the plot did not update correctly when using newer versions of `ipywidgets` and `matplotlib`.
* Fixed a bug in the video export that led to one frame less being exported than intended.
* Fixed a bug which prevented the range selector widget from updating when the dataset to be plotted is changed. Previously, on some supported versions of `matplotlib` it would no longer update the figure. This is now fixed.
* Force distance models now raise a `ValueError` exception when simulated for invalid parameter values.
* Force distance models now have a non-zero lower bound for the contour length (`Lc`), persistence length (`Lp`), stretch modulus (`St`) and Boltzmann constant times temperature (`kT`) instead of a lower bound of zero.
* Force distance fits now raise a `RuntimeError` if any of the returned simulation values are `NaN`.
* Fixed a bug that resulted in profile likelihood automatically failing when an attempted step exceeded the bounds where the model could be simulated.

#### Breaking changes

* To disable image alignment for `lk.CorrelatedStack`, the alignment argument has to be provided as a keyword argument (e.g. `lk.CorrelatedStack("filename.tiff", align=False)` rather than `lk.CorrelatedStack("filename.tiff", False)`).
* Removed deprecated argument `roi` from `CorrelatedStack.export_tiff`. Use `CorrelatedStack.crop_by_pixels()` to select the ROI before exporting.
* `CorrelatedStack.frame_timestamp_ranges()` is now a method rather than a property. This was done for API consistency with the API for `Scan`. Please refer to [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/stable/tutorial/correlatedstacks.html#correlated-stacks) for more information.
* Removed public attributes `CorrelatedStack.start_idx` and `CorrelatedStack.stop_idx` and made them protected.
* The property `sample_rate` of `Continuous` data now returns a `float` instead of an `int`.
* Changed the error type when attempting to access undefined per-pixel timestamps in `Kymo` from `AttributeError` to `NotImplementedError`.
* `KymoWidgetGreedy` now enforces using keywords for all arguments after the first two (`kymo` and `channel`).
* The following `KymoWidgetGreedy` attributes/functions have been removed (replaced with private API): `adding`, `algorithm_parameters`, `axis_aspect_ratio`, `output_filename`, `plotted_lines`, `show_lines`, `create_algorithm_sliders()`, `refine()`, `show()`, `track_all()`, `track_kymo()` and `update_lines()`.
* The `track_width` argument of `refine_tracks_centroid()` expects values in physical units whereas the deprecated `refine_lines_centroid()` expected the `line_width` argument in pixel units.
* Removed default value provided for `driving_frequency_guess` in `lk.calibrate_force()`.
* It is now mandatory to supply a `sample_rate` when calling `lk.calibrate_force()`.
* It is now mandatory to supply a `sample_rate` when calling `lumicks.pylake.force_calibration.touchdown.touchdown()`.

#### Deprecations

* `Scan.save_tiff()` and `Kymo.save_tiff()` were deprecated and replaced with `Scan.export_tiff()` and `Kymo.export_tiff()` to more clearly communicate that the data is exported to a different format.
* Deprecated `export_video_red()`, `export_video_green()`, `export_video_blue()`, and `export_video_rgb()` methods for `Scan`. These methods have been replaced with a single `export_video(channel=color)` method.
* In the functions `Scan.frame_timestamp_ranges()` and `Kymo.line_timestamp_ranges()`, the parameter `exclude` was deprecated in favor of `include_dead_time` for clarity.
* Deprecated `KymoTrackGroup.remove_lines_in_rect()`; use `KymoTrackGroup.remove_tracks_in_rect()` instead (see below).
* Deprecated the `line_width` argument of `track_greedy()`; use `track_width` instead.
* Deprecated `filter_lines()`; use `filter_tracks()` instead.
* Deprecated `refine_lines_centroid()`; use `refine_tracks_centroid()` instead. *Note: the `track_width` argument of `refine_tracks_centroid()` expects values in physical units whereas the previous `refine_lines_centroid()` expected the `line_width` argument in pixel units.*
* Deprecated `refine_lines_gaussian()`; use `refine_tracks_gaussian()` instead.
* Deprecated `KymoWidget.save_lines()`; use `KymoWidget.save_tracks()` instead.

#### Other changes

* Added default values for the `track_greedy()` arguments `track_width` and `pixel_threshold`.
* Renamed classes/methods/functions dealing with tracked particles. This change was made to avoid ambiguity with regard to the term *"line"*. Now, a *"line"* refers to a single scan pass of the confocal mirror during imaging. A *"track"* refers to the coordinates of tracked particles from a kymograph. *Note: Any breaking changes or deprecations to the public API are noted above. The renamed classes/functions below are considered internal API and subject to change without notice; these classes should not be constructed manually:*
    * `KymoLine` was renamed to `KymoTrack`
    * `KymoLineGroup` was renamed to `KymoTrackGroup`
    * `export_kymolinegroup_to_csv()` was renamed to `export_kymotrackgroup_to_csv()`
    * `import_kymolinegroup_from_csv()` was renamed to `import_kymotrackgroup_from_csv()`
* Updated the `KymoWidgetGreedy` UI to reflect changes in terminology.
* Made `ipywidgets>=7.0.0` and `notebook>=4.4.1` optional dependencies for `pip`.
* Made `notebook>=4.4.1` a mandatory dependency for `conda` release.
* `Pylake` now depends on `h5py>=3.4, <4`. This change is required to support lazy loading for `TimeSeries`.

## v0.12.1 | 2022-06-21

#### New features

* Added `Scan.pixel_time_seconds` and `Kymo.pixel_time_seconds` to obtain the pixel dwell time. See [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) for more information.
* Allow cropping `CorrelatedStack` using multidimensional indexing, i.e. `stack[start_frame : end_frame, start_row : end_row, start_column : end_column]`. See [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.
* Added `KymoLine.estimate_diffusion()` which provides additional information regarding the diffusion constant estimation. This dataclass can be converted to floating point to get just the estimate, but also contains the number of points used to compute the estimate, and the number of lags used in the analysis. In addition to that, it provides a `std_err` field which reports the standard error for the estimate.
* Added generalized least squares method as `method` option for `KymoLine.estimate_diffusion()`. Please refer to the [kymotracker documentation](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html#studying-diffusion-processes) for more information.
* Added offline piezo tracking functionality (documentation pending).
* Added `lk.benchmark()` that can be used to estimate the performance of your computer with various pylake tasks.

#### Bug fixes

* Changed the internal calculation of the `extent` argument in `Kymo.plot()` such that the spatial limits are now defined at the center of the pixel (same functionality that is used for the temporal axis). Previously the limits were defined at the edge of the pixel resulting in a subtle misalignment between the coordinates of the tracked lines and the actual image.
* Fixed an issue where the pixel dwell time stored in TIFFs exported from Pylake could be incorrect. Prior to this fix, Pylake exported the value entered in the Bluelake UI as pixel dwell time. In reality, the true pixel dwell time is a multiple of the acquisition sample rate. After the fix, TIFF files correctly report the achieved pixel dwell time.
* Changed the internal calculation of the hydrodynamically correct force spectrum. Before this change, the computation of the power spectral density relied on the frequencies always being positive. While this change does not affect any results, it allows evaluating the power spectral density for negative frequencies.
* Fixed an issue where evaluating the hydrodynamically correct spectrum up to very high frequencies could lead to precision losses. These precision losses typically occur at frequencies upwards of 30 kHz and manifest themselves as visible discontinuities.
* Perform better input validation on the kymotracking functions `track_greedy` and `track_lines`. The line width and pixel threshold must be larger than zero. The diffusion parameter must be positive. Previously, failure to provide values respecting these limits would produce cryptic error messages.
* Perform better input validation on `refine_lines_centroid`. Line width must now be at least one pixel. Previously, negative values produced a cryptic error message, while a line width smaller than one pixel would silently result in no refinement taking place.
* Fixed bug in force calibration convenience function where setting `fixed_alpha` or `fixed_diode` to zero resulted in those parameters still being fitted.  After this change, setting `fixed_alpha` to zero will result in the diode model having a fixed `alpha` of zero, whereas setting `f_diode` to zero raises an exception.
* Include one extra sample when requesting frame timestamp ranges from a scan (`Scan.frame_timestamp_ranges(exclude=True)`). Previously, when slicing using these timestamps, you would omit the last sample of the scan. Now this sample will be included.

#### Deprecations

* `KymoLine.estimate_diffusion_ols()` is now deprecated; use `KymoLine.estimate_diffusion(method="ols")` instead. This new method returns the same estimate of the diffusion coefficient as before but includes additional information about the fit.

## v0.12.0 | 2022-04-21

#### New features

* Support negating channels (e.g. `neg_force = - file.force1x`). See [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#exporting-h5-files) for more information.
* Allow applying color intensity adjustments on images using `lk.ColorAdjustment()`. See [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html#correlating-scans) and [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.
* Added `DwelltimeModel` to fit dwelltimes to an exponential (mixture) distribution. For more information, see the tutorials section on [Population Dynamics](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/population_dynamics.html)
* Allow slicing directly with an object with a `.start` and `.stop` property. See [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#markers) for more information.
* Allow boolean array indexing on `Slice` (e.g. `file.force1x[file.force1x.data > 5]`. See [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#boolean-array-indexing) for more information.
* When performing arithmetic on `Slice`, the calibration is propagated if it is still valid.
* Allow applying a gamma adjustment on images using `lk.ColorAdjustment()`. See [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html#correlating-scans) and [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.
* Added `lk.dsdna_odijk()` and `lk.ssdna_fjc()` convenience functions to build Fd models for dsDNA and ssDNA with parameter defaults based on literature.

#### Bug fixes

* Fixed a minor bug in `KymoLineGroup.fit_binding_times()`. Previously, the binding time for all lines in the group were used for the analysis. However, lines which start in the first frame of the kymo or end in the last frame have ambiguous dwelltimes as the start or end of the line is not known definitively. Now, the default behavior is to exclude these lines from the analysis. This behavior can be overridden with the keyword argument `exclude_ambiguous_dwells=False`. In general, this bug would lead to only very minor biases in the results unless the number of dwells to be excluded is large relative to the total number.
* Fixed bug in `vmax` handling for `CorrelatedStack`. Before `vmax` values were scaled to the maximally possible range of values for the image instead of the actual intensity value. Note that use of `vmax` and `vmin` is deprecated and one should use `adjustment=lk.ColorAdjustment(min, max)` for color adjustments. See [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.
* Fixed a bug in the kymotracker in which the plotted aspect ratio did not match the requested `axis_aspect_ratio` argument.

#### Deprecations

* Deprecated `red_image`, `green_image`, `blue_image`, and `rgb_image` properties for `Scan` and `Kymo`. These data should now be accessed using the `get_image(channel="{color}")` method (where `"{color}"` can be `"red"`, `"green"`, `"blue"`, or `"rgb"`).

#### Breaking changes

* Changed the frame indexing convention for plotting confocal scans to match `CorrelatedStack.plot()`. Previously, `Scan.plot(frame=1)` referred to the first frame in the stack. Now, indexing starts at `0`.
* Requesting a frame outside of the available range for `Scan.plot()` now throws an `IndexError`.
* Removed deprecated properties `scan_width_um`, `json`, `has_force`, `has_fluorescence` from confocal classes.

#### Other changes

* Changed titles for all plots of `Scan` and `CorrelatedStack` images to be consistent. First frame is titled as `"[frame 1 / N]"` and last frame is titled as `"[frame N / N]"`.
* The returned type from `KymoLineGroup.fit_binding_times()` has been changed to `DwelltimeModel`. Note, this class has the same public attributes and methods as the previously returned `BindingDwelltimes` class; however the `plot()` method has been deprecated and renamed to `DwelltimeModel.hist()`. This new method name more closely describes the actual functionality and also unifies the API with `GaussianMixtureModel`.
* `KymoLine` is now immutable.
* Removed `examples` directory. Application examples can be found in the [online documentation](https://lumicks-pylake.readthedocs.io/en/stable/examples/index.html).

## v0.11.1 | 2022-02-22

#### New features

* Added support for Python 3.10.
* Added `CorrelatedStack.define_tether()` which can be used to define the endpoints of the tether between two beads and return image data rotated such that the tether is horizontal. Check out the [documentation](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html) for more information.
* Added function to correlate `Scan` frames to channel data. See [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html#correlating-scans).
* Added `CorrelatedStack.crop_and_rotate()` for interactive editing of the image stack. Actions include scrolling through image frames with the mouse wheel, and left-clicking to define the tether coordinates, and right-click/drag to define a cropping region. Check out the [documentation](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html#image-stack-editor) for more information.
* Added `KymoLineGroup.plot_binding_histogram()` to plot histograms of binding events for tracked lines. See [Kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html#plotting-binding-histograms) for more details.
* Allow fixing the photon background parameter in `refine_lines_gaussian()`. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html#maximum-likelihood-estimation) for more information.
* Added option to specify a custom label when plotting fit with `FdFit.plot()`. See the tutorial section on [Fd Fitting](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdfitting.html#plotting-the-data) for more information.
* Added functionality to slice `Scan` objects by frame indices. See [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html).
* Added convenience function which allows users to perform a force calibration procedure with a single function call `calibrate_force()`. See [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html#more-convenient-calibration).
* Added `Kymo.crop_and_calibrate()` for interactive cropping of a kymograph. If the optional `tether_length_kbp` argument is supplied, the resulting kymograph will be automatically calibrated to this length. Check out the [documentation](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html#calibrating-to-base-pairs) for more information.
* Added fallback to the function `Kymo.plot_with_force()` when only low-frequency force data is available.
* Added option to undo/redo actions in the kymotracker widget.
* Added option to fit peaks simultaneously in `refine_lines_gaussian()` using the flag `overlap_strategy="multiple"`. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* Added ability to manually connect two lines from any points (not just the ends) in the kymotracker widget.

#### Bug fixes

* Fixed issue which resulted in the offset parameter added by `Model.subtract_independent_offset()` not having a unit associated with it.
* Fixed bug which resulted in erroneous standard errors on parameter estimates computed from an `FdFit` with fixed parameters. For such a fitting problem, the covariance matrix was evaluated for the unconstrained problem (without the fixed parameter constraints). As a result, standard errors were always overestimated. Note that uncertainty estimation by profile likelihood was unaffected.
* Fixed issue which resulted in overly stringent positional tolerance when using the kymotracker widget. This tolerance has now been made proportional to the axis viewport.
* Removed `axial` parameter from `lk.ActiveCalibrationModel()` as we do not support active force calibration in the axial direction.
* Improved default scaling behaviour for `CorrelatedStack.plot_correlated()` and `Scan.plot_correlated()`. It now ensures the ratio between the image and temporal plot is according to the aspect ratio of the scan or stack.
* Slicing `CorrelatedStack` in reverse (i.e., `stack[5:3]`) or resulting in an empty stack (i.e., `stack[5:5]`) now throws an exception.
* Resolved `DeprecationWarning` for `tifffile.imsave()` and `tifffile.TiffWriter.save()` with `tifffile >= 2020.9.30`.
* Fixed a bug in `refine_lines_gaussian()` which incorrectly rounded the pixel position instead of flooring it. For pixels with a subpixel position larger than half the pixel, this resulted in shifting the window by one pixel in the positive direction. Typically, this would have little or no effect since the majority of the peak should still be covered.

#### Deprecations

* Deprecated `CorrelatedStack.timestamps` and replaced with `CorrelatedStack.frame_timestamp_ranges`. The reason for this change is that per-pixel timestamps are not defined for camera based images; therefore, this previous use was not in line with the use of the `timestamps` property of confocal image classes. This change also brings consistency with `Scan.frame_timestamp_ranges`.
* Deprecated `plot_red()`, `plot_green()`, `plot_blue()`, and `plot_rgb()` methods for `PointScan`, `Scan`, and `Kymo`. These methods have been replaced with a single `.plot(channel={color})` method.

## v0.11.0 | 2021-12-07

#### New force calibration features

* Added support for active force calibration. For more information, please read: [active force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html#active-calibration).
* Added support for axial force calibration. See [axial calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html#axial-calibration) for more information.
* Added support for using near-surface corrections for lateral and axial force calibration, please read: [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html#faxen-s-law).
* Added function to compute viscosity of water at specific temperature. Please refer to: [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html) for more information.
* Added parameters describing the inferred driving peak (`driving_amplitude`, `driving_frequency`, `driving_power`) when performing active force calibration to `CalibrationResults`.

#### Other new features

* Added `Kymo.calibrate_to_kbp()` for calibrating the position axis of a kymograph from microns to kilobase-pairs. **Note: this calibration is applied to the full kymograph, so one should crop to the bead edges with `Kymo.crop_by_distance()` before calling this method.**
* Added `CorrelatedStack.get_image()` to get the image stack data as an `np.ndarray`.
* Allow setting custom slider ranges for the algorithm parameters in the kymotracker widget. Please refer to [kymotracker widget](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html#using-the-kymotracker-widget) for more information.
* Added function `Scan.frame_timestamp_ranges()` to obtain the start and stop timestamp of each frame in a `Scan`. Please refer to [Confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) for more information.

#### Bug fixes

* Fixed issue in force calibration where the analytical fit would sometimes fail when the corner frequency is below the lower fitting bound. What would happen is that the analytical fit resulted in a negative term of which the square root was taken to obtain the corner frequency. Now this case is gracefully handled by setting the initial guess halfway between the lowest frequency in the power spectrum and zero.
* Fixed issue that led to `DeprecationWarning` in the kymotracker widget.
* Fixed error in kymotracking documentation. The tutorial previously indicated the incorrect number of samples for `sample_from_image`.
* Fixed bug that would implicitly convert `Kymograph` and `Scan` `timestamps` to floating point values. Converting them to floating point values leads to a loss of precision. For more information see [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#channels) for more information.
* Fixed bug where color-aligned data was returned from `TiffFrame.data` although alignment was not requested (e.g., `CorrelatedStack("filename.tiff", align=False)`). This bug was introduced in `v0.10.1`.

#### Deprecations

* `CorrelatedStack.raw` has been deprecated and will be removed in a future release. Use `CorrelatedStack.get_image()` instead.

#### Breaking changes

* Changed default for `viscosity` in force calibration models. When omitted, `pylake` will use the viscosity of water calculated from the temperature. Note that this results in the default (when no viscosity or temperature is set) changing from `1.002e-3`  to `1.00157e-3 Pa*s`. Please refer to [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html) for more information.
* Units are now included in the headers for kymograph traces exported from the widget or with `KymoLineGroup.save()` (either `um` or `kbp` depending on the calibration of the kymograph). Any code that hardcoded the header names directly should be updated.
* Added more input validation for model parameters when performing force calibration. We now force bead and sample density to be more than `100 kg/m³` when specified. Temperature should be specified between `5` and `90 °C` and viscosity should be bigger than `0.0003 Pa*s` (viscosity of water at 90 degrees Celsius).

## v0.10.1 | 2021-10-27

#### New features

* Added support for axial force detection (i.e. force detection along the Z axis). The high-frequency data can be accessed with `f.force1z` and `f.force2z` while the downsampled low-frequency channels can be accessed with `f.downsampled_force1z` and `f.downsampled_force2z`. The calibrations can be accessed with `f.force1z.calibration` and `f.force2z.calibration`. The Z component is *not* factored in the calculation of the total force `f.downsampled_force1` and `f.downsampled_force2`.
* Added `KymoLineGroup.fit_binding_times()` to allow for dwelltime analysis of bound states found using kymotracker. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* Allow using data acquired with a fast force sensor by passing `fast_sensor=True` when creating a `PassiveCalibrationModel`.
* Allow using hydrodynamically correct power spectrum when performing force calibration. See [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html) for more information.
* Added ability to crop a `CorrelatedStack`. See [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.

#### Bug fixes

* Show an error message when user attempts to refine lines before tracking or loading them so the kymotracker widget does not become unresponsive.
* Force calibration models now throw an error when a bead diameter of less than `10^-2` microns is used (rather than produce `NaN` results).
* Fixed bug that prevented export of `CorrelatedStack` if the alignment matrices were missing from the metadata.
* Fixed bug where the exported metadata fields were erroneously exported as `"Applied channel X alignment"` for `CorrelatedStack` where the alignment was not actually applied.
* Fixed error in optimal point determination for MSD estimator. Note that this bug led to an error being thrown when the number of points in the trace was exactly 4. Now it results in a more informative error message (`You need at least 5 time points to estimate the number of points to include in the fit.`).

#### Improvements

* Switched to trust region reflective algorithm for fitting thermal calibration spectrum. This results in fewer optimization failures.
* Pylake now ensures that `f_diode` stays below the Nyquist frequency during fitting.
* Implemented a bias correction for the thermal calibration. Note that this typically leads to a small correction unless you use a very low number of points per block.

#### Deprecations

* `CorrelatedStack.from_data()` has been renamed to `CorrelatedStack.from_dataset()` for consistency with `BaseScan.from_dataset()`.

## v0.10.0 | 2021-08-20

Important notice: This release contains an important fix that could lead to timestamp corruption when slicing kymographs. If you are still on version `0.8.2` or `0.9.0` we highly recommend updating or not using the kymograph slicing functionality (e.g. using the syntax `kymo["0s":"5s"]`). Please refer to the `Bug fixes` section for more information.

#### New features

* Added option to exclude ranges with potential noise peaks from the calibration routines. Please refer to [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html) for more information.
* Added `crop_by_distance` to `Kymo` to allow cropping Kymographs by distance. Please refer to [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html#kymo-data-and-details).
* Added `refine_lines_gaussian()` for refining lines detected by the kymotracking algorithm using gaussian localization. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.

#### Bug fixes

* Fixed bug in `Kymo` which resulted in erroneous timestamps and line times after slicing the Kymograph by time (e.g. `Kymo["1s":"5s"]`). The reason for this was imprecision in the timestamp reconstruction that occurred when slicing the data. This in turn led to an erroneous reconstruction of the timestamps. **As a result, any downstream analysis that relies on the time axis of a Kymograph post-slicing cannot be trusted (MSD analysis, plotting, tracking, attributes such as `.line_time_seconds`, etc.) in pylake v0.8.2 and v0.9.0.** Only those two versions are affected. Note that regular image reconstruction was not affected. Timestamps of unsliced kymographs constructed directly from the `.h5` files are also not affected.
* Fixed an error in the documentation. In v0.9.0 `PowerSpectrum.power` was changed to represent power in `V^2/Hz` instead of `0.5 V^2/Hz`. However, the docs were not appropriately updated to reflect this change in the model equation that's fitted to the spectrum. This is mitigated now.
* Fixed bug in `plot_with_force` which caused an exception on Kymographs with a partial last pixel.

#### Breaking changes

* Removed the option to specify a custom `reduce_timestamps` function in `kymo.downsampled_by`. The reason for the removal is that by downsampling repeatedly with different `reduce` functions for the timestamps, the data can end up in an inconsistent state. Additionally, the timestamps of the original object (read directly from the h5 file) are defined as the mean of the pixel timestamps; this definition is now consistent regardless of downsampling state.

## v0.9.0 | 2021-07-29

`Pylake v0.9` provides several new features. Starting from `pylake v0.9`, the kymotracker will handle units for you. From now on, all you have to worry about is physical quantities rather than pixels. Please see the updated [example on Cas9 binding](https://lumicks-pylake.readthedocs.io/en/latest/examples/cas9_kymotracking/cas9_kymotracking.html) for a demonstration of this. In addition to that, you can now [infer diffusion constants from diffusive kymograph traces](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html#studying-diffusion-processes).

For convenience, we added the option to [simulate force-distance models](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdfitting.html#simulating-the-model) without having to fit them first and directly do [arithmetic with channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#arithmetic). `Pylake` also supports [baseline corrected force data](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdcurves.html#baseline-correction) from `Bluelake` now.

Unfortunately, some of these improvements required some breaking changes so please see the detailed changelog entries for more information.

#### New features

* Added `Kymo.downsampled_by()` for downsampling Kymographs in space and time. See [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html) for more information.
* Added option to stitch Kymograph lines visually via the Jupyter notebook widget.
* Added Mean Square Displacement (MSD) and diffusion constant estimation to `KymoLine`. For more information, please refer to [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html)
* Added `FdCurve.with_baseline_corrected_x()` to return a baseline corrected version of the FD curve if the corrected data is available. **Note: currently the baseline is only calculated for the x-component of the force channel in Bluelake. Therefore baseline corrected `FdCurve` instances use only the x-component of the force channel, unlike default `FdCurve`s which use the full magnitude of the force channel by default.**
* Added ability to perform arithmetic on `Slice` (e.g. `(f.force1x - f.force2x) / 2`). For more information see [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#exporting-h5-files) for more information.
* Allow simulating force model with a custom set of parameters (see the tutorial section on [Fd Fitting](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdfitting.html) for more information).
* Added a slider to set the algorithm parameter `velocity` to the kymotracker widget.

#### Bug fixes

* Fixed bug in kymotracker which could result in `sample_from_image` returning erroneous values in some cases.
  When sampling the image, we sample pixels in a region around the traced line.
  This sampling procedure is constrained to stay within the image bounds.
  Previously, we used the incorrect axis to clamp the pixel index along the positional axis.
  As a consequence, kymographs that are wider (in terms of number of pixels on the positional axis) than they are long (in terms of number of pixels along the time axis) would only have an accurate sampling in the top half of the kymograph.
  The lower portion of the kymograph would result in zero counts.
* Fixed bug which could lead to bias when tracking lines using `track_greedy()` with the rectangle tool.
  Selecting which region to track used to pass that region specifically to the tracking algorithm.
  This means that the algorithm is unable to sample the image outside of the passed region.
  Therefore, this can lead to a bias when the line to be tracked is near the edge of the selected region.
  In the updated version, all image processing steps that depend on the image use the full image.
* Fixed bug which could lead to bias when tracking lines using `track_lines()` with the rectangle tool.
  Selecting which region to track used to pass that region specifically to the tracking algorithm.
  This means that the blurring steps involved in this algorithm become biased near the edges (since they do not get contributions from outside the selected areas, while they should).
  In the updated version, all image processing steps that depend on the image use the full image.
* Fixed a bug in `Kymo.plot_with_force()` which resulted in the plotting function throwing an error for Kymographs with an incomplete final line.
* Fixed a bug in the plotting order of `CalibrationResults.plot()`. Previously, when plotting after performing a force calibration, the model fit was erroneously plotted first (while the legend indicated that the model fit was plotted last). The results of the calibration itself are unchanged.
* Resolved `DeprecationWarning` with `tifffile >= 2021.7.2`.
* Fixed a bug in `CalibrationResults.ps_model_fit` which resulted in its attribute `num_points_per_block` to be `1` rather than the number of points per block the model was fitted to. Note that this does not affect the calibration results as the calibration procedure internally used the correct number of points per block.

#### Breaking changes

* **Changed `PowerSpectrum.power` to actually reflect power in `V^2/Hz`. Before it was expressed in `0.5 V^2/Hz`.**
* Dropped support for Python 3.6.
* Pylake now depends on `numpy>=1.20`. This change is required to use a different fft normalization in the force calibration tests.
* The attribute `image_data` in `KymoLine` is now private.
* Make kymotracker functions `track_greedy()`, `track_lines()`, and class `KymoWidgetGreedy` take `Kymo` and a channel (e.g. "red") as their input.
  The advantage of this is that now units of time (seconds) and space (microns) are propagated automatically to the tracked `KymoLines`.
  See the [Cas9 kymotracking example](https://lumicks-pylake.readthedocs.io/en/latest/examples/cas9_kymotracking/cas9_kymotracking.html) or the [kymotracking tutorial](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* `KymoLineGroup.save()` and `KymoWidgetGreedy.save_lines()` no longer take `dx` and `dt` arguments.
  Instead, the correct time and position calibration is now passed automatically to these functions. See [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html) for more information.
* Express kymotracker algorithm parameters `line_width`, `sigma`, `velocity` and `diffusion` in physical units rather than pixels. Prior to this change, the units of the kymotracking algorithm were in pixels. Note that if you want to reproduce your earlier results multiply `line_width` and `sigma` by `kymo.pixelsize_um[0]`, `velocity` by `kymo.pixelsize_um[0] / kymo.line_time_seconds` and `diffusion` by `kymo.pixelsize_um[0] ** 2 / kymo.line_time_seconds`.
* In the FD Fitter, `Parameters.keys()` is now a member function instead of a property (used to be invoked as `parameter.keys`) to be consistent with dictionary.
* `Slice.downsampled_like()` now returns both the downsampled `Slice` and a copy of the low frequency reference `Slice` cropped such that both instances have exactly the same timestamps. The reason for this is that the first two samples of the low frequency trace can typically not be reconstructed (since there is no high frequency data for those available). This led to confusion, since now the trace `downsampled_like` produces is shorter than the input. By returning both, this problem is mitigated. Please refer to [Files and Channels](https://lumicks-pylake.readthedocs.io/en/excluded_ranges/tutorial/file.html#downsampling) for an example of its updated use.
* Optimization settings are now passed to `fit_power_spectrum()` as keyword arguments instead of using the class `lk.CalibrationSettings`.
* Renamed `CalibrationResults.ps_model_fit` and `CalibrationResults.ps_fitted` to `CalibrationResults.ps_model` and `CalibrationResults.ps_data` for clarity.
* Drop units from parameter names in `CalibrationResults`. Note that the unit is still available in the `.unit` attribute of a calibration parameter.
* Accessing an element from a force calibration performed with Pylake now returns a `CalibrationParameter` instead of a `float`. The calibration value can be accessed as `calibration["kappa"].value`, while the unit can be accessed in the `.unit` property.

## v0.8.2 | 2021-04-30

#### New features

* Added `GaussianMixtureModel` to fit a Gaussian Mixture Model to channel data. For more information, see the tutorials section on [Population Dynamics](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/population_dynamics.html)
* Added `Scan.size_um` and `Kymo.size_um` to return the scanned size along each dimension. Use these properties to access the scan sizes along each axis for [scans](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) and [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html).
* Added force calibration functionality to Pylake. Please refer to [force calibration](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/force_calibration.html) for more information.
* Added support for new metadata format in order to handle widefield TIFF files recorded in Bluelake 1.7.

#### Bug fixes

* Fixed bug in kymotracker which could result in a line being extended one pixel too far in either direction. Reason for the bug was that a blurring step in the peak-finding routine was being applied on both axes, while it should have only been applied to one axis. Note that while this bug affects peak detection (finding one too many), it should not affect peak localization as that is performed in a separate step.
* Fixed bug in kymotracker slicing which could result in one line too many or too few being included. The bug was caused by using the timestamp corresponding to one sample beyond the last pixel of the line as "start of the next line" without accounting for the dead time that may be there.

#### Deprecations

* `Scan.scan_width_um` and `Kymo.scan_width_um` have been deprecated. Use `Scan.size_um` and `Kymo.size_um` to get the scan sizes. When performing a scan, Bluelake determines an appropriate scan width based on the desired pixel size and the scan width entered in the UI. The property `scan_width_um` returned the scan size entered by the user, rather than the actual scan size achieved. To obtain the achieved scan size, use `Scan.size_um` and `Kymo.size_um`.

## v0.8.1 | 2021-02-17

#### New features

* Added widget to graphically slice `FdCurve` by distance in Jupyter Notebooks. It can be opened by calling `pylake.FdDistanceRangeSelector(fdcurves)`. For more information, see the tutorials section on [notebook widgets](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html).
* Added `FdCurve.range_selector()` and `FdCurve.distance_range_selector()`. See [Notebook widgets](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html#range-selection-by-distance) for more information.
* Added `center_point_um` property to `PointScan`, `Kymo` and `Scan` classes. Use these properties to access the metadata for [scans](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) and [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html) instead of the deprecated `json` field.
* Added `scan_width_um` property to `Kymo` and `Scan` classes. Use these properties to access the metadata for [scans](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) and [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html). instead of the deprecated `json` field.
* Added `FdCurve.with_offset()` to `FdCurve` to add offsets to force and distance.
* Added `FdEnsemble` to be able to process multiple `FdCurve` instances simultaneously. See [FD Curves](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdcurves.html#fd-ensembles) for more information.
* Added `FdEnsemble.align_linear()` to align F,d curves in an ensemble by correcting for a constant offset in force and distance using two linear regressions. See [FD curves](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdcurves.html#fd-ensembles) for more information.
* Added `CorrelatedStack.export_tiff()` for exporting aligned image stacks. See [Correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html#correlated-stacks) for more information.

#### Bug fixes

* Fixed bug when using continuous channels which lead to excessive memory usage and degraded performance.
* Fixed `Slice.downsampled_over()` to ignore gaps rather than result in an unhandled exception. Previously when you downsampled a `TimeSeries` channel which had a gap in its data, `Slice.downsampled_over()` would try to compute the mean of an empty subsection, which raises an exception. Now this case is gracefully handled.

#### Breaking changes

* Deprecated `json` attribute in confocal classes `PointScan`, `Scan`, and `Kymo`. **Note: The format of the raw metadata exported from Bluelake is likely to change in future releases and therefore should not be accessed directly. Instead, use the accessor properties, as documented for [scans](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) and [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html).**
* `Slice.range_selector()` is now a method instead of a property.
* Deprecated `has_force` and `has_fluorescence` properties in confocal classes `PointScan`, `Scan`, and `Kymo`.
* Renamed `fd_selector.py` to `range_selector.py`.
* `FdRangeSelectorWidget` is no longer public.
* Renamed `FDCurve` and `FDSlice` to `FdCurve` and `FdSlice`.

#### Other

* Added documentation for the Kymotracker widget. See the [Cas9 kymotracking example](https://lumicks-pylake.readthedocs.io/en/latest/examples/cas9_kymotracking/cas9_kymotracking.html) or the [kymotracking tutorial](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.

## v0.7.2 | 2020-01-14

#### New features

* Support Python 3.9 (this required bumping the `h5py` requirement to >= 3.0).
* Added `refine_lines_centroid()` for refining lines detected by the kymotracking algorithm. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* Added `Kymo.plot_with_force()` for plotting a kymograph and corresponding force channel downsampled to the same time ranges of the scan lines.
* Added `Kymo.plot_with_position_histogram()` and `Kymo.plot_with_time_histogram()` for plotting a kymograph and corresponding histogram along selected axis.
* Added `KymoLineGroup.save()` for saving tracked Kymograph traces to a file. See [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* Allow `nbAgg` backend to be used for interactive plots in Jupyter notebooks (i.e. `%matplotlib notebook`). Note that this backend doesn't work for JupyterLab (please see the [FAQ](https://lumicks-pylake.readthedocs.io/en/simplify_widgets/install.html#frequently-asked-questions) for more information).
* Show `downsampled_force` channels when printing a `File`.

#### Bug fixes

* Fixed exception message in `Slice.downsampled_to()` which erroneously suggested to use `force=True` when downsampling a variable frequency channel, while the correct argument is `method="force"`.
* Fixed bug in `Kymo` plot functions which incorrectly set the time limits. Now, pixel centers are aligned with the mean time for each line.

#### Other

* Include `ipywidgets` as a dependency so users don't have to install it themselves.
* Switch `opencv` dependency to headless version.

## v0.7.1 | 2020-11-19

* Add `start` and `stop` property to `Slice`. These return the timestamp in nanoseconds.
* Add `start` argument to `Slice.plot()` which allows you to use a specific timestamp as time point zero in a plot. See [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#channels) for more information.
* Added `Slice.downsampled_like` to downsample a high frequency channel according to the timestamps of a low frequency channel, using the same downsampling method as Bluelake.
* Fixed bug in `Kymo` plotting functions. Previously, the time limits were calculated including the fly-in/out times which could lead to subtle discrepancies when comparing against force channels. These dead times are now omitted.
* Add workaround for `Scan` and `Kymo` which could prevent valid scans and kymos from being opened when the `start` timestamp of a scan or kymo had a timestamp earlier than the actual start of the timeline channels. The cause of this time difference was the lack of quantization of a delay when acquiring STED images. This delay resulted in a subsample offset between the `Scan`/`Kymo` start time and the start of the timeline data which was falsely detected as a corrupted `Scan` or `Kymo`.

## v0.7.0 | 2020-11-04

#### New features

* Added `seconds` attribute to `Slice`. See [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#channels) for more information.
* Added `downsampled_to` to `Slice` for downsampling channel data to a new sampling frequency. See section on downsampling in [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#downsampling).
* Added `save_as` to `File` for exporting compressed HDF5 files or omitting specific channels from an HDF5 file. See tutorial on [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#exporting-h5-files) for more information.
* Added `Scan.export_video_red`, `Scan.export_video_green`, `Scan.export_video_blue` and `Scan.export_video_rgb` to export multi-frame videos to video formats or GIFs. See tutorial on [confocal images](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/images.html) for more information.
* Added widget to graphically slice a `Slice` in Jupyter Notebooks. It can be opened by calling `channel.range_selector`. For more information, see [notebook widgets](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html).
* Added profile likelihood method to FdFitter. See [confidence intervals and standard errors](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdfitting.html#confidence-intervals-and-standard-errors) for more information.
* Added support for RGB images to `CorrelatedStack`.
* Added image alignment to `CorrelatedStack`. Image alignment is enabled by default and will automatically align the color channels for multi-channel images, provided that the alignment metadata from Bluelake is available. Image alignment can be disabled by specifying `align=False` when loading the `CorrelatedStack`. For more information see [correlated stacks](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/correlatedstacks.html).
* Exposed low-level kymograph API (still in alpha status!). See tutorial on [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html).
* Added `Kymo.line_time_seconds` for obtaining the time step between two consecutive kymograph lines.
* Added `Kymo.pixelsize_um` and `Scan.pixelsize_um` for obtaining the pixel size for various axes.

#### Bug fixes

* Improved accuracy of covariance matrix computation. To compute the covariance matrix of the parameter estimates, it is required to estimate the standard deviation of the residuals. This calculation was previously biased by not correctly taking into account the number of degrees of freedom. This is fixed now.
* Fixed bug in `CorrelatedStack.plot_correlated` which lead to the start index of the frame being added twice when the movie had been sliced.
* Fixed bug in `File.scans` so that a warning is generated when a scan is missing metadata. Other scans that can be loaded properly are still accessible.
* Fixed bug in `plot_correlated` which did not allow plotting camera images correlated to channel data when the channel data did not completely cover the camera image stack.

#### Breaking changes

* Fixed bug in `downsampled_over` which affects people who used it with `where="center"`. With these settings the function returns timestamps at the center of the ranges being downsampled over. In the previous version, this timestamp included the end timestamp (i.e. t1 <= t <= t2), now it has been changed to exclude the end timestamp (i.e. t1 <= t < t2) making it consistent with `downsampled_by` for integer downsampling rates.
* Fixed bug in `File.point_scans` to return `PointScan` instances. Previously, attempts to access this property would cause an error due to missing `PointScan.from_dataset` method. Note that the `__init__` method arguments of `PointScan` have been changed to be in line with `Kymo` and `Scan`.

## v0.6.2 | 2020-09-21

* Support plotting Z-axis scans. Z-axis scans would previously throw an exception due to how the physical dimensions were fetched. This issue is now resolved.
* Add slicing (by time) for `FdCurve`.
* Add widget to graphically slice `FdCurve` in Jupyter Notebooks. It can be opened by calling `pylake.FdRangeSelector(fdcurves)`. For more information, see the tutorials section on [notebook widgets](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html).
* Fixed bug in `FdRangeSelectorWidget` that prevented drawing to the correct axes when other axes has focus.
* Fixed displayed coordinates to correctly reflect position in `Kymo.plot_red()`, `Kymo.plot_green()`, `Kymo.plot_blue()` and `Kymo.plot_rgb()`. The data origin (e.g. `kymo.red_image[0, 0]`) is displayed on the top left of the image in these plots, whereas previously this was not reflected correctly in the coordinates displayed on the plot axes (placing the coordinate origin at the bottom left).

## v0.6.1 | 2020-08-31

* Added inverted simplified Marko Siggia model with only entropic contributions to `FdFitter`.
* Change exception that was being raised on non-API field access such as `Calibration`, `Marker`, `FD Curve`, `Kymograph` and `Scan` to a `FutureWarning` rather than an exception.

## v0.6.0 | 2020-08-18

* Plot and return images and timestamps for scans using physical coordinate system rather than fast and slow scanning axis. In v5.0, this resulted in a failed reconstruction and `Scan.pixels_per_line` being defined as pixels along the x axis. `Scan.pixels_per_line` and `Kymo.pixels_per_line` now return the number of pixels along the fast axis. This fix was also applied to the timestamps. In the previous version, for a multi-frame scan where the y-axis is the fast axis, you could incorrectly get the time between pixels on the fast axis by calculating `scan.timestamps[0, 0, 1] - scan.timestamps[0, 0, 0]`. In the new version, this is `scan.timestamps[0, 1, 0] - scan.timestamps[0, 0, 0]` (note that the image array convention is `[frame, height, width]`). **Note that this is a breaking change affecting scans with the fast axis in y direction!**
* Verify starting timestamp when reconstructing `Kymo` or `Scan`. In those cases, scans cannot be reliably reconstructed from the exported data and an error is thrown. For kymos, the first (partial) line is omitted and a warning is issued. **Note that scans where the scan was initiated before the exported time window cannot be reconstructed! For kymos, the first line cannot be reconstructed if the export window does not cover the start of the kymograph.**
* Add literature page to the documentation.
* Fix docstring for `Fit.plot()`.
* Optimized reconstruction algorithm for sum.

## v0.5.0 | 2020-06-08

* Added F, d Fitting functionality (beta, see docs tutorial section `Fd Fitting` and examples `Twistable Worm-Like-Chain Fitting` and `RecA Fd Fitting`).
* Fixed an issue which prevented images from being reconstructed when a debugger is attached. Problem resided in `reconstruct_image` which threw an exception when attempting to resize a `numpy` array while the debugger was holding a reference to it.
* Fixed bug that lead to timestamps becoming floating point values when using `channel.downsampled_over`.

## v0.4.1 | 2020-03-23

* Drop `matplotlib` < 3 requirement.
* Add functionality which redirects users to the API when accessing particular fields, e.g. accessing `file["FD curve"]` will throw an error and redirect users to use `file.fdcurves`.
* Add API for markers, i.e. `file.markers` returns a dictionary of markers (see docs tutorials section: Files and Channels).
* Bugfix `CorrelatedStack.plot()` which resulted in the function throwing an error rather than showing a single frame.
* Add canvas draw call to `CorrelatedStack.plot_correlated()` to make sure the plot is also interactive when it is not run from an interactive notebook.

## v0.4.0 | 2020-01-21

* Add calibration data as attribute of force channels (see docs tutorials section: Files and Channels).
* Fixed bug which produced shifted timestamps when slicing continuous channel with time values between two data points.
* Export pixel size and dwell time to exported TIFFs.
* Implement slicing for kymographs (see docs tutorials section: Kymographs).
* Show keys in a group when printing the group.
* Allow iteration over groups.
* Fix bug which could cause axes to be inverted for reconstructed images.
* Add functionality to correlate images recorded from cameras with timeline data (see docs tutorial section: Correlated stacks).

## v0.3.1 | 2019-03-27

* Loading scans and kymographs is now much faster
* Improved perfomance of slicing continuous channels
* Fixed `Unknown channel kind` error with the v2 file format
* Fixed deprecation warnings with h5py v2.9

## v0.3.0 | 2018-12-04

* TIFF scan exports now default to using 32bit floats
* Support for photon time tag data has been added (Bluelake version 1.5.0 and higher, HDF5 file format version 2)
* File properties like `File.kymos`, `File.scans` now return empty `dict` objects instead of an error if there are no such items in the file

## v0.2.0 | 2018-07-27

* Channel slices can be downsampled: `lf_force = hf_force.downsampled_by(factor=20)`
* `FdCurve`s now support subtraction, e.g. `fd = f.fdcurves["measured"] - f.fdcurves["baseline"]`
* Scans and kymos now have a `.timestamps` property with per-pixel timestamps with the same shape as the image arrays
* Added Matlab compatibility examples to the repository in `examples/matlab`
* `h5py` >= v2.8 is now required

## v0.1.0 | 2018-06-20

* Initial release
