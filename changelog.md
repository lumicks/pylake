# Changelog

## v0.9.0 | t.b.d.

#### New features

* Added `Kymo.downsampled_by()` for downsampling Kymographs in space and time. See [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html) for more information.
* Added option to stitch Kymograph lines via the Jupyter notebook widget.
* Added Mean Square Displacement (MSD) and diffusion constant estimation to `KymoLine`. For more information, please refer to [kymotracking](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html)
* Added `FdCurve.with_baseline_corrected_x()` to return a baseline corrected version of the FD curve if the corrected data is available. **Note: currently the baseline is only calculated for the x-component of the force channel in Bluelake. Therefore baseline corrected `FdCurve` instances use only the x-component of the force channel, unlike default `FdCurve`s which use the full magnitude of the force channel by default.**
* Added ability to perform arithmetic on `Slice` (e.g. `(f.force1x - f.force2x) / 2`). For more information see [files and channels](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/file.html#exporting-h5-files) for more information.
* Allow simulating force model with a custom set of parameters (see tutorial section [Fd Fitting](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/fdfitting.html)).
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
  This means that the blurring steps involved in this algorithm become biased (since they do not get contributions from outside the selected areas, while they should).
  In the updated version, all image processing steps that depend on the image use the full image.
* Fixed a bug in the plotting order of `CalibrationResults.plot()`. Previously, when plotting after performing a force calibration, the model fit was erroneously plotted first (while the legend indicated that the model fit was plotted last). The results of the calibration itself are unchanged.
* Resolved `DeprecationWarning` with `tifffile >= 2021.7.2`.
* Fixed a bug in `CalibrationResults.ps_model_fit` which resulted in its attribute `num_points_per_block` to be `1` rather than the number of points per block the model was fitted to. Note that this does not affect the calibration results as the calibration procedure internally used the correct number of points per block.
* Fixed a bug in `Kymo.plot_with_force()` which resulted in the plotting function throwing an error for Kymographs with an incomplete final line.

#### Breaking changes

* **Changed `PowerSpectrum.power` to actually reflect power in `V^2/Hz`. Before it was expressed in `0.5 V^2/Hz`.**
* Dropped support for Python 3.6.
* Pylake now depends on `numpy>=1.20`. This change is required to use a different fft normalization in the force calibration tests.
* The attribute `image_data` in `KymoLine` is now private.
* Make kymotracker functions `track_greedy()`, `track_lines()`, and class `KymoWidgetGreedy` take `Kymo` and a channel (e.g. "red") as their input.
  The advantage of this is that now units of time (seconds) and space (microns) are propagated automatically to the tracked `KymoLine`s.
  See the [Cas9 kymotracking example](https://lumicks-pylake.readthedocs.io/en/latest/examples/cas9_kymotracking/cas9_kymotracking.html) or the [kymotracking tutorial](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymotracking.html) for more information.
* `KymoLineGroup.save()` and `KymoWidgetGreedy.save_lines()` no longer take `dx` and `dt` arguments.
  Instead, the correct time and position calibration is now passed automatically to these functions. See [kymographs](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/kymographs.html) for more information.
* Express kymotracker algorithm parameters `line_width`, `sigma`, `velocity` and `diffusion` in physical units rather than pixels. Prior to this change, the units of the kymotracking algorithm were in pixels. Note that if you want to reproduce your earlier results multiply `line_width` and `sigma` by `kymo.pixelsize_um[0]`, `velocity` by `kymo.pixelsize_um[0] / kymo.line_time_seconds` and `diffusion` by `kymo.pixelsize_um[0] ** 2 / kymo.line_time_seconds`.
* `Parameters.keys()` is now a member function instead of a property (used to be invoked as `parameter.keys`) to be consistent with dictionary.
* `Slice.downsampled_like()` now returns both the downsampled `Slice` and a copy of the low frequency reference `Slice` cropped such that both instances have exactly the same timestamps.
* Optimization settings are now passed to `fit_power_spectrum()` as keyword arguments instead of using the class `lk.CalibrationSettings`.
* Renamed `CalibrationResults.ps_model_fit` and `CalibrationResults.ps_fitted` to `CalibrationResults.ps_model` and `CalibrationResults.ps_data` for clarity.

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
