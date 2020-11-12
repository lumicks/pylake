# Changelog

## v0.7.1 | t. b. d.
* Add workaround for `Scan` and `Kymo` which could prevent valid scans and kymos from being opened when the `start` timestamp of a scan or kymo had a value before the actual start of the timeline channels. The cause of this subsample time difference was the lack of quantization of a delay when acquiring STED images.
* Fixed bug in `Kymo` plotting functions. Previously, the time limits were calculated using the fly-in/out times which could lead to subtle discrepancies when comparing against force channels. These dead times are now omitted.
* Added `Slice.downsampled_like` to downsample a high frequency channel according to the timestamps of a low frequency channel, using the same downsampling method as Bluelake.
* Add `start` and `stop` property to `Slice`.

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
* Add slicing (by time) for `FDCurve`.
* Add widget to graphically slice `FDCurve` in Jupyter Notebooks. It can be opened by calling `pylake.FdRangeSelector(fdcurves)`. For more information, see the tutorials section on [notebook widgets](https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html).
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
* `FDCurve`s now support subtraction, e.g. `fd = f.fdcurves["measured"] - f.fdcurves["baseline"]`
* Scans and kymos now have a `.timestamps` property with per-pixel timestamps with the same shape as the image arrays
* Added Matlab compatibility examples to the repository in `examples/matlab`
* `h5py` >= v2.8 is now required

## v0.1.0 | 2018-06-20

* Initial release
