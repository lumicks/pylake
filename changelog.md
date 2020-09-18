# Changelog

## v0.6.2 | t.b.d.

* Support plotting Z-axis scans. Z-axis scans would previously throw an exception due to how the physical dimensions were fetched. This issue is now resolved.
* Add slicing (by time) for `FDCurve`.
* Add widget to slice Fd's with from Jupyter Notebooks.
* Fixed bug in `FdRangeSelectorWidget` that prevented drawing to the correct axes when other axes has focus.

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
