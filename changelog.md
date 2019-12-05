# Changelog

## v0.4.0 | 2019-12-05

* Add calibration data as attribute of force channels.
* Take into account slicing of channels when returning force calibration data.
* Fixed slicing behaviour when slicing continuous time channels within a timestep.
* Export pixel size and dwell time to TIFFs exported with pylake.
* Implement slicing for kymographs.
* Show keys in a group when printing the group.
* Allow iteration over groups.
* Fix bug which could cause axes to be inverted for images reconstructed in pylake.
* Add functionality to correlate images recorded from cameras with timeline data.

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
* `FDCurve`s now support subtraction, e.g. `fd = f.fdcurves["measured"] - f.fdcurves["basline"]`
* Scans and kymos now have a `.timestamps` property with per-pixel timestamps with the same shape as the image arrays
* Added Matlab compatibility examples to the repository in `examples/matlab`
* `h5py` >= v2.8 is now required

## v0.1.0 | 2018-06-20

* Initial release
