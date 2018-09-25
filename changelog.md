# Changelog

## v0.3.0 | In development

* TIFF scan exports now default to using 32bit floats

## v0.2.0 | 2018-07-27

* Channel slices can be downsampled: `lf_force = hf_force.downsampled_by(factor=20)`
* `FDCurve`s now support subtraction, e.g. `fd = f.fdcurves["measured"] - f.fdcurves["basline"]`
* Scans and kymos now have a `.timestamps` property with per-pixel timestamps with the same shape as the image arrays
* Added Matlab compatibility examples to the repository in `examples/matlab`
* `h5py` >= v2.8 is now required

## v0.1.0 | 2018-06-20

* Initial release
