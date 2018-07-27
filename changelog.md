# Changelog

## v0.2.0 | In development

* Channel slices can be downsampled: `lf_force = hf_force.downsampled_by(20)`.
* `FDCurve`s now support subtraction, e.g. `fd = file.fdcurves["measured"] - file.fdcurves["basline"]`
* Scans and kymos now have a `.timestamps` property with per-pixel timestamps with the same shape as the image arrays.
* `h5py` >= v2.8 is now required.

## v0.1.0 | 2018-06-20

* Initial release.
