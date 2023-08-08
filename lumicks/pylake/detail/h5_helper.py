import warnings
from fnmatch import fnmatch


def _write_numerical_data(
    lk_file, out_file, name, node, compression_level, crop_time_range, verbose
):
    """Write numerical data"""
    if crop_time_range:
        sliced = lk_file[name][slice(*crop_time_range)]
        if not sliced:
            if verbose:
                print(f"{name} dropped from dataset (no data within time window)")
        else:
            sliced._src.to_dataset(
                out_file,
                name,
                compression="gzip",
                compression_opts=compression_level,
            )
    else:
        out_file.create_dataset(
            name, data=node, compression="gzip", compression_opts=compression_level
        )
        out_file[name].attrs.update(node.attrs)


def _write_cropped_metadata(lk_file, out_file, name, node, crop_time_range, verbose):
    """Write non-numerical data"""

    def write_node():
        out_file.create_dataset(name, data=node)
        out_file[name].attrs.update(node.attrs)

    if not crop_time_range:
        write_node()
    else:
        # Override time ranges. Items know how to crop themselves.
        try:
            start, stop = (
                getattr(lk_file[name][slice(*crop_time_range)], field)
                for field in ("start", "stop")
            )
            if stop >= crop_time_range[0] and start < crop_time_range[1] and (stop - start) > 0:
                write_node()
                out_file[name].attrs["Start time (ns)"] = start
                out_file[name].attrs["Stop time (ns)"] = stop
            else:
                if verbose:
                    print(f"{name} removed from file (out of cropping range)")
        except (IndexError, TypeError):
            if verbose:
                print(f"{name} not cropped")


def write_h5(
    lk_file,
    output_filename,
    compression_level=5,
    omit_data=None,
    *,
    crop_time_range=None,
    verbose=False,
):
    """Write a modified h5 file to disk.

    Parameters
    ----------
    lk_file : lk.File
        pylake file handle
    output_filename : str | os.PathLike
        Output file name.
    compression_level : int
        Compression level for gzip compression.
    omit_data : str or iterable of str, optional
        Which data sets to omit. Should be a set of h5 paths.
    crop_time_range : tuple of np.int64
        Specify a time interval to crop to (tuple of a start and stop time). Interval must be
        specified in nanoseconds since epoch (the same format as timestamps).
    verbose : bool, optional.
        Print verbose output. Default: False.
    """
    import h5py

    omit_data = {omit_data} if isinstance(omit_data, str) else omit_data
    h5_file = lk_file.h5

    with h5py.File(output_filename, "w") as out_file:

        def traversal_function(name, node):
            if omit_data and any([fnmatch(name, o) for o in omit_data]):
                if verbose:
                    print(f"Omitted {name} from export")
                return

            if isinstance(node, h5py.Dataset):
                if node.dtype.kind == "O":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore",
                            category=FutureWarning,
                            message="Direct access to this field is deprecated",
                        )

                        _write_cropped_metadata(
                            lk_file, out_file, name, node, crop_time_range, verbose
                        )
                else:
                    _write_numerical_data(
                        lk_file, out_file, name, node, compression_level, crop_time_range, verbose
                    )

            else:
                out_file.create_group(f"{name}")
                out_file[name].attrs.update(node.attrs)

        h5_file.visititems(traversal_function)
        out_file.attrs.update(h5_file.attrs)
