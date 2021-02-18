import h5py
from fnmatch import fnmatch


def write_h5(h5_file, output_filename, compression_level=5, omit_data={}):
    """Write a modified h5 file to disk.

    Parameters
    ----------
    h5_file : h5py.File
        loaded h5 file
    output_filename : str
        Output file name.
    compression_level : int
        Compression level for gzip compression.
    omit_data : Set[str]
        Which data sets to omit. Should be a set of h5 paths.
    """

    with h5py.File(output_filename, "w") as out_file:

        def traversal_function(name, node):
            if any([fnmatch(name, o) for o in omit_data]):
                print(f"Omitted {name} from export")
                return

            if isinstance(node, h5py.Dataset):
                if node.dtype.kind == "O":
                    # Non-numerical data doesn't support compression
                    out_file.create_dataset(name, data=node)
                else:
                    # Numerical data can benefit a lot from compression
                    out_file.create_dataset(
                        name, data=node, compression="gzip", compression_opts=compression_level
                    )
            else:
                out_file.create_group(f"{name}")

            out_file[name].attrs.update(node.attrs)

        h5_file.visititems(traversal_function)
        out_file.attrs.update(h5_file.attrs)
