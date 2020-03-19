class Marker:
    def __init__(self, file, marker_data):
        self.file = file
        self.start = marker_data["Start time (ns)"]
        self.stop = marker_data["Stop time (ns)"]

    @staticmethod
    def from_dataset(h5py_dset, file):
        """
        Construct Marker class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing Marker information
        file : lumicks.pylake.File
            The parent file.
        """
        return Marker(file, h5py_dset.attrs)
