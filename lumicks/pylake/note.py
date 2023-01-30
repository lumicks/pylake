import json


class Note:
    def __init__(self, file, note_data, json):
        self.file = file
        self.start = note_data["Start time (ns)"]
        self.stop = note_data["Stop time (ns)"]
        self.name = json["name"]
        self.text = json["Note text"]

    @staticmethod
    def from_dataset(h5py_dset, file):
        """
        Construct Note class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing Note information
        file : lumicks.pylake.File
            The parent file.
        """
        return Note(file, h5py_dset.attrs, json.loads(h5py_dset[()]))
