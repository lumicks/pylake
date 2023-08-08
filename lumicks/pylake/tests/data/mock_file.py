import json

import numpy as np

# We generate mock data files for different versions of the Bluelake HDF5 file
# format:


class MockDataFile_v1:
    def __init__(self, file):
        import h5py

        self.file = h5py.File(file, "w")

    def get_file_format_version(self):
        return 1

    def write_metadata(self):
        self.file.attrs["Bluelake version"] = "unknown"
        self.file.attrs["File format version"] = self.get_file_format_version()
        self.file.attrs["Experiment"] = "test"
        self.file.attrs["Description"] = "test"
        self.file.attrs["GUID"] = "invalid"
        self.file.attrs["Export time (ns)"] = -1

    def make_continuous_channel(self, group, name, start, dt, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group][name] = data
        dset = self.file[group][name]
        dset.attrs["Start time (ns)"] = start
        dset.attrs["Stop time (ns)"] = start + len(data) * dt
        dset.attrs["Sample rate (Hz)"] = 1 / dt * 1e9
        return dset

    def make_timeseries_channel(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        compound_type = np.dtype([("Timestamp", np.int64), ("Value", float)])
        self.file[group][name] = np.array(data, compound_type)
        dset = self.file[group][name]
        return dset

    def make_timetags_channel(self, group, name, data):
        raise NotImplementedError


class MockDataFile_v2(MockDataFile_v1):
    # CAVE: Please respect that `MockDataFile_v2` is used by another Lumicks project

    def get_file_format_version(self):
        return 2

    def make_calibration_data(self, calibration_idx, group, attributes):
        if "Calibration" not in self.file:
            self.file.create_group("Calibration")

        # Numeric value converted to string
        if calibration_idx not in self.file["Calibration"]:
            self.file["Calibration"].create_group(calibration_idx)

        # e.g. Force 1x, Force 1y ... etc
        if group not in self.file["Calibration"][calibration_idx]:
            self.file["Calibration"][calibration_idx].create_group(group)

        # Attributes
        field = self.file["Calibration"][calibration_idx][group]
        for i, v in attributes.items():
            field.attrs[i] = v

    def make_fd(self, fd_name=None, metadata={}, attributes={}):
        if "FD Curve" not in self.file:
            self.file.create_group("FD Curve")

        if fd_name:
            dset = self.file["FD Curve"].create_dataset(fd_name, data=metadata)
            for i, v in attributes.items():
                dset.attrs[i] = v

    def make_marker(self, marker_name, attributes, payload=None):
        if "Marker" not in self.file:
            self.file.create_group("Marker")

        if marker_name not in self.file["Marker"]:
            payload_string = f', "payload":{payload}' if payload else ""
            dset = self.file["Marker"].create_dataset(
                marker_name, data=f'{{"name":"{marker_name}"{payload_string}}}'
            )

            for i, v in attributes.items():
                dset.attrs[i] = v

    def make_note(self, note_name, attributes, note_text):
        if "Note" not in self.file:
            self.file.create_group("Note")

        if note_name not in self.file["Note"]:
            payload = {"name": note_name, "Note text": note_text}
            dset = self.file["Note"].create_dataset(note_name, data=json.dumps(payload))

            for i, v in attributes.items():
                dset.attrs[i] = v

    def make_continuous_channel(self, group, name, start, dt, data):
        dset = super().make_continuous_channel(group, name, start, dt, data)
        dset.attrs["Kind"] = "Continuous"

    def make_timeseries_channel(self, group, name, data):
        dset = super().make_timeseries_channel(group, name, data)
        dset.attrs["Kind"] = b"TimeSeries"

    def make_json_data(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group].create_dataset(name, data=data)
        return self.file[group][name]

    def make_timetags_channel(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group][name] = data
        dset = self.file[group][name]
        dset.attrs["Kind"] = "TimeTags"
        return dset
