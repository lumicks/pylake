import numpy as np

from .kymo import Kymo
from .detail.image import reconstruct_image, reconstruct_num_frames


class Scan(Kymo):
    """A confocal scan exported from Bluelake

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing kymo information
    file : lumicks.pylake.File
        The parent file. Used to loop up channel data
    """
    def __init__(self, h5py_dset, file):
        super().__init__(h5py_dset, file)
        self._num_frames = self.json["scan count"]
        if len(self.json["scan volume"]["scan axes"]) > 2:
            raise RuntimeError("3D scans are not supported")

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels=({self.pixels_per_line}, {self.lines_per_frame}))"

    @property
    def num_frames(self):
        if self._num_frames == 0:
            self._num_frames = reconstruct_num_frames(self.infowave.data, self.pixels_per_line,
                                                      self.lines_per_frame)
        return self._num_frames

    @property
    def lines_per_frame(self):
        return self.json["scan volume"]["scan axes"][1]["num of pixels"]

    def _image(self, color):
        if color not in self._cache:
            photon_counts = getattr(self, f"{color}_photon_count").data
            self._cache[color] = reconstruct_image(photon_counts, self.infowave.data,
                                                   self.pixels_per_line, self.lines_per_frame)
        return self._cache[color]

    def _timestamps(self, sample_timestamps):
        return reconstruct_image(sample_timestamps, self.infowave.data, self.pixels_per_line,
                                 self.lines_per_frame, reduce=np.mean)

    def _plot(self, image, **kwargs):
        import matplotlib.pyplot as plt

        frame = np.clip(kwargs.pop("frame", 1), 1, self.num_frames)
        if self.num_frames != 1:
            image = image[frame - 1]

        x_um = self.json["scan volume"]["scan axes"][0]["scan width (um)"]
        y_um = self.json["scan volume"]["scan axes"][1]["scan width (um)"]
        default_kwargs = dict(
            extent=[0, x_um, 0, y_um],
            aspect=(image.shape[0] / image.shape[1]) * (x_um / y_um)
        )

        plt.imshow(image, **{**default_kwargs, **kwargs})
        plt.xlabel(r"x ($\mu$m)")
        plt.ylabel(r"y ($\mu$m)")
        if self.num_frames == 1:
            plt.title(self.name)
        else:
            plt.title(f"{self.name} [frame {frame}/{self.num_frames}]")
