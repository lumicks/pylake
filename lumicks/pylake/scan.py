import json
import numpy as np

from .kymo import Kymo
from .detail.image import reconstruct_image, reconstruct_num_frames


class Scan(Kymo):
    """A confocal scan exported from Bluelake

    Parameters
    ----------
    name : str
        Kymograph name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing kymograph-specific metadata.
    """
    def __init__(self, name, file, start, stop, json):
        super().__init__( name, file, start, stop, json)
        self._num_frames = self.json["scan count"]
        if len(self.json["scan volume"]["scan axes"]) > 2:
            raise RuntimeError("3D scans are not supported")

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels=({self.pixels_per_line}, {self.lines_per_frame}))"

    def __getitem__(self, item):
        raise NotImplementedError("Indexing and slicing are not implemented for scans")

    @property
    def num_frames(self):
        if self._num_frames == 0:
            self._num_frames = reconstruct_num_frames(self.infowave.data, self.pixels_per_line,
                                                      self.lines_per_frame)
        return self._num_frames

    @property
    def lines_per_frame(self):
        return self._get_axis_metadata(1)["num of pixels"]

    def _image(self, color):
        if color not in self._cache:
            photon_counts = getattr(self, f"{color}_photon_count").data
            self._cache[color] = reconstruct_image(photon_counts, self.infowave.data,
                                                   self.pixels_per_line, self.lines_per_frame)
        return self._cache[color]

    def _timestamps(self, sample_timestamps):
        return reconstruct_image(sample_timestamps, self.infowave.data, self.pixels_per_line,
                                 self.lines_per_frame, reduce=np.mean)

    def _plot(self, image, frame=1, **kwargs):
        import matplotlib.pyplot as plt

        frame = np.clip(frame, 1, self.num_frames)
        if self.num_frames != 1:
            image = image[frame - 1]

        x_um = self._get_axis_metadata(0)["scan width (um)"]
        y_um = self._get_axis_metadata(1)["scan width (um)"]
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
