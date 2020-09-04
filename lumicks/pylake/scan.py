import numpy as np

from .kymo import axis_label
from .kymo import Kymo
from .detail.image import reconstruct_image_sum, reconstruct_image, reconstruct_num_frames


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
        return self.json["scan volume"]["scan axes"][1]["num of pixels"]

    def _to_spatial(self, data):
        """If the first axis of the reconstruction has a higher physical axis number than the second, we flip the axes.

        Checks whether the axes should be flipped w.r.t. the reconstruction. Reconstruction always produces images
        with the slow axis first, and the fast axis second. Depending on the order of axes scanned, this may not
        coincide with physical axes. The axes should always be ordered from the lowest physical axis number to higher.
        Here X, Y, Z correspond to axis number 0, 1 and 2. So for an YZ scan, we'd want Y on the X axis."""
        physical_axis = [axis["axis"] for axis in self.json["scan volume"]["scan axes"]]
        if physical_axis[0] > physical_axis[1]:
            new_axis_order = np.arange(len(data.shape), dtype=np.int)
            new_axis_order[-1], new_axis_order[-2] = new_axis_order[-2], new_axis_order[-1]
            return np.transpose(data, new_axis_order)
        else:
            return data

    def _image(self, color):
        if color not in self._cache:
            photon_counts = getattr(self, f"{color}_photon_count").data
            self._cache[color] = reconstruct_image_sum(photon_counts, self.infowave.data, self.pixels_per_line,
                                                       self.lines_per_frame)
            self._cache[color] = self._to_spatial(self._cache[color])

        return self._cache[color]

    def _timestamps(self, sample_timestamps):
        timestamps = reconstruct_image(sample_timestamps, self.infowave.data, self.pixels_per_line,
                                       self.lines_per_frame, reduce=np.mean)

        return self._to_spatial(timestamps)

    def _plot(self, image, frame=1, **kwargs):
        import matplotlib.pyplot as plt

        frame = np.clip(frame, 1, self.num_frames)
        if self.num_frames != 1:
            image = image[frame - 1]

        axes = self._ordered_axes()
        x_um = axes[0]["scan width (um)"]
        y_um = axes[1]["scan width (um)"]
        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            extent=[0, x_um, y_um, 0],
            aspect=(image.shape[0] / image.shape[1]) * (x_um / y_um)
        )

        plt.imshow(image, **{**default_kwargs, **kwargs})
        plt.xlabel(f"{axis_label[axes[0]['axis']]} ($\\mu m$)")
        plt.ylabel(f"{axis_label[axes[1]['axis']]} ($\\mu m$)")
        if self.num_frames == 1:
            plt.title(self.name)
        else:
            plt.title(f"{self.name} [frame {frame}/{self.num_frames}]")
