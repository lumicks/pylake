from .kymo import Kymo


class Scan(Kymo):
    """A confocal scan exported from Bluelake

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing kymo information
    file : lumicks.hdf5.File
        The parent file. Used to loop up channel data
    """
    def __init__(self, h5py_dset, file):
        super().__init__(h5py_dset, file)

    @property
    def red_image(self):
        return super().red_image.T

    @property
    def green_image(self):
        return super().green_image.T

    @property
    def blue_image(self):
        return super().blue_image.T

    def _plot(self, image, **kwargs):
        import matplotlib.pyplot as plt

        x_um = self.json["scan volume"]["scan axes"][0]["scan width (um)"]
        y_um = self.json["scan volume"]["scan axes"][1]["scan width (um)"]
        default_kwargs = dict(
            extent=[0, x_um, 0, y_um],
            aspect=(image.shape[0] / image.shape[1]) * (x_um / y_um)
        )

        plt.imshow(image, **{**default_kwargs, **kwargs})
        plt.xlabel(r"x ($\mu$m)")
        plt.ylabel(r"y ($\mu$m)")
        plt.title(self.name)
