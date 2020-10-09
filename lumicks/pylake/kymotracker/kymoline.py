import numpy as np
import matplotlib.pyplot as plt


class KymoLine:
    """A line on a kymograph"""
    __slots__ = ['time_idx', 'coordinate_idx', 'image_data']

    def __init__(self, time_idx, coordinate_idx, image_data=None):
        self.time_idx = list(time_idx)
        self.coordinate_idx = list(coordinate_idx)
        self.image_data = image_data

    def append(self, time_idx, coordinate_idx):
        """Append time, coordinate pair to the KymoLine"""
        self.time_idx.append(time_idx)
        self.coordinate_idx.append(coordinate_idx)

    def with_offset(self, time_offset, coordinate_offset):
        """Returns an offset version of the KymoLine"""
        return KymoLine([time_idx + time_offset for time_idx in self.time_idx],
                        [coordinate_idx + coordinate_offset for coordinate_idx in self.coordinate_idx], self.image_data)

    def __add__(self, other):
        """Concatenate two KymoLines"""
        return KymoLine(self.time_idx + other.time_idx, self.coordinate_idx + other.coordinate_idx, self.image_data)

    def __getitem__(self, item):
        return np.squeeze(np.array(np.vstack((self.time_idx[item], self.coordinate_idx[item]))).transpose())

    def in_rect(self, rect):
        """Check whether any point of this KymoLine falls in the rect given in rect.

        Parameters
        ----------
        rect : Tuple[Tuple[float, float], Tuple[float, float]]
            Only perform tracking over a subset of the image. Pixel coordinates should be given as:
            ((min_time, min_coord), (max_time, max_coord)).
        """
        time_idx = np.array(self.time_idx)
        coordinate_idx = np.array(self.coordinate_idx)
        time_match = np.logical_and(time_idx < rect[1][0], time_idx >= rect[0][0])
        coord_match = np.logical_and(coordinate_idx < rect[1][1], coordinate_idx >= rect[0][1])
        return np.any(np.logical_and(time_match, coord_match))

    def sample_from_image(self, num_pixels, reduce=np.sum):
        """Sample from image using coordinates from this KymoLine.

        This function samples data from the image given in data based on the points in this KymoLine. It samples
        from [time, position - num_pixels : position + num_pixels + 1] and then applies the function sum.

        Parameters
        ----------
        num_pixels : int
            Number of pixels in either direction to include in the sample
        reduce : callable
            Function evaluated on the sample. (Default: np.sum which produces sum of photon counts).
        """
        if self.image_data is None:
            raise RuntimeError("No image data associated with this KymoLine")

        y_size = self.image_data.shape[1]

        # Time and coordinates are being cast to an integer since we use them to index into a data array.
        return [reduce(self.image_data[max(int(c) - num_pixels, 0):min(int(c) + num_pixels + 1, y_size), int(t)])
                for t, c in zip(self.time_idx, self.coordinate_idx)]

    def extrapolate(self, forward, n_estimate, extrapolation_length):
        """This function linearly extrapolates a track segment towards positive time.

        Parameters
        ----------
        forward: boolean
            extrapolate forward (True) or backward in time (False)
        n_estimate: int
            Number of points to use for linear regression.
        extrapolation_length: float
            How far to extrapolate.
        """
        assert n_estimate > 1, "Too few time points to extrapolate"
        assert len(self.time_idx) > 1, "Cannot extrapolate linearly with less than one time point"

        time_idx = np.array(self.time_idx)
        coordinate_idx = np.array(self.coordinate_idx)

        if forward:
            coeffs = np.polyfit(time_idx[-n_estimate:], coordinate_idx[-n_estimate:], 1)
            return np.array([time_idx[-1] + extrapolation_length,
                             coordinate_idx[-1] + coeffs[0] * extrapolation_length])
        else:
            coeffs = np.polyfit(time_idx[:n_estimate], coordinate_idx[:n_estimate], 1)
            return np.array([time_idx[0] - extrapolation_length,
                             coordinate_idx[0] - coeffs[0] * extrapolation_length])

    def __len__(self):
        return len(self.coordinate_idx)
