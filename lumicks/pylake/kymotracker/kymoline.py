import numpy as np


def export_kymolinegroup_to_csv(filename, kymoline_group, dt, dx, delimiter, sampling_width):
    """Export KymoLineGroup to a csv file.

    Parameters
    ----------
    filename : str
        Filename to output KymoLineGroup to.
    kymoline_group : KymoLineGroup
        Kymograph traces to export.
    dt : float
        Calibration for the time axis.
    dx : float
        Calibration for the coordinate axis.
    delimiter : str
        Which delimiter to use in the csv file.
    sampling_width : int or None
        When supplied, this will sample the source image around the kymograph line and export the summed intensity with
        the image. The value indicates the number of pixels in either direction to sum over.
    """
    if not kymoline_group:
        raise RuntimeError("No kymograph traces to export")

    idx = np.hstack([np.full(len(line), idx) for idx, line in enumerate(kymoline_group)])
    coords_idx = np.hstack([line.coordinate_idx for line in kymoline_group])
    times_idx = np.hstack([line.time_idx for line in kymoline_group])

    data, header, fmt = [], [], []

    def store_column(column_title, format_string, new_data):
        data.append(new_data)
        header.append(column_title)
        fmt.append(format_string)

    store_column("line index", "%d", idx)
    store_column("time (pixels)", "%.18e", times_idx)
    store_column("coordinate (pixels)", "%.18e", coords_idx)

    if dt:
        store_column("time", "%.18e", [times_idx * dt])

    if dx:
        store_column("coordinate", "%.18e", [coords_idx * dx])

    if sampling_width is not None:
        store_column(f"counts (summed over {2 * sampling_width + 1} pixels)", "%d",
                     np.hstack([line.sample_from_image(sampling_width) for line in kymoline_group]))

    data = np.vstack(data).T
    np.savetxt(filename, data, fmt=fmt, header=delimiter.join(header), delimiter=delimiter)


def import_kymolinegroup_from_csv(filename, image, delimiter=';'):
    """Import kymolines from csv

    Parameters
    ----------
    filename : str
        filename to import from
    image : array_like
        2D image that these lines were tracked from
    delimiter : str
        A delimiter that delimits the column data.

    The file format contains a series of columns as follows:
    line index, time (pixels), coordinate (pixels), time (optional), coordinate (optional), sampled_counts (optional)"""
    data = np.loadtxt(filename, delimiter=delimiter)
    assert len(data.shape) == 2, "Invalid file format"
    assert data.shape[0] > 2, "Invalid file format"

    indices = data[:, 0]
    lines = np.unique(indices)
    return KymoLineGroup([KymoLine(data[indices == k, 1], data[indices == k, 2], image) for k in lines])


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

    def interpolate(self):
        """Interpolate Kymoline to whole pixel values"""
        interpolated_time = np.arange(int(np.min(self.time_idx)), int(np.max(self.time_idx)) + 1, 1)
        interpolated_coord = np.interp(interpolated_time, self.time_idx, self.coordinate_idx)
        return KymoLine(interpolated_time, interpolated_coord, self.image_data)

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


class KymoLineGroup:
    """Kymograph lines"""
    def __init__(self, kymo_lines):
        self._src = kymo_lines

    def __iter__(self):
        return self._src.__iter__()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return KymoLineGroup(self._src[item])
        else:
            return self._src[item]

    def __setitem__(self, item, value):
        raise NotImplementedError("Cannot overwrite KymoLines.")

    def __len__(self):
        return len(self._src)

    def extend(self, other):
        if isinstance(other, self.__class__):
            self._src.extend(other._src)
        elif isinstance(other, KymoLine):
            self._src.extend([other])
        else:
            raise TypeError(f"You can only extend a {self.__class__} with a {self.__class__} or "
                            f"{KymoLine}")

    def remove_lines_in_rect(self, rect):
        """Removes traces that fall in a particular region. Note that if any point on a line falls inside the selected
        region it will be removed.

        Parameters
        ----------
        rect : array_like
            Array of 2D coordinates
        """
        if rect[0][0] > rect[1][0]:
            rect[0][0], rect[1][0] = rect[1][0], rect[0][0]

        if rect[0][1] > rect[1][1]:
            rect[0][1], rect[1][1] = rect[1][1], rect[0][1]

        self._src = [line for line in self._src if not line.in_rect(rect)]

    def __repr__(self):
        return f"{self.__class__.__name__}(N={len(self._src)})"

    def save(self, filename, dt=None, dx=None, delimiter=';', sampling_width=None):
        """Export kymograph lines to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph traces to.
        dt : float
            Calibration for the time axis.
        dx : float
            Calibration for the coordinate axis.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph line and export the summed intensity
            with the image. The value indicates the number of pixels in either direction to sum over.
        """
        export_kymolinegroup_to_csv(filename, self._src, dt, dx, delimiter, sampling_width)
