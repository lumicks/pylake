import re
import pathlib
import itertools
from copy import copy

from ..__about__ import __version__
from ..detail.utilities import replace_key_aliases
from .detail.peakfinding import _sum_track_signal
from ..population.dwelltime import DwelltimeModel
from .detail.msd_estimation import *
from .detail.localization_models import LocalizationModel, CentroidLocalizationModel


def _read_txt(file, delimiter):
    """Read text-based csv file with KymoTrack data

    Parameters
    ----------
    file : str | os.PathLike | StringIO
        path to a file to read, or handle to read from directly
    delimiter : str
        Delimiter to use.

    Raises
    ------
    IOError
        If the file is in the wrong format.
    """
    try:
        with open(pathlib.Path(file), "r") as f:
            header_lines = [f.readline(), f.readline()]
    except TypeError:
        # Direct StringIO
        header_lines = [file.readline(), file.readline()]

    # from v0.13.0, exported CSV files have an additional header line
    # with the pylake version and CSV version (starting at 2)
    version_header = re.search(
        r"Exported with pylake v([\d\.]*) \| track coordinates v(\d)", header_lines[0]
    )

    # We had a version header
    if version_header:
        header_lines = header_lines[1:]  # Consumed the header line
        pylake_version = version_header.group(1)
        csv_version = int(version_header.group(2))
    else:
        pylake_version = None
        csv_version = 1

    data = {}
    try:
        raw_data = np.loadtxt(file, delimiter=delimiter, unpack=True)
    except ValueError:
        raise IOError("Invalid file format!")

    header = header_lines[0].rstrip().split(delimiter)
    track_idx = raw_data[0, :]
    unique_tracks = np.unique(track_idx)
    for key, col in zip(header, raw_data):
        data[key] = [col[np.argwhere(track_idx == idx).flatten()] for idx in unique_tracks]

    return data, pylake_version, csv_version


def export_kymotrackgroup_to_csv(
    filename, kymotrack_group, delimiter, sampling_width, *, correct_origin=None
):
    """Export KymoTrackGroup to a csv file.

    Parameters
    ----------
    filename : str | os.PathLike
        Filename to output KymoTrackGroup to.
    kymotrack_group : KymoTrackGroup
        Kymograph tracks to export.
    delimiter : str
        Which delimiter to use in the csv file.
    sampling_width : int or None
        If specified, this will sample the source image around the kymograph track and export the
        summed intensity with the image. The value indicates the number of pixels in either
        direction to sum over.
    correct_origin : bool, optional
        Use the correct pixel origin when sampling from image. Kymotracks are defined with the
        origin of each image pixel defined at the center. Earlier versions of the method that
        samples photon counts around the track had a bug which assumed the origin at the edge
        of the pixel. Setting this flag to `True` produces the correct behavior. The default is
        set to `None` which reproduces the old behavior and results in a warning, while `False`
        reproduces the old behavior without a warning.

    Raises
    ------
    NotImplementedError
        If group contains tracks from more than one source kymograph.
    """
    if not kymotrack_group:
        raise RuntimeError("No kymograph tracks to export")

    kymotrack_group._validate_single_source(
        "Exporting a group with tracks from more than a single source kymograph"
    )

    time_units = "seconds"
    position_units = kymotrack_group._calibration_info["unit"]

    idx = np.hstack([np.full(len(track), idx) for idx, track in enumerate(kymotrack_group)])
    coords_idx = np.hstack([track.coordinate_idx for track in kymotrack_group])
    times_idx = np.hstack([track.time_idx for track in kymotrack_group])

    position = np.hstack([track.position for track in kymotrack_group])
    seconds = np.hstack([track.seconds for track in kymotrack_group])
    minimum_observable_duration = np.hstack(
        [np.full(len(track), track._minimum_observable_duration) for track in kymotrack_group]
    )

    data, column_titles, fmt = [], [], []

    def store_column(column_title, format_string, new_data):
        data.append(new_data)
        column_titles.append(column_title)
        fmt.append(format_string)

    store_column("track index", "%d", idx)
    store_column("time (pixels)", "%.18e", times_idx)
    store_column("coordinate (pixels)", "%.18e", coords_idx)

    store_column(f"time ({time_units})", "%.18e", seconds)
    store_column(f"position ({position_units})", "%.18e", position)

    if sampling_width is not None:
        store_column(
            f"counts (summed over {2 * sampling_width + 1} pixels)",
            "%d",
            np.hstack(
                [
                    track.sample_from_image(sampling_width, correct_origin=correct_origin)
                    for track in kymotrack_group
                ]
            ),
        )

    if None not in minimum_observable_duration:
        store_column(
            f"minimum observable duration ({time_units})", "%.6e", minimum_observable_duration
        )
    else:
        warnings.warn(
            RuntimeWarning(
                "Loaded tracks have no minimum length metadata defined (CSV exported with "
                "pylake <  1.2.0) therefore some analyses may not be possible with this file. To "
                "remedy this error and add the required metadata, use `lk.filter_tracks()` on the "
                "tracks from the widget with the desired minimum track length and resave "
                "using the `KymoTrackGroup.save()` method. Note that the minimum length should be "
                "greater or equal to the minimum length used for the original tracking. For more "
                "information refer to "
                "https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html#migrating-old-track-files "
            )
        )

    version_header = f"Exported with pylake v{__version__} | track coordinates v4\n"
    header = version_header + delimiter.join(column_titles)
    data = np.vstack(data).T
    np.savetxt(filename, data, fmt=fmt, header=header, delimiter=delimiter)


def _check_summing_mismatch(track, sampling_width):
    """Checks calling sample_from_image on a loaded track reproduces the same result as is in the
    file."""
    wrong_kymo_warning = (
        "Photon counts do not match the photon counts found in the file. It is "
        "possible that the loaded kymo or channel doesn't match the one used to "
        "create this file."
    )
    try:
        if not np.allclose(
            track.sample_from_image((sampling_width - 1) // 2, correct_origin=True),
            track.photon_counts,
        ):
            if np.allclose(
                track.sample_from_image((sampling_width - 1) // 2, correct_origin=False),
                track.photon_counts,
            ):
                return RuntimeWarning(
                    "Photon counts do not match the photon counts found in the file. Prior to "
                    "Pylake v1.1.0, the method `sample_from_image` had a bug that assumed the "
                    "origin of a pixel to be at the edge rather than the center of the pixel. "
                    "Consequently, the sampled window could be off by one pixel. This file was "
                    "likely created using the incorrect origin. "
                    "Note that Pylake loaded the counts found in the file as is, so if the "
                    "used summing window was very small, there may be a bias in the counts."
                    "To recreate these counts without bias invoke:"
                    f"`track.sample_from_image({(sampling_width - 1) // 2}, correct_origin=True)`"
                )
            else:
                return RuntimeWarning(wrong_kymo_warning)
    except IndexError:
        return RuntimeWarning(wrong_kymo_warning)


def import_kymotrackgroup_from_csv(filename, kymo, channel, delimiter=";"):
    """Import a KymoTrackGroup from a csv file.

    The file format contains a series of columns as follows:
    track index, time (pixels), coordinate (pixels), time (optional), coordinate (optional),
    sampled_counts (optional), minimum length

    Parameters
    ----------
    filename : str | os.PathLike
        filename to import from.
    kymo : Kymo
        kymograph instance that the CSV data was tracked from.
    channel : str
        color channel that was used for tracking.
    delimiter : str
        The string used to separate columns.

    Returns
    -------
    kymotrack_group : KymoTrackGroup
        Tracked kymograph lines that were stored in the file.

    Raises
    ------
    IOError
        If the file format is not as expected.
    """

    # TODO: File format validation could use some improvement
    data, pylake_version, csv_version = _read_txt(filename, delimiter)

    mandatory_fields = ["time (pixels)", "coordinate (pixels)"]
    if not all(field_name in data for field_name in mandatory_fields):
        raise IOError("Invalid file format!")

    # We get a list of time coordinates per track
    for track_time in data["time (pixels)"]:
        if np.any(np.floor(track_time) != track_time):
            warnings.warn(
                RuntimeWarning(
                    "File contains non-integer time indices; round-off errors may have occurred "
                    "when loading the data"
                ),
                stacklevel=2,
            )

    def create_track(time, coord, min_length=None, counts=None):
        if min_length is not None:
            min_length = float(np.unique(min_length).squeeze())

        if counts is not None:
            coord = CentroidLocalizationModel(coord * kymo.pixelsize_um, counts)

        return KymoTrack(time.astype(int), coord, kymo, channel, min_length)

    if csv_version == 3:
        warnings.warn(
            RuntimeWarning(
                "This CSV file is from a version in which the minimum observable track duration "
                "was incorrectly rounded to an integer. This may result in errors in certain "
                "analyses. To remedy this error, use `lk.filter_tracks()` on the "
                "tracks from the widget with the desired minimum track length and resave "
                "using the `KymoTrackGroup.save()` method. Note that the minimum length should be "
                "greater or equal to the minimum length used for the original tracking. For more "
                "information refer to "
                "https://lumicks-pylake.readthedocs.io/en/latest/tutorial/nbwidgets.html#migrating-old-track-files"
            )
        )
        min_duration_field = "minimum_length (-)"
    else:
        min_duration_field = "minimum observable duration (seconds)"

    count_field = [key for key in data.keys() if "counts" in key]
    sampling_width = None
    if count_field:
        count_field = count_field[0]
        if match := re.findall(r"over (\d*) pixels", count_field):
            sampling_width = int(match[0])

    tracks = []
    resampling_mismatch = None
    for track_idx in range(len(data[mandatory_fields[0]])):
        tracks.append(
            create_track(
                time=data[mandatory_fields[0]][track_idx],
                coord=data[mandatory_fields[1]][track_idx],
                min_length=data[min_duration_field][track_idx]
                if min_duration_field in data
                else None,
                counts=data[count_field][track_idx] if count_field else None,
            )
        )

        if sampling_width is not None:
            resampling_mismatch = _check_summing_mismatch(tracks[-1], sampling_width)

    if resampling_mismatch:
        warnings.warn(resampling_mismatch, stacklevel=2)

    return KymoTrackGroup(tracks)


class KymoTrack:
    """A tracked particle on a kymograph.

    Parameters
    ----------
    time_idx : array_like
        Frame time indices. Note that these should be of integer type.
    localization : LocalizationModel or array_like
        LocalizationModel instance containing localization parameters
        or list of (sub)pixel coordinates to be converted to spatial
        position via calibration with pixel size.
    kymo : Kymo
        Kymograph instance.
    channel : {"red", "green", "blue"}
        Color channel to analyze.
    minimum_observable_duration : float or None
        Minimum observable dwelltime when this track was tracked. Note that filtering may change
        this minimum time.

    Raises
    ------
    TypeError
        If time indices are not of integer type.
    """

    __slots__ = ["_time_idx", "_localization", "_kymo", "_channel", "_minimum_observable_duration"]

    def __init__(self, time_idx, localization, kymo, channel, minimum_observable_duration):
        self._kymo = kymo
        self._channel = channel
        self._time_idx = np.asarray(time_idx)
        self._minimum_observable_duration = minimum_observable_duration

        if np.any(self._time_idx) and not np.issubdtype(self._time_idx.dtype, np.integer):
            raise TypeError(f"Time indices should be of integer type, got {self._time_idx.dtype}.")

        self._localization = (
            localization
            if isinstance(localization, LocalizationModel)
            else LocalizationModel(np.array(localization) * self._pixelsize)
        )

    @property
    def _image(self):
        return self._kymo.get_image(self._channel)

    @property
    def photon_counts(self):
        """Photon counts

        Provides an estimate of the photon counts for each point in the track. The value is
        calculated differently depending on the refinement method:

        - for a tracks originating from tracking without additional refinement or from centroid
          refinement, the counts are estimated by summing an odd number of pixels around the peak
          position.
        - for tracks originating from Gaussian refinement, the photon count estimate is given
          by the fitted integrated area of the peak.
        """
        try:
            return self._localization.total_photons
        except AttributeError:
            raise AttributeError("Photon counts are unavailable for this KymoTrack.")

    def __str__(self):
        return f"KymoTrack(N={len(self._time_idx)})"

    def _with_coordinates(self, time_idx, localization):
        """Return a copy of the KymoTrack with new spatial/temporal coordinates."""
        return KymoTrack(
            time_idx,
            localization,
            self._kymo,
            self._channel,
            self._minimum_observable_duration,
        )

    def _with_minimum_time(self, minimum_observable_duration):
        return KymoTrack(
            self._time_idx,
            self._localization,
            self._kymo,
            self._channel,
            minimum_observable_duration,
        )

    @classmethod
    def _from_centroid_estimate(
        cls, time_idx, coordinate_idx, kymo, channel, half_width_pixels, minimum_observable_duration
    ):
        """Return a KymoTrack including sampled photon counts.

        Parameters
        ----------
        time_idx : array_like
            Frame time indices. Note that these should be of integer type.
        coordinate_idx : numpy.ndarray
            (Sub)pixel coordinates to be converted to spatial position via calibration with pixel
            size.
        kymo : Kymo
            Kymograph instance.
        channel : {"red", "green", "blue"}
            Color channel to analyze.
        half_width_pixels : int
           Number of pixels in either direction to include in the photon count sum.
        minimum_observable_duration : float
            Minimum observable time.

        Raises
        ------
        TypeError
            If frame time indices are not of integer type.
        """
        return cls(
            time_idx,
            CentroidLocalizationModel(
                coordinate_idx * kymo.pixelsize[0],
                np.array(
                    _sum_track_signal(
                        kymo.get_image(channel),
                        half_width_pixels,
                        time_idx,
                        coordinate_idx,
                        correct_origin=True,
                    )
                ),
            ),
            kymo,
            channel,
            minimum_observable_duration,
        )

    def _flip(self, kymo):
        """Return a copy flipped vertically.

        Parameters
        ----------
        kymo : lumicks.pylake.Kymo
            Flipped kymograph
        """

        return KymoTrack(
            self.time_idx,
            self._localization._flip(self._kymo.pixelsize[0] * (self._kymo.pixels_per_line - 1)),
            kymo,
            self._channel,
            self._minimum_observable_duration,
        )

    def with_offset(self, time_offset, coordinate_offset):
        """Returns an offset version of the KymoTrack"""
        # Convert from image units to (integer rounded toward zero) pixels
        time_pixel_offset = int(time_offset / self._line_time_seconds)
        coordinate_pixel_offset = int(coordinate_offset / self._pixelsize)

        return self._with_coordinates(
            self.time_idx + time_pixel_offset,
            self.coordinate_idx + coordinate_pixel_offset,
        )

    def __add__(self, other):
        """Concatenate two KymoTracks"""
        try:
            localization = self._localization + other._localization
        except TypeError:
            # If one of them is a refined localization and the other is not, fall back to
            # a non-refined localization.
            localization = np.hstack((self.coordinate_idx, other.coordinate_idx))

        return self._with_coordinates(np.hstack((self.time_idx, other.time_idx)), localization)

    def __getitem__(self, item):
        return np.squeeze(
            np.array(np.vstack((self.time_idx[item], self.coordinate_idx[item]))).transpose()
        )

    @property
    def time_idx(self):
        """Return temporal coordinates in units of pixels."""
        return self._time_idx

    @property
    def coordinate_idx(self):
        """Return spatial coordinates in units of pixels.

        Coordinates are defined w.r.t. pixel centers (i.e. 0, 0 is the center of the first pixel).
        """
        return self._localization.position / self._kymo.pixelsize[0]

    @property
    def seconds(self):
        """The tracked temporal coordinates in seconds."""
        return self._line_time_seconds * self.time_idx

    @property
    def position(self):
        """The tracked positional coordinates. The units depend on the spatial units of the tracked kymo."""
        return self._localization.position

    @property
    def duration(self):
        """Track duration in seconds"""
        return self.seconds[-1] - self.seconds[0]

    @property
    def _line_time_seconds(self):
        """Source kymograph line time in seconds."""
        return self._kymo.line_time_seconds

    @property
    def _pixelsize(self):
        """Kymograph (spatial) pixel size in physical units."""
        return self._kymo.pixelsize[0]

    def _model_fit(self, node_idx):
        """Extract fitted model from localization model at particular node if available"""
        num_coords = len(self)
        node_idx = num_coords + node_idx if node_idx < 0 else node_idx
        if not 0 <= node_idx < num_coords:
            raise IndexError(
                f"Node index is out of range of the KymoTrack. Kymotrack has length {num_coords}."
            )

        pixelsize, num_pixels = self._kymo.pixelsize[0], self._kymo.pixels_per_line
        coords = np.arange(0.0, num_pixels * pixelsize, 0.1 * pixelsize)
        model = self._localization.evaluate(coords, node_idx, pixelsize)

        return coords, model

    def plot_fit(self, node_idx, *, fit_kwargs=None, data_kwargs=None, show_data=True):
        """Plot the localization model

        Plots the model fit for a particular localization. Note that currently, this fit is only
        available for Gaussian refinement.

        Parameters
        ----------
        node_idx : int
            Index of the track node to plot (corresponding to a specific scan line).
        fit_kwargs : dict
            Dictionary containing arguments to be passed to the fit plot.
        data_kwargs : dict
            Dictionary containing arguments to be passed to the data plot.
        show_data : bool
            Plot the data.

        Raises
        ------
        NotImplementedError
            if localization was not performed using Gaussian localization
        IndexError
            if requesting a fit for an index out of bounds of the KymoTrack.
        """
        import matplotlib.pyplot as plt

        model_fit = self._model_fit(node_idx)  # Verifies whether model exists for this index
        aliases = ["color", "c"]
        if show_data:
            plt.plot(
                np.arange(0.0, self._kymo.pixels_per_line) * self._kymo.pixelsize[0],
                self._image[:, self.time_idx[node_idx]],
                **{"color": "#c8c8c8"} | replace_key_aliases(data_kwargs or {}, aliases),
            )

        plt.plot(*model_fit, **{"color": "C0"} | replace_key_aliases(fit_kwargs or {}, aliases))
        plt.xlabel(f"Position [{self._kymo._calibration.unit_label}]")
        plt.ylabel("Photon counts [#]")

    def _check_ends_are_defined(self):
        """Checks if beginning and end of the track are not in the first/last frame."""
        return self.time_idx[0] > 0 and self.time_idx[-1] < self._image.shape[1] - 1

    def in_rect(self, rect, all_points=False):
        """Check whether points of this KymoTrack fall in the given rect.

        Parameters
        ----------
        rect : Tuple[Tuple[float, float], Tuple[float, float]]
            Coordinates should be given as ((min_time, min_coord), (max_time, max_coord)).
        all_points : bool
            Require that all points fall inside the rectangle.

        Returns
        -------
        is_inside : bool
        """
        time_match = np.logical_and(self.seconds < rect[1][0], self.seconds >= rect[0][0])
        position_match = np.logical_and(self.position < rect[1][1], self.position >= rect[0][1])
        criterion = np.all if all_points else np.any
        return criterion(np.logical_and(time_match, position_match))

    def interpolate(self):
        """Interpolate KymoTrack to whole pixel values"""
        interpolated_time = np.arange(int(np.min(self.time_idx)), int(np.max(self.time_idx)) + 1, 1)
        interpolated_coord = np.interp(interpolated_time, self.time_idx, self.coordinate_idx)
        return self._with_coordinates(interpolated_time, interpolated_coord)

    def _split(self, node):
        """Split track.

        Splits a track at `node`. Returns two tracks if successful.

        Parameters
        ----------
        node : int
            Node index to split at

        Raises
        ------
        ValueError
            If asked to split at a point that would result in an empty track.
        """
        node = np.clip(node, 0, len(self))
        before = self._with_coordinates(self.time_idx[:node], self._localization[:node])
        after = self._with_coordinates(self.time_idx[node:], self._localization[node:])

        if not before or not after:
            raise ValueError("Invalid split point. This split would result in an empty track")

        return before, after

    def sample_from_image(self, num_pixels, reduce=np.sum, *, correct_origin=None):
        """Sample from image using coordinates from this KymoTrack.

        This function samples data from the image given in data based on the points in this
        KymoTrack. It samples from `[time, position - num_pixels : position + num_pixels + 1]` and
        then applies the function sum.

        Parameters
        ----------
        num_pixels : int
            Number of pixels in either direction to include in the sample
        reduce : callable
            Function evaluated on the sample. (Default: np.sum which produces sum of photon counts).
        correct_origin : bool, optional
            Use the correct pixel origin when sampling from image. Kymotracks are defined with the
            origin of each image pixel defined at the center. Earlier versions of the method that
            samples photon counts around the track had a bug which assumed the origin at the edge
            of the pixel. Setting this flag to `True` produces the correct behavior. The default is
            set to `None` which reproduces the old behavior and results in a warning, while `False`
            reproduces the old behavior without a warning.
        """
        if correct_origin is None:
            warnings.warn(
                RuntimeWarning(
                    "Prior to version 1.1.0 the method `sample_from_image` had a bug that assumed "
                    "the origin of a pixel to be at the edge rather than the center of the pixel. "
                    "Consequently, the sampled window could frequently be off by one pixel. To get "
                    "the correct behavior and silence this warning, specify `correct_origin=True`. "
                    "The old (incorrect) behavior is maintained until the next major release to "
                    "ensure backward compatibility. To silence this warning use "
                    "`correct_origin=False`."
                ),
                stacklevel=2,
            )

        # Time and coordinates are being cast to an integer since we use them to index into a data
        # array. Note that coordinate pixel centers are defined at integer coordinates.
        return _sum_track_signal(
            self._image,
            num_pixels,
            self.time_idx,
            self.coordinate_idx,
            reduce=reduce,
            correct_origin=correct_origin,
        )

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

        Raises
        ------
        ValueError
            If there are insufficient points to extrapolate a linear curve from (two points). This
            can be due to too few points being available, or `n_estimate` being smaller than two.
        RuntimeError
            If this line consists of fewer than two points, it cannot be extrapolated.
        """
        if n_estimate < 2:
            raise ValueError("Cannot extrapolate linearly with fewer than two timepoints")

        if len(self.time_idx) < 2:
            raise RuntimeError("Cannot extrapolate linearly with fewer than two timepoints")

        time_idx = np.array(self.time_idx)
        coordinate_idx = np.array(self.coordinate_idx)

        if forward:
            coeffs = np.polyfit(time_idx[-n_estimate:], coordinate_idx[-n_estimate:], 1)
            return np.array(
                [
                    time_idx[-1] + extrapolation_length,
                    coordinate_idx[-1] + coeffs[0] * extrapolation_length,
                ]
            )
        else:
            coeffs = np.polyfit(time_idx[:n_estimate], coordinate_idx[:n_estimate], 1)
            return np.array(
                [
                    time_idx[0] - extrapolation_length,
                    coordinate_idx[0] - coeffs[0] * extrapolation_length,
                ]
            )

    def __len__(self):
        return len(self.coordinate_idx)

    def plot(self, *, show_outline=True, show_labels=True, axes=None, **kwargs):
        """A simple line plot to visualize the track coordinates.

        Parameters
        ----------
        show_outline : bool
            Whether to show black outline around the track.
        show_labels : bool
            Whether to add axes labels.
        axes : matplotlib.axes.Axes, optional
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe

        from ..detail.plotting import get_axes

        ax = get_axes(axes)

        if show_outline:
            linewidth = kwargs.get("lw", kwargs.get("linewidth", plt.rcParams["lines.linewidth"]))

            ax.plot(
                self.seconds,
                self.position,
                path_effects=[pe.Stroke(foreground="k", linewidth=linewidth * 2.5)],
                **{**kwargs, "label": "__no_label__"},
            )

        ax.plot(self.seconds, self.position, path_effects=[pe.Normal()], **kwargs)

        if show_labels:
            ax.set_ylabel(f"position ({self._kymo._calibration.unit_label})")
            ax.set_xlabel("time (s)")

    def msd(self, max_lag=None):
        r"""Estimate the Mean Square Displacement (MSD) for various time lags.

        The estimator for the MSD (:math:`\rho`) is defined as:

        .. math::
            \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n} \left(r_{i+n} - r_{i}\right)^2

        Note: This estimator leads to highly correlated estimates, which should not be treated
        as independent measurements. See [1]_ for more information.

        Parameters
        ----------
        max_lag : int
            Maximum lag to include.

        Returns
        -------
        lag_time : np.ndarray
            array of lag times
        msd : np.ndarray
            array of mean square distance estimates for the corresponding lag times

        References
        ----------
        .. [1] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
               single-particle tracking. Physical Review E, 85(6), 061916.
        """
        frame_lag, msd = calculate_msd(
            np.array(self.time_idx, dtype=int),
            np.array(self.position),
            max_lag if max_lag else len(self.time_idx),
        )

        return frame_lag * self._line_time_seconds, msd

    def plot_msd(self, max_lag=None, **kwargs):
        r"""Plot Mean Squared Differences of this trace

        The estimator for the MSD (:math:`\rho`) is defined as:

        .. math::
            \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n} \left(r_{i+n} - r_{i}\right)^2

        Note: This estimator leads to highly correlated estimates, which should not be treated
        as independent measurements. See [2]_ for more information.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to include. When omitted, an optimal number of lags is chosen [2]_.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.

        References
        ----------
        .. [2] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
               single-particle tracking. Physical Review E, 85(6), 061916.
        """
        import matplotlib.pyplot as plt

        frame_idx, positions = np.array(self.time_idx, dtype=int), np.array(self.position)
        max_lag = max_lag if max_lag else determine_optimal_points(frame_idx, positions)[0]
        lag_time, msd = self.msd(max_lag)
        plt.plot(lag_time, msd, **kwargs)
        plt.xlabel("Lag time [s]")
        plt.ylabel(f"Mean Squared Displacement [{self._kymo._calibration.unit_label}²]")

    def estimate_diffusion(
        self,
        method,
        max_lag=None,
        localization_variance=None,
        variance_of_localization_variance=None,
    ):
        r"""Estimate diffusion constant

        There are three algorithms to determine diffusion constants:

            - CVE: Covariance based estimator. Works directly on the tracks.
            - OLS: Ordinary least-squares estimator based on Mean Squared Displacements (MSD).
            - GLS: Generalized least-squares estimator based on Mean Squared Displacements (MSD).

        1. Covariance based estimator (CVE)

        The covariance-based diffusion estimator provides a simple unbiased estimator of diffusion.
        This estimator was introduced in the work of Vestergaard et al [5]_. The correction for
        missing data was introduced in [6]_. The CVE is unbiased and practically optimal when the
        signal-to-noise ratio (SNR) is bigger than 1. In this context, the SNR is defined
        as: :math:`\sqrt{D \Delta t} / \sigma`. Note that for 1D confocal scans, we neglect the
        effect of motion blur (R=0).

        2. MSD-based estimators (OLS, GLS)

        The estimator for the MSD (:math:`\rho`) is defined as:

        .. math::

            \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n} \left(r_{i+n} - r_{i}\right)^2

        In a diffusion problem, the MSD can be fitted to a linear curve.

        .. math::

            \textrm{intercept} =& 2 d (\sigma^2 - 2 R D dt)

            \textrm{slope} =& 2 d D dt

        Here :math:`d` is the dimensionality of the problem. :math:`D` is the diffusion constant.
        :math:`R` is a motion blur constant. :math:`dt` is the time step and :math:`\sigma`
        represents the dynamic localization error.

        One aspect that is import to consider is that this estimator uses every data point multiple
        times. As a consequence the elements of rho_n are highly correlated. This means that
        including more points doesn't necessarily make the estimates better and can actually make
        the estimate worse.

        There are two ways around this. Either you determine an optimal number of points to use
        in the estimation procedure (ols) [3]_ or you take into account the covariances present in
        the mean squared difference estimates (gls) [4]_.

        The standard error returned by this method is based on [4]_.

        Note that this estimation procedure should only be used for pure diffusion in the absence
        of drift.

        Parameters
        ----------
        method : {"cve", "ols", "gls"}
            - "cve" : Covariance based estimator [5]_. Optimal if SNR > 1.
            - "ols" : Ordinary least squares [3]_. Determines optimal number of lags.
            - "gls" : Generalized least squares [4]_. Takes into account covariance matrix (slower).
              Can only be used when track is equidistantly sampled.
        max_lag : int (optional)
            Number of lags to include when using an MSD-based estimator. When omitted, the method
            will choose an appropriate number of lags to use. For the cve estimator this argument
            is ignored.
            When the method chosen is "ols" an optimal number of lags is estimated as determined by
            [3]_. When the method is set to "gls" all lags are included.
        localization_variance : float (optional)
            Estimate of the localization variance. This value can be obtained from estimating an
            ensemble diffusion constant using `cve`. This parameter is only used when method="cve".
        variance_of_localization_variance : float (optional)
            Estimate of the variance of the localization variance estimate. This value can be
            obtained from estimating an ensemble diffusion constant using `cve`. This parameter is
            only used when method="cve".

        Raises
        ------
        ValueError
            if `method == "cve"`, but the `KymoTrack` has fewer than 3 time intervals defined.
        ValueError
            if `method == "cve"`, the source kymograph does not have a clearly defined motion blur
            constant, and a localization variance is supplied as an argument. As a result, the
            diffusion constant cannot be reliably calculated. In this case, do not provide a
            localization uncertainty.
        NotImplementedError
            if the tracked kymograph had disjoint time intervals (such as for a temporally
            downsampled kymograph).

        Warns
        -----
        RuntimeWarning
            if `method == "cve"` and the source kymograph does not have a clearly defined motion
            blur constant. As a result, the localization variance and standard error for the
            diffusion constant will not be available. Estimates that are unavailable are returned
            as `np.nan`.

        References
        ----------
        .. [3] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
               single-particle tracking. Physical Review E, 85(6), 061916.
        .. [4] Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of
               self-diffusion coefficients from molecular dynamics simulations. The Journal of
               Chemical Physics, 153(2), 024116.
        .. [5] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
               diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
               022726.
        .. [6] Vestergaard, C. L. (2016). Optimizing experimental parameters for tracking of
               diffusing particles. Physical Review E, 94(2), 022401.
        """
        if method not in ("cve", "gls", "ols"):
            raise ValueError('Invalid method selected. Method must be "cve", "gls" or "ols"')

        if not self._kymo.contiguous:
            raise NotImplementedError(
                "Estimating diffusion constants from data which has been integrated over disjoint "
                "sections of time is not supported. To estimate diffusion constants, do not "
                "downsample the kymograph temporally prior to tracking."
            )

        frame_idx, positions = np.array(self.time_idx, dtype=int), np.array(self.position)
        unit_labels = {
            "unit": f"{self._kymo._calibration.unit}^2 / s",
            "unit_label": f"{self._kymo._calibration.unit_label}²/s",
        }

        if method == "cve":
            try:
                blur = self._kymo.motion_blur_constant
            except NotImplementedError:
                if localization_variance:
                    raise ValueError(
                        "Cannot compute diffusion constant reliably for a kymograph that does not"
                        "have a clearly defined motion blur constant and the localization variance "
                        "is provided. Omit the localization variance to calculate a diffusion "
                        "constant."
                    )

                warnings.warn(
                    RuntimeWarning(
                        "Motion blur cannot be taken into account for this type of Kymo. As a "
                        "consequence, not all estimates will be available."
                    )
                )
                blur = np.nan

            return estimate_diffusion_cve(
                frame_idx,
                positions,
                self._line_time_seconds,
                **unit_labels,
                blur_constant=blur,
                localization_var=localization_variance,
                var_of_localization_var=variance_of_localization_variance,
            )

        if localization_variance is not None or variance_of_localization_variance is not None:
            raise NotImplementedError(
                "Passing in a localization error is only supported for method=`cve`."
            )

        max_lag = (
            max_lag
            if max_lag
            else (
                determine_optimal_points(frame_idx, positions)[0]
                if method == "ols"
                else len(frame_idx)
            )
        )

        return estimate_diffusion_constant_simple(
            frame_idx,
            positions,
            self._line_time_seconds,
            max_lag,
            method,
            **unit_labels,
        )


class KymoTrackGroup:
    """Tracks on a kymograph.

    :class:`KymoTrackGroup` instances are typically returned from tracking or refinement algorithms. They
    contain a number of :class:`KymoTrack` instances.

    Parameters
    ----------
    kymo_tracks : list of :class:`KymoTrack`
        Kymograph tracks.

    Raises
    ------
    ValueError
        If the :class:`KymoTrack` instances weren't tracked on the same source kymograph.
    ValueError
        If the :class:`KymoTrack` instances provided in the list are not unique.

    Examples
    --------
    ::

        from lumicks import pylake

        tracks = lk.track_greedy(kymo, channel="red", pixel_threshold=5)

        tracks[1]  # Extract the second `KymoTrack` from the group.
        tracks[-1]  # Extract the last `KymoTrack` from the group.
        tracks[1:3]  # Extract a `KymoTrackGroup` with the second and third kymotrack in the group.

        # You can also perform boolean array indexing. For example, one can extract a
        # `KymoTrackGroup` containing all tracks with more than 3 points.
        tracks[[len(track) > 3 for track in tracks]]

        # Or index by a list or numpy array
        tracks[[1, 3]]  # Extract the second and fourth track.
        tracks[np.asarray([1, 3])]  # Same as above.
    """

    def __init__(self, kymo_tracks):
        self._src = kymo_tracks
        self._kymos = self._validate_compatible_sources()

    def _validate_compatible_sources(self, additional_tracks=()):
        """Check that source kymos for all tracks (including in self) are compatible.

        Parameters
        ----------
        additional_tracks : KymoTrackGroup, optional
            Additional tracks to be added to the current instance.

        Returns
        -------
        tuple
            Tuple of source kymograph instances
        """
        tracks = list(itertools.chain(self, additional_tracks))
        if not len(tracks):
            return ()

        if len(set(tracks)) != len(tracks):
            raise ValueError(
                "Cannot extend this KymoTrackGroup with a KymoTrack that is already part of the group"
                if additional_tracks
                else "Some tracks appear multiple times. The provided tracks must be unique."
            )

        channels = {track._channel for track in tracks}
        if len(channels) > 1:
            raise ValueError("All tracks must be from the same color channel.")

        kymos = tuple({track._kymo: None for track in tracks}.keys())
        if len(calibrations := set([kymo._calibration.unit for kymo in kymos])) > 1:
            raise ValueError(
                f"All tracks must be calibrated in the same units, got {calibrations}."
            )

        return kymos

    def _validate_single_source(self, method_description):
        if not self:
            raise RuntimeError("No kymo associated with this empty group (no tracks available)")

        if (n_kymos := len(self._kymos)) > 1:
            raise NotImplementedError(
                f"{method_description} is not supported. "
                f"This group contains tracks from {n_kymos} source kymographs."
            )

    def _validate_single_linetime_pixelsize(self):
        """Check that image acquisition attributes (scan line times and pixel sizes)
        are the same for all source kymos.

        Returns
        -------
        bool
            If validity criteria are met (all source kymographs have the same line times and pixel sizes)
        str
            Error message to be raised if validity criteria are not met
        """

        line_times = {kymo.line_time_seconds for kymo in self._kymos}
        pixel_sizes = {kymo.pixelsize_um[0] for kymo in self._kymos}

        line_times_err_msg = (
            ""
            if len(line_times) == 1
            else (
                f"All source kymographs must have the same line times, got {sorted(line_times)} seconds."
            )
        )

        px_size_err_msg = (
            ""
            if len(pixel_sizes) == 1
            else (
                "All source kymographs must have the same pixel sizes, "
                f"got {sorted(pixel_sizes)} {self._calibration_info['unit']}."
            )
        )

        if line_times_err_msg or px_size_err_msg:
            raise ValueError(" ".join([line_times_err_msg, px_size_err_msg]))

    def __iter__(self):
        return self._src.__iter__()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return KymoTrackGroup(self._src[item])

        try:
            return self._src[item]
        except TypeError:  # Not an integer
            return KymoTrackGroup([self._src[idx] for idx in np.arange(len(self))[item]])

    def __setitem__(self, item, value):
        raise NotImplementedError("Cannot overwrite KymoTracks.")

    def __bool__(self):
        return bool(self._src)

    def __copy__(self):
        return KymoTrackGroup(copy(self._src))

    def __add__(self, other):
        new_group = copy(self)
        new_group.extend(other)
        return new_group

    def _get_track_by_id(self, python_id):
        """Finds a track by its python identity

        Parameters
        ----------
        python_id : int
            python identity

        Returns
        -------
        track : KymoTrack or None
        """
        return next((track for track in self._src if id(track) == python_id), None)

    def remove(self, track: KymoTrack):
        """Remove a KymoTrack from the KymoTrackGroup

        Parameters
        ----------
        track : KymoTrack
            track to remove from the group
        """
        self._src.remove(track)

    @property
    def _channel(self):
        try:
            return self[0]._channel
        except IndexError:
            raise RuntimeError("No channel associated with this empty group (no tracks available)")

    @property
    def _calibration_info(self):
        try:
            kymo = self._kymos[0]
            return {"unit": kymo._calibration.unit, "unit_label": kymo._calibration.unit_label}
        except IndexError:
            raise RuntimeError("No kymo associated with this empty group (no tracks available)")

    def _flip(self):
        """Return a flipped copy of this KymoTrackGroup.

        Raises
        ------
        NotImplementedError
            If group contains tracks from more than one source kymograph.
        """
        self._validate_single_source("Flipping")
        flipped_kymo = self._kymos[0].flip()
        return KymoTrackGroup([track._flip(flipped_kymo) for track in self])

    def _split_track(self, track, split_node, min_length):
        """Split a track at a particular node

        Splits a track at index `split_node`. Modifies the `KymoTrackGroup` in-place. The track to
        be split is removed and the split tracks are appended to the list of `KymoTracks` if they
        are long enough for inclusion (length equal or larger than `min_length`).

        Parameters
        ----------
        track : KymoTrack
            Track to split
        split_node : int
            Index of the node in the track to split at
        min_length : int
            Minimum length of a track. Tracks shorter than this will be discarded.

        Raises
        ------
        ValueError
            If asked to split at a point that would result in an empty track.
        """
        new_tracks = [t for t in track._split(split_node) if len(t) >= min_length]

        self._src.remove(track)
        self._src.extend(new_tracks)

    def _merge_tracks(self, starting_track, starting_node, ending_track, ending_node):
        """Connect two tracks from any given nodes, removing the points in between.

        Parameters
        ----------
        starting_track: KymoTrack
            First track to connect
        starting_node : int
            Index of the node in the track to connect from
        ending_track: KymoTrack
            Second track to connect
        ending_node: int
            Index of the node in the track to connect to

        Raises
        ------
        RuntimeError
            If the two tracks do not belong to the same group.
        ValueError
            If the two points we are attempting to connect are part of the same frame.
        """
        if starting_track not in self._src or ending_track not in self._src:
            raise RuntimeError("Both tracks need to be part of this group to be merged")

        starting_node = int(starting_node)
        ending_node = int(ending_node)

        start_time_idx = starting_track.time_idx[starting_node]
        end_time_idx = ending_track.time_idx[ending_node]
        if start_time_idx == end_time_idx:
            raise ValueError("Cannot connect two points with the same time index.")

        # ensure that tracks are properly ordered so resulting merge track
        # has coordinates sorted in ascending time indices
        if start_time_idx > end_time_idx:
            starting_track, ending_track = ending_track, starting_track
            starting_node, ending_node = ending_node, starting_node

        # up to and *including* starting_node
        first_half = starting_track._with_coordinates(
            starting_track.time_idx[: starting_node + 1],
            starting_track._localization[: starting_node + 1],
        )

        last_half = ending_track._with_coordinates(
            ending_track.time_idx[ending_node:],
            ending_track._localization[ending_node:],
        )

        self._src[self._src.index(starting_track)] = first_half + last_half
        if starting_track != ending_track:
            self.remove(ending_track)

    def __len__(self):
        return len(self._src)

    def extend(self, other):
        """Extend this group with additional `KymoTrack` instances.

        Parameters
        ----------
        other : `KymoTrack` or `KymoTrackGroup`
            `Kymograph` tracks to extend this group with.

        Raises
        ------
        TypeError
            If the data to extend this group with isn't a KymoTrack or a KymoTrackGroup
        ValueError
            If the `KymoTrack` instances that we want to extend this one with weren't tracked on
            the same source kymograph.
        ValueError
            If any of the `KymoTrack` instaces we are trying to extend this group with are already
            part of this group.
        """

        if not other:
            return self

        if not (isinstance(other, KymoTrack) or isinstance(other, self.__class__)):
            raise TypeError(
                f"You can only extend a {self.__class__.__name__} with a {self.__class__.__name__} "
                f"or {KymoTrack.__name__}"
            )

        other = self.__class__([other]) if isinstance(other, KymoTrack) else other
        self._kymos = self._validate_compatible_sources(other)
        self._src.extend(other._src)

    def remove_tracks_in_rect(self, rect, all_points=False):
        """Removes tracks that fall in a particular region.

        Parameters
        ----------
        rect : array_like
            Array of 2D coordinates in time and space units (not pixels)
        all_points : bool
            Only remove tracks that are completely inside the rectangle.
        """
        if rect[0][0] > rect[1][0]:
            rect[0][0], rect[1][0] = rect[1][0], rect[0][0]

        if rect[0][1] > rect[1][1]:
            rect[0][1], rect[1][1] = rect[1][1], rect[0][1]

        self._src = [track for track in self._src if not track.in_rect(rect, all_points)]

    def filter(self, *, minimum_length=1, minimum_duration=0):
        """Remove tracks shorter than specified criteria from the list.

        Parameters
        ----------
        minimum_length : int, optional
            Minimum number of tracked points for the track to be accepted (default: 1).
        minimum_duration : seconds, optional
            Minimum duration in seconds for a track to be accepted (default: 0).
        """
        from .kymotracker import filter_tracks  # local import to avoid circular import

        self._src = filter_tracks(
            self, minimum_length=minimum_length, minimum_duration=minimum_duration
        )._src

    def __repr__(self):
        return f"{self.__class__.__name__}(N={len(self)})"

    def plot(self, *, show_outline=True, show_labels=True, axes=None, **kwargs):
        """Plot the track coordinates for all tracks in the group.

        Parameters
        ----------
        show_outline : bool
            Whether to show black outline around the tracks.
        show_labels : bool
            Whether to add axes labels.
        axes : matplotlib.axes.Axes, optional
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        from ..detail.plotting import get_axes

        ax = get_axes(axes)
        for track in self:
            track.plot(show_outline=show_outline, show_labels=False, axes=ax, **kwargs)

        if show_labels:
            ax.set_ylabel(f"position ({self._calibration_info['unit_label']})")
            ax.set_xlabel("time (s)")

    def _tracks_in_frame(self, frame_idx):
        """Grab tracks which are in a particular kymograph frame.

        Parameters
        ----------
        frame_idx : int
            Kymograph frame

        Returns
        -------
        list of tuple of int
            Returns a list of tuples. This list has one entry for each track that intersects with
            this kymograph frame. The tuple consists of the track index and the index of which
            coordinate inside that track corresponds to this kymograph frame.

        Raises
        ------
        NotImplementedError
            If this group has multiple kymographs associated with it.
        """
        if not self:
            return []  # _validate_single_source throws for empty groups

        self._validate_single_source("_tracks_in_frame")

        tracks_in_frame = []
        for track_idx, track in enumerate(self):
            track_frame_idx = np.where(track.time_idx == frame_idx)[0]
            if track_frame_idx.size > 0:
                tracks_in_frame.append((track_idx, track_frame_idx))

        return tracks_in_frame

    def plot_fit(self, frame_idx, *, fit_kwargs=None, data_kwargs=None, show_track_index=True):
        """Plot the localization model fits for a particular kymograph frame.

        Note that currently, this fit is only available for Gaussian refinement.

        Parameters
        ----------
        frame_idx : int
            Index of the kymograph frame to plot.
        fit_kwargs : dict
            Dictionary containing arguments to be passed to the fit plot.
        data_kwargs : dict
            Dictionary containing arguments to be passed to the data plot.
        show_track_index : bool
            Display track index in the plot.

        Raises
        ------
        ValueError
            If the group is empty.
        NotImplementedError
            If this group has multiple kymographs associated with it.
        NotImplementedError
            If localization was not performed using Gaussian localization.
        IndexError
            If requesting a fit for a frame index out of bounds of the KymoTrack.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            tracks = lk.track_greedy(kymo, channel="red", pixel_threshold=5)
            refined = lk.refine_tracks_gaussian(tracks, window=10, refine_missing_frames=True, overlap_strategy="multiple")

            # Plot the localization fits for kymograph frame 5
            refined.plot_fit(5)

            # Using ipywidgets and jupyter, it is easy to make an interactive version of this plot.
            import ipywidgets as widgets

            plt.figure()
            def plot_fit(frame):
                plt.cla()  # Clear the current axis
                refined.plot_fit(frame, show_track_index=True)

            widgets.interact(plot_fit, frame=widgets.IntSlider(0, max=kymo.shape[1] - 1));
        """
        import matplotlib.pyplot as plt

        self._validate_single_source("plot_fit")

        kymo = self._kymos[0]  # We can rely on there being one (_validate_single_source)
        num_frames = kymo.shape[1]
        frame_idx = num_frames + frame_idx if frame_idx < 0 else frame_idx
        if not 0 <= frame_idx < num_frames:
            raise IndexError(
                f"Frame index is out of range of the kymograph. Kymograph length is {num_frames}."
            )

        kymo_data = kymo.get_image(self._channel)[:, frame_idx]
        data_kwargs = {"color": "#c8c8c8"} | replace_key_aliases(data_kwargs or {}, ["color", "c"])
        plt.plot(np.arange(kymo.pixels_per_line) * kymo.pixelsize[0], kymo_data, **data_kwargs)

        # Put the numeric values above the baseline
        y_coord = np.min(kymo_data) + 0.05 * (np.max(kymo_data) - np.min(kymo_data))
        for track_idx, track_frame_idx in self._tracks_in_frame(frame_idx):
            self[track_idx].plot_fit(track_frame_idx, show_data=False, fit_kwargs=fit_kwargs)
            if show_track_index:
                plt.text(
                    self[track_idx].position[track_frame_idx],
                    y_coord,
                    str(track_idx),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

    def save(self, filename, delimiter=";", sampling_width=None, *, correct_origin=None):
        """Export kymograph tracks to a csv file.

        Parameters
        ----------
        filename : str | os.PathLike
            Filename to output kymograph tracks to.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph track and export the summed intensity
            with the image. The value indicates the number of pixels in either direction to sum over.
        correct_origin : bool, optional
            Use the correct pixel origin when sampling from image. Kymotracks are defined with the
            origin of each image pixel defined at the center. Earlier versions of the method that
            samples photon counts around the track had a bug which assumed the origin at the edge
            of the pixel. Setting this flag to `True` produces the correct behavior. The default is
            set to `None` which reproduces the old behavior and results in a warning, while `False`
            reproduces the old behavior without a warning.
        """
        export_kymotrackgroup_to_csv(
            filename, self, delimiter, sampling_width, correct_origin=correct_origin
        )

    def _tracks_by_kymo(self):
        """Find tracks for each `Kymo` in the group.

        Returns
        -------
        dict of KymoTrackGroup
            returns a dictionary where the keys are Kymos which the associated tracks
        """
        groups = [
            KymoTrackGroup([track for track in self if track._kymo == kymo]) for kymo in self._kymos
        ]

        indices = [
            [j for j, track in enumerate(self) if track._kymo == kymo] for kymo in self._kymos
        ]

        return groups, indices

    @staticmethod
    def _extract_dwelltime_data_from_groups(groups, exclude_ambiguous_dwells, *, observed_minimum):
        """Compute data needed for dwelltime analysis from a dictionary of KymoTrackGroups.

        This data consists of dwelltimes and observation limits per track. Note that dwelltimes of zero
        are automatically dropped.

        Parameters
        ----------
        groups : iterable of KymoTrackGroup
            An iterable which provides a sequence of KymoTrackGroup. Note that each group can only
            have one `Kymo` associated with it.
        exclude_ambiguous_dwells : bool
            Determines whether to exclude dwelltimes which are not exactly determined. If `True`,
            tracks which start in the first frame or end in the last frame of the kymograph are not
            used in the analysis, since the exact start/stop times of the binding event are not
            definitively known.
        observed_minimum : bool
            Whether to enable the legacy mode that incorrectly uses the minimum dwelltime observed
            for each Kymo, rather than the true minimum observable time.

        Returns
        -------
        dwelltimes : numpy.ndarray
            Dwelltimes
        min_obs : numpy.ndarray
            List of minimum observation times extracted from the kymos
        max_obs : numpy.ndarray
            List of maximum observation time
        removed_zeros : bool
            Whether zeroes were dropped
        time_step : numpy.ndarray
            Discretization timesteps

        Raises
        ------
        ValueError
            if one of the KymoTrackGroups has more than one `Kymo` associated with it
        """
        removed_zeros = False

        def extract_dwelltime_data(group):
            nonlocal removed_zeros

            if len(group._kymos) > 1:
                raise ValueError("This group has more than one Kymo associated with it.")

            tracks = (
                tuple(filter(KymoTrack._check_ends_are_defined, group))
                if exclude_ambiguous_dwells
                else group
            )
            dwelltimes_sec = np.array([track.duration for track in tracks])
            nonzero_dwelltimes_sec = dwelltimes_sec[dwelltimes_sec > 0]
            removed_zeros = removed_zeros or len(nonzero_dwelltimes_sec) != len(dwelltimes_sec)

            # Gracefully handle empty groups
            if nonzero_dwelltimes_sec.size == 0:
                return np.empty((4, 0))

            min_observation_time = (
                np.min(nonzero_dwelltimes_sec)
                if observed_minimum
                else np.array(
                    [
                        track._minimum_observable_duration
                        for track, dwelltime in zip(tracks, dwelltimes_sec)
                        if dwelltime > 0
                    ]
                )
            )
            if np.all(min_observation_time) is None:
                raise RuntimeError(
                    "Minimum observation time unavailable in KymoTrackGroup (tracking was "
                    "performed prior to version `1.2.0`). The minimum observable time will be "
                    "estimated as the duration of the shortest track. This approximation can cause "
                    "bias for data containing few events per kymograph. To properly calculate this "
                    "lower bound, use lk.filter_tracks() before analyzing further. If you are "
                    "confident that each group contains sufficient events and wish to suppress "
                    "this warning, pass `observed_minimum=True` to "
                    "`KymoTrackGroup.fit_binding_times()`. See the online documentation for more "
                    "information."
                )
            max_observation_time = group[0]._image.shape[1] * group[0]._line_time_seconds

            return np.vstack(
                list(
                    np.broadcast(
                        nonzero_dwelltimes_sec,
                        min_observation_time,
                        max_observation_time,
                        group[0]._line_time_seconds,
                    )
                )
            ).T

        dwelltimes, min_obs, max_obs, time_step = np.hstack(
            [extract_dwelltime_data(tracks) for tracks in groups]
        )

        return dwelltimes, min_obs, max_obs, removed_zeros, time_step

    def fit_binding_times(
        self,
        n_components,
        *,
        exclude_ambiguous_dwells=True,
        tol=None,
        max_iter=None,
        observed_minimum=None,
        discrete_model=None,
    ):
        """Fit the distribution of bound dwelltimes to an exponential (mixture) model.

        Parameters
        ----------
        n_components : int
            Number of components in the model. Currently only values of {1, 2} are supported.
        exclude_ambiguous_dwells : bool
            Determines whether to exclude dwelltimes which are not exactly determined. If `True`,
            tracks which start in the first frame or end in the last frame of the kymograph are not
            used in the analysis, since the exact start/stop times of the binding event are not
            definitively known.
        tol : float
            The tolerance for optimization convergence. This parameter is forwarded as the `ftol`
            argument to `scipy.minimize(method="L-BFGS-B")`.
        max_iter : int
            The maximum number of iterations to perform. This parameter is forwarded as the
            `maxiter` argument to `scipy.minimize(method="L-BFGS-B")`.
        observed_minimum : bool
            Use duration of shortest track as minimum observable time. This is a legacy mode that
            can lead to biases when performing global analysis on data with very few events per
            kymograph. This behavior is currently the default to preserve backward compatibility,
            but will be removed in the next major release. Set to `False` to specifically enable
            the proper behavior.
        discrete_model : bool, optional
            Take into account discretization when fitting dwell times. Prior to Pylake `v1.2.0`,
            dwell times were fitted using a continuous model. In reality, the observed dwell times
            are discretized by the time difference between successive frames. To take into account
            the temporal discretization of the data, specify `discrete_model=True`. Currently, this
            argument is set to `False` by default for backward compatibility, but in the next
            major version of Pylake (`v2.0.0`), it will default to `True`.
        """
        if not len(self):
            raise RuntimeError("No tracks available for analysis")

        if observed_minimum is None:
            warnings.warn(
                UserWarning(
                    "Prior to version 1.2.0 the method `fit_binding_times` had an issue that "
                    "assumed that the shortest track duration is indicative of the minimum "
                    "observable dwell time. This assumption is valid for kymographs with many "
                    "events but problematic when few events occur per kymograph. In this case, "
                    "binding times will be underestimated. The correct calculation of the minimum "
                    "observable time is based on the image acquisition parameters and minimum "
                    "track length. The old (incorrect) behavior is maintained as default until the "
                    "next major release (2.0.0) to ensure backward compatibility. To silence this "
                    "warning, but use the deprecated behavior use `observed_minimum=True`. To "
                    "enable the recommended method of estimating the minimum observable dwell "
                    "time use `observed_minimum=False`."
                )
            )
            observed_minimum = True

        if discrete_model is None:
            warnings.warn(
                UserWarning(
                    "Prior to version 1.2.0 the method `fit_binding_times` fitted a continuous "
                    "dwell-time distribution to the experimental data. In reality, dwell times are "
                    "a discrete multiple of the kymograph line time. This effect is most "
                    "noticeable for short dwell times. To take the discretization into account "
                    "pass the parameter `discrete_model=True` to this function. In the next major "
                    "version of Pylake (`2.0.0`), using the discrete model will become the "
                    "default. Until then, the continuous model is still used for backward "
                    "compatibility."
                )
            )
            discrete_model = False

        if n_components not in (1, 2):
            raise ValueError(
                "Only 1- and 2-component exponential distributions are currently supported."
            )

        groups, _ = self._tracks_by_kymo()
        (
            dwelltimes,
            min_obs,
            max_obs,
            removed_zeros,
            time_step,
        ) = self._extract_dwelltime_data_from_groups(
            groups, exclude_ambiguous_dwells, observed_minimum=observed_minimum
        )

        if removed_zeros:
            warnings.warn(
                RuntimeWarning(
                    "Some dwell times are zero. A dwell time of zero indicates that some of the "
                    "tracks were only observed in a single frame. For these samples it is not "
                    "possible to actually determine a dwell time. Therefore these samples are "
                    "dropped from the analysis. If you wish to not see this warning, filter the "
                    "tracks with `lk.filter_tracks` with a minimum length of 2 samples."
                ),
                stacklevel=2,
            )

        if dwelltimes.size == 0:
            raise RuntimeError("No tracks available for analysis")

        return DwelltimeModel(
            dwelltimes,
            n_components,
            min_observation_time=min_obs,
            max_observation_time=max_obs,
            tol=tol,
            max_iter=max_iter,
            discretization_timestep=time_step if discrete_model else None,
        )

    def _histogram_binding_events(self, kind, bins=10):
        """Make histogram of bound events.

        Parameters
        ----------
        kind : {'binding', 'all'}
            Type of events to count. 'binding' counts only the first position of each track whereas
            'all' counts positions of all time points in each track.
        bins : int or sequence of scalars
            Definition of the histogram bins; passed to `np.histogram()`. If `int`, defines the total
            number of bins. If sequence of scalars, defines the edges of the bins.
        """
        if kind == "all":
            slc = slice(None)
        elif kind == "binding":
            slc = slice(0, 1)
        else:
            raise ValueError(f"`kind` argument '{kind}' must be 'all' or 'binding'.")
        tracks = [track.position[slc] for track in self]

        if not tracks:
            raise RuntimeError("No tracks available for analysis")

        events = np.hstack(tracks)

        pos_range = (0, self[0]._pixelsize * self[0]._image.shape[0])
        return np.histogram(events, bins=bins, range=pos_range)

    def plot_binding_histogram(self, kind, bins=10, **kwargs):
        """Plot histogram of bound events.

        Parameters
        ----------
        kind : {'binding', 'all'}
            Type of events to count. 'binding' counts only the first position of each track whereas
            'all' counts positions of all time points in each track.
        bins : int or sequence of scalars
            Definition of the histogram bins; passed to `np.histogram()`. When an integer is supplied
            to the `bins` argument, the full position range is used to calculate the bin edges (this
            is equivalent to using `np.histogram(data, bins=n, range=(0, max_position))`). If a
            sequence of scalars is provided, they directly define the edges of the bins.
        **kwargs
            Keyword arguments forwarded to `plt.bar()`.
        """
        import matplotlib.pyplot as plt

        counts, edges = self._histogram_binding_events(kind, bins)
        widths = np.diff(edges)
        plt.bar(edges[:-1], counts, width=widths, align="edge", **kwargs)
        plt.ylabel("Counts")
        plt.xlabel(f"Position ({self._calibration_info['unit_label']})")

    def _histogram_binding_profile(self, n_time_bins, bandwidth, n_position_points, roi=None):
        """Calculate a Kernel Density Estimate (KDE) of binding density along the tether for time bins.

        First the kymograph is binned along the temporal axis. In the case of non-integer `frames / bins`,
        remaining frames are discarded. Next, for each time bin, a KDE is calculated along the spatial axis.

        Parameters
        ----------
        n_time_bins : int
            Requested number of time bins.
        bandwidth : float
            KDE bandwidth; units are in the physical spatial units of the kymograph.
        n_position_points : int
            Length of the returned density array(s).
        roi: list or None
            ROI coordinates as `[[min_time, min_position], [max_time, max_position]]`.
        """
        from sklearn.neighbors import KernelDensity

        self._validate_single_source("Binding profile")
        _kymo = self._kymos[0]

        if n_time_bins == 0:
            raise ValueError("Number of time bins must be > 0.")
        if n_position_points < 2:
            raise ValueError("Number of spatial bins must be >= 2.")

        if roi is None:
            n_rows, n_frames = _kymo.get_image(self._channel).shape
            start_frame = 0
            min_position = 0
            max_position = n_rows * _kymo.pixelsize[0]
        else:
            (min_time, min_position), (max_time, max_position) = roi
            n_frames = np.ceil((max_time - min_time) / _kymo.line_time_seconds)
            start_frame = min_time // _kymo.line_time_seconds

        try:
            bin_size = n_frames // n_time_bins
            bin_edges = np.arange(start_frame, start_frame + n_frames, step=bin_size)
        except ZeroDivisionError:
            raise ValueError("Number of time bins must be <= number of frames.")

        frames = np.hstack([track.time_idx for track in self])
        positions = np.hstack([track.position for track in self])
        bin_labels = np.digitize(frames, bin_edges, right=False)

        x = np.linspace(min_position, max_position, n_position_points)[:, np.newaxis]
        densities = []
        for bin_index in np.arange(n_time_bins) + 1:
            binned_positions = positions[bin_labels == bin_index][:, np.newaxis]
            try:
                # Each estimate is normalized to integrate to 1. For proper comparison
                # need to weight each by the number of data points used to estimate
                kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(binned_positions)
                densities.append(np.exp(kde.score_samples(x)) * binned_positions.size)
            except ValueError as err:
                if len(binned_positions) == 0:
                    densities.append(np.zeros(x.size))
                else:
                    raise err

        # "normalize" such that the highest peak == 1. This helps with plotting such that
        # the offset between bins does not strongly depend on the number of bins or data
        y = densities / np.max(densities)

        return x.squeeze(), y

    def estimate_diffusion(self, method, *args, min_length=None, **kwargs):
        r"""Estimate diffusion constant for each track in the group.

        Calls :meth:`KymoTrack.estimate_diffusion` for each track. See the documentation for that
        method for more detailed information and references.

        Parameters
        ----------
        method : {"cve", "ols", "gls"}
            - "cve" : Covariance based estimator. Optimal if SNR > 1.
            - "ols" : Ordinary least squares. Determines optimal number of lags.
            - "gls" : Generalized least squares. Takes into account covariance matrix (slower). Can
              only be used when track is equidistantly sampled.
        min_length : None or int (optional)
            Discards tracks shorter than a certain length from the analysis. If `None` (the default)
            tracks shorter than 3 points if `method == "cve"` or 5 points if `method == "ols" or "gls"`
            will be discarded.
        **kwargs :
            forwarded to :meth:`KymoTrack.estimate_diffusion`

        Raises
        ------
        ValueError
            if `method == "cve"`, the source kymograph does not have a clearly defined motion blur
            constant, and a localization variance is supplied as an argument. As a result, the
            diffusion constant cannot be reliably calculated. In this case, do not provide a
            localization uncertainty.
        NotImplementedError
            if the tracked kymograph had disjoint time intervals (such as for a temporally
            downsampled kymograph).

        Warns
        -----
        RuntimeWarning
            if `method == "cve"` and the source kymograph does not have a clearly defined motion
            blur constant. As a result, the localization variance and standard error for the
            diffusion constant will not be available. Estimates that are unavailable are returned
            as `np.nan`.
        RuntimeWarning
            if some tracks were discarded because they were shorter than the specified `min_length`.
        """
        required_length = (3 if method == "cve" else 5) if min_length is None else min_length
        filtered_tracks = [track for track in self if len(track) >= required_length]
        n_discarded = len(self) - len(filtered_tracks)

        if n_discarded and min_length is None:
            warnings.warn(
                f"{n_discarded} tracks were shorter than the specified min_length "
                "and discarded from the analysis.",
                RuntimeWarning,
                stacklevel=2,
            )

        return [k.estimate_diffusion(method, *args, **kwargs) for k in filtered_tracks]

    def ensemble_diffusion(self, method, *, max_lag=None):
        """Determine ensemble based diffusion estimates.

        Determines ensemble based diffusion estimates for the entire group of KymoTracks. This
        method assumes that all tracks experience the same diffusion and computes an averaged
        diffusion estimate.

        Parameters
        ----------
        method : {"cve", "ols"}
            - "cve" : Covariance based estimator. Optimal if SNR > 1. Ensemble average is
              determined by determining the weighted average of the individual track estimates. The
              standard error is computed by determining the weighted average of the associated
              standard errors for each estimate (Equation 57 and 58 from Vestergaard [7]_). See
              :meth:`KymoTrack.estimate_diffusion` for more detailed information and references.
            - "ols" : Ordinary least squares. Determines the ensemble mean squared displacements for
              the entire KymoTrackGroup and estimates a diffusion constant for it. See
              :meth:`KymoTrack.estimate_diffusion` for more detailed information and references.
        max_lag : int
            Maximum number of lags to include when using the ordinary least squares method (OLS).

        Raises
        ------
        ValueError
            if `method == "ols"` and the source kymographs do not have the same line times
            or pixel sizes.

        Warns
        -----
        RuntimeWarning
            if `method == "cve"` and the source kymograph does not have a clearly defined motion
            blur constant. As a result, the localization variance and standard error for the
            diffusion constant will not be available. If only one track is available, the standard
            error on the diffusion constant will also not be available. Estimates that are
            unavailable are returned as `np.nan`.
        RuntimeWarning
            if `method == "cve"` and the source kymographs do not have the same line times
            or pixel sizes. As a result, the localization variance and variance of the localization
            variance are not available. Estimates that are unavailable are returned as `np.nan`.

        References
        ----------
        .. [7] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
               diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
               022726.
        """
        if method == "cve":
            try:
                self._validate_single_linetime_pixelsize()
                is_valid = True
            except ValueError:
                warnings.warn(
                    RuntimeWarning(
                        "Localization variances cannot be reliably calculated for an ensemble of "
                        "tracks from kymographs with different line times or pixel sizes."
                    ),
                )
                is_valid = False
            return ensemble_cve(self, calculate_localization_var=is_valid)
        elif method == "ols":
            self._validate_single_linetime_pixelsize()
            return ensemble_ols(self, max_lag)
        else:
            raise ValueError(f'Invalid method ({method}) selected. Method must be "cve" or "ols".')

    def ensemble_msd(self, max_lag=None, min_count=2) -> EnsembleMSD:
        r"""This method returns the weighted average of the Mean Squared Displacement (MSD) for all
        tracks in this group.

        This method determines the MSDs per track and determines the weighted average of them.
        The intrinsic assumption made when computing this quantity is that all tracks are
        independent and  all trajectories sample the same environment and undergo the same type of
        diffusion.

        The estimator for the MSD (:math:`\rho`) is defined as:

        .. math::
            \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n} \left(r_{i+n} - r_{i}\right)^2

        For a diffusion process :math:`\rho_n` is gamma distributed. From the additivity of
        independent gamma distributions we know that for :math:`N_T` tracks of equal length [1]_:

        .. math::
            E[\rho^{ens}_n] = \rho_n

        and

        .. math::
            E[(\sigma^{ens})^2] = \frac{\sigma}{N_T}

        In reality, the tracks will not have equal length, therefore the returned statistics are
        weighted by the number of samples that contributed to the estimate. If all the tracks were
        of equal length with no missing data points, the weighting will have no effect on the
        estimates.

        Note: This estimator leads to highly correlated estimates, which should not be treated
        as independent measurements. See [1]_ for more information.

        Parameters
        ----------
        max_lag : int
            Maximum number of lags to compute.
        min_count : int
            If fewer than `min_count` tracks contribute to the MSD at a particular lag then that lag
            is omitted.

        References
        ----------
        .. [1] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
               single-particle tracking. Physical Review E, 85(6), 061916.
        """
        track_msds = [
            calculate_msd_counts(np.array(track.time_idx, dtype=int), track.position, max_lag)
            for track in self._src
        ]

        self._validate_single_linetime_pixelsize()

        return calculate_ensemble_msd(
            line_msds=track_msds,
            time_step=self._kymos[0].line_time_seconds,
            min_count=min_count,
            **self._calibration_info,
        )
