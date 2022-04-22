from copy import copy
from sklearn.neighbors import KernelDensity
from .detail.msd_estimation import *
from .detail.localization_models import LocalizationModel
from lumicks.pylake.population.dwelltime import DwelltimeModel


def export_kymolinegroup_to_csv(filename, kymoline_group, delimiter, sampling_width):
    """Export KymoLineGroup to a csv file.

    Parameters
    ----------
    filename : str
        Filename to output KymoLineGroup to.
    kymoline_group : KymoLineGroup
        Kymograph traces to export.
    delimiter : str
        Which delimiter to use in the csv file.
    sampling_width : int or None
        When supplied, this will sample the source image around the kymograph line and export the summed intensity with
        the image. The value indicates the number of pixels in either direction to sum over.
    """
    if not kymoline_group:
        raise RuntimeError("No kymograph traces to export")

    time_units = "seconds"
    position_units = kymoline_group[0]._kymo._calibration.unit

    idx = np.hstack([np.full(len(line), idx) for idx, line in enumerate(kymoline_group)])
    coords_idx = np.hstack([line.coordinate_idx for line in kymoline_group])
    times_idx = np.hstack([line.time_idx for line in kymoline_group])

    position = np.hstack([line.position for line in kymoline_group])
    seconds = np.hstack([line.seconds for line in kymoline_group])

    data, header, fmt = [], [], []

    def store_column(column_title, format_string, new_data):
        data.append(new_data)
        header.append(column_title)
        fmt.append(format_string)

    store_column("line index", "%d", idx)
    store_column("time (pixels)", "%.18e", times_idx)
    store_column("coordinate (pixels)", "%.18e", coords_idx)

    store_column(f"time ({time_units})", "%.18e", seconds)
    store_column(f"position ({position_units})", "%.18e", position)

    if sampling_width is not None:
        store_column(
            f"counts (summed over {2 * sampling_width + 1} pixels)",
            "%d",
            np.hstack([line.sample_from_image(sampling_width) for line in kymoline_group]),
        )

    data = np.vstack(data).T
    np.savetxt(filename, data, fmt=fmt, header=delimiter.join(header), delimiter=delimiter)


def import_kymolinegroup_from_csv(filename, kymo, channel, delimiter=";"):
    """Import kymolines from csv

    Parameters
    ----------
    filename : str
        filename to import from.
    kymo : Kymo
        kymograph instance that these lines were tracked from.
    channel : str
        color channel that these lines were tracked from.
    delimiter : str
        A delimiter that delimits the column data.

    The file format contains a series of columns as follows:
    line index, time (pixels), coordinate (pixels), time (optional), coordinate (optional), sampled_counts (optional)"""
    data = np.loadtxt(filename, delimiter=delimiter)
    assert len(data.shape) == 2, "Invalid file format"
    assert data.shape[0] > 2, "Invalid file format"

    indices = data[:, 0]
    lines = np.unique(indices)

    return KymoLineGroup(
        [KymoLine(data[indices == k, 1], data[indices == k, 2], kymo, channel) for k in lines]
    )


class KymoLine:
    """A line on a kymograph.

    Parameters
    ----------
    time_idx : array-like
        Frame time indices.
    localization : LocalizationModel or array-like
        LocalizationModel instance containing localization parameters
        or list of (sub)pixel coordinates to be converted to spatial
        position via calibration with pixel size.
    kymo : lk.kymo.Kymo
        Kymograph instance.
    channel : {"red", "green", "blue"}
        Color channel to analyze
    """

    __slots__ = ["_time_idx", "_localization", "_kymo", "_channel"]

    def __init__(self, time_idx, localization, kymo, channel):
        self._kymo = kymo
        self._channel = channel
        self._time_idx = np.asarray(time_idx)
        self._localization = (
            localization
            if isinstance(localization, LocalizationModel)
            else LocalizationModel(np.array(localization) * self._pixelsize)
        )

    @property
    def _image(self):
        return self._kymo.get_image(self._channel)

    def _with_coordinates(self, time_idx, localization):
        """Return a copy of the KymoLine with new spatial/temporal coordinates."""
        return KymoLine(
            time_idx,
            localization,
            self._kymo,
            self._channel,
        )

    def with_offset(self, time_offset, coordinate_offset):
        """Returns an offset version of the KymoLine"""
        # Convert from image units to (integer rounded toward zero) pixels
        time_pixel_offset = int(time_offset / self._line_time_seconds)
        coordinate_pixel_offset = int(coordinate_offset / self._pixelsize)

        return self._with_coordinates(
            self.time_idx + time_pixel_offset,
            self.coordinate_idx + coordinate_pixel_offset,
        )

    def __add__(self, other):
        """Concatenate two KymoLines"""
        return self._with_coordinates(
            np.hstack((self.time_idx, other.time_idx)),
            np.hstack((self.coordinate_idx, other.coordinate_idx)),
        )

    def __getitem__(self, item):
        return np.squeeze(
            np.array(np.vstack((self.time_idx[item], self.coordinate_idx[item]))).transpose()
        )

    @property
    def time_idx(self):
        return self._time_idx

    @property
    def coordinate_idx(self):
        """Return spatial coordinates in units of pixels."""
        return self._localization.position / self._kymo.pixelsize[0]

    @property
    def seconds(self):
        return self._line_time_seconds * self.time_idx

    @property
    def position(self):
        return self._localization.position

    @property
    def _line_time_seconds(self):
        """Kymograph line (frame) time in seconds."""
        return self._kymo.line_time_seconds

    @property
    def _pixelsize(self):
        """Kymograph (spatial) pixel size in physical units."""
        return self._kymo.pixelsize[0]

    def _check_ends_are_defined(self):
        """Checks if beginning and end of the line are not in the first/last frame."""
        return self.time_idx[0] > 0 and self.time_idx[-1] < self._image.shape[1] - 1

    def in_rect(self, rect):
        """Check whether any point of this KymoLine falls in the rect given in rect.

        Parameters
        ----------
        rect : Tuple[Tuple[float, float], Tuple[float, float]]
            Coordinates should be given as ((min_time, min_coord), (max_time, max_coord)).
        """
        time_match = np.logical_and(self.seconds < rect[1][0], self.seconds >= rect[0][0])
        position_match = np.logical_and(self.position < rect[1][1], self.position >= rect[0][1])
        return np.any(np.logical_and(time_match, position_match))

    def interpolate(self):
        """Interpolate Kymoline to whole pixel values"""
        interpolated_time = np.arange(int(np.min(self.time_idx)), int(np.max(self.time_idx)) + 1, 1)
        interpolated_coord = np.interp(interpolated_time, self.time_idx, self.coordinate_idx)
        return self._with_coordinates(interpolated_time, interpolated_coord)

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
        # Time and coordinates are being cast to an integer since we use them to index into a data array.
        return [
            reduce(self._image[max(int(c) - num_pixels, 0) : int(c) + num_pixels + 1, int(t)])
            for t, c in zip(self.time_idx, self.coordinate_idx)
        ]

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

    def msd(self, max_lag=None):
        """Estimate the Mean Square Displacement (MSD) for various time lags.

        The estimator for the MSD (rho) is defined as:

            rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

        Note: This estimator leads to highly correlated estimates, which should not be treated
        as independent measurements. See [1] for more information.

        1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
        single-particle tracking. Physical Review E, 85(6), 061916.

        Parameters
        ----------
        max_lag : int
            Maximum lag to include.

        Returns
        -------
        lag_time : array_like
            array of lag times
        msd : array_like
            array of mean square distance estimates for the corresponding lag times
        """
        frame_lag, msd = calculate_msd(
            np.array(self.time_idx, dtype=int),
            np.array(self.position),
            max_lag if max_lag else len(self.time_idx),
        )

        return frame_lag * self._line_time_seconds, msd

    def plot_msd(self, max_lag=None, **kwargs):
        """Plot Mean Squared Differences of this trace

        The estimator for the MSD (rho) is defined as:

            rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

        Note: This estimator leads to highly correlated estimates, which should not be treated
        as independent measurements. See [1] for more information.

        1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
        single-particle tracking. Physical Review E, 85(6), 061916.

        Parameters
        ----------
        max_lag : int (optional)
            Maximum lag to include. When omitted, an optimal number of lags is chosen [1].
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        frame_idx, positions = np.array(self.time_idx, dtype=int), np.array(self.position)
        max_lag = max_lag if max_lag else determine_optimal_points(frame_idx, positions)[0]
        lag_time, msd = self.msd(max_lag)
        plt.plot(lag_time, msd, **kwargs)
        plt.xlabel("Lag time [s]")
        plt.ylabel(f"Mean Squared Displacement [{self._kymo._calibration.unit_label}$^2$]")

    def estimate_diffusion_ols(self, max_lag=None):
        """Perform an unweighted fit to the MSD estimates to obtain a diffusion constant.

        The estimator for the MSD (rho) is defined as:

          rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

        In a diffusion problem, the MSD can be fitted to a linear curve.

            intercept = 2 * d * (sigma**2 - 2 * R * D * delta_t)
            slope = 2 * d * D * delta_t

        Here d is the dimensionality of the problem. D is the diffusion constant. R is a motion blur
        constant. delta_t is the time step and sigma represents the dynamic localization error.

        One aspect that is import to consider is that this estimator uses every data point multiple
        times. As a consequence the elements of rho_n are highly correlated. This means that
        including more points doesn't necessarily make the estimates better and can actually make
        the estimate worse. It is therefore a good idea to estimate an appropriate number of MSD
        estimates to use. See [1] for more information on this.

        Note that this estimation procedure should only be used for pure diffusion in the absence
        of drift.

        1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
        single-particle tracking. Physical Review E, 85(6), 061916.

        Parameters
        ----------
        max_lag : int (optional)
            Number of lags to include. When omitted, the method uses an optimal number of lags
            as determined by [1].
        """
        frame_idx, positions = np.array(self.time_idx, dtype=int), np.array(self.position)
        max_lag = max_lag if max_lag else determine_optimal_points(frame_idx, positions)[0]
        return estimate_diffusion_constant_simple(
            frame_idx, positions, self._line_time_seconds, max_lag
        )


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

    def __copy__(self):
        return KymoLineGroup(copy(self._src))

    def _concatenate_lines(self, starting_line, ending_line):
        """Concatenate two lines together.

        Parameters
        ----------
        starting_line : KymoLine
            Note that this line has to start before the second one.
        ending_line : KymoLine
        """
        if starting_line not in self._src or ending_line not in self._src:
            raise RuntimeError("Both lines need to be part of this group to be concatenated")

        if starting_line.seconds[-1] >= ending_line.seconds[0]:
            raise RuntimeError("First line needs to end before the second starts for concatenation")

        self._src[self._src.index(starting_line)] = starting_line + ending_line
        self._src.remove(ending_line)

    def _merge_lines(self, starting_line, starting_node, ending_line, ending_node):
        starting_node = int(starting_node) + 1
        ending_node = int(ending_node)

        first_half = starting_line._with_coordinates(
            starting_line.time_idx[:starting_node],
            starting_line.coordinate_idx[:starting_node],
        )

        last_half = ending_line._with_coordinates(
            ending_line.time_idx[ending_node:],
            ending_line.coordinate_idx[ending_node:],
        )

        self._src[self._src.index(starting_line)] = first_half + last_half
        if starting_line != ending_line:
            self._src.remove(ending_line)

    def __len__(self):
        return len(self._src)

    def extend(self, other):
        if isinstance(other, self.__class__):
            self._src.extend(other._src)
        elif isinstance(other, KymoLine):
            self._src.extend([other])
        else:
            raise TypeError(
                f"You can only extend a {self.__class__} with a {self.__class__} or " f"{KymoLine}"
            )

    def remove_lines_in_rect(self, rect):
        """Removes traces that fall in a particular region. Note that if any point on a line falls
        inside the selected region it will be removed.

        Parameters
        ----------
        rect : array_like
            Array of 2D coordinates in time and space units (not pixels)
        """
        if rect[0][0] > rect[1][0]:
            rect[0][0], rect[1][0] = rect[1][0], rect[0][0]

        if rect[0][1] > rect[1][1]:
            rect[0][1], rect[1][1] = rect[1][1], rect[0][1]

        self._src = [line for line in self._src if not line.in_rect(rect)]

    def __repr__(self):
        return f"{self.__class__.__name__}(N={len(self._src)})"

    def save(self, filename, delimiter=";", sampling_width=None):
        """Export kymograph lines to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph traces to.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph line and export the summed intensity
            with the image. The value indicates the number of pixels in either direction to sum over.
        """
        export_kymolinegroup_to_csv(filename, self._src, delimiter, sampling_width)

    def fit_binding_times(
        self, n_components, *, exclude_ambiguous_dwells=True, tol=None, max_iter=None
    ):
        """Fit the distribution of bound dwelltimes to an exponential (mixture) model.

        Parameters
        ----------
        n_components : int
            Number of components in the model. Currently only values of {1, 2} are supported.
        exclude_ambiguous_dwells : bool
            Determines whether to exclude dwelltimes which are not exactly determined. If `True`, tracks which
            start in the first frame or end in the last frame of the kymograph are not used in the analysis,
            since the exact start/stop times of the binding event are not definitively known.
        tol : float
            The tolerance for optimization convergence. This parameter is forwarded as the `ftol` argument
            to `scipy.minimize(method="L-BFGS-B")`.
        max_iter : int
            The maximum number of iterations to perform. This parameter is forwarded as the `maxiter` argument
            to `scipy.minimize(method="L-BFGS-B")`.
        """

        if n_components not in (1, 2):
            raise ValueError(
                "Only 1- and 2-component exponential distributions are currently supported."
            )

        lines = filter(KymoLine._check_ends_are_defined, self) if exclude_ambiguous_dwells else self
        dwelltimes_sec = np.hstack([line.seconds[-1] - line.seconds[0] for line in lines])

        min_observation_time = np.min(dwelltimes_sec)
        max_observation_time = self[0]._image.shape[1] * self[0]._line_time_seconds

        return DwelltimeModel(
            dwelltimes_sec,
            n_components,
            min_observation_time=min_observation_time,
            max_observation_time=max_observation_time,
            tol=tol,
            max_iter=max_iter,
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
        events = np.hstack([track.position[slc] for track in self])

        image = self[0]._image
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
        plt.xlabel(f"Position ({self[0]._kymo._calibration.unit_label})")

    def _histogram_binding_profile(self, n_time_bins, bandwidth, n_position_points):
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
        """
        if n_time_bins == 0:
            raise ValueError("Number of time bins must be > 0.")
        if n_position_points < 2:
            raise ValueError("Number of spatial bins must be >= 2.")

        n_rows, n_frames = self[0]._image.shape
        position_max = n_rows * self[0]._pixelsize
        try:
            bin_size = n_frames // n_time_bins
            bin_edges = np.arange(n_frames, step=bin_size)
        except ZeroDivisionError:
            raise ValueError("Number of time bins must be <= number of frames.")

        frames = np.hstack([line.time_idx for line in self])
        positions = np.hstack([line.position for line in self])
        bin_labels = np.digitize(frames, bin_edges, right=False)

        x = np.linspace(0, position_max, n_position_points)[:, np.newaxis]
        densities = []
        for bin_index in np.arange(n_time_bins) + 1:
            binned_positions = positions[bin_labels == bin_index][:, np.newaxis]
            try:
                kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(binned_positions)
                densities.append(np.exp(kde.score_samples(x)))
            except ValueError as err:
                if len(binned_positions) == 0:
                    densities.append(np.zeros(x.size))
                else:
                    raise err

        return x.squeeze(), densities
