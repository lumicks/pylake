from copy import copy
from deprecated.sphinx import deprecated
from sklearn.neighbors import KernelDensity
from .detail.msd_estimation import *
from .detail.localization_models import LocalizationModel
from .. import __version__
from ..population.dwelltime import DwelltimeModel


def export_kymotrackgroup_to_csv(filename, kymotrack_group, delimiter, sampling_width):
    """Export KymoTrackGroup to a csv file.

    Parameters
    ----------
    filename : str
        Filename to output KymoTrackGroup to.
    kymotrack_group : KymoTrackGroup
        Kymograph tracks to export.
    delimiter : str
        Which delimiter to use in the csv file.
    sampling_width : int or None
        If, this will sample the source image around the kymograph track and export the summed intensity with
        the image. The value indicates the number of pixels in either direction to sum over.
    """
    if not kymotrack_group:
        raise RuntimeError("No kymograph tracks to export")

    time_units = "seconds"
    position_units = kymotrack_group._kymo._calibration.unit

    idx = np.hstack([np.full(len(track), idx) for idx, track in enumerate(kymotrack_group)])
    coords_idx = np.hstack([track.coordinate_idx for track in kymotrack_group])
    times_idx = np.hstack([track.time_idx for track in kymotrack_group])

    position = np.hstack([track.position for track in kymotrack_group])
    seconds = np.hstack([track.seconds for track in kymotrack_group])

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
            np.hstack([track.sample_from_image(sampling_width) for track in kymotrack_group]),
        )

    version_header = f"Exported with pylake v{__version__} | track coordinates v2\n"
    header = version_header + delimiter.join(column_titles)
    data = np.vstack(data).T
    np.savetxt(filename, data, fmt=fmt, header=header, delimiter=delimiter)


def import_kymotrackgroup_from_csv(filename, kymo, channel, delimiter=";"):
    """Import a KymoTrackGroup from a csv file.

    The file format contains a series of columns as follows:
    track index, time (pixels), coordinate (pixels), time (optional), coordinate (optional), sampled_counts (optional)

    Parameters
    ----------
    filename : str
        filename to import from.
    kymo : Kymo
        kymograph instance that the CSV data was tracked from.
    channel : str
        color channel that was used for tracking.
    delimiter : str
        The string used to separate columns.
    """

    data = np.loadtxt(filename, delimiter=delimiter)
    assert len(data.shape) == 2, "Invalid file format"
    assert data.shape[0] > 2, "Invalid file format"

    indices = data[:, 0]
    tracks = np.unique(indices)

    return KymoTrackGroup(
        [KymoTrack(data[indices == k, 1], data[indices == k, 2], kymo, channel) for k in tracks]
    )


class KymoTrack:
    """A tracked particle on a kymograph.

    Parameters
    ----------
    time_idx : array_like
        Frame time indices.
    localization : LocalizationModel or array_like
        LocalizationModel instance containing localization parameters
        or list of (sub)pixel coordinates to be converted to spatial
        position via calibration with pixel size.
    kymo : Kymo
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
        """Return a copy of the KymoTrack with new spatial/temporal coordinates."""
        return KymoTrack(
            time_idx,
            localization,
            self._kymo,
            self._channel,
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
        """The tracked temporal coordinates in seconds."""
        return self._line_time_seconds * self.time_idx

    @property
    def position(self):
        """The tracked positional coordinates. The units depend on the spatial units of the tracked kymo."""
        return self._localization.position

    @property
    def _line_time_seconds(self):
        """Source kymograph line time in seconds."""
        return self._kymo.line_time_seconds

    @property
    def _pixelsize(self):
        """Kymograph (spatial) pixel size in physical units."""
        return self._kymo.pixelsize[0]

    def _check_ends_are_defined(self):
        """Checks if beginning and end of the track are not in the first/last frame."""
        return self.time_idx[0] > 0 and self.time_idx[-1] < self._image.shape[1] - 1

    def in_rect(self, rect):
        """Check whether any point of this KymoTrack falls in the rect given in rect.

        Parameters
        ----------
        rect : Tuple[Tuple[float, float], Tuple[float, float]]
            Coordinates should be given as ((min_time, min_coord), (max_time, max_coord)).
        """
        time_match = np.logical_and(self.seconds < rect[1][0], self.seconds >= rect[0][0])
        position_match = np.logical_and(self.position < rect[1][1], self.position >= rect[0][1])
        return np.any(np.logical_and(time_match, position_match))

    def interpolate(self):
        """Interpolate KymoTrack to whole pixel values"""
        interpolated_time = np.arange(int(np.min(self.time_idx)), int(np.max(self.time_idx)) + 1, 1)
        interpolated_coord = np.interp(interpolated_time, self.time_idx, self.coordinate_idx)
        return self._with_coordinates(interpolated_time, interpolated_coord)

    def sample_from_image(self, num_pixels, reduce=np.sum):
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
        lag_time : array_like
            array of lag times
        msd : array_like
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
        plt.ylabel(f"Mean Squared Displacement [{self._kymo._calibration.unit_label}$^2$]")

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
        as: :math:`\sqrt{D \Delta t} / \sigma`.

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
            "unit_label": f"{self._kymo._calibration.unit_label}$^2$/s",
        }

        if method == "cve":
            # We hardcode the blur constant for confocal for now (no motion blur)
            return estimate_diffusion_cve(
                frame_idx,
                positions,
                self._line_time_seconds,
                **unit_labels,
                blur_constant=0,
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

    @deprecated(
        reason=(
            'This method is replaced by :meth:`KymoTrack.estimate_diffusion(method="ols") '
            "<lumicks.pylake.kymotracker.KymoTrack.estimate_diffusion()>` to allow more "
            "flexibility in the choice of algorithms and provide additional metadata."
        ),
        action="always",
        version="0.12.1",
    )
    def estimate_diffusion_ols(self, max_lag=None):
        """Perform an unweighted fit to the MSD estimates to obtain a diffusion constant"""
        return self.estimate_diffusion("ols", max_lag).value


class KymoTrackGroup:
    """Tracks on a kymograph."""

    def __init__(self, kymo_tracks):
        self._src = kymo_tracks
        if self:
            self._validate_single_source(kymo_tracks)

    def _validate_single_source(self, kymo_tracks):
        kymos = set([track._kymo._id for track in kymo_tracks])
        channels = set([track._channel for track in kymo_tracks])

        assert len(kymos) == 1, "All tracks must have the same source kymograph."
        assert len(channels) == 1, "All tracks must be from the same color channel."

        return next(iter(kymos)), next(iter(channels))

    def __iter__(self):
        return self._src.__iter__()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return KymoTrackGroup(self._src[item])
        else:
            return self._src[item]

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

    @property
    def _kymo(self):
        try:
            return self[0]._kymo
        except IndexError:
            raise RuntimeError("No kymo associated with this empty group (no tracks available)")

    @property
    def _channel(self):
        try:
            return self[0]._channel
        except IndexError:
            raise RuntimeError("No channel associated with this empty group (no tracks available)")

    def _concatenate_tracks(self, starting_track, ending_track):
        """Concatenate two tracks together.

        Parameters
        ----------
        starting_track : KymoTrack
            Note that this track has to start before the second one.
        ending_track : KymoTrack
        """
        if starting_track not in self._src or ending_track not in self._src:
            raise RuntimeError("Both tracks need to be part of this group to be concatenated")

        if starting_track.seconds[-1] >= ending_track.seconds[0]:
            raise RuntimeError(
                "First track needs to end before the second starts for concatenation"
            )

        self._src[self._src.index(starting_track)] = starting_track + ending_track
        self._src.remove(ending_track)

    def _merge_tracks(self, starting_track, starting_node, ending_track, ending_node):
        """Connect two tracks from any given nodes, removing the points in between.

        Note: Any specialized refinement (eg. gaussian refinement) will be lost when
        tracks are merged.

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
        """
        if starting_track not in self._src or ending_track not in self._src:
            raise RuntimeError("Both tracks need to be part of this group to be merged")

        starting_node = int(starting_node)
        ending_node = int(ending_node)

        start_time_idx = starting_track.time_idx[starting_node]
        end_time_idx = ending_track.time_idx[ending_node]
        assert start_time_idx != end_time_idx, "Cannot connect two points with the same time index."

        # ensure that tracks are properly ordered so resulting merge track
        # has coordinates sorted in ascending time indices
        if start_time_idx > end_time_idx:
            starting_track, ending_track = ending_track, starting_track
            starting_node, ending_node = ending_node, starting_node

        # up to and *including* starting_node
        first_half = starting_track._with_coordinates(
            starting_track.time_idx[: starting_node + 1],
            starting_track.coordinate_idx[: starting_node + 1],
        )

        last_half = ending_track._with_coordinates(
            ending_track.time_idx[ending_node:],
            ending_track.coordinate_idx[ending_node:],
        )

        self._src[self._src.index(starting_track)] = first_half + last_half
        if starting_track != ending_track:
            self._src.remove(ending_track)

    def __len__(self):
        return len(self._src)

    def extend(self, other):
        if not other:
            return self

        if not (isinstance(other, KymoTrack) or isinstance(other, self.__class__)):
            raise TypeError(
                f"You can only extend a {self.__class__.__name__} with a {self.__class__.__name__} "
                f"or {KymoTrack.__name__}"
            )

        other = self.__class__([other]) if isinstance(other, KymoTrack) else other
        other_kymo, other_channel = self._validate_single_source(other)
        if self:
            assert self._kymo._id == other_kymo, "All tracks must have the same source kymograph."
            assert self._channel == other_channel, "All tracks must be from the same color channel."

        self._src.extend(other._src)

    @deprecated(
        reason=(
            "This method will be removed in a future release. Use `remove_tracks_in_rect()` instead."
        ),
        action="always",
        version="0.13.0",
    )
    def remove_lines_in_rect(self, rect):
        self.remove_tracks_in_rect(rect)

    def remove_tracks_in_rect(self, rect):
        """Removes tracks that fall in a particular region. Note that if any point on a track falls
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

        self._src = [track for track in self._src if not track.in_rect(rect)]

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
            ax.set_ylabel(f"position ({self._kymo._calibration.unit_label})")
            ax.set_xlabel("time (s)")

    def save(self, filename, delimiter=";", sampling_width=None):
        """Export kymograph tracks to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph tracks to.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph track and export the summed intensity
            with the image. The value indicates the number of pixels in either direction to sum over.
        """
        export_kymotrackgroup_to_csv(filename, self, delimiter, sampling_width)

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

        tracks = (
            filter(KymoTrack._check_ends_are_defined, self) if exclude_ambiguous_dwells else self
        )
        dwelltimes_sec = np.array([track.seconds[-1] - track.seconds[0] for track in tracks])

        nonzero_dwelltimes_sec = dwelltimes_sec[dwelltimes_sec > 0]
        if len(nonzero_dwelltimes_sec) != len(dwelltimes_sec):
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

        if nonzero_dwelltimes_sec.size == 0:
            raise RuntimeError("No tracks available for analysis")

        min_observation_time = np.min(nonzero_dwelltimes_sec)
        max_observation_time = self[0]._image.shape[1] * self[0]._line_time_seconds

        return DwelltimeModel(
            nonzero_dwelltimes_sec,
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
        plt.xlabel(f"Position ({self._kymo._calibration.unit_label})")

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
        if n_time_bins == 0:
            raise ValueError("Number of time bins must be > 0.")
        if n_position_points < 2:
            raise ValueError("Number of spatial bins must be >= 2.")

        if roi is None:
            n_rows, n_frames = self._kymo.get_image(self._channel).shape
            start_frame = 0
            min_position = 0
            max_position = n_rows * self._kymo.pixelsize[0]
        else:
            (min_time, min_position), (max_time, max_position) = roi
            n_rows = np.ceil((max_position - min_position) / self._kymo.pixelsize[0])
            n_frames = np.ceil((max_time - min_time) / self._kymo.line_time_seconds)
            start_frame = min_time // self._kymo.line_time_seconds

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

        References
        ----------
        .. [7] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
               diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
               022726.
        """
        if method == "cve":
            return ensemble_cve(self)
        elif method == "ols":
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

        src_calibration = self._kymo._calibration
        return calculate_ensemble_msd(
            line_msds=track_msds,
            time_step=self._kymo.line_time_seconds,
            unit=src_calibration.unit,
            unit_label=src_calibration.unit_label,
            min_count=min_count,
        )
