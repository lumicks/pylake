import re

import numpy as np
import scipy
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
from lumicks.pylake.kymotracker.kymotracker import *
from lumicks.pylake.tests.data.mock_confocal import generate_kymo
from lumicks.pylake.kymotracker.detail.localization_models import GaussianLocalizationModel


def test_kymotrack_interpolation(blank_kymo, blank_kymo_track_args):
    time_idx = np.array([1, 3, 5])
    coordinate_idx = np.array([1.0, 3.0, 3.0])
    kymotrack = KymoTrack(time_idx, coordinate_idx, *blank_kymo_track_args)
    interpolated = kymotrack.interpolate()
    np.testing.assert_equal(interpolated.time_idx, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])

    # Test whether concatenation still works after interpolation
    np.testing.assert_equal((interpolated + kymotrack).time_idx, [1, 2, 3, 4, 5, 1, 3, 5])
    np.testing.assert_allclose(
        (interpolated + kymotrack).coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0]
    )


def test_refinement_2d():
    time_idx = np.array([1, 2, 3, 4, 5])
    coordinate_idx = np.array([1, 2, 3, 3, 3])

    # Draw image with a deliberate offset
    offset = 2
    data = np.zeros((7, 7))
    data[coordinate_idx + offset, time_idx] = 5
    data[coordinate_idx - 1 + offset, time_idx] = 1
    data[coordinate_idx + 1 + offset, time_idx] = 1

    kymo = generate_kymo(
        "",
        data,
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    track = KymoTrack(time_idx[::2], coordinate_idx[::2], kymo, "red", kymo.line_time_seconds)
    refined_track = refine_tracks_centroid([track], 5, bias_correction=False)[0]
    np.testing.assert_allclose(refined_track.time_idx, time_idx)
    np.testing.assert_allclose(refined_track.coordinate_idx, coordinate_idx + offset)

    # Test whether concatenation still works after refinement
    np.testing.assert_allclose(
        (refined_track + track).time_idx, np.hstack((time_idx, time_idx[::2]))
    )
    np.testing.assert_allclose(
        (refined_track + track).coordinate_idx,
        np.hstack((coordinate_idx + offset, coordinate_idx[::2])),
    )


@pytest.mark.parametrize("loc, ref_counts", [(25.3, 29), (25.5, 29), (26.25, 28), (23.6, 27)])
def test_refinement_track(loc, ref_counts):
    xx = np.arange(0, 50) - loc
    image = np.exp(-0.3 * xx * xx)
    # real kymo pixel values are integer photon counts
    # multiply by some value and convert to int, otherwise kymo.red_image is zeros
    image = np.array(image * 10).astype(int)

    kymo = generate_kymo(
        "",
        np.expand_dims(image, 1),
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    track = refine_tracks_centroid([KymoTrack([0], [25], kymo, "red", 1)], 5)[0]
    np.testing.assert_allclose(track.coordinate_idx, loc, rtol=1e-2)
    np.testing.assert_equal(track.photon_counts, ref_counts)


@pytest.mark.parametrize("loc, ref_count", [(25.3, 29), (25.5, 29), (26.25, 28), (23.6, 27)])
def test_refinement_with_background(loc, ref_count):
    xx = np.arange(0, 50) - loc
    background = 50
    image = np.array((np.exp(-0.3 * xx * xx)) * 10 + background).astype(int)
    kymo = generate_kymo("", np.expand_dims(image, 1), pixel_size_nm=1000)

    # Without bias correction, we should see worse quality estimates
    tracks = [KymoTrack([0], [25], kymo, "red", 1)]
    refinement_width = 5
    track = refine_tracks_centroid(tracks, refinement_width, bias_correction=False)[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(track.coordinate_idx, loc, rtol=1e-2)

    # With correction, this should resolve
    track = refine_tracks_centroid(tracks, refinement_width, bias_correction=True)[0]
    np.testing.assert_allclose(track.coordinate_idx, loc, rtol=1e-2)
    np.testing.assert_equal(track.photon_counts, ref_count + background * refinement_width)


def test_refinement_error(kymo_integration_test_data):
    args = [
        [0],
        [25],
        kymo_integration_test_data,
        "red",
        kymo_integration_test_data.line_time_seconds,
    ]
    with pytest.raises(
        ValueError, match=re.escape("track_width must at least be 3 pixels (0.150 [um])")
    ):
        refine_tracks_centroid([KymoTrack(*args)], 0.149)[0]

    # This should be fine though
    refine_tracks_centroid([KymoTrack(*args)], 0.15)[0]


def test_centroid_refinement_multiple_sources(kymogroups_2tracks, kymogroups_close_tracks):
    tracks1, *_ = kymogroups_2tracks
    tracks2 = kymogroups_close_tracks
    assert [id(k) for k in tracks1._kymos] != [id(k) for k in tracks2._kymos]

    tracks3 = tracks1[:1] + tracks2 + tracks1[1:]

    refined_tracks1 = refine_tracks_centroid(tracks1)
    refined_tracks2 = refine_tracks_centroid(tracks2)
    refined_tracks3 = refine_tracks_centroid(tracks3)

    reference_refined_tracks = refined_tracks1[:1] + refined_tracks2 + refined_tracks1[1:]

    for track, ref_track in zip(refined_tracks3, reference_refined_tracks):
        np.testing.assert_allclose(track.position, ref_track.position)


@pytest.mark.parametrize("fit_mode", ["ignore", "simultaneous"])
def test_gaussian_refinement(kymogroups_2tracks, fit_mode):
    tracks, gapped_tracks, mixed_tracks = kymogroups_2tracks

    # full data, no overlap
    refined = refine_tracks_gaussian(
        tracks, window=3, refine_missing_frames=True, overlap_strategy=fit_mode
    )
    assert np.allclose(
        refined[0].position, [3.54796254, 3.52869381, 3.51225177, 3.38714711, 3.48588436]
    )
    assert np.allclose(
        refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495, 4.99092852]
    )

    # initial guess for sigma
    refined = refine_tracks_gaussian(
        tracks, window=3, refine_missing_frames=True, overlap_strategy=fit_mode, initial_sigma=0.250
    )
    assert np.allclose(
        refined[0].position, [3.54796279, 3.52869369, 3.51225138, 3.46877412, 3.48588434]
    )
    assert np.allclose(
        refined[1].position, [4.96700218, 4.99771571, 5.04086917, 5.00664717, 4.9909296]
    )

    # all frames overlap, therefore skipped and result is empty
    with pytest.warns(UserWarning):
        refined = refine_tracks_gaussian(
            tracks, window=10, refine_missing_frames=True, overlap_strategy="skip"
        )
    assert len(refined) == 0

    with pytest.raises(ValueError, match="Invalid overlap strategy selected."):
        refine_tracks_gaussian(
            tracks, window=3, refine_missing_frames=True, overlap_strategy="something"
        )

    # gapped data, fill in missing frames
    refined = refine_tracks_gaussian(
        gapped_tracks, window=3, refine_missing_frames=True, overlap_strategy="skip"
    )
    assert np.allclose(
        refined[0].position, [3.54796254, 3.52869381, 3.51225177, 3.38714711, 3.48588436]
    )
    assert np.allclose(
        refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495, 4.99092852]
    )

    # gapped data, skip missing frames
    refined = refine_tracks_gaussian(
        gapped_tracks, window=3, refine_missing_frames=False, overlap_strategy=fit_mode
    )
    assert np.allclose(refined[0].position, [3.54796254, 3.52869381, 3.38714711, 3.48588436])
    assert np.allclose(refined[1].position, [4.96700319, 4.99771575, 5.0066495, 4.99092852])

    # mixed length tracks, no overlap
    refined = refine_tracks_gaussian(
        mixed_tracks, window=3, refine_missing_frames=True, overlap_strategy="skip"
    )
    assert np.allclose(refined[0].position, [3.52869383, 3.51225048])
    assert np.allclose(
        refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495, 4.99092852]
    )

    # mixed length tracks, track windows overlap
    with pytest.warns(UserWarning):
        refined = refine_tracks_gaussian(
            mixed_tracks, window=10, refine_missing_frames=True, overlap_strategy="skip"
        )
    # all frames in mixed_tracks[0] overlap with second track, all skipped
    assert len(refined) == 1
    # 2 frames in mixed_tracks[1] overlap with first track, 2 skipped, 3 fitted
    assert np.allclose(refined[0].position, [4.94659924, 5.00920806, 4.97724526])


@pytest.mark.parametrize("fit_mode", ["ignore", "simultaneous"])
def test_gaussian_refinement_fixed_background(kymogroups_2tracks, fit_mode):
    tracks, _, _ = kymogroups_2tracks

    refined = refine_tracks_gaussian(
        tracks,
        window=3,
        refine_missing_frames=True,
        overlap_strategy=fit_mode,
        fixed_background=1.0,
    )
    assert np.allclose(
        refined[0].position,
        [3.54875771, 3.52793245, 3.56789807, 3.46844518, 3.48508813],
    )
    assert np.allclose(
        refined[1].position,
        [4.96956982, 4.99811141, 5.02009032, 5.01614766, 4.99094119],
    )


def test_no_swap_gaussian_refinement():
    """This test ensures that tracks don't swap during a Gaussian refinement. What can happen
    during refinement is that a fluorophore blinks, and that the refinement then interpolates
    over this blinking state. When this happens, the optimization problem is poorly defined,
    since we have one more Gaussian in the model than needed. To ensure that we don't start
    swapping Gaussians around (making the localization jump between tracks), we enforce that
    they stay within limits dictated by their order."""

    def gen_gaussians(locs):
        x = np.arange(0, 20, 1)
        return np.random.poisson(
            200 * np.sum(np.vstack([scipy.stats.norm.pdf(x, loc=loc) for loc in locs]), axis=0) + 10
        )

    np.random.seed(31415)
    locations = (5, 9, 12, 16)
    kymo = _kymo_from_array(
        np.vstack((gen_gaussians(locations[:-1]), gen_gaussians(locations))).T,
        "r",
        1,
    )
    group = KymoTrackGroup(
        [
            KymoTrack(np.array([0, 1]), np.array([loc, loc]), kymo, "red", kymo.line_time_seconds)
            for loc in locations
        ]
    )

    refined_group = refine_tracks_gaussian(
        group, window=10, refine_missing_frames=True, overlap_strategy="simultaneous"
    )

    # We want the ones that have signal to move by less than a pixel. Note that the position of
    # the last track is arbitrary (no signal there) and therefore not part of the test.
    assert all(abs(np.diff(track.coordinate_idx)) < 1 for track in refined_group[:-1])


@pytest.mark.filterwarnings("ignore:overlap_strategy=")
@pytest.mark.parametrize(
    "fit_mode, ref_pos1, ref_pos2",
    [
        (
            "simultaneous",
            [5.24719911, 5.08517919, 4.75665519, 4.87242494, 4.62156363],
            [3.32774207, 3.42563247, 3.37547213, 3.61309564, 3.22537837],
        ),
        (
            "multiple",
            [5.24723138, 5.08524557, 4.6939314, 4.84496914, 4.78668516],
            [3.32775782, 3.42564736, 3.33315701, 3.60090496, 3.26356061],
        ),
    ],
)
def test_gaussian_refinement_overlap(kymogroups_close_tracks, fit_mode, ref_pos1, ref_pos2):
    refined = refine_tracks_gaussian(
        kymogroups_close_tracks,
        window=15,
        refine_missing_frames=True,
        overlap_strategy=fit_mode,
        fixed_background=1.0,
    )

    assert np.allclose(refined[0].position, ref_pos1)
    assert np.allclose(refined[1].position, ref_pos2)


def test_gaussian_refinement_multiple_sources(kymogroups_2tracks, kymogroups_close_tracks):
    tracks1, *_ = kymogroups_2tracks
    tracks2 = kymogroups_close_tracks
    assert [id(k) for k in tracks1._kymos] != [id(k) for k in tracks2._kymos]

    tracks3 = tracks1[:1] + tracks2 + tracks1[1:]

    refinement_kwargs = {
        "window": 3,
        "refine_missing_frames": False,
        "overlap_strategy": "simultaneous",
    }

    refined_tracks1 = refine_tracks_gaussian(tracks1, **refinement_kwargs)
    refined_tracks2 = refine_tracks_gaussian(tracks2, **refinement_kwargs)
    refined_tracks3 = refine_tracks_gaussian(tracks3, **refinement_kwargs)

    reference_refined_tracks = refined_tracks1[:1] + refined_tracks2 + refined_tracks1[1:]

    for track, ref_track in zip(refined_tracks3, reference_refined_tracks):
        np.testing.assert_allclose(track.position, ref_track.position)


@pytest.mark.filterwarnings("ignore:There were")
def test_gaussian_refinement_min_time_ordering():
    kymo = _kymo_from_array(
        np.ones((50, 50)), pixel_size_um=1, line_time_seconds=2, color_format="r"
    )

    k1 = KymoTrack(np.arange(5), np.zeros(5) + 2 * np.random.rand(5), kymo, "red", 5)
    k2 = KymoTrack(np.arange(2, 7), np.ones(5) * 7 + 2 * np.random.rand(5), kymo, "red", 1)
    k3 = KymoTrack(np.arange(5), np.ones(5) * 14 + 2 * np.random.rand(5), kymo, "red", 2)
    ktg = KymoTrackGroup([k1, k2, k3])

    refined = refine_tracks_gaussian(
        ktg, overlap_strategy="skip", window=3, refine_missing_frames=False
    )
    for track, ref in zip(refined, (5, 1, 2)):
        if len(track) > 0:
            assert np.all(track._minimum_observable_duration == ref)


@pytest.mark.filterwarnings("ignore:There were")
def test_gaussian_refinement_min_time_ordering_skip_tracks():
    kymo = _kymo_from_array(
        np.ones((50, 50)), pixel_size_um=1, line_time_seconds=2, color_format="r"
    )
    k1 = KymoTrack(np.arange(5), np.zeros(5), kymo, "red", 4)
    k2 = KymoTrack(np.arange(5), np.zeros(5), kymo, "red", 5)
    k3 = KymoTrack(np.arange(5), np.ones(5) * 10, kymo, "red", 6)
    ktg = KymoTrackGroup([k1, k2, k3])
    refined = refine_tracks_gaussian(
        ktg, overlap_strategy="skip", window=3, refine_missing_frames=False
    )
    assert refined[0]._minimum_observable_duration == 6
    assert len(refined) == 1


def test_no_model_fit(blank_kymo, blank_kymo_track_args):
    with pytest.raises(
        NotImplementedError, match="No model fit available for this localization method."
    ):
        KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args)._model_fit(1)


@pytest.mark.parametrize("method", ["_model_fit", "plot_fit"])
def test_gaussian_model_fit(method):
    pixel_size_um = 2.0
    kymo_data = [0, 1, 2, 1, 2, 1, 0]
    kymo = _kymo_from_array(np.tile(kymo_data, (4, 1)).T, "r", 1, pixel_size_um=pixel_size_um)
    gauss_loc = GaussianLocalizationModel(
        position=np.array([2.0, 2.0]),
        total_photons=np.array([20, 30]),
        sigma=np.array([1.0, 1.0]),
        background=np.array([10, 15]),
        _overlap_fit=np.array([True, True]),
    )
    track = KymoTrack(np.array([1, 3]), gauss_loc, kymo, "red", kymo.line_time_seconds)
    tested_method = getattr(track, method)
    ref_coords = np.arange(0, len(kymo_data) * pixel_size_um, 0.1 * pixel_size_um)

    for node_idx in (0, 1):
        coords, data = track._model_fit(node_idx)
        tested_method(node_idx)
        np.testing.assert_allclose(coords, ref_coords)
        np.testing.assert_allclose(data, gauss_loc.evaluate(coords, node_idx, pixel_size_um))

    for node_idx in (-1, -2):
        coords, data = track._model_fit(node_idx)
        tested_method(node_idx)
        np.testing.assert_allclose(coords, ref_coords)
        np.testing.assert_allclose(data, gauss_loc.evaluate(coords, 2 + node_idx, pixel_size_um))

    for node_idx in (-3, 2):
        with pytest.raises(
            IndexError, match="Node index is out of range of the KymoTrack. Kymotrack has length 2"
        ):
            track._model_fit(node_idx)
            tested_method(node_idx)


def test_gaussian_refinement_plotting():
    kymo = _kymo_from_array(np.tile([0, 1, 2, 1, 2, 1, 0], (4, 1)).T, "r", 1, pixel_size_um=1)
    group = KymoTrackGroup(
        [
            KymoTrack(np.array([0, 2]), np.array([2, 2]), kymo, "red", kymo.line_time_seconds),
            KymoTrack(
                np.array([0, 1, 2]), np.array([4, 4, 4]), kymo, "red", kymo.line_time_seconds
            ),
        ]
    )

    # Only Gaussian refinement has a model visualization available
    for to_be_plotted in (group, group[0]):
        with pytest.raises(
            NotImplementedError, match="No model fit available for this localization method."
        ):
            to_be_plotted.plot_fit(0)

    refined = refine_tracks_gaussian(
        group, window=2, refine_missing_frames=False, overlap_strategy="simultaneous"
    )

    for kymo_frame_idx in range(-4, 4):
        refined.plot_fit(frame_idx=kymo_frame_idx)

    for kymo_frame_idx in (-5, 5):
        with pytest.raises(
            IndexError, match="Frame index is out of range of the kymograph. Kymograph length is 4"
        ):
            refined.plot_fit(frame_idx=kymo_frame_idx)

    with pytest.raises(
        RuntimeError,
        match=re.escape("No kymo associated with this empty group (no tracks available)"),
    ):
        KymoTrackGroup([]).plot_fit(0)


def test_filter_tracks(blank_kymo, blank_kymo_track_args):
    k1 = KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args)
    k2 = KymoTrack([2, 3], [1, 2], *blank_kymo_track_args)
    k3 = KymoTrack([2, 3, 4, 5], [1, 2, 4, 5], *blank_kymo_track_args)
    tracks = KymoTrackGroup([k1, k2, k3])
    assert len(filter_tracks(tracks, minimum_length=5)) == 0

    # We compare the positions since a track doesn't have a proper equality operator.
    filtered = filter_tracks(tracks, minimum_length=5)
    assert all(
        [
            np.array_equal(track1.position, track2.position)
            for track1, track2 in zip(filtered, [k1, k3])
        ]
    )
    assert all([t._minimum_observable_duration == 4 * k1._kymo.line_time_seconds for t in filtered])

    # Ensure that if we filter again with a shorter filter, we don't reduce the minimum observable
    # time, since we wouldn't recover lines.
    twice = filter_tracks(filtered, minimum_length=2)
    assert all([t._minimum_observable_duration == 4 * k1._kymo.line_time_seconds for t in twice])

    filtered = filter_tracks(tracks, minimum_length=2)
    assert all(
        [
            np.array_equal(track1.position, track2.position)
            for track1, track2 in zip(filtered, [k1, k2, k3])
        ]
    )
    assert all([t._minimum_observable_duration == k1._kymo.line_time_seconds for t in filtered])

    # Ensure that a non-identified minimum time still gets handled gracefully.
    ktg = KymoTrackGroup([KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red", None)])
    filtered = filter_tracks(ktg, minimum_length=2)
    np.testing.assert_allclose(
        filtered[0]._minimum_observable_duration, blank_kymo.line_time_seconds
    )


@pytest.mark.parametrize(
    "minimum_duration, ref_length, ref_minimum, ref_minimum_dt2",
    [
        [1.0, 5, 1.0, 2.0],
        [1.1, 4, 2.0, 2.0],  # Going over slightly bumps the minimum time to the next frame
        [2.0, 4, 2.0, 2.0],
        [2.1, 2, 3.0, 4.0],  # Last line gets a bigger bump in minimum time once we filter
    ],
)
def test_filter_by_duration(
    blank_kymo_track_args, minimum_duration, ref_length, ref_minimum, ref_minimum_dt2
):
    k1 = KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args)  # duration = 2
    k2 = KymoTrack([2, 3], [1, 2], *blank_kymo_track_args)  # duration = 1
    k3 = KymoTrack([2, 3, 5], [1, 2, 5], *blank_kymo_track_args)  # duration = 3

    # Track with different line time. We use two line times because we want to test that the
    # minimum observable time gets bumped correctly to the next full timestep.
    kymo2 = _kymo_from_array(np.zeros((3, 3)), "r", line_time_seconds=2.0, pixel_size_um=50)
    k4 = KymoTrack([2, 3], [1, 2, 5], kymo2, "red", 0)  # duration = 2
    k5 = KymoTrack([2, 6], [1, 2, 5], kymo2, "red", 0)  # duration = 6

    tracks = KymoTrackGroup([k1, k2, k3, k4, k5])

    filtered = filter_tracks(tracks, minimum_length=0, minimum_duration=minimum_duration)
    assert len(tracks) == 5  # return new instance

    tracks.filter(minimum_length=0, minimum_duration=minimum_duration)
    assert len(tracks) == ref_length  # in-place modification

    for filtered_tracks in (filtered, tracks):
        assert len(filtered_tracks) == ref_length
        np.testing.assert_allclose(filtered_tracks[0]._minimum_observable_duration, ref_minimum)
        np.testing.assert_allclose(
            filtered_tracks[-1]._minimum_observable_duration, ref_minimum_dt2
        )


def test_empty_group():
    """Validate that the refinement methods don't fail when applied to an empty group"""
    tracks = KymoTrackGroup([])

    result = refine_tracks_gaussian(tracks, 5, False, "simultaneous")
    assert id(tracks) != result  # Validate that we get a new object
    assert isinstance(result, KymoTrackGroup)
    assert len(result) == 0

    result = refine_tracks_centroid(tracks, 5)
    assert id(tracks) != result
    assert isinstance(result, KymoTrackGroup)
    assert len(result) == 0


def test_bias_corrected_refinement_background(kymogroups_2tracks):
    tracks, _, _ = kymogroups_2tracks

    refined = refine_tracks_centroid(
        tracks,
        track_width=2,
        bias_correction=True,
    )

    assert np.allclose(
        refined[0].position,
        [3.53199999, 3.56773634, 3.48390805, 3.33971131, 3.48724068],
    )
    assert np.allclose(
        refined[1].position,
        [5.06666451, 4.83968723, 4.9625, 5.088, 5.02488394],
    )
