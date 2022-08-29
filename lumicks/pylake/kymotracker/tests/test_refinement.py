import pytest
import numpy as np
from lumicks.pylake.kymotracker.kymotracker import *
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def test_kymotrack_interpolation(blank_kymo):
    time_idx = np.array([1.0, 3.0, 5.0])
    coordinate_idx = np.array([1.0, 3.0, 3.0])
    kymotrack = KymoTrack(time_idx, coordinate_idx, blank_kymo, "red")
    interpolated = kymotrack.interpolate()
    np.testing.assert_allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])

    # Test whether concatenation still works after interpolation
    np.testing.assert_allclose(
        (interpolated + kymotrack).time_idx, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0]
    )
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

    track = KymoTrack(time_idx[::2], coordinate_idx[::2], kymo, "red")
    refined_track = refine_tracks_centroid([track], 5)[0]
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


@pytest.mark.parametrize("loc", [25.3, 25.5, 26.25, 23.6])
def test_refinement_track(loc, inv_sigma=0.3):
    # TODO: test does not run, same name as below
    xx = np.arange(0, 50) - loc
    image = np.exp(-inv_sigma * xx * xx)
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

    track = refine_tracks_centroid([KymoTrack([0], [25], kymo, "red")], 5)[0]
    np.testing.assert_allclose(track.coordinate_idx, loc, rtol=1e-2)


def test_refinement_error(kymo_integration_test_data):
    with pytest.raises(ValueError, match="track_width should be larger than zero"):
        refine_tracks_centroid([KymoTrack([0], [25], kymo_integration_test_data, "red")], 0)[0]

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="line_width may not be smaller than 1"):
            refine_lines_centroid([KymoTrack([0], [25], kymo_integration_test_data, "red")], 0)[0]


@pytest.mark.parametrize("fit_mode", ["ignore", "multiple"])
def test_gaussian_refinement(kymogroups_2tracks, fit_mode):
    tracks, gapped_tracks, mixed_tracks = kymogroups_2tracks

    # test aliasing
    with pytest.warns(DeprecationWarning):
        refined = refine_lines_gaussian(
            tracks, window=3, refine_missing_frames=True, overlap_strategy=fit_mode
        )
        assert np.allclose(
            refined[0].position, [3.54796254, 3.52869381, 3.51225177, 3.38714711, 3.48588436]
        )
        assert np.allclose(
            refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495, 4.99092852]
        )

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

    # invalid overlap strategy
    with pytest.raises(AssertionError):
        refined = refine_tracks_gaussian(
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


@pytest.mark.parametrize("fit_mode", ["ignore", "multiple"])
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


def test_gaussian_refinement_overlap(kymogroups_close_tracks):
    refined = refine_tracks_gaussian(
        kymogroups_close_tracks,
        window=15,
        refine_missing_frames=True,
        overlap_strategy="multiple",
        fixed_background=1.0,
    )
    assert np.allclose(
        refined[0].position,
        [5.24723138, 5.08524557, 4.6939314, 4.84496914, 4.78668516],
    )
    assert np.allclose(
        refined[1].position,
        [3.32775782, 3.42564736, 3.33315701, 3.60090496, 3.26356061],
    )


def test_filter_tracks(blank_kymo):
    k1 = KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red")
    k2 = KymoTrack([2, 3], [1, 2], blank_kymo, "red")
    k3 = KymoTrack([2, 3, 4, 5], [1, 2, 4, 5], blank_kymo, "red")
    tracks = KymoTrackGroup([k1, k2, k3])
    assert len(filter_tracks(tracks, 5)) == 0
    assert all([track1 == track2 for track1, track2 in zip(filter_tracks(tracks, 5), [k1, k3])])
    assert all([track1 == track2 for track1, track2 in zip(filter_tracks(tracks, 2), [k1, k2, k3])])

    with pytest.warns(DeprecationWarning):
        filter_lines(tracks, 5)
