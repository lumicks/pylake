import io
import re
import inspect
from copy import copy
from pathlib import Path

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import (
    KymoTrack,
    KymoTrackGroup,
    _read_txt,
    import_kymotrackgroup_from_csv,
)
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def compare_kymotrack_group(group1, group2):
    assert len(group1) == len(group2)
    attributes = (
        "coordinate_idx",
        "time_idx",
        "position",
        "seconds",
        "_minimum_observable_duration",
    )
    for track1, track2 in zip(group1, group2):
        for attr in attributes:
            attr1, attr2 = getattr(track1, attr), getattr(track2, attr)
            np.testing.assert_allclose(attr1, attr2)
            if not (np.isscalar(attr1) and np.isscalar(attr2)):
                assert len(attr1) == len(attr2)


@pytest.mark.parametrize(
    "dt, dx, delimiter, sampling_width, sampling_outcome",
    [
        [int(1e9), 1.0, ";", 0, 2],
        [int(2e9), 1.0, ";", 0, 2],
        [int(1e9), 2.0, ";", 0, 2],
        [int(1e9), 1.0, ",", 0, 2],
        [int(1e9), 1.0, ";", 1, 3],
        [int(1e9), 2.0, ";", None, None],
        [np.int64(12800), 1.0, ";", 0, 2],  # realistic infowave dt
    ],
)
def test_kymotrackgroup_io(tmpdir_factory, dt, dx, delimiter, sampling_width, sampling_outcome):
    track_coordinates = [
        ((1, 2, 3), (2, 3, 4)),
        ((2, 3, 4), (3, 4, 5)),
        ((3, 4, 5), (4, 5, 6)),
        ((4, 5, 6), (5, 6, 7)),
    ]
    test_data = np.zeros((8, 8))
    for time_idx, position_idx in track_coordinates:
        test_data[np.array(position_idx).astype(int), np.array(time_idx).astype(int)] = 2
        test_data[np.array(position_idx).astype(int) - 1, np.array(time_idx).astype(int)] = 1

    kymo = generate_kymo(
        "test",
        test_data,
        pixel_size_nm=dx * 1000,
        start=np.int64(20e9),
        dt=dt,
        samples_per_pixel=5,
        line_padding=3,
    )

    tracks = KymoTrackGroup(
        [
            KymoTrack(
                np.array(time_idx), np.array(position_idx), kymo, "red", kymo.line_time_seconds
            )
            for time_idx, position_idx in track_coordinates
        ]
    )

    # Test round trip through the API
    testfile = f"{tmpdir_factory.mktemp('pylake')}/test.csv"
    tracks.save(testfile, delimiter, sampling_width, correct_origin=True)
    imported_tracks = import_kymotrackgroup_from_csv(testfile, kymo, "red", delimiter=delimiter)
    data, pylake_version, csv_version = _read_txt(testfile, delimiter)

    compare_kymotrack_group(tracks, imported_tracks)

    for track1, time in zip(tracks, data["time (seconds)"]):
        np.testing.assert_allclose(track1.seconds, time)

    for track1, coord in zip(tracks, data["position (um)"]):
        np.testing.assert_allclose(track1.position, coord)

    if sampling_width is None:
        assert len([key for key in data.keys() if "counts" in key]) == 0
    else:
        count_field = [key for key in data.keys() if "counts" in key][0]
        for track1, imported_track, cnt in zip(tracks, imported_tracks, data[count_field]):
            np.testing.assert_allclose([sampling_outcome] * len(track1.coordinate_idx), cnt)
            np.testing.assert_allclose(imported_track.photon_counts, cnt)


def test_export_sources(tmpdir_factory):
    kymo1 = _kymo_from_array(np.random.poisson(5, (5, 5, 3)), "rgb", 1e-4, start=20e9)
    kymo2 = copy(kymo1)
    tracks1 = KymoTrackGroup([KymoTrack(np.arange(3), np.arange(3), kymo1, "red", 0)])
    tracks2 = KymoTrackGroup([KymoTrack(np.arange(5), np.arange(5), kymo2, "red", 0)])
    tracks3 = tracks1 + tracks2

    testfile = f"{tmpdir_factory.mktemp('pylake')}/failed_test.csv"

    # can export group with single source
    tracks1.save(testfile, ";")

    # cannot export group with more than one source
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Exporting a group with tracks from more than a single source kymograph is not supported. "
            "This group contains tracks from 2 source kymographs."
        ),
    ):
        tracks3.save(testfile, ";")


@pytest.mark.parametrize(
    "delimiter, sampling_width, correct_origin",
    [[";", 0, True], [",", 0, True], [";", 1, True], [";", None, True]],
)
def test_roundtrip_without_file(
    delimiter, sampling_width, correct_origin, kymo_integration_test_data, kymo_integration_tracks
):
    # Validate that this also works when providing a string handle (this is the API LV uses).

    def get_args(func):
        return list(inspect.signature(func).parameters.keys())

    # This helps us ensure that if we get additional arguments to this function, we don't forget to
    # add them to the parametrization here.
    assert set(get_args(KymoTrackGroup.save)[2:]) == set(get_args(test_roundtrip_without_file)[:-2])

    with io.StringIO() as s:
        kymo_integration_tracks.save(
            s, delimiter=delimiter, sampling_width=sampling_width, correct_origin=correct_origin
        )
        string_representation = s.getvalue()

    with io.StringIO(string_representation) as s:
        read_tracks = import_kymotrackgroup_from_csv(
            s, kymo_integration_test_data, "red", delimiter=delimiter
        )

    compare_kymotrack_group(kymo_integration_tracks, read_tracks)


def test_photon_count_validation(kymo_integration_test_data, kymo_integration_tracks):
    with io.StringIO() as s:
        kymo_integration_tracks.save(s, sampling_width=0, correct_origin=False)
        biased_tracks = s.getvalue()

    with io.StringIO() as s:
        kymo_integration_tracks.save(s, sampling_width=0, correct_origin=True)
        good_tracks = s.getvalue()

    with pytest.warns(
        RuntimeWarning,
        match="origin of a pixel to be at the edge rather than the center of the pixel",
    ):
        _ = import_kymotrackgroup_from_csv(
            io.StringIO(biased_tracks), kymo_integration_test_data, "red"
        )

    # We can also fail by having the wrong kymo
    with pytest.warns(
        RuntimeWarning,
        match="loaded kymo or channel doesn't match the one used to create this file",
    ):
        _ = import_kymotrackgroup_from_csv(
            io.StringIO(good_tracks), kymo_integration_test_data, "green"
        )

    # Or by having the wrong one where it actually completely fails to sample. This tests whether
    # the exception inside import_kymotrackgroup_from_csv is correctly caught and handled
    with pytest.warns(
        RuntimeWarning,
        match="loaded kymo or channel doesn't match the one used to create this file",
    ):
        _ = import_kymotrackgroup_from_csv(
            io.StringIO(good_tracks), kymo_integration_test_data[:"1s"], "red"
        )

    # Control for good tracks
    import_kymotrackgroup_from_csv(io.StringIO(good_tracks), kymo_integration_test_data, "red")


@pytest.mark.parametrize(
    "version, read_with_version",
    [[0, False], [1, False], [2, True], [3, True], [4, True]],
)
def test_csv_version(version, read_with_version, recwarn):
    # Test that header is parsed properly on CSV import
    # Version 2 has 2 header lines, <2 only has 1 header line

    track_coordinates = [
        ((1, 2, 3), (2, 3, 4)),
        ((2, 3, 4), (3, 4, 5)),
    ]

    test_data = np.zeros((8, 8))
    for time_idx, position_idx in track_coordinates:
        test_data[np.array(position_idx).astype(int), np.array(time_idx).astype(int)] = 2
        test_data[np.array(position_idx).astype(int) - 1, np.array(time_idx).astype(int)] = 1

    kymo = generate_kymo(
        "test",
        test_data,
        pixel_size_nm=1.0 * 1000,
        start=np.int64(20e9),
        dt=np.int64(12800),
        samples_per_pixel=5,
        line_padding=3,
    )

    testfile = Path(__file__).parent / f"./data/tracks_v{version}.csv"
    imported_tracks = import_kymotrackgroup_from_csv(testfile, kymo, "red", delimiter=";")

    match version:
        case 3:
            assert "minimum observable track duration" in str(recwarn[0].message)
        case 4:
            assert "loaded kymo or channel doesn't match the one used to create this file" in str(
                recwarn[0].message
            )

    data, pylake_version, csv_version = _read_txt(testfile, ";")

    if read_with_version:
        assert pylake_version is not None
    else:
        assert pylake_version is None

    for j, track in enumerate(imported_tracks):
        np.testing.assert_allclose(track.time_idx, data["time (pixels)"][j])
        np.testing.assert_allclose(track.coordinate_idx, data["coordinate (pixels)"][j])


@pytest.mark.parametrize("filename", ["csv_bad_format.csv", "csv_unparseable.csv"])
def test_bad_csv(filename, blank_kymo):
    with pytest.raises(IOError, match="Invalid file format!"):
        file = Path(__file__).parent / "data" / filename
        import_kymotrackgroup_from_csv(file, blank_kymo, "red", delimiter=";")


def test_min_obs_csv_regression(tmpdir_factory, blank_kymo):
    """This tests a regression where saving a freshly imported older file does not function"""
    testfile = Path(__file__).parent / f"./data/tracks_v0.csv"
    with pytest.warns(
        RuntimeWarning,
        match="loaded kymo or channel doesn't match the one used to create this file",
    ):
        imported_tracks = import_kymotrackgroup_from_csv(testfile, blank_kymo, "red", delimiter=";")

    out_file = f"{tmpdir_factory.mktemp('pylake')}/no_min_lengths.csv"

    err_msg = "Loaded tracks have no minimum length metadata defined"
    with pytest.warns(RuntimeWarning, match=err_msg):
        imported_tracks.save(out_file, ";", None, correct_origin=True)

    out_file2 = f"{tmpdir_factory.mktemp('pylake')}/no_min_lengths2.csv"
    with pytest.warns(RuntimeWarning, match=err_msg):
        import_kymotrackgroup_from_csv(out_file, blank_kymo, "red", delimiter=";").save(out_file2)
