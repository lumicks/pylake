import pytest
import numpy as np
import inspect
import re
import io
from pathlib import Path
from copy import copy
from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import (
    KymoTrack,
    KymoTrackGroup,
    import_kymotrackgroup_from_csv,
)
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def read_txt(testfile, delimiter, with_version=True):
    raw_data = np.loadtxt(testfile, delimiter=delimiter, unpack=True)
    with open(testfile, "r") as f:
        data = {}

        # from v0.13.0, exported CSV files have an additional header line
        # with the pylake version and CSV version (starting at 2)
        if with_version:
            version_header = re.search(
                r"Exported with pylake v([\d\.]*) \| track coordinates v(\d)", f.readline()
            )
            pylake_version = version_header.group(1)
            csv_version = int(version_header.group(2))
        else:
            pylake_version = None
            csv_version = 1

        header = f.readline().rstrip().split(delimiter)
        track_idx = raw_data[0, :]
        for key, col in zip(header, raw_data):
            data[key] = [
                col[np.argwhere(track_idx == idx).flatten()] for idx in np.unique(track_idx)
            ]

        return data, pylake_version, csv_version


def compare_kymotrack_group(group1, group2):
    assert len(group1) == len(group2)
    for track1, track2 in zip(group1, group2):
        for property in ("coordinate_idx", "time_idx", "position", "seconds"):
            np.testing.assert_allclose(getattr(track1, property), getattr(track2, property))


@pytest.mark.parametrize(
    "dt, dx, delimiter, sampling_width, sampling_outcome",
    [
        [int(1e9), 1.0, ";", 0, 2],
        [int(2e9), 1.0, ";", 0, 2],
        [int(1e9), 2.0, ";", 0, 2],
        [int(1e9), 1.0, ",", 0, 2],
        [int(1e9), 1.0, ";", 1, 3],
        [int(1e9), 2.0, ";", None, None],
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
            KymoTrack(np.array(time_idx), np.array(position_idx), kymo, "red")
            for time_idx, position_idx in track_coordinates
        ]
    )

    # Test round trip through the API
    testfile = f"{tmpdir_factory.mktemp('pylake')}/test.csv"
    tracks.save(testfile, delimiter, sampling_width, correct_origin=True)
    imported_tracks = import_kymotrackgroup_from_csv(testfile, kymo, "red", delimiter=delimiter)
    data, pylake_version, csv_version = read_txt(testfile, delimiter)

    compare_kymotrack_group(tracks, imported_tracks)

    for track1, time in zip(tracks, data["time (seconds)"]):
        np.testing.assert_allclose(track1.seconds, time)

    for track1, coord in zip(tracks, data["position (um)"]):
        np.testing.assert_allclose(track1.position, coord)

    if sampling_width is None:
        assert len([key for key in data.keys() if "counts" in key]) == 0
    else:
        count_field = [key for key in data.keys() if "counts" in key][0]
        for track1, cnt in zip(tracks, data[count_field]):
            np.testing.assert_allclose([sampling_outcome] * len(track1.coordinate_idx), cnt)


def test_export_sources(tmpdir_factory):
    kymo1 = _kymo_from_array(np.random.poisson(5, (5, 5, 3)), "rgb", 1e-4, start=20e9)
    kymo2 = copy(kymo1)
    tracks1 = KymoTrackGroup([KymoTrack(np.arange(3), np.arange(3), kymo1, "red")])
    tracks2 = KymoTrackGroup([KymoTrack(np.arange(5), np.arange(5), kymo2, "red")])
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
    delimiter, sampling_width, correct_origin, kymo_integration_test_data
):
    # Validate that this also works when providing a string handle (this is the API LV uses).

    def get_args(func):
        return list(inspect.signature(func).parameters.keys())

    # This helps us ensure that if we get additional arguments to this function, we don't forget to
    # add them to the parametrization here.
    assert set(get_args(KymoTrackGroup.save)[2:]) == set(get_args(test_roundtrip_without_file)[:-1])

    track_coordinates = [
        ((1, 2, 3), (2, 3, 4)),
        ((2, 3, 4), (3, 4, 5)),
        ((3, 4, 5), (4, 5, 6)),
        ((4, 5, 6), (5, 6, 7)),
    ]

    tracks = KymoTrackGroup(
        [
            KymoTrack(np.array(time_idx), np.array(position_idx), kymo_integration_test_data, "red")
            for time_idx, position_idx in track_coordinates
        ]
    )

    with io.StringIO() as s:
        tracks.save(
            s, delimiter=delimiter, sampling_width=sampling_width, correct_origin=correct_origin
        )
        string_representation = s.getvalue()

    with io.StringIO(string_representation) as s:
        read_tracks = import_kymotrackgroup_from_csv(
            s, kymo_integration_test_data, "green", delimiter=delimiter
        )

    compare_kymotrack_group(tracks, read_tracks)


@pytest.mark.parametrize(
    "version, read_with_version",
    [[0, False], [1, False], [2, True]],
)
def test_csv_version(version, read_with_version):
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
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=3,
    )

    testfile = Path(__file__).parent / f"./data/tracks_v{version}.csv"
    imported_tracks = import_kymotrackgroup_from_csv(testfile, kymo, "red", delimiter=";")

    data, pylake_version, csv_version = read_txt(testfile, ";", read_with_version)

    if read_with_version:
        assert pylake_version is not None
    else:
        assert pylake_version is None

    for j, track in enumerate(imported_tracks):
        np.testing.assert_allclose(track.time_idx, data["time (pixels)"][j])
        np.testing.assert_allclose(track.coordinate_idx, data["coordinate (pixels)"][j])


@pytest.mark.parametrize("filename", ["csv_bad_format.csv", "csv_unparseable.csv"])
def test_bad_csv(filename, tmpdir_factory, blank_kymo):
    with pytest.raises(IOError, match="Invalid file format!"):
        file = Path(__file__).parent / "data" / filename
        import_kymotrackgroup_from_csv(file, blank_kymo, "red", delimiter=";")
