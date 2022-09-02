import pytest
import numpy as np
import re
from pathlib import Path
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
    tracks.save(testfile, delimiter, sampling_width)
    imported_tracks = import_kymotrackgroup_from_csv(testfile, kymo, "red", delimiter=delimiter)

    # Test raw fields
    data, pylake_version, csv_version = read_txt(testfile, delimiter)
    assert len(imported_tracks) == len(tracks)

    for track1, track2 in zip(tracks, imported_tracks):
        np.testing.assert_allclose(np.array(track1.coordinate_idx), np.array(track2.coordinate_idx))
        np.testing.assert_allclose(np.array(track1.time_idx), np.array(track2.time_idx))

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
