import pytest
import numpy as np
from lumicks.pylake.kymotracker.kymoline import (
    KymoLine,
    KymoLineGroup,
    import_kymolinegroup_from_csv,
)
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def read_txt(testfile, delimiter):
    raw_data = np.loadtxt(testfile, delimiter=delimiter, unpack=True)
    with open(testfile, "r") as f:
        data = {}
        header = f.readline().rstrip().split(delimiter)
        line_idx = raw_data[0, :]
        for key, col in zip(header, raw_data):
            data[key] = [col[np.argwhere(line_idx == idx).flatten()] for idx in np.unique(line_idx)]

        return data


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
def test_kymolinegroup_io(
    tmpdir_factory, dt, dx, delimiter, sampling_width, sampling_outcome
):

    line_coordinates = [
        ((1, 2, 3), (2, 3, 4)),
        ((2, 3, 4), (3, 4, 5)),
        ((3, 4, 5), (4, 5, 6)),
        ((4, 5, 6), (5, 6, 7)),
    ]
    test_data = np.zeros((8, 8))
    for time_idx, position_idx in line_coordinates:
        test_data[np.array(position_idx).astype(int), np.array(time_idx).astype(int)] = 2
        test_data[np.array(position_idx).astype(int) - 1, np.array(time_idx).astype(int)] = 1

    kymo = generate_kymo(
        "test",
        test_data,
        pixel_size_nm=dx * 1000,
        start=np.int64(20e9),
        dt=dt,
        samples_per_pixel=5,
        line_padding=3
    )

    lines = KymoLineGroup(
        [
            KymoLine(
                np.array(time_idx),
                np.array(position_idx),
                kymo,
                "red"
            )
            for time_idx, position_idx in line_coordinates
        ]
    )

    # Test round trip through the API
    testfile = f"{tmpdir_factory.mktemp('pylake')}/test.csv"
    lines.save(testfile, delimiter, sampling_width)
    read_file = import_kymolinegroup_from_csv(testfile, kymo, "red", delimiter=delimiter)

    # Test raw fields
    data = read_txt(testfile, delimiter)
    assert len(read_file) == len(lines)

    for line1, line2 in zip(lines, read_file):
        np.testing.assert_allclose(np.array(line1.coordinate_idx), np.array(line2.coordinate_idx))
        np.testing.assert_allclose(np.array(line1.time_idx), np.array(line2.time_idx))

    for line1, time in zip(lines, data["time (seconds)"]):
        np.testing.assert_allclose(line1.seconds, time)

    for line1, coord in zip(lines, data["position (um)"]):
        np.testing.assert_allclose(line1.position, coord)

    if sampling_width is None:
        assert len([key for key in data.keys() if "counts" in key]) == 0
    else:
        count_field = [key for key in data.keys() if "counts" in key][0]
        for line1, cnt in zip(lines, data[count_field]):
            np.testing.assert_allclose([sampling_outcome] * len(line1.coordinate_idx), cnt)
