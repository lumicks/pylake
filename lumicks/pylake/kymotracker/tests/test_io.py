from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup, import_kymolinegroup_from_csv
import numpy as np
import pytest
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


@pytest.fixture(scope="session")
def kymolinegroup_io_data():
    test_data = np.zeros((8, 8))

    test_img = CalibratedKymographChannel("test", data=test_data, time_step_ns=100e9, pixel_size=2)
    k1 = KymoLine([1, 2, 3], np.array([2, 3, 4]), test_img)
    k2 = KymoLine([2, 3, 4], np.array([3, 4, 5]), test_img)
    k3 = KymoLine([3, 4, 5], np.array([4, 5, 6]), test_img)
    k4 = KymoLine([4, 5, 6], np.array([5, 6, 7]), test_img)
    lines = KymoLineGroup([k1, k2, k3, k4])

    for k in lines:
        test_data[k.coordinate_idx, k.time_idx] = 2
        test_data[np.array(k.coordinate_idx) - 1, k.time_idx] = 1

    return test_img, lines


def read_txt(testfile, delimiter):
    raw_data = np.loadtxt(testfile, delimiter=delimiter, unpack=True)
    with open(testfile, "r") as f:
        data = {}
        header = f.readline().rstrip().split(delimiter)
        line_idx = raw_data[0, :]
        for key, col in zip(header, raw_data):
            data[key] = [col[np.argwhere(line_idx == idx).flatten()] for idx in np.unique(line_idx)]

        return data


@pytest.mark.parametrize("dt, dx, delimiter, sampling_width, sampling_outcome",
                         [[int(1e9), 1.0, ';', 0, 2],
                          [int(2e9), 1.0, ';', 0, 2],
                          [int(1e9), 2.0, ';', 0, 2],
                          [int(1e9), 1.0, ',', 0, 2],
                          [int(1e9), 1.0, ';', 1, 3],
                          [int(1e9), 2.0, ';', None, None]])
def test_kymolinegroup_io(tmpdir_factory, kymolinegroup_io_data, dt, dx, delimiter, sampling_width, sampling_outcome):
    test_img, lines = kymolinegroup_io_data

    kymo = generate_kymo(
        "test",
        test_img.data,
        dx*1000,
        start=4,
        dt=dt,
        samples_per_pixel=5,
        line_padding=3
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

    for line1, time in zip(lines, data["time"]):
        np.testing.assert_allclose(line1.seconds, time)

    for line1, coord in zip(lines, data["position"]):
        np.testing.assert_allclose(line1.position, coord)

    if sampling_width is None:
        assert len([key for key in data.keys() if "counts" in key]) == 0
    else:
        count_field = [key for key in data.keys() if "counts" in key][0]
        for line1, cnt in zip(lines, data[count_field]):
            np.testing.assert_allclose([sampling_outcome] * len(line1.coordinate_idx), cnt)
