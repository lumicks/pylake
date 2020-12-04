from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup, import_kymolinegroup_from_csv
import numpy as np
import pytest


@pytest.fixture(scope="session")
def kymolinegroup_io_data():
    test_img = np.zeros((8, 8))

    k1 = KymoLine([1, 2, 3], [2, 3, 4], test_img)
    k2 = KymoLine([2, 3, 4], [3, 4, 5], test_img)
    k3 = KymoLine([3, 4, 5], [4, 5, 6], test_img)
    k4 = KymoLine([4, 5, 6], [5, 6, 7], test_img)
    lines = KymoLineGroup([k1, k2, k3, k4])

    for k in lines:
        test_img[k.coordinate_idx, k.time_idx] = 2
        test_img[np.array(k.coordinate_idx) - 1, k.time_idx] = 1

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
                         [[1.0, 1.0, ';', 0, 2],
                          [2.0, 1.0, ';', 0, 2],
                          [1.0, 2.0, ';', 0, 2],
                          [1.0, 1.0, ',', 0, 2],
                          [1.0, 1.0, ';', 1, 3],
                          [None, None, ';', 1, 3],
                          [None, None, ';', None, 3],
                          [1.0, 2.0, ';', None, None],
                          [None, 1.0, ';', None, None],
                          [1.0, None, ';', None, None]])
def test_kymolinegroup_io(tmpdir_factory, kymolinegroup_io_data, dt, dx, delimiter, sampling_width, sampling_outcome):
    test_img, lines = kymolinegroup_io_data

    # Test round trip through the API
    testfile = f"{tmpdir_factory.mktemp('pylake')}/test.csv"
    lines.save(testfile, dt, dx, delimiter, sampling_width)
    read_file = import_kymolinegroup_from_csv(testfile, test_img, delimiter=delimiter)

    # Test raw fields
    data = read_txt(testfile, delimiter)
    assert len(read_file) == len(lines)

    for line1, line2 in zip(lines, read_file):
        assert np.allclose(np.array(line1.coordinate_idx), np.array(line2.coordinate_idx))
        assert np.allclose(np.array(line1.time_idx), np.array(line2.time_idx))

    if not dt:
        assert "time" not in data
    else:
        for line1, time in zip(lines, data["time"]):
            assert np.allclose(np.array(line1.time_idx) * dt, time)

    if not dx:
        assert "coordinate" not in data
    else:
        for line1, coord in zip(lines, data["coordinate"]):
            assert np.allclose(np.array(line1.coordinate_idx) * dx, coord)

    if sampling_width is None:
        assert len([key for key in data.keys() if "counts" in key]) == 0
    else:
        count_field = [key for key in data.keys() if "counts" in key][0]
        for line1, cnt in zip(lines, data[count_field]):
            assert np.allclose([sampling_outcome] * len(line1.coordinate_idx), cnt)
