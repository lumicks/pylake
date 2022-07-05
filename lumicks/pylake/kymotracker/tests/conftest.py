import pytest
import numpy as np
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup
from lumicks.pylake.tests.data.mock_confocal import generate_kymo
from .data.generate_gaussian_data import read_dataset as read_dataset_gaussian


def raw_test_data():
    test_data = np.ones((30, 30))
    test_data[10, 10:20] = 10
    test_data[11, 10:20] = 30
    test_data[12, 10:20] = 10

    test_data[20, 15:25] = 10
    test_data[21, 15:25] = 20
    test_data[22, 15:25] = 10
    return test_data


def raw_test_data_lengths():
    test_data = np.ones((30, 30))
    test_data[10, 5:20] = 10
    test_data[11, 5:20] = 30
    test_data[12, 5:20] = 10

    test_data[20, 15:25] = 10
    test_data[21, 15:25] = 20
    test_data[22, 15:25] = 10
    return test_data


@pytest.fixture
def kymo_integration_test_data():
    def make_kymo(data):
        return generate_kymo(
            "test",
            data,
            pixel_size_nm=5000,
            start=int(4e9),
            dt=int(5e9 / 100),
            samples_per_pixel=3,
            line_padding=5,
        )

    return {"standard": make_kymo(raw_test_data()), "diff_len": make_kymo(raw_test_data_lengths())}


@pytest.fixture
def gaussian_1d():
    return read_dataset_gaussian("gaussian_data_1d.npz")


@pytest.fixture
def two_gaussians_1d():
    return read_dataset_gaussian("two_gaussians_1d.npz")


@pytest.fixture
def blank_kymo():
    return generate_kymo(
        "",
        np.ones((1, 10)),
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )


@pytest.fixture
def kymogroups_2lines():
    _, _, photon_count, parameters = read_dataset_gaussian("kymo_data_2lines.npz")
    pixel_size = parameters[0].pixel_size
    centers = [p.center / pixel_size for p in parameters]

    kymo = generate_kymo(
        "",
        photon_count,
        pixel_size_nm=pixel_size * 1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )
    _, n_frames = kymo.get_image("red").shape

    lines = KymoLineGroup(
        [KymoLine(np.arange(0.0, n_frames), np.full(n_frames, c), kymo, "red") for c in centers]
    )

    # introduce gaps into tracked lines
    use_frames = np.array([0, 1, -2, -1])
    gapped_lines = KymoLineGroup(
        [
            KymoLine(line.time_idx[use_frames], line.coordinate_idx[use_frames], kymo, "red")
            for line in lines
        ]
    )

    # crop the ends of initial lines and make new set of lines with one cropped and the second full
    truncated_lines = KymoLineGroup(
        [
            KymoLine(np.arange(1.0, n_frames - 2), np.full(n_frames - 3, c), kymo, "red")
            for c in centers
        ]
    )
    mixed_lines = KymoLineGroup([truncated_lines[0], lines[1]])

    return lines, gapped_lines, mixed_lines


@pytest.fixture
def kymogroups_close_lines():
    _, _, photon_count, parameters = read_dataset_gaussian("two_gaussians_1d.npz")
    pixel_size = parameters[0].pixel_size
    centers = [p.center / pixel_size for p in parameters]

    kymo = generate_kymo(
        "",
        photon_count,
        pixel_size_nm=pixel_size * 1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )
    _, n_frames = kymo.get_image("red").shape

    lines = KymoLineGroup(
        [KymoLine(np.arange(0.0, n_frames), np.full(n_frames, c), kymo, "red") for c in centers]
    )

    return lines
