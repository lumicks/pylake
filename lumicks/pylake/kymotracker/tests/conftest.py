import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
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


@pytest.fixture
def kymo_integration_test_data():
    return generate_kymo(
        "test",
        raw_test_data(),
        pixel_size_nm=50,
        start=int(4e9),
        dt=int(5e9 / 100),
        samples_per_pixel=3,
        line_padding=5,
    )


@pytest.fixture
def kymo_integration_tracks(kymo_integration_test_data):
    track_coordinates = [
        (np.arange(10, 20), np.full(10, 11)),
        (np.arange(15, 25), np.full(10, 21.51)),
    ]

    tracks = KymoTrackGroup(
        [
            KymoTrack(
                np.array(time_idx), np.array(position_idx), kymo_integration_test_data, "red", 0.1
            )
            for time_idx, position_idx in track_coordinates
        ]
    )

    return tracks


@pytest.fixture
def kymo_pixel_calibrations():
    image = raw_test_data()
    background = np.random.uniform(1, 10, size=image.size).reshape(image.shape)

    kymo_um = generate_kymo(
        "test",
        image + background,
        pixel_size_nm=50,
        start=int(4e9),
        dt=int(5e9 / 100),
        samples_per_pixel=3,
        line_padding=5,
    )

    kymo_kbp = kymo_um.calibrate_to_kbp(kymo_um.pixelsize_um[0] * kymo_um.pixels_per_line / 0.34)
    kymo_px = _kymo_from_array(image + background, "r", kymo_um.line_time_seconds)

    return kymo_um, kymo_kbp, kymo_px


@pytest.fixture
def gaussian_1d():
    return read_dataset_gaussian("gaussian_data_1d.npz")


@pytest.fixture
def two_gaussians_1d():
    return read_dataset_gaussian("two_gaussians_1d.npz")


@pytest.fixture
def blank_kymo():
    kymo = generate_kymo(
        "",
        np.ones((1, 10)),
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )
    kymo._motion_blur_constant = 0
    return kymo


@pytest.fixture
def blank_kymo_track_args(blank_kymo):
    return [blank_kymo, "red", blank_kymo.line_time_seconds]


@pytest.fixture
def kymogroups_2tracks():
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

    tracks = KymoTrackGroup(
        [
            KymoTrack(
                np.arange(0, n_frames), np.full(n_frames, c), kymo, "red", kymo.line_time_seconds
            )
            for c in centers
        ]
    )

    # introduce gaps into tracks
    use_frames = np.array([0, 1, -2, -1])
    gapped_tracks = KymoTrackGroup(
        [
            KymoTrack(
                track.time_idx[use_frames],
                track.coordinate_idx[use_frames],
                kymo,
                "red",
                kymo.line_time_seconds,
            )
            for track in tracks
        ]
    )

    # crop the ends of initial tracks and make new set of tracks with one cropped and the second full
    truncated_tracks = KymoTrackGroup(
        [
            KymoTrack(
                np.arange(1, n_frames - 2),
                np.full(n_frames - 3, c),
                kymo,
                "red",
                kymo.line_time_seconds,
            )
            for c in centers
        ]
    )
    mixed_tracks = KymoTrackGroup([truncated_tracks[0], tracks[1]])

    return tracks, gapped_tracks, mixed_tracks


@pytest.fixture
def kymogroups_close_tracks():
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

    return KymoTrackGroup(
        [
            KymoTrack(
                np.arange(0, n_frames), np.full(n_frames, c), kymo, "red", kymo.line_time_seconds
            )
            for c in centers
        ]
    )


@pytest.fixture
def simulate_dwelltimes():
    def simulate_poisson(scale, num_samples, min_time=0, max_time=np.inf):
        samples = np.array([])
        for _ in range(100):
            new_samples = np.random.exponential(scale, num_samples)
            samples = np.hstack(
                (
                    samples,
                    new_samples[np.logical_and(new_samples >= min_time, new_samples < max_time)],
                )
            )
            if samples.size > num_samples:
                return samples[:num_samples]
        else:
            raise RuntimeError("Generated fewer samples than intended.")

    return simulate_poisson
