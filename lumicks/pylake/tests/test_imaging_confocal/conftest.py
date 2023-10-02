import numpy as np
import pytest

from ..data.mock_file import MockDataFile_v2
from ..data.mock_confocal import generate_scan_json, generate_kymo_with_ref

start = np.int64(20e9)
dt = np.int64(62.5e6)


@pytest.fixture(scope="module")
def test_kymo():
    # RGB Kymo with infowave as expected from BL
    image = np.random.poisson(5, size=(5, 10, 3))

    kymo, ref = generate_kymo_with_ref(
        "tester",
        image,
        pixel_size_nm=100,
        start=start,
        dt=dt,
        samples_per_pixel=4,
        line_padding=50,
    )

    return kymo, ref


@pytest.fixture(scope="module")
def truncated_kymo():
    image = np.random.poisson(5, size=(5, 4, 3))

    kymo, ref = generate_kymo_with_ref(
        "truncated",
        image,
        pixel_size_nm=100,
        start=start,
        dt=dt,
        samples_per_pixel=4,
        line_padding=50,
    )
    kymo.start = start - 62500000
    return kymo, ref


@pytest.fixture(scope="module")
def downsampling_kymo():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo, ref = generate_kymo_with_ref(
        "downsampler",
        image,
        pixel_size_nm=100,
        start=1592916040906356300,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    return kymo, ref


@pytest.fixture(scope="module")
def downsampled_results():
    time_factor = 2
    position_factor = 2
    time_image = np.array(
        [
            [12, 12, 6],
            [0, 0, 6],
            [12, 0, 18],
            [12, 24, 6],
            [12, 24, 6],
        ]
    )
    position_image = np.array(
        [
            [0, 12, 0, 12, 0, 12, 0],
            [12, 12, 12, 12, 12, 12, 0],
        ]
    )
    both_image = np.array([[12, 12, 12], [24, 24, 24]])

    return time_factor, position_factor, time_image, position_image, both_image


@pytest.fixture(scope="module")
def cropping_kymo():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [12, 12, 12, 12, 0, 6, 0],
            [24, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo, ref = generate_kymo_with_ref(
        "cropper",
        image,
        pixel_size_nm=100,
        start=1592916040906356300,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    return kymo, ref


@pytest.fixture(scope="module")
def kymo_h5_file(tmpdir_factory, test_kymo):
    kymo, ref = test_kymo
    dt = ref.timestamps.dt
    start = ref.start

    mock_class = MockDataFile_v2

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join(f"kymo_{mock_class.__class__.__name__}.h5"))
    mock_file.write_metadata()

    json_kymo = generate_scan_json(
        [
            {
                "axis": 0,
                "num of pixels": ref.metadata.num_pixels[0],
                "pixel size (nm)": ref.metadata.pixelsize_um[0],
            }
        ]
    )

    for color in ("Red", "Green", "Blue"):
        mock_file.make_continuous_channel(
            "Photon count",
            color,
            np.int64(start),
            dt,
            getattr(kymo, f"{color.lower()}_photon_count").data,
        )
    mock_file.make_continuous_channel(
        "Info wave", "Info wave", np.int64(start), dt, ref.infowave.data.data
    )

    ds = mock_file.make_json_data("Kymograph", kymo.name, json_kymo)
    ds.attrs["Start time (ns)"] = np.int64(start)
    ds.attrs["Stop time (ns)"] = np.int64(start + len(ref.infowave.data.data) * dt)

    # Force channel that overlaps kymo; step from high to low force
    # We want two lines of the kymo to have a force of 30, the other 10.
    # Force starts 5 samples before the kymograph.
    # A kymotrack line is 20 samples long, with a 50 sample dead time on either side.
    # The pause before the third line starts after 245 samples.
    force_start_offset = 5
    lines_in_first_step = 2
    padding_count = lines_in_first_step * ref.infowave.line_padding
    pixels_count = ref.metadata.pixels_per_line * ref.infowave.samples_per_pixel
    cutoff = force_start_offset + (padding_count + pixels_count) * lines_in_first_step

    iw_mask = np.hstack((np.zeros(force_start_offset), np.copy(ref.infowave.data.data)))
    force_data = np.zeros(len(ref.infowave.data.data) + force_start_offset)
    force_data[:cutoff][iw_mask[:cutoff] > 0] = 30
    force_data[cutoff:][iw_mask[cutoff:] > 0] = 10

    # start force channel before infowave
    force_start = np.int64(ds.attrs["Start time (ns)"] - (dt * force_start_offset))
    mock_file.make_continuous_channel("Force HF", "Force 2x", force_start, dt, force_data)

    # LF force for plotting, 1 sample per pixel
    # integer values within data, zeros within deadtime
    n_pad = ref.infowave.line_padding
    n_samples = ref.infowave.samples_per_pixel
    n_pixels = ref.metadata.pixels_per_line
    n_lines = ref.metadata.lines_per_frame
    n_dead_pixels = n_pad // n_samples

    force_line = np.hstack(
        [np.zeros(n_dead_pixels + 1), np.ones(n_pixels), np.zeros(n_dead_pixels)]
    )
    lf_force = np.hstack([force_line * j for j in range(1, n_lines + 1)])
    timestamps = start + (np.arange(lf_force.size, dtype=np.int64) * n_samples * dt)

    mock_file.make_timeseries_channel(
        "Force LF", "Force 2y", [(ts, pt) for ts, pt in zip(timestamps, lf_force)]
    )

    return mock_file.file
