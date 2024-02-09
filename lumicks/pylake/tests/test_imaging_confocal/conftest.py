import pathlib
from itertools import permutations

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.point_scan import PointScan
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP

from ..data.mock_file import MockDataFile_v2
from ..data.mock_confocal import (
    MockConfocalFile,
    generate_scan_json,
    generate_kymo_with_ref,
    generate_scan_with_ref,
)

start = np.int64(20e9)
dt = np.int64(62.5e6)
axes_map = {"X": 0, "Y": 1, "Z": 2}
channel_map = {"r": 0, "g": 1, "b": 2}


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
def bead_kymo():
    data = np.load(pathlib.Path(__file__).parent / "data" / "bead_kymo.npz")

    return _kymo_from_array(
        data["rgb"],
        color_format="rgb",
        line_time_seconds=data["line_time_seconds"],
        exposure_time_seconds=0,
        pixel_size_um=data["pixelsize_um"],
    )


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


@pytest.fixture(scope="module")
def test_point_scan():
    n_samples = 90
    data = {c: np.random.poisson(15, n_samples) for c in ("red", "green", "blue")}

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [],
        [],
        [],
        infowave=np.zeros(data["red"].shape),
        red_photon_counts=data["red"],
        green_photon_counts=data["green"],
        blue_photon_counts=data["blue"],
    )
    point_scan = PointScan("PointScan1", mock_file, start, stop, metadata)

    reference = {
        "data": data,
        "timestamps": np.arange(n_samples, dtype=np.int64) * dt + start,
        "dt": dt,
        "start": start,
    }

    return point_scan, reference


@pytest.fixture(scope="module")
def test_scans():
    image = np.random.poisson(5, size=(4, 5, 3))
    return {
        (name := f"fast {axes[0]} slow {axes[1]}"): generate_scan_with_ref(
            name,
            image,
            pixel_sizes_nm=[50, 50],
            axes=[axes_map[k] for k in axes],
            start=start,
            dt=dt,
            samples_per_pixel=5,
            line_padding=50,
            multi_color=True,
        )
        for axes in permutations(axes_map.keys(), 2)
    }


@pytest.fixture(scope="module")
def test_scans_multiframe():
    image = np.random.poisson(5, size=(10, 4, 5, 3))
    return {
        (name := f"fast {axes[0]} slow {axes[1]} multiframe"): generate_scan_with_ref(
            name,
            image,
            pixel_sizes_nm=[50, 50],
            axes=[axes_map[k] for k in axes],
            start=start,
            dt=dt,
            samples_per_pixel=5,
            line_padding=50,
            multi_color=True,
        )
        for axes in permutations(axes_map.keys(), 2)
    }


@pytest.fixture(scope="module")
def test_scan_missing_channels():
    empty = Slice(Continuous([], start=start, dt=dt))

    def make_data(*missing_channels):
        image = np.random.poisson(5, size=(4, 5, 3))
        for channel in missing_channels:
            image[:, :, channel_map[channel[0]]] = 0

        scan, ref = generate_scan_with_ref(
            f"missing {', '.join(missing_channels)}",
            image,
            pixel_sizes_nm=[50, 50],
            axes=[1, 0],
            start=start,
            dt=dt,
            samples_per_pixel=5,
            line_padding=50,
            multi_color=True,
        )

        for channel in missing_channels:
            setattr(scan.file, f"{channel}_photon_count", empty)

        return scan, ref

    return {key: make_data(*key) for key in [("red",), ("red", "blue"), ("red", "green", "blue")]}


@pytest.fixture(scope="module")
def test_scan_truncated():
    image = np.random.poisson(5, size=(2, 4, 5, 3))
    scan, ref = generate_scan_with_ref(
        "truncated",
        image,
        pixel_sizes_nm=[50, 50],
        axes=[1, 0],
        start=start,
        dt=dt,
        samples_per_pixel=5,
        line_padding=50,
        multi_color=True,
    )
    scan.start = start - dt
    return scan, ref


@pytest.fixture(scope="module")
def test_scan_sted_bug():
    image = np.random.poisson(5, size=(2, 4, 5, 3))
    scan, ref = generate_scan_with_ref(
        "sted bug",
        image,
        pixel_sizes_nm=[50, 50],
        axes=[1, 0],
        start=start,
        dt=dt,
        samples_per_pixel=5,
        line_padding=50,
        multi_color=True,
    )
    corrected_start = scan.red_photon_count.timestamps[5]

    # start *between* samples
    scan.start = corrected_start - np.int64(dt - 1e5)
    return scan, ref, corrected_start


@pytest.fixture(scope="module")
def test_scan_slicing():
    start = _FIRST_TIMESTAMP + 100
    dt = np.int64(1e7)
    line_padding = 10

    scan, ref = generate_scan_with_ref(
        "slicing",
        np.random.poisson(5, size=(10, 2, 2, 3)),
        pixel_sizes_nm=[50, 50],
        axes=[1, 0],
        start=start - line_padding * dt,
        dt=dt,
        samples_per_pixel=15,
        line_padding=line_padding,
        multi_color=True,
    )
    return scan, ref


@pytest.fixture
def grab_tiff_tags():
    def grab_tags(file):
        from ast import literal_eval

        import tifffile

        tiff_tags = []
        with tifffile.TiffFile(file) as tif:
            for page in tif.pages:
                page_tags = {}
                for tag in page.tags.values():
                    name, value = tag.name, tag.value
                    try:
                        page_tags[name] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        page_tags[name] = value
                tiff_tags.append(page_tags)
        return tiff_tags

    return grab_tags
