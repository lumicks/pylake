import pytest
import numpy as np
from lumicks.pylake.point_scan import PointScan
from lumicks.pylake.kymo import Kymo
from lumicks.pylake.scan import Scan
from ..data.mock_file import MockDataFile_v2
from ..data.mock_confocal import MockConfocalFile, generate_scan_json, generate_image_data


start = np.int64(20e9)
dt = np.int64(62.5e6)

# fmt: off
@pytest.fixture(scope="module")
def reference_counts():
    return np.uint32([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0,
                      0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 8, 0])

reference_infowave = np.uint8([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                               0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2,
                               1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                               0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2])
# fmt: on


@pytest.fixture(scope="module")
def reference_timestamps():
    stop = start + len(reference_infowave) * dt
    return np.arange(start, stop, 6.25e7).astype(np.int64)


@pytest.fixture(scope="module")
def test_point_scans(reference_counts):
    point_scans = {}

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [],
        [],
        [],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    point_scans["PointScan1"] = PointScan("PointScan1", mock_file, start, stop, metadata)

    return point_scans


@pytest.fixture(scope="module")
def test_kymos(reference_counts):
    kymos = {}

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [0],
        [5],
        [10.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    kymos["Kymo1"] = Kymo("Kymo1", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [0],
        [5],
        [10.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    kymos["truncated_kymo"] = Kymo("truncated", mock_file, start - 62500000, stop, metadata)


    # RGB Kymo with infowave as expected from BL
    image = np.random.poisson(5, size=(5, 10, 3))
    infowave, red_photon_count = generate_image_data(image[:, :, 0], 4, 50)
    _, green_photon_count = generate_image_data(image[:, :, 1], 4, 50)
    _, blue_photon_count = generate_image_data(image[:, :, 2], 4, 50)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [0],
        [5],
        [100],
        infowave=infowave,
        red_photon_counts=red_photon_count,
        green_photon_counts=green_photon_count,
        blue_photon_counts=blue_photon_count,
    )
    kymos["noise"] = Kymo("noise", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_image(
        np.ones(shape=(5, 4, 3)),
        [10.0],
        start,
        dt,
        [0],
        samples_per_pixel=5,
        line_padding=3
    )
    kymos["slicing_regression"] = Kymo("slicing_regression", mock_file, start, stop, metadata)

    return kymos


@pytest.fixture(scope="module")
def test_scans(reference_counts):
    scans = {}

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["fast Y slow X"] = Scan("fast Y slow X", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 3],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["fast Y slow X multiframe"] = Scan(
        "fast Y slow X multiframe", mock_file, start, stop, metadata
    )

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [0, 2],
        [4, 3],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["fast X slow Z multiframe"] = Scan(
        "fast X slow Z multiframe", mock_file, start, stop, metadata
    )

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 2],
        [4, 3],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["fast Y slow Z multiframe"] = Scan(
        "fast Y slow Z multiframe", mock_file, start, stop, metadata
    )

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["truncated_scan"] = Scan("truncated", mock_file, start - 62500000, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    middle = scans["fast Y slow X"].red_photon_count.timestamps[5]
    scans["sted bug"] = Scan("sted bug", mock_file, middle - 62400000, stop, metadata)

    image = np.random.poisson(10, (10, 3, 4))
    mock_file, metadata, stop = MockConfocalFile.from_image(
        image,
        pixel_sizes_nm=[5, 5],
        start=start,
        dt=dt,
        axes=[0, 1],
        samples_per_pixel=5,
        line_padding=3,
    )
    scans["multiframe_poisson"] = Scan("multiframe_poisson", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=None,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    scans["red channel missing"] = Scan("red channel missing", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=None,
        green_photon_counts=reference_counts,
        blue_photon_counts=None,
    )
    scans["rb channels missing"] = Scan("rb channels missing", mock_file, start, stop, metadata)

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [1, 0],
        [4, 5],
        [191.0, 197.0],
        infowave=reference_infowave,
        red_photon_counts=None,
        green_photon_counts=None,
        blue_photon_counts=None,
    )
    scans["all channels missing"] = Scan("all channels missing", mock_file, start, stop, metadata)

    return scans


@pytest.fixture(scope="module")
def kymo_h5_file(tmpdir_factory, reference_counts):
    mock_class = MockDataFile_v2

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join(f"kymo_{mock_class.__class__.__name__}.h5"))
    mock_file.write_metadata()

    json_kymo = generate_scan_json([{"axis": 0, "num of pixels": 5, "pixel size (nm)": 10.0}])

    # Generate lines at 1 Hz
    for color in ("Red", "Green", "Blue"):
        mock_file.make_continuous_channel(
            "Photon count", color, np.int64(start), dt, reference_counts
        )
    mock_file.make_continuous_channel(
        "Info wave", "Info wave", np.int64(start), dt, reference_infowave
    )

    ds = mock_file.make_json_data("Kymograph", "Kymo1", json_kymo)
    ds.attrs["Start time (ns)"] = np.int64(start)
    ds.attrs["Stop time (ns)"] = np.int64(start + len(reference_infowave) * dt)

    # Force channel that overlaps kymo; step from high to low force
    # We want two lines of the kymo to have a force of 30, the other 10. Force starts 5 samples
    # before the kymograph. First kymotrack is 15 samples long, second is 16 samples long, which
    # means the third line starts after 31 + 5 = 36 samples
    force_data = np.hstack((np.ones(37) * 30, np.ones(33) * 10))
    force_start = np.int64(ds.attrs["Start time (ns)"] - (dt * 5))  # before infowave
    mock_file.make_continuous_channel("Force HF", "Force 2x", force_start, dt, force_data)
    mock_file.make_continuous_channel("Force HF", "Force 1x", 1, 10, np.arange(5.0))

    return mock_file.file