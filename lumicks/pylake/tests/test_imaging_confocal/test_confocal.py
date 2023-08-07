import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.detail.confocal import ScanAxis, ScanMetaData

from ..data.mock_confocal import generate_kymo, generate_scan, generate_scan_json


@pytest.mark.parametrize(
    "axis, num_pixels, pixel_size_um, label",
    [(0, 10, 0.5, "X"), (1, 5, 0.4, "Y"), (2, 15, 0.35, "Z")],
)
def test_scanaxis(axis, num_pixels, pixel_size_um, label):
    scanaxis = ScanAxis(axis, num_pixels, pixel_size_um)

    assert scanaxis.axis == axis
    assert scanaxis.num_pixels == num_pixels
    assert scanaxis.pixel_size_um == pixel_size_um
    assert scanaxis.axis_label == label


sa0 = ScanAxis(0, 10, 0.5)
sa1 = ScanAxis(1, 5, 0.4)
sa2 = ScanAxis(2, 15, 0.36)
cp0 = np.array([0.1, 1.0, 10.0])
cp1 = np.array([10.0, 1.0, 0.1])


@pytest.mark.parametrize(
    "scan_axes, scan_order, center_point, num_frames",
    [
        ([sa0], [0], cp0, 10),
        ([sa0, sa1], [0, 1], cp1, 20),
        ([sa1, sa0], [1, 0], cp0, 30),
        ([sa2, sa0], [1, 0], cp1, 40),
    ],
)
def test_scanmetadata(scan_axes, scan_order, center_point, num_frames):
    metadata = ScanMetaData(scan_axes, center_point, num_frames)

    assert metadata.scan_axes is scan_axes
    assert metadata.center_point_um is center_point
    assert metadata.num_frames == num_frames
    assert metadata.num_axes == len(scan_axes)
    assert metadata.fast_axis == scan_axes[0].axis_label
    assert len(metadata.scan_order) == len(scan_order)
    np.testing.assert_equal(metadata.scan_order, scan_order)
    assert np.all(
        [oa is sa for oa, sa in zip(metadata.ordered_axes, [scan_axes[i] for i in scan_order])]
    )


def test_scanmetadata_with_num_frames():
    scan_axes = [sa0, sa1]
    center_point_um = [0.1, 1.0, 10.0]
    metadata = ScanMetaData(scan_axes, center_point_um, 0)
    assert metadata.num_frames == 0
    metadata = metadata.with_num_frames(10)
    metadata.scan_axes is scan_axes
    metadata.center_point_um is center_point_um
    metadata.num_frames == 10


@pytest.mark.parametrize(
    "axes, shape, pixel_sizes_nm, scan_order",
    [
        ([0], [10], [500], [0]),
        ([0, 1], [10, 5], [500, 400], [0, 1]),
        ([1, 0], [5, 10], [400, 500], [1, 0]),
        ([2, 0], [15, 10], [360, 500], [1, 0]),
    ],
)
def test_scanmetadata_from_json(axes, shape, pixel_sizes_nm, scan_order):
    scan_axes = [
        ScanAxis(axis, num_pixels, pixel_size_nm * 1e-3)
        for axis, num_pixels, pixel_size_nm in zip(axes, shape, pixel_sizes_nm)
    ]
    json_string = generate_scan_json(
        [
            {
                "axis": axis,
                "num of pixels": num_pixels,
                "pixel size (nm)": pixel_size,
            }
            for pixel_size, axis, num_pixels in zip(pixel_sizes_nm, axes, shape)
        ]
    )
    metadata = ScanMetaData.from_json(json_string)
    assert metadata.scan_axes == scan_axes
    # center_point is hard coded in function `generate_scan_json()`
    assert metadata.center_point_um == {"x": 58.075877109272604, "y": 31.978375270573267, "z": 0}
    # num_frames / scan count is hard coded in function `generate_scan_json()`
    assert metadata.num_frames == 0
    assert metadata.num_axes == len(scan_axes)
    assert metadata.fast_axis == scan_axes[0].axis_label
    assert len(metadata.scan_order) == len(scan_order)
    np.testing.assert_equal(metadata.scan_order, scan_order)
    assert np.all(
        [oa == sa for oa, sa in zip(metadata.ordered_axes, [scan_axes[i] for i in scan_order])]
    )


@pytest.mark.parametrize(
    "item, has_timestamps",
    [
        (generate_kymo("Mock", np.ones((5, 5))), True),
        (generate_scan("Mock", np.ones((5, 5)), [1, 1]), True),
        (_kymo_from_array(np.ones((5, 5)), "r", 1), False),
    ],
)
def test_immutable_returns(item, has_timestamps):
    """Ensure that users cannot modify data in the cache."""
    red = item.get_image("red")

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        red[2, 2] = 100
    assert item.get_image("red")[2, 2] == 1

    if has_timestamps:
        ts = item.timestamps
        ref_ts = ts[2, 2]

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            ts[2, 2] = 100
        assert item.timestamps[2, 2] == ref_ts
