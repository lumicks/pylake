import pytest
import matplotlib.pyplot as plt
import numpy as np
from lumicks import pylake
from matplotlib.testing.decorators import cleanup
from ..data.mock_confocal import generate_scan


def test_scans(test_scans):
    scan = test_scans["fast Y slow X"]
    assert repr(scan) == "Scan(pixels=(4, 5))"

    # fmt: off
    reference_timestamps = np.array([[2.006250e+10, 2.025000e+10, 2.043750e+10, 2.062500e+10],
                                    [2.084375e+10, 2.109375e+10, 2.128125e+10, 2.146875e+10],
                                    [2.165625e+10, 2.187500e+10, 2.206250e+10, 2.225000e+10],
                                    [2.243750e+10, 2.262500e+10, 2.284375e+10, 2.309375e+10],
                                    [2.328125e+10, 2.346875e+10, 2.365625e+10, 2.387500e+10]])
    # fmt: on

    np.testing.assert_allclose(scan.timestamps, np.transpose(reference_timestamps))
    assert scan.num_frames == 1
    with pytest.deprecated_call():
        scan.json
    with pytest.deprecated_call():
        assert scan.has_fluorescence
    with pytest.deprecated_call():
        assert not scan.has_force
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 5
    assert len(scan.infowave) == 64
    assert scan.rgb_image.shape == (4, 5, 3)
    assert scan.red_image.shape == (4, 5)
    assert scan.blue_image.shape == (4, 5)
    assert scan.green_image.shape == (4, 5)
    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [197 / 1000, 191 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.197 * 5, 0.191 * 4])
    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(scan.scan_width_um, [0.197 * 5 + 0.5, 0.191 * 4 + 0.5])

    scan = test_scans["fast Y slow X multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    with pytest.deprecated_call():
        scan.json
    with pytest.deprecated_call():
        assert scan.has_fluorescence
    with pytest.deprecated_call():
        assert not scan.has_force
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 64
    assert scan.rgb_image.shape == (2, 4, 3, 3)
    assert scan.red_image.shape == (2, 4, 3)
    assert scan.blue_image.shape == (2, 4, 3)
    assert scan.green_image.shape == (2, 4, 3)
    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [197 / 1000, 191 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.197 * 3, 0.191 * 4])

    scan = test_scans["fast X slow Z multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
    reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    with pytest.deprecated_call():
        scan.json
    with pytest.deprecated_call():
        assert scan.has_fluorescence
    with pytest.deprecated_call():
        assert not scan.has_force
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 64
    assert scan.rgb_image.shape == (2, 3, 4, 3)
    assert scan.red_image.shape == (2, 3, 4)
    assert scan.blue_image.shape == (2, 3, 4)
    assert scan.green_image.shape == (2, 3, 4)
    assert scan.fast_axis == "X"
    np.testing.assert_allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.191 * 4, 0.197 * 3])

    scan = test_scans["fast Y slow Z multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
    reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    with pytest.deprecated_call():
        scan.json
    with pytest.deprecated_call():
        assert scan.has_fluorescence
    with pytest.deprecated_call():
        assert not scan.has_force
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 64
    assert scan.rgb_image.shape == (2, 3, 4, 3)
    assert scan.red_image.shape == (2, 3, 4)
    assert scan.blue_image.shape == (2, 3, 4)
    assert scan.green_image.shape == (2, 3, 4)
    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.191 * 4, 0.197 * 3])


def test_slicing(test_scans):
    scan0 = test_scans["multiframe_poisson"]
    assert scan0.num_frames == 10

    def compare_frames(original_frames, new_scan):
        assert new_scan.num_frames == len(original_frames)
        for new_frame_index, index in enumerate(original_frames):
            frame = scan0.red_image[index]
            new_frame = (
                new_scan.red_image[new_frame_index]
                if new_scan.num_frames > 1
                else new_scan.red_image
            )
            np.testing.assert_equal(frame, new_frame)

    compare_frames([0], scan0[0])  # first frame
    compare_frames([9], scan0[-1])  # last frame
    compare_frames([3], scan0[3])  # single frame
    compare_frames([2, 3, 4], scan0[2:5])  # slice
    compare_frames([0, 1, 2], scan0[:3])  # from beginning
    compare_frames([8, 9], scan0[8:])  # until end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[:])  # all
    compare_frames([7], scan0[-3])  # negative index
    compare_frames([0, 1, 2, 3], scan0[:-6])  # until negative index
    compare_frames([5, 6, 7], scan0[5:-2])  # mixed sign indices
    compare_frames([6, 7], scan0[-4:-2])  # full negative slice

    compare_frames([3, 4], scan0[2:6][1:3])  # iterative slicing

    compare_frames([1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[1:100])  # test clamping past the end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], scan0[-100:9])  # test clamping past the beginning
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[-100:100])  # test clamping both dirs

    # reverse slice
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan0[5:3]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan0[-2:-4]

    # empty slice
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan0[5:5]


def test_damaged_scan(test_scans):
    # Assume the user incorrectly exported only a partial scan (62500000 is the time step)
    scan = test_scans["truncated_scan"]
    with pytest.raises(RuntimeError):
        scan.red_image.shape

    # Test for workaround for a bug in the STED delay mechanism which could result in scan start times ending up
    # within the sample time.
    scan = test_scans["sted bug"]
    middle = test_scans["fast Y slow X"].red_photon_count.timestamps[5]
    scan.red_image.shape  # should not raise, but change the start appropriately to work around sted bug
    np.testing.assert_allclose(scan.start, middle)


@cleanup
def test_plotting(test_scans):
    scan = test_scans["fast Y slow X multiframe"]
    scan.plot(channel="blue")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), scan.blue_image[0])
    np.testing.assert_allclose(image.get_extent(), [0, 0.197 * 3, 0.191 * 4, 0])
    plt.close()

    scan = test_scans["fast X slow Z multiframe"]
    scan.plot(channel="rgb")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), scan.rgb_image[0] / np.max(scan.rgb_image[0]))
    np.testing.assert_allclose(image.get_extent(), [0, 0.191 * 4, 0.197 * 3, 0])
    plt.close()

    scan = test_scans["fast Y slow Z multiframe"]
    scan.plot(channel="rgb")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), scan.rgb_image[0] / np.max(scan.rgb_image[0]))
    np.testing.assert_allclose(image.get_extent(), [0, 0.191 * 4, 0.197 * 3, 0])
    plt.close()


@cleanup
def test_deprecated_plotting(test_scans):
    scan = test_scans["fast Y slow X multiframe"]
    with pytest.deprecated_call():
        scan.plot_red()
    with pytest.deprecated_call():
        scan.plot_green()
    with pytest.deprecated_call():
        scan.plot_blue()
    with pytest.deprecated_call():
        scan.plot_rgb()


def test_save_tiff(tmpdir_factory, test_scans):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    scan = test_scans["fast Y slow X"]
    scan.save_tiff(f"{tmpdir}/single_frame.tiff")
    assert stat(f"{tmpdir}/single_frame.tiff").st_size > 0

    scan = test_scans["fast Y slow X multiframe"]
    scan.save_tiff(f"{tmpdir}/multi_frame.tiff")
    assert stat(f"{tmpdir}/multi_frame.tiff").st_size > 0


def test_movie_export(tmpdir_factory, test_scans):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    scan = test_scans["fast Y slow X multiframe"]
    scan.export_video_red(f"{tmpdir}/red.gif", 0, 4)
    assert stat(f"{tmpdir}/red.gif").st_size > 0
    scan.export_video_rgb(f"{tmpdir}/rgb.gif", 0, 4)
    assert stat(f"{tmpdir}/rgb.gif").st_size > 0


@pytest.mark.parametrize(
    "dim_x, dim_y, line_padding, start, dt, samples_per_pixel",
    [
        (5, 6, 3, 14, 4, 4),
        (3, 4, 60, 1592916040906356300, 12800, 30),
        (3, 2, 60, 1592916040906356300, 12800, 3000),
    ],
)
def test_single_frame_times(dim_x, dim_y, line_padding, start, dt, samples_per_pixel):
    img = np.ones((dim_x, dim_y))
    scan = generate_scan(
        "test",
        img,
        [1, 1],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    frame_times = scan.frame_timestamp_ranges()
    assert len(frame_times) == 1
    assert frame_times[0][0] == start + line_padding * dt
    line_time = dt * (img.shape[1] * samples_per_pixel + 2 * line_padding) * img.shape[0]
    assert frame_times[0][1] == start + line_time - line_padding * dt - dt

    # For the single frame case, there is no dead time, so these are identical
    frame_times_inclusive = scan.frame_timestamp_ranges(exclude=False)
    assert len(frame_times_inclusive) == 1
    assert frame_times_inclusive[0][0] == frame_times[0][0]
    assert frame_times_inclusive[0][1] == frame_times[0][1]


@pytest.mark.parametrize(
    "dim_x, dim_y, frames, line_padding, start, dt, samples_per_pixel",
    [
        (5, 6, 3, 3, 14, 4, 4),
        (3, 4, 4, 60, 1592916040906356300, 12800, 30),
        (3, 2, 2, 60, 1592916040906356300, 12800, 3000),
    ],
)
def test_multiple_frame_times(dim_x, dim_y, frames, line_padding, start, dt, samples_per_pixel):
    img = np.ones((dim_x, dim_y, frames))
    scan = generate_scan(
        "test",
        img,
        [1, 1],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    frame_times = scan.frame_timestamp_ranges()

    line_time = dt * (img.shape[2] * samples_per_pixel + 2 * line_padding) * img.shape[1]
    assert len(frame_times) == scan.num_frames
    assert frame_times[0][0] == start + line_padding * dt
    assert frame_times[0][1] == start + line_time - line_padding * dt - dt
    assert frame_times[1][0] == start + line_padding * dt + line_time
    assert frame_times[1][1] == start + 2 * line_time - line_padding * dt - dt
    assert frame_times[-1][0] == start + line_padding * dt + (len(frame_times) - 1) * line_time
    assert frame_times[-1][1] == start + len(frame_times) * line_time - line_padding * dt - dt

    frame_times_inclusive = scan.frame_timestamp_ranges(exclude=False)

    # Start times should be the same
    assert len(frame_times_inclusive) == scan.num_frames
    assert frame_times_inclusive[0][0] == frame_times[0][0]
    assert frame_times_inclusive[1][0] == frame_times[1][0]
    assert frame_times_inclusive[-1][0] == frame_times[-1][0]

    assert frame_times_inclusive[0][1] == frame_times[1][0]
    assert frame_times_inclusive[1][1] == frame_times[2][0]
    assert frame_times_inclusive[-1][1] == frame_times[-1][0] + (
        frame_times[1][0] - frame_times[0][0]
    )
