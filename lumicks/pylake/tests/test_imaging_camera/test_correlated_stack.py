import numpy as np
import json
import tifffile
from pathlib import Path
import pytest
from lumicks.pylake.correlated_stack import CorrelatedStack
from lumicks.pylake.detail.widefield import TiffStack
from lumicks.pylake import channel
import matplotlib as mpl
from matplotlib.testing.decorators import cleanup
from ..data.mock_widefield import MockTiffFile, make_alignment_image_data, make_frame_times


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_correlated_stack(shape):
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6)), align_requested=False
    )
    stack = CorrelatedStack.from_dataset(fake_tiff)

    assert stack[0].start == 10
    assert stack[1].start == 20
    assert stack[-1].start == 60
    assert stack[0].num_frames == 1

    assert stack[0].stop == 18
    assert stack[-1].stop == 68

    assert stack[1:2].stop == 28
    assert stack[1:3].stop == 38
    assert stack[1:2].num_frames == 1
    assert stack[1:3].num_frames == 2

    assert stack[3:5][0].start == 40
    assert stack[3:5][1].start == 50
    assert stack[3:5][0].num_frames == 1

    with pytest.raises(IndexError):
        stack[3:5][2]

    assert stack[2:5][1:2].start == 40
    assert stack[2:5][1:3]._get_frame(1).start == 50

    with pytest.raises(IndexError):
        stack[::2]

    with pytest.raises(IndexError):
        stack[1:2]._get_frame(1).stop

    # Integration test whether slicing from the stack object actually provides you with correct slices
    np.testing.assert_allclose(stack[2:5].start, 30)
    np.testing.assert_allclose(stack[2:5].stop, 58)

    # Test iterations
    np.testing.assert_allclose([x.start for x in stack], [10, 20, 30, 40, 50, 60])
    np.testing.assert_allclose([x.start for x in stack[1:]], [20, 30, 40, 50, 60])
    np.testing.assert_allclose([x.start for x in stack[:-1]], [10, 20, 30, 40, 50])
    np.testing.assert_allclose([x.start for x in stack[2:4]], [30, 40])
    np.testing.assert_allclose([x.start for x in stack[2]], [30])


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_slicing(shape):
    image = [np.random.poisson(10, size=shape) for _ in range(10)]
    times = make_frame_times(10)
    fake_tiff = TiffStack(MockTiffFile(data=image, times=times), align_requested=False)
    stack0 = CorrelatedStack.from_dataset(fake_tiff)

    def compare_frames(original_frames, new_stack):
        assert new_stack.num_frames == len(original_frames)
        for new_frame_index, index in enumerate(original_frames):
            frame = stack0._get_frame(index).data
            new_frame = new_stack._get_frame(new_frame_index).data
            np.testing.assert_equal(frame, new_frame)

    compare_frames([0], stack0[0])  # first frame
    compare_frames([9], stack0[-1])  # last frame
    compare_frames([3], stack0[3])  # single frame
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], stack0[:])  # all frames
    compare_frames([3, 4, 5], stack0[3:6])  # normal slice
    compare_frames([0, 1, 2], stack0[:3])  # from beginning
    compare_frames([6, 7, 8, 9], stack0[6:])  # until end
    compare_frames([0, 1, 2, 3, 4, 5], stack0[:-4])  # until negative index
    compare_frames([5, 6, 7], stack0[5:-2])  # mixed sign indices
    compare_frames([6, 7], stack0[-4:-2])  # negative indices slice

    compare_frames([3, 4], stack0[2:6][1:3])  # iterative slicing

    compare_frames([1, 2, 3, 4, 5, 6, 7, 8, 9], stack0[1:100])  # test clamping past the end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], stack0[-100:9])  # test clamping past the beginning
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], stack0[-100:100])  # test clamping both dirs

    # reverse slice
    with pytest.raises(NotImplementedError, match="Reverse slicing is not supported"):
        stack0[5:2]

    # reverse slice, negative indices
    with pytest.raises(NotImplementedError, match="Reverse slicing is not supported"):
        stack0[-1:-3]

    # empty slice
    with pytest.raises(NotImplementedError, match="Slice is empty"):
        stack0[5:5]


def test_deprecated_timestamps():
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones((5, 4, 3))] * 6, times=make_frame_times(6)),
        align_requested=False,
    )
    stack = CorrelatedStack.from_dataset(fake_tiff)
    with pytest.deprecated_call():
        stack.timestamps


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_correlation(shape):
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))

    # Test image stack without dead time
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6, step=10)),
        align_requested=False,
    )
    stack = CorrelatedStack.from_dataset(fake_tiff)
    np.testing.assert_allclose(
        np.hstack([cc[x.start : x.stop].data for x in stack[2:4]]), np.arange(30, 50, 2)
    )

    # Test image stack with dead time
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6)), align_requested=False
    )
    stack = CorrelatedStack.from_dataset(fake_tiff)

    np.testing.assert_allclose(
        np.hstack([cc[x.start : x.stop].data for x in stack[2:4]]),
        np.hstack([np.arange(30, 38, 2), np.arange(40, 48, 2)]),
    )

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a stack.
    ch = cc.downsampled_over(stack[0:3].frame_timestamp_ranges)
    np.testing.assert_allclose(
        ch.data,
        [
            np.mean(np.arange(10, 18, 2)),
            np.mean(np.arange(20, 28, 2)),
            np.mean(np.arange(30, 38, 2)),
        ],
    )
    np.testing.assert_allclose(ch.timestamps, [(10 + 16) / 2, (20 + 26) / 2, (30 + 36) / 2])

    ch = cc.downsampled_over(stack[1:4].frame_timestamp_ranges)
    np.testing.assert_allclose(
        ch.data,
        [
            np.mean(np.arange(20, 28, 2)),
            np.mean(np.arange(30, 38, 2)),
            np.mean(np.arange(40, 48, 2)),
        ],
    )
    np.testing.assert_allclose(ch.timestamps, [(20 + 26) / 2, (30 + 36) / 2, (40 + 46) / 2])

    with pytest.raises(TypeError):
        cc.downsampled_over(stack[1:4])

    with pytest.raises(ValueError):
        cc.downsampled_over(stack[1:4].frame_timestamp_ranges, where="up")

    with pytest.raises(AssertionError):
        cc["0ns":"20ns"].downsampled_over(stack[3:4].frame_timestamp_ranges)

    with pytest.raises(AssertionError):
        cc["40ns":"70ns"].downsampled_over(stack[0:1].frame_timestamp_ranges)

    assert stack[0]._get_frame(0).start == 10
    assert stack[1]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(1).start == 30

    # Regression test downsampled_over losing precision due to reverting to double rather than int64.
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 1588267266006287100, 1000))
    ch = cc.downsampled_over([(1588267266006287100, 1588267266006287120)], where="left")
    assert int(ch.timestamps[0]) == 1588267266006287100


def test_name_change_from_data():
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones((5, 4, 3))], times=make_frame_times(1)), align_requested=False
    )
    with pytest.deprecated_call():
        CorrelatedStack.from_data(fake_tiff)


def test_stack_roi():
    first_page = np.arange(60).reshape((6, 10))
    data = np.stack([first_page + (j * 60) for j in range(3)], axis=2)
    stack_0 = TiffStack(MockTiffFile([data], times=make_frame_times(1)), align_requested=False)

    # recursive cropping
    stack_1 = stack_0.with_roi([1, 7, 3, 6])
    np.testing.assert_equal(stack_1.get_frame(0).data, data[3:6, 1:7, :])

    stack_2 = stack_1.with_roi([3, 6, 0, 3])
    np.testing.assert_equal(stack_2.get_frame(0).data, data[3:6, 4:7, :])

    stack_3 = stack_2.with_roi([1, 2, 1, 2])
    np.testing.assert_equal(stack_3.get_frame(0).data, data[4:5, 5:6, :])

    # negative indices
    with pytest.raises(ValueError):
        stack_4 = stack_0.with_roi([-5, 4, 1, 2])

    # out of bounds
    with pytest.raises(ValueError):
        stack_5 = stack_0.with_roi([0, 11, 1, 2])


def test_deprecate_raw():
    fake_tiff = TiffStack(
        MockTiffFile(data=[np.ones((5, 4, 3))], times=make_frame_times(1)), align_requested=False
    )
    stack = CorrelatedStack.from_dataset(fake_tiff)

    with pytest.deprecated_call():
        stack.raw


@cleanup
def test_plot_correlated():
    cc = channel.Slice(
        channel.Continuous(np.arange(10, 80, 2), 10, 2), {"y": "mock", "title": "mock"}
    )

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(
        MockTiffFile(
            data=[
                np.zeros((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)) * 2,
                np.ones((3, 3)) * 3,
                np.ones((3, 3)) * 4,
                np.ones((3, 3)) * 5,
            ],
            times=make_frame_times(7, step=10),
        ),
        align_requested=False,
    )

    CorrelatedStack.from_dataset(fake_tiff)[3:5].plot_correlated(cc)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((3, 3)) * 3)

    CorrelatedStack.from_dataset(fake_tiff)[3:5].plot_correlated(cc, frame=1)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((3, 3)) * 4)


def test_plot_correlated_smaller_channel():
    from matplotlib.backend_bases import MouseEvent

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(
        MockTiffFile(
            data=[
                np.zeros((3, 3)),
                np.ones((3, 3)),
                np.ones((3, 3)) * 2,
                np.ones((3, 3)) * 3,
                np.ones((3, 3)) * 4,
                np.ones((3, 3)) * 5,
            ],
            times=make_frame_times(7, step=10),
        ),
        align_requested=False,
    )

    # Add test for when there's only a subset in terms of channel data
    cc = channel.Slice(
        channel.Continuous(np.arange(10, 80, 2), 30, 2), {"y": "mock", "title": "mock"}
    )

    with pytest.warns(UserWarning):
        CorrelatedStack.from_dataset(fake_tiff).plot_correlated(cc)

    def mock_click(fig, data_position):
        pos = fig.axes[0].transData.transform(data_position)
        fig.canvas.callbacks.process(
            "button_press_event", MouseEvent("button_press_event", fig.canvas, pos[0], pos[1], 1)
        )
        images = [
            obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)
        ]
        assert len(images) == 1
        return images[0].get_array()

    np.testing.assert_allclose(mock_click(mpl.pyplot.gcf(), np.array([0, 40])), np.ones((3, 3)) * 2)
    np.testing.assert_allclose(
        mock_click(mpl.pyplot.gcf(), np.array([10.1e-9, 40])), np.ones((3, 3)) * 3
    )


def test_cropping(rgb_tiff_file, gray_tiff_file):
    for filename in (rgb_tiff_file, gray_tiff_file):
        for align in (True, False):
            stack = CorrelatedStack(filename, align=True)
            cropped = stack.crop_by_pixels(25, 50, 25, 50)
            np.testing.assert_allclose(
                cropped._get_frame(0).data,
                stack._get_frame(0).data[25:50, 25:50],
                err_msg=f"failed on {Path(filename).name}, align={align}, frame.data",
            )
            np.testing.assert_allclose(
                cropped._get_frame(0).raw_data,
                stack._get_frame(0).raw_data[25:50, 25:50],
                err_msg=f"failed on {Path(filename).name}, align={align}, frame.raw_data",
            )
            stack.src._tiff_file.close()


def test_cropping_then_export(
    rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi
):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        stack = stack.crop_by_pixels(10, 190, 20, 80)

        stack.export_tiff(savename)
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60
        stack.src._tiff_file.close()


def test_get_image():
    # grayscale image - multiple frames
    data = [np.full((2, 2), j) for j in range(3)]
    times = make_frame_times(3)

    fake_tiff = TiffStack(MockTiffFile(data, times), align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image())

    # grayscale image - single frame
    fake_tiff = TiffStack(MockTiffFile([data[0]], [times[0]]), align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    np.testing.assert_array_equal(data[0], stack.get_image())

    # RGB image - multiple frames
    rgb_data = np.stack([np.full((2, 2), j) for j in range(3)], axis=2)
    data = [rgb_data] * 3

    fake_tiff = TiffStack(MockTiffFile(data, times), align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)

    for j, color in enumerate(("red", "green", "blue")):
        np.testing.assert_array_equal(
            stack.get_image(channel=color), j, err_msg=f"failed on {color}"
        )
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image())
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image(channel="rgb"))

    # RGB image - multiple frames
    fake_tiff = TiffStack(MockTiffFile([data[0]], [times[0]]), align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)

    for j, color in enumerate(("red", "green", "blue")):
        np.testing.assert_array_equal(
            stack.get_image(channel=color), data[0][:, :, j], err_msg=f"failed on {color}"
        )
    np.testing.assert_array_equal(data[0], stack.get_image())
    np.testing.assert_array_equal(data[0], stack.get_image("rgb"))


def test_define_tether():
    from lumicks.pylake.detail.widefield import TransformMatrix

    def make_stack(data, description, bit_depth):
        tiff = TiffStack(
            MockTiffFile(
                data=[data],
                times=make_frame_times(1),
                description=json.dumps(description),
                bit_depth=bit_depth,
            ),
            align_requested=True,
        )
        return CorrelatedStack.from_dataset(tiff)

    rot = TransformMatrix.rotation(25, (100, 50))
    horizontal_spot_coordinates = [(50, 50), (100, 50), (150, 50)]
    spot_coordinates = rot.warp_coordinates(horizontal_spot_coordinates)
    tether_ends = np.array((spot_coordinates[0], spot_coordinates[-1]))
    warp_parameters = {
        "red_warp_parameters": {"Tx": 0, "Ty": 0, "theta": 0},
        "blue_warp_parameters": {"Tx": 0, "Ty": 0, "theta": 0},
    }

    def make_test_data(version, bit_depth, camera):
        _, image, description, bit_depth = make_alignment_image_data(
            spot_coordinates, version=version, bit_depth=bit_depth, camera=camera, **warp_parameters
        )
        ref_image, _, ref_description, _ = make_alignment_image_data(
            horizontal_spot_coordinates,
            version=version,
            bit_depth=bit_depth,
            camera=camera,
            **warp_parameters,
        )

        test_stack = make_stack(image, description, bit_depth)
        ref_stack = make_stack(ref_image, ref_description, bit_depth)
        return test_stack, ref_stack

    def check_result(stack, ref_stack, bit_depth):
        # calculate error as max absolute difference
        # as fraction of bit depth, allow some room for interpolation error
        frame = np.atleast_3d(stack._get_frame(0).data)[:, :, 0].astype(float)
        ref = np.atleast_3d(ref_stack._get_frame(0).data)[:, :, 0].astype(float)
        assert np.max(np.abs(frame - ref)) / (2 ** bit_depth - 1) < 0.055

    # IRM - grayscale
    stack, ref_stack = make_test_data(1, 8, "irm")
    stack = stack.define_tether(*tether_ends)
    check_result(stack, ref_stack, 8)
    np.testing.assert_allclose(
        stack.src._tether.ends, (horizontal_spot_coordinates[0], horizontal_spot_coordinates[-1])
    )

    # WT - RGB
    stack, ref_stack = make_test_data(1, 16, "wt")
    stack = stack.define_tether(*tether_ends)
    check_result(stack, ref_stack, 16)
    np.testing.assert_allclose(
        stack.src._tether.ends, (horizontal_spot_coordinates[0], horizontal_spot_coordinates[-1])
    )

    # test crop/tether permutations
    original_stack, original_ref_stack = make_test_data(1, 16, "wt")
    crop_coordinates = (25, 175, 25, 75)
    bad_tether_ends = tether_ends + [[5, -10], [5, 10]]
    offset_tether_ends = tether_ends - (25, 25)

    # tether -> crop
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.define_tether(*tether_ends)
    stack = stack.crop_by_pixels(*crop_coordinates)
    check_result(stack, ref_stack, 16)

    # crop -> tether
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.crop_by_pixels(*crop_coordinates)
    stack = stack.define_tether(*offset_tether_ends)
    check_result(stack, ref_stack, 16)

    # tether -> tether
    stack = original_stack.define_tether(*bad_tether_ends)
    correct_points = stack.src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    stack = stack.define_tether(*correct_points)
    check_result(stack, original_ref_stack, 16)

    # tether -> tether -> crop
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.define_tether(*bad_tether_ends)
    correct_points = stack.src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    stack = stack.define_tether(*correct_points)
    stack = stack.crop_by_pixels(*crop_coordinates)
    check_result(stack, ref_stack, 16)

    # crop -> tether -> tether
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.crop_by_pixels(*crop_coordinates)
    cropped_offset_tether_ends = offset_tether_ends + [[5, -10], [5, 10]]
    stack = stack.define_tether(*cropped_offset_tether_ends)
    correct_points = stack.src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    offset_correct_points = [np.array(pp) - (25, 25) for pp in correct_points]
    stack = stack.define_tether(*offset_correct_points)
    check_result(stack, ref_stack, 16)

    # tether -> crop -> tether
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.define_tether(*bad_tether_ends)
    stack = stack.crop_by_pixels(*crop_coordinates)
    correct_points = stack.src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    offset_correct_points = [np.array(pp) - (25, 25) for pp in correct_points]
    stack = stack.define_tether(*offset_correct_points)
    check_result(stack, ref_stack, 16)
