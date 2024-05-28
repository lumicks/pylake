import re
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt

from lumicks.pylake import ImageStack, CorrelatedStack, channel
from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.detail.widefield import TiffStack
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP
from lumicks.pylake.kymotracker.kymotracker import track_greedy

from ..data.mock_widefield import MockTiffFile, make_frame_times, make_alignment_image_data


def to_tiff(image, description, bit_depth, start_time=1, num_images=2):
    return MockTiffFile(
        data=[image] * num_images,
        times=make_frame_times(num_images, start=start_time),
        description=description,
        bit_depth=bit_depth,
    )


def test_correlated_stack_deprecation(rgb_tiff_file):
    with pytest.warns(DeprecationWarning):
        cs = CorrelatedStack(str(rgb_tiff_file), align=True)
        cs.close()


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_image_stack(shape):
    fake_tiff = TiffStack(
        [MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6))], align_requested=False
    )
    stack = ImageStack.from_dataset(fake_tiff)

    assert stack.name is None
    assert stack[0].start == 10
    assert stack[1].start == 20
    assert stack[-1].start == 60
    assert stack[0].num_frames == 1

    assert stack[0].stop == 18
    assert stack[-1].stop == 68

    # Test if tuple of size one is correctly interpreted
    assert stack[0,].start == 10
    assert stack[1,].start == 20
    assert stack[-1,].start == 60
    assert stack[0,].num_frames == 1

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

    # Test slicing with steps (slice & int)
    assert stack[::2][0].start == 10
    assert stack[::2][1].start == 30
    assert stack[::3][1].start == 40
    assert stack[::4][1].start == 50
    assert stack[::2][-1].start == 50
    assert stack[::3][-1].start == 40
    assert stack[::4][-1].start == 50
    assert stack[-10::2][1].start == 30
    assert stack[-10::3][1].start == 40
    assert stack[-10::4][1].start == 50
    assert stack[:10:2][-1].start == 50
    assert stack[:10:3][-1].start == 40
    assert stack[:10:4][-1].start == 50
    assert stack[3:5:2][0].start == 40
    assert stack[3:6:2][0].start == 40
    assert stack[3:6:2][1].start == 60
    # Test slicing with steps (slice & slice)
    assert stack[::2][0::].start == 10
    assert stack[::2][1::].start == 30
    assert stack[::3][1::].start == 40
    assert stack[::4][1::].start == 50
    assert stack[::2][-1::].start == 50
    assert stack[::3][-1::].start == 40
    assert stack[::4][-1::].start == 50
    # Test slicing with steps (correct number of frames)
    assert stack[::2].num_frames == 3
    assert stack[::3].num_frames == 2
    assert stack[::4].num_frames == 2
    assert stack[-10::2].num_frames == 3
    assert stack[-10::3].num_frames == 2
    assert stack[-10::4].num_frames == 2
    assert stack[:10:2].num_frames == 3
    assert stack[:10:3].num_frames == 2
    assert stack[:10:4].num_frames == 2
    assert stack[3:5:2].num_frames == 1
    assert stack[3:6:2].num_frames == 2

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


def test_warning_unequal_exposure():
    times = make_frame_times(6)
    times[1][2] = times[1][2] / 2  # Ensure unequal exposure
    fake_tiff = TiffStack(
        tiff_files=[MockTiffFile(data=[np.ones((4, 3, 3))] * 6, times=times)],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    with pytest.warns(RuntimeWarning, match="image stack contains a non-constant exposure time"):
        stack.frame_timestamp_ranges()


def test_stack_from_dataset():
    frame_times = make_frame_times(2)
    mock_stack = TiffStack(
        [MockTiffFile([np.zeros((3, 3, 3))] * 2, frame_times)],
        align_requested=False,
    )

    stack = ImageStack.from_dataset(mock_stack, name="hello", start_idx=1, stop_idx=2, step=3)
    assert stack.name == "hello"
    assert stack._start_idx == 1
    assert stack._stop_idx == 2
    assert stack._step == 3


def test_stack_name_from_file(rgb_tiff_file):
    cs = ImageStack(str(rgb_tiff_file), align=True)
    assert cs.name == "rgb_single"
    cs.close()


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_slicing(shape):
    image = [np.random.poisson(10, size=shape) for _ in range(10)]
    start = _FIRST_TIMESTAMP + 100
    times = make_frame_times(10, step=int(0.8e9), start=start, frame_time=int(1e9))
    fake_tiff = TiffStack([MockTiffFile(data=image, times=times)], align_requested=False)
    stack0 = ImageStack.from_dataset(fake_tiff)

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

    # With steps
    compare_frames([0, 2, 4, 6, 8], stack0[::2])  # all frames with steps
    compare_frames([0, 3, 6, 9], stack0[::3])  # all frames with steps
    compare_frames([3, 5], stack0[3:6:2])  # normal slice with steps
    compare_frames([0, 2], stack0[:3:2])  # from beginning with steps
    compare_frames([1, 5, 9], stack0[1::2][::2])  # recursive slice with steps
    compare_frames([3, 4], stack0[2:6][1:3])  # iterative slicing

    compare_frames([1, 2, 3, 4, 5, 6, 7, 8, 9], stack0[1:100])  # test clamping past the end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], stack0[-100:9])  # test clamping past the beginning
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], stack0[-100:100])  # test clamping both dirs

    compare_frames([2], stack0["2s":"2.81s"])
    compare_frames([2], stack0["2s":"3.8s"])
    compare_frames([2, 3], stack0["2s":"3.81s"])
    compare_frames([1, 2], stack0["1s":"3s"])
    compare_frames([2, 3, 4, 5, 6, 7, 8, 9], stack0["2s":])  # until end
    compare_frames([3, 4, 5, 6, 7, 8, 9], stack0["2.1s":])  # from beginning
    compare_frames([0, 1], stack0[:"1.81s"])  # until end
    compare_frames([0], stack0[:"1.8s"])  # from beginning

    compare_frames([3, 4, 5], stack0["2s":"5.81s"]["1s":"3.81s"])  # iterative
    compare_frames([3, 4], stack0["2s":"5.81s"]["1s":"3.80s"])  # iterative
    compare_frames([3, 4, 5], stack0["2s":"5.81s"]["1s":])  # iterative
    compare_frames([2, 4, 6, 8], stack0["2s"::2])  # include steps
    compare_frames([2, 4], stack0["2s":"5.81s"][::2])  # iterative with steps

    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], stack0[:"-0.9s"])  # negative indexing with time
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7], stack0[:"-1s"])  # negative indexing with time
    compare_frames([8, 9], stack0["-2s":])  # negative indexing with time
    compare_frames([9], stack0["-1.79s":])  # negative indexing with time
    compare_frames([2, 3, 4], stack0["2s":"5.81s"][:"-0.9s"])  # iterative with from end
    compare_frames([2, 3], stack0["2s":"5.81s"][:"-1s"])  # iterative with from end

    # Slice by timestamps
    compare_frames([2, 3], stack0[start + int(2e9) : start + int(4e9)])
    compare_frames([2, 3], stack0[start + int(2e9) : start + int(4.8e9)])
    compare_frames([2, 3, 4], stack0[start + int(2e9) : start + int(4.81e9)])
    compare_frames([0, 1, 2, 3, 4], stack0[: start + int(4.81e9)])
    compare_frames([5, 6, 7, 8, 9], stack0[start + int(5e9) :])
    compare_frames(
        [2, 3, 4], stack0[start + int(2e9) : start + int(4.81e9)][start : start + int(100e9)]
    )
    compare_frames(
        [3], stack0[start + int(2e9) : start + int(4.81e9)][start + int(3e9) : start + int(3.81e9)]
    )

    # empty slices
    for s in [
        slice(5, 2),
        slice(-3, 0),
        slice(-1, -3),
        slice(5, 5),
        slice(5, 5, -1),
        slice(-5, -3, -1),
        slice("2s", "2.8s"),
    ]:
        with pytest.raises(NotImplementedError, match="Slice is empty"):
            stack0[s]

    # reverse slices
    for s in [slice(5, 2, -1), slice(-3, 0, -1), slice(-1, -3, -1), slice(-3, -5, -1)]:
        with pytest.raises(NotImplementedError, match="Reverse slicing is not supported"):
            stack0[s]


@pytest.mark.parametrize("shape", [(3, 3), (5, 4, 3)])
def test_correlation(shape):
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))

    # Test image stack without dead time
    fake_tiff = TiffStack(
        [MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6))],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    np.testing.assert_allclose(
        np.hstack([cc[x.start : x.stop].data for x in stack[2:4]]),
        np.hstack((np.arange(30, 38, 2), np.arange(40, 48, 2))),
    )

    # Test image stack with dead time
    fake_tiff = TiffStack(
        [MockTiffFile(data=[np.ones(shape)] * 6, times=make_frame_times(6))], align_requested=False
    )
    stack = ImageStack.from_dataset(fake_tiff)

    np.testing.assert_allclose(
        np.hstack([cc[x.start : x.stop].data for x in stack[2:4]]),
        np.hstack([np.arange(30, 38, 2), np.arange(40, 48, 2)]),
    )

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a stack.
    ch = cc.downsampled_over(stack[0:3].frame_timestamp_ranges())
    np.testing.assert_allclose(
        ch.data,
        [
            np.mean(np.arange(10, 18, 2)),
            np.mean(np.arange(20, 28, 2)),
            np.mean(np.arange(30, 38, 2)),
        ],
    )
    np.testing.assert_allclose(ch.timestamps, [(10 + 16) / 2, (20 + 26) / 2, (30 + 36) / 2])

    ch = cc.downsampled_over(stack[1:4].frame_timestamp_ranges())
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
        cc.downsampled_over(stack[1:4].frame_timestamp_ranges(), where="up")

    with pytest.raises(RuntimeError, match="No overlap between range and selected channel."):
        cc["0ns":"20ns"].downsampled_over(stack[3:4].frame_timestamp_ranges())

    with pytest.raises(RuntimeError, match="No overlap between range and selected channel."):
        cc["40ns":"70ns"].downsampled_over(stack[0:1].frame_timestamp_ranges())

    assert stack[0]._get_frame(0).start == 10
    assert stack[1]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(0).start == 20
    assert stack[1:3]._get_frame(1).start == 30

    # Regression test downsampled_over losing precision due to reverting to double rather than int64.
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 1588267266006287100, 1000))
    ch = cc.downsampled_over([(1588267266006287100, 1588267266006287120)], where="left")
    assert int(ch.timestamps[0]) == 1588267266006287100


def test_stack_roi():
    first_page = np.arange(60).reshape((6, 10))
    data = np.stack([first_page + (j * 60) for j in range(3)], axis=2)
    stack_0 = TiffStack([MockTiffFile([data], times=make_frame_times(1))], align_requested=False)

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
    stack_5 = stack_0.with_roi([0, 11, 1, 2])
    np.testing.assert_equal(stack_5.get_frame(0).data, data[1:2, 0:11, :])


def test_roi_defaults():
    first_page = np.arange(60).reshape((6, 10))
    data = np.stack([first_page + (j * 60) for j in range(3)], axis=2)
    stack_0 = TiffStack([MockTiffFile([data], times=make_frame_times(1))], align_requested=False)

    np.testing.assert_equal(stack_0.with_roi([None, 7, 3, 6]).get_frame(0).data, data[3:6, :7])
    np.testing.assert_equal(stack_0.with_roi([1, None, 3, 6]).get_frame(0).data, data[3:6, 1:])
    np.testing.assert_equal(stack_0.with_roi([1, 7, None, 6]).get_frame(0).data, data[:6, 1:7])
    np.testing.assert_equal(stack_0.with_roi([1, 7, 3, None]).get_frame(0).data, data[3:, 1:7])
    np.testing.assert_equal(stack_0.with_roi([None, None, 3, 6]).get_frame(0).data, data[3:6, :])
    np.testing.assert_equal(stack_0.with_roi([1, 7, None, None]).get_frame(0).data, data[:, 1:7])


def test_image_stack_plotting(rgb_alignment_image_data):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    image = stack.plot(channel="blue", frame=0)
    assert id(image) == id(plt.gca().get_images()[0])
    ref_image = stack._get_frame(0)._get_plot_data(channel="blue")
    np.testing.assert_allclose(image.get_array(), ref_image)
    plt.close()

    image = stack.plot(channel="rgb", frame=0)
    assert id(image) == id(plt.gca().get_images()[0])
    ref_image = stack._get_frame(0)._get_plot_data(channel="rgb")
    np.testing.assert_allclose(image.get_array(), ref_image)
    plt.close()

    image = stack.plot(channel="blue", frame=0)

    # Update existing image handle (should be ok)
    image = stack.plot(channel="blue", frame=0, image_handle=image)
    assert id(image) == id(plt.gca().get_images()[0])

    with pytest.raises(
        ValueError, match="Supplied image_handle with a different axes than the provided axes"
    ):
        stack.plot(channel="blue", frame=0, image_handle=image, axes=plt.axes(label="a new axes"))
    # Plot to a new axis
    axes = plt.axes([0, 0, 1, 1], label="axes")
    not_the_right_axes = plt.axes([0, 0, 1, 1], label="newer axes")  # We shouldn't draw to this one
    assert id(plt.gca()) == id(not_the_right_axes)
    assert id(stack.plot(channel="blue", frame=0, axes=axes).axes) == id(axes)

    with pytest.raises(IndexError, match="Frame index out of range"):
        stack.plot(channel="blue", frame=4)
    with pytest.raises(IndexError, match="Frame index out of range"):
        stack.plot(channel="blue", frame=-1)


def test_plot_correlated():
    cc = channel.Slice(
        channel.Continuous(np.arange(10, 80, 2), 10, 2), {"y": "mock", "title": "mock"}
    )

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(
        [
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
            )
        ],
        align_requested=False,
    )

    ImageStack.from_dataset(fake_tiff)[3:5].plot_correlated(cc)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((3, 3)) * 3)

    ImageStack.from_dataset(fake_tiff)[3:5].plot_correlated(cc, frame=1)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((3, 3)) * 4)


def test_plot_correlated_smaller_channel():
    from matplotlib.backend_bases import MouseEvent

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(
        [
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
            )
        ],
        align_requested=False,
    )

    # Add test for when there's only a subset in terms of channel data
    cc = channel.Slice(
        channel.Continuous(np.arange(10, 80, 2), 30, 2), {"y": "mock", "title": "mock"}
    )

    with pytest.warns(UserWarning):
        ImageStack.from_dataset(fake_tiff).plot_correlated(cc)

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


def test_stack_name(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            "lumicks.pylake.image_stack.TiffStack.from_file",
            lambda *args, **kwargs: TiffStack(
                [MockTiffFile([np.zeros((3, 3, 3))] * 2, make_frame_times(2))],
                align_requested=False,
            ),
        )

        stack = ImageStack("test.tiff")
        assert stack.name == "test"

        stack = ImageStack("test.tiff", "test2.tiff")
        assert stack.name == "Multi-file stack"


def test_cropping(rgb_tiff_file, gray_tiff_file):
    for filename in (rgb_tiff_file, gray_tiff_file):
        for align in (True, False):
            stack = ImageStack(filename, align=True)
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
            stack.close()


def test_cropping_then_export(
    rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi
):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = ImageStack(str(filename))
        stack = stack.crop_by_pixels(10, 190, 20, 80)

        stack.export_tiff(savename)
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60

        stack.close()


def test_get_image():
    # grayscale image - multiple frames
    data = [np.full((2, 2), j) for j in range(3)]
    times = make_frame_times(3)

    fake_tiff = TiffStack([MockTiffFile(data, times)], align_requested=False)
    stack = ImageStack.from_dataset(fake_tiff)
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image())

    # grayscale image - single frame
    fake_tiff = TiffStack([MockTiffFile([data[0]], [times[0]])], align_requested=False)
    stack = ImageStack.from_dataset(fake_tiff)
    np.testing.assert_array_equal(data[0], stack.get_image())

    # RGB image - multiple frames
    rgb_data = np.stack([np.full((2, 2), j) for j in range(3)], axis=2)
    data = [rgb_data] * 3

    fake_tiff = TiffStack([MockTiffFile(data, times)], align_requested=False)
    stack = ImageStack.from_dataset(fake_tiff)

    for j, color in enumerate(("red", "green", "blue")):
        np.testing.assert_array_equal(
            stack.get_image(channel=color), j, err_msg=f"failed on {color}"
        )
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image())
    np.testing.assert_array_equal(np.stack(data, axis=0), stack.get_image(channel="rgb"))

    # RGB image - multiple frames
    fake_tiff = TiffStack([MockTiffFile([data[0]], [times[0]])], align_requested=False)
    stack = ImageStack.from_dataset(fake_tiff)

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
            [
                MockTiffFile(
                    data=[data],
                    times=make_frame_times(1),
                    description=description,
                    bit_depth=bit_depth,
                )
            ],
            align_requested=True,
        )
        return ImageStack.from_dataset(tiff)

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
        assert np.max(np.abs(frame - ref)) / (2**bit_depth - 1) < 0.055

    # IRM - grayscale
    stack, ref_stack = make_test_data(1, 8, "irm")
    stack = stack.define_tether(*tether_ends)
    check_result(stack, ref_stack, 8)
    np.testing.assert_allclose(
        stack._src._tether.ends, (horizontal_spot_coordinates[0], horizontal_spot_coordinates[-1])
    )

    # WT - RGB
    stack, ref_stack = make_test_data(1, 16, "wt")
    stack = stack.define_tether(*tether_ends)
    check_result(stack, ref_stack, 16)
    np.testing.assert_allclose(
        stack._src._tether.ends, (horizontal_spot_coordinates[0], horizontal_spot_coordinates[-1])
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
    correct_points = stack._src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    stack = stack.define_tether(*correct_points)
    check_result(stack, original_ref_stack, 16)

    # tether -> tether -> crop
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.define_tether(*bad_tether_ends)
    correct_points = stack._src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    stack = stack.define_tether(*correct_points)
    stack = stack.crop_by_pixels(*crop_coordinates)
    check_result(stack, ref_stack, 16)

    # crop -> tether -> tether
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.crop_by_pixels(*crop_coordinates)
    cropped_offset_tether_ends = offset_tether_ends + [[5, -10], [5, 10]]
    stack = stack.define_tether(*cropped_offset_tether_ends)
    correct_points = stack._src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    offset_correct_points = [np.array(pp) - (25, 25) for pp in correct_points]
    stack = stack.define_tether(*offset_correct_points)
    check_result(stack, ref_stack, 16)

    # tether -> crop -> tether
    ref_stack = original_ref_stack.crop_by_pixels(*crop_coordinates)
    stack = original_stack.define_tether(*bad_tether_ends)
    stack = stack.crop_by_pixels(*crop_coordinates)
    correct_points = stack._src._tether.rot_matrix.warp_coordinates([p for p in tether_ends])
    offset_correct_points = [np.array(pp) - (25, 25) for pp in correct_points]
    stack = stack.define_tether(*offset_correct_points)
    check_result(stack, ref_stack, 16)


@pytest.mark.parametrize(
    "scaling, ref_point1, ref_point2",
    [
        ([1.0, 1.0], [1.0, 2.0], [4.0, 5.0]),
        ([2.0, 4.0], [1.0 / 2.0, 2.0 / 4.0], [4.0 / 2.0, 5.0 / 4.0]),
        ([3.0, 1.0], [1.0 / 3.0, 2.0], [4.0 / 3.0, 5.0]),
        (None, [1.0, 2.0], [4.0, 5.0]),
    ],
)
def test_calibrated_tether(monkeypatch, scaling, ref_point1, ref_point2):
    """The actual tether functionality is tested in test_define_tether. This one merely tests
    whether the calibration is correctly taken into account."""
    stack = _create_random_stack((30, 20), {})

    def check_points(_, points):
        point1, point2 = points
        np.testing.assert_allclose(point1, ref_point1)
        np.testing.assert_allclose(point2, ref_point2)
        return stack

    with monkeypatch.context() as m:
        m.setattr("lumicks.pylake.image_stack.TiffStack.with_tether", check_points)
        m.setattr("lumicks.pylake.ImageStack.pixelsize_um", scaling)

        stack.define_tether([1.0, 2.0], [4.0, 5.0])


def test_image_stack_plot_rgb_absolute_color_adjustment(rgb_alignment_image_data):
    """Tests whether we can set an absolute color range for RGB plots."""
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    fig = plt.figure()
    lb = np.array([10000.0, 20000.0, 5000.0])
    ub = np.array([30000.0, 40000.0, 50000.0])
    stack.plot(channel="rgb", frame=0, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), np.clip((warped_image.astype(float) - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_image_stack_plot_channels_absolute_color_adjustment(rgb_alignment_image_data):
    """Tests whether we can set an absolute color range for separate channel plots."""
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    lbs = np.array([10000.0, 20000.0, 5000.0])
    ubs = np.array([30000.0, 40000.0, 50000.0])
    for channel_idx, (lb, ub, channel) in enumerate(zip(lbs, ubs, ("red", "green", "blue"))):
        fig = plt.figure()
        stack.plot(channel=channel, frame=0, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), warped_image[:, :, channel_idx])
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_image_stack_plot_rgb_percentile_color_adjustment(rgb_alignment_image_data):
    """Tests whether we can set a percentile color range for RGB plots."""
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    fig = plt.figure()
    lb, ub = np.array([94, 93, 95]), np.array([95, 98, 97])
    stack.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    raw_image = stack[0].get_image(channel="rgb")
    image = plt.gca().get_images()[0]
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(raw_image, 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    np.testing.assert_allclose(image.get_array(), np.clip((raw_image - lb) / (ub - lb), 0, 1))
    plt.close(fig)


def test_image_stack_plot_single_channel_percentile_color_adjustment(rgb_alignment_image_data):
    """Tests whether we can set a percentile color range for separate channel plots."""
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    lbs, ubs = np.array([94, 93, 95]), np.array([95, 98, 97])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        stack.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), stack[0].get_image(channel=channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        stack.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), stack[0].get_image(channel=channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_single_channel_image_adjustment(gray_alignment_image_data):
    """Tests whether we can set color ranges for single channel images."""
    reference_image, warped_image, description, bit_depth = gray_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 2,
                times=make_frame_times(2),
                description=description,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)

    lbs, ubs = np.array([94, 93, 95]), np.array([95, 98, 97])
    lbs_ref, ubs_ref = [*lbs, 94], [*ubs, 95]
    for lb, ub, channel in zip(lbs_ref, ubs_ref, ("red", "green", "blue", "rgb")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        stack.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), stack[0].get_image(channel=channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color value correctly
        fig = plt.figure()
        stack.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), stack[0].get_image(channel=channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_invalid_slicing():
    n_frames = 3
    data = [np.random.rand(5, 4) for j in range(n_frames)]
    stack = ImageStack.from_dataset(
        TiffStack([MockTiffFile(data, times=make_frame_times(n_frames))], align_requested=False)
    )
    with pytest.raises(
        IndexError, match="Only three indices are accepted when slicing an ImageStack."
    ):
        stack[1:3, :, :, 2]

    with pytest.raises(IndexError, match="Slice steps are not supported when indexing"):
        stack[1:3, ::3, ::3]


@pytest.mark.parametrize(
    "frame_slice,axis1_slice,axis2_slice, dims",
    (
        (slice(1, 3), slice(3, 6), slice(3, 5), (8, 9)),
        # Test single element access
        (slice(1, 3), 3, slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, 6), 3, (8, 9)),
        # Test open ranges
        (slice(None, 3), slice(3, 6), slice(3, 5), (8, 9)),
        (slice(1, None), slice(3, 6), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(None, 6), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, None), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, 6), slice(None, 5), (8, 9)),
        (slice(1, 3), slice(3, 6), slice(3, None), (8, 9)),
        (slice(1, 3), slice(3, 6), slice(3, 5), (8, 9, 3)),
        # Test three color images
        (slice(None, 3), slice(3, 6), slice(3, 5), (8, 9, 3)),
        (slice(1, None), slice(3, 6), slice(3, 5), (8, 9, 3)),
        (slice(1, 3), slice(None, 6), slice(3, 5), (8, 9, 3)),
        (slice(1, 3), slice(3, None), slice(3, 5), (8, 9, 3)),
        (slice(1, 3), slice(3, 6), slice(None, 5), (8, 9, 3)),
        (slice(1, 3), slice(3, 6), slice(3, None), (8, 9, 3)),
        # Test ranges over the end
        (slice(1, 13), slice(3, 6), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, 16), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, 6), slice(3, 15), (8, 9)),
        # Test negative indices
        (slice(1, -1), slice(3, 6), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, -2), slice(3, 5), (8, 9)),
        (slice(1, 3), slice(3, 6), slice(3, -2), (8, 9)),
        # Test negative indices beyond the start
        (slice(-100, 3), slice(-100, 6), slice(3, 15), (8, 9)),
        (slice(-100, -1), slice(-100, 6), slice(3, 5), (8, 9)),
        (slice(-100, -1), slice(3, 6), slice(-100, 5), (8, 9)),
    ),
)
def test_multidim_slicing(frame_slice, axis1_slice, axis2_slice, dims):
    n_frames = 6
    data = [np.random.rand(*dims) for _ in range(n_frames)]
    stack = ImageStack.from_dataset(
        TiffStack([MockTiffFile(data, times=make_frame_times(n_frames))], align_requested=False)
    )

    def validate_img_and_shape(stack, img):
        np.testing.assert_allclose(stack.get_image(), img)

        # Note that when indexing the numpy array, all dimensions with length one get dropped when
        # slicing the raw image.
        stack_shape = np.array(stack.shape)
        np.testing.assert_allclose(stack_shape[stack_shape > 1], img.shape)

    # Frame stack
    validate_img_and_shape(
        stack[frame_slice, axis1_slice, axis2_slice],
        stack[frame_slice].get_image()[:, axis1_slice, axis2_slice],
    )

    # All frames
    validate_img_and_shape(
        stack[:, axis1_slice, axis2_slice],
        stack[:].get_image()[:, axis1_slice, axis2_slice],
    )

    # Only first axis
    validate_img_and_shape(
        stack[frame_slice, axis1_slice],
        stack[frame_slice].get_image()[:, axis1_slice],
    )

    # Only second axis
    validate_img_and_shape(
        stack[frame_slice, :, axis2_slice],
        stack[frame_slice].get_image()[:, :, axis2_slice],
    )


def test_alignment_multistack_failure_modes(
    rgb_alignment_image_data,
    rgb_alignment_image_data_offset,
    gray_alignment_image_data,
):
    """Check whether we enforce that the metadata agrees."""

    def check_error(dataset1, dataset2, error_message):
        with pytest.raises(ValueError, match=re.escape(error_message)):
            TiffStack([to_tiff(*dataset1[1:]), to_tiff(*dataset2[1:])], align_requested=False)

    check_error(
        rgb_alignment_image_data,
        rgb_alignment_image_data_offset,
        "Alignment matrices must be the same for stacks to be merged. The alignment matrix for "
        "channel 0 is different.",
    )
    check_error(
        rgb_alignment_image_data, gray_alignment_image_data, "Cannot mix RGB and non-RGB stacks"
    )

    # Drop the metadata entirely
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    rgb_alignment_image_data_no_metadata = reference_image, warped_image, "", bit_depth

    check_error(
        rgb_alignment_image_data,
        rgb_alignment_image_data_no_metadata,
        "Alignment matrices must be the same for stacks to be merged. The following alignment "
        "matrices were found in one stack but not the other {0, 1, 2}.",
    )


def test_time_ordering_stack(rgb_alignment_image_data):
    """Check whether we enforce a correct time order when setting up a stack from multiple files."""

    timestamps = range(0, 40, 10)
    t1, t2, t3, t4 = (to_tiff(*rgb_alignment_image_data[1:], t, 1) for t in timestamps)
    stack = TiffStack([t3, t2, t1, t4], align_requested=True)

    for idx, (frame, ts) in enumerate(zip((t1, t2, t3, t4), timestamps)):
        assert stack.get_frame(idx).start == ts


def test_malformed_timestamps_image_stack(rgb_alignment_image_data):
    """We need good timestamps for image stack functionalities, so we should validate that
    we have those"""

    class MockProperty:
        @property
        def value(self):
            return "okay"

    t1, t2, t3 = (to_tiff(*rgb_alignment_image_data[1:], t) for t in range(0, 30, 10))
    t2.pages[0].tags.pop("DateTime")
    t3.pages[0].tags["DateTime"] = MockProperty()

    for test_frame in (t2, t3):
        with pytest.raises(RuntimeError, match="The timestamp data was incorrectly formatted"):
            TiffStack([t1, test_frame], align_requested=True)


def test_alignment_multistack(rgb_alignment_image_data, gray_alignment_image_data):
    t_starts = np.arange(10) * 100
    files = [
        to_tiff(*rgb_alignment_image_data[1:], start_time=t_start, num_images=5)
        for t_start in t_starts
    ]
    full_stack = TiffStack(files, align_requested=True)

    ref_ts = np.hstack([start + np.arange(0, 50, 10) for start in np.arange(10) * 100])
    for idx, timestamp in enumerate(ref_ts):
        assert full_stack.get_frame(idx).start == timestamp

    full_stack_idx = 0
    for file in files:
        current_stack = TiffStack([file], align_requested=True)
        for idx in range(current_stack.num_frames):
            np.testing.assert_allclose(
                full_stack.get_frame(full_stack_idx).data, current_stack.get_frame(idx).data
            )
            full_stack_idx += 1

    assert full_stack.num_frames == len(ref_ts)


@pytest.mark.parametrize("steps", [2, 4])
def test_frame_timestamp_ranges_include_true(steps):
    stack = ImageStack.from_dataset(
        TiffStack(
            [MockTiffFile(data=[np.ones((5, 5))] * steps, times=make_frame_times(steps, step=6))],
            align_requested=False,
        )
    )

    for include, ranges in zip(
        [True, False],
        [[(10, 20), (20, 30), (30, 40), (40, 50)], [(10, 16), (20, 26), (30, 36), (40, 46)]],
    ):
        np.testing.assert_allclose(
            stack.frame_timestamp_ranges(include_dead_time=include), ranges[:steps]
        )


def test_frame_timestamp_ranges_snapshot():
    stack = ImageStack.from_dataset(
        TiffStack(
            [MockTiffFile(data=[np.ones((5, 5))] * 1, times=make_frame_times(1, step=6))],
            align_requested=False,
        )
    )

    for include, ranges in zip([True, False], [[(10, 20)], [(10, 16)]]):
        np.testing.assert_allclose(stack.frame_timestamp_ranges(include_dead_time=include), ranges)


@pytest.mark.parametrize(
    "num_frames, dims, ref_pixelsize_um, ref_size_um",
    [
        (3, (10, 5, 3), 0.1, (0.5, 1.0)),
        (3, (10, 5), 0.1, (0.5, 1.0)),
        (1, (10, 5), 0.2, (1.0, 2.0)),
        (1, (10, 5), None, (1.0, 2.0)),
    ],
)
def test_pixel_calibration(num_frames, dims, ref_pixelsize_um, ref_size_um):
    description = (
        {"Pixel calibration (nm/pix)": 1000 * ref_pixelsize_um} if ref_pixelsize_um else {}
    )
    fake_tiff = TiffStack(
        [to_tiff(np.zeros(dims), description, 16, start_time=1, num_images=num_frames)],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    if ref_pixelsize_um:
        np.testing.assert_allclose(stack.pixelsize_um, [ref_pixelsize_um] * 2)
        np.testing.assert_allclose(stack.size_um, ref_size_um)
        np.testing.assert_allclose(
            stack[:, :3, :4].size_um, [ref_pixelsize_um * 4, ref_pixelsize_um * 3]
        )
        np.testing.assert_allclose(stack._pixel_calibration_factors, stack.pixelsize_um)
    else:
        assert stack.size_um is None
        assert stack.pixelsize_um is None
        np.testing.assert_allclose(stack._pixel_calibration_factors, [1.0, 1.0])


def _create_random_stack(img_shape, description):
    return ImageStack.from_dataset(
        TiffStack(
            [to_tiff(np.random.rand(*img_shape), description, 16, start_time=1, num_images=2)],
            align_requested=False,
        )
    )


def test_pixel_calibration():
    # Test calibrated
    stack = _create_random_stack((3, 5), {"Pixel calibration (nm/pix)": 500})
    image = stack.plot()
    np.testing.assert_allclose(image.get_extent(), [-0.5, 0.5 * 5 - 0.5, 0.5 * 3 - 0.5, -0.5])

    # Test uncalibrated
    stack = _create_random_stack((3, 5), {})
    image = stack.plot()
    np.testing.assert_allclose(image.get_extent(), [-0.5, 4.5, 2.5, -0.5])


@pytest.mark.parametrize(
    "x_min, x_max, y_min, y_max, scale_x, scale_y",
    [
        (2, 4, 1, 5, 1, 1),  # No scaling
        (2, 4, 1, 5, 2, 4),
        (None, 4, 1, 5, 2, 4),
        (2, None, 1, 5, 2, 4),
        (2, 4, None, 3, 2, 4),
        (2, 4, 1, None, 2, 4),
        (None, None, None, None, 2, 4),
    ],
)
def test_calibrated_crop(monkeypatch, x_min, x_max, y_min, y_max, scale_x, scale_y):
    def scale(x, scale_factor):
        return x * scale_factor if x else None

    stack = _create_random_stack((30, 20), {})
    with monkeypatch.context() as m:
        m.setattr("lumicks.pylake.ImageStack.pixelsize_um", [scale_x, scale_y])
        ref_img = stack.get_image("red")[:, y_min:y_max, x_min:x_max]
        np.testing.assert_allclose(
            stack._crop(
                scale(x_min, scale_x),
                scale(x_max, scale_x),
                scale(y_min, scale_y),
                scale(y_max, scale_y),
            ).get_image("red"),
            ref_img,
        )


@pytest.mark.parametrize(
    "position, half_width, num_images, tether_start, half_window, pixel_size",
    [
        (9, 1, 10, 0, 0, None),
        (6, 1, 15, 2, 0, None),  # Validate line not all the way to edge
        (6, 1, 15, 1, 2, None),  # Validate summing wider area
        (6, 1, 15, 1, 2, 2),  # Validate calibration
    ],
)
def test_integration_test_to_kymo(
    position, half_width, num_images, tether_start, half_window, pixel_size
):
    """The individual methods for converting to a Kymo and converting from a stack to a Kymo have
    already been individually tested. This function specifically tests whether the API works end
    to end."""
    img = np.ones((20, 30))
    img[7:14, position - half_width : position + half_width + 1] = 5
    description = {"Pixel calibration (nm/pix)": 1000 * pixel_size} if pixel_size else {}
    stack = ImageStack.from_dataset(
        TiffStack(
            [to_tiff(img, description, 16, start_time=1, num_images=num_images)],
            align_requested=False,
        ),
    )

    pixelsize = pixel_size if pixel_size else 1
    tether_start, half_window = 2, 1
    with_tether = stack.define_tether(
        (tether_start * pixelsize, 10 * pixelsize), (26 * pixelsize, 10 * pixelsize)
    )
    kymo = with_tether.to_kymo(half_window=half_window, reduce=np.sum)
    lines = track_greedy(kymo, "red", pixel_threshold=4)
    np.testing.assert_allclose(
        lines[0].position, [(position - tether_start) * pixelsize] * num_images
    )
    np.testing.assert_allclose(np.max(kymo.get_image("red")), 5 * (1 + 2 * half_window))
    np.testing.assert_equal(kymo.pixelsize_um[0], pixel_size)


@pytest.mark.filterwarnings(
    r"ignore:File does not contain alignment matrices. Only raw data is available"
)
@pytest.mark.parametrize("include_dead_time", [True, False])
def test_legacy_exposure_handling(tmpdir_factory, reference_data, include_dead_time):
    """For confocal Scans exported with Pylake `<v1.3.2` timestamps contained the start and end time
    of the exposure, rather than the full frame. This behaviour was inconsistent with Bluelake and
    therefore changed. This test makes sure that such old files produce correct frame times in
    Pylake."""
    im = ImageStack(Path(__file__).parent / "data/tiff_from_scan_v1_3_1.tiff")
    assert im._src._description._legacy_exposure is True
    ts_ranges = im.frame_timestamp_ranges(include_dead_time=include_dead_time)
    np.testing.assert_equal(ts_ranges, reference_data(ts_ranges, test_name="dead_time"))

    # Save it again
    tmpdir = tmpdir_factory.mktemp("legacy_exposures")
    tmp_file = tmpdir.join(f"include_dead_time={include_dead_time}.tiff")
    im.export_tiff(tmp_file)
    im.close()

    # Test whether the round trip results in valid results
    im2 = ImageStack(tmp_file)
    ts_ranges2 = im2.frame_timestamp_ranges(include_dead_time=include_dead_time)
    np.testing.assert_equal(ts_ranges2, ts_ranges)

    # Verify that this migrated the file, and we are no longer using legacy reading for this
    assert im2._src._description._legacy_exposure is False
    im2.close()


def test_tiffstack_automatic_cleanup(gray_tiff_file_multi):
    im = TiffStack.from_file(gray_tiff_file_multi, False)
    weakref_file = weakref.ref(im._tiff_files[0])
    handle = im._tiff_files[0]._src.filehandle
    assert not handle.closed
    del im
    assert handle.closed
    assert not weakref_file()

    im = TiffStack.from_file(gray_tiff_file_multi, False)
    handle = im._tiff_files[0]._src.filehandle
    im2 = im.with_roi((1, 3, 1, 3))
    weakref_file = weakref.ref(im._tiff_files[0])
    assert not handle.closed
    del im
    assert weakref_file()  # im2 should keep it alive
    assert not handle.closed
    del im2
    assert handle.closed
    assert not weakref_file()

    im = TiffStack.from_file(gray_tiff_file_multi, False)
    handle = im._tiff_files[0]._src.filehandle
    assert not handle.closed
    im.close()
    assert handle.closed
    with pytest.raises(
        IOError,
        match=r"The file handle for this TiffStack \(gray_multi.tiff\) has already been closed.",
    ):
        im.get_frame(0)


def test_imagestack_explicit_close(gray_tiff_file_multi):
    im = ImageStack(gray_tiff_file_multi)
    handle = im._src._tiff_files[0]._src.filehandle
    derived_im = im.crop_by_pixels(1, 3, 1, 3)
    assert not handle.closed
    im.close()
    assert handle.closed

    for current_stack in (im, derived_im):
        with pytest.raises(
            IOError,
            match=r"The file handle for this TiffStack \(gray_multi.tiff\) has already been closed.",
        ):
            current_stack.get_image("rgb")


def test_two_color(gb_tiff_file_single, gb_tiff_file_multi, bg_tiff_file_single):
    for filename, reference_image in (gb_tiff_file_single, gb_tiff_file_multi):
        im = ImageStack(filename)

        normalization = 1.0 / np.max(np.abs(reference_image))
        np.testing.assert_allclose(
            im.get_image() * normalization,
            reference_image * normalization,
            atol=0.05,
        )
        im.close()


def test_invalid_order(bg_tiff_file_single):
    with pytest.raises(
        RuntimeError, match=re.escape("Wavelengths are not in descending order [525.0, 600.0]")
    ):
        ImageStack(bg_tiff_file_single[0])


@pytest.mark.parametrize("align_export, align_load", [(False, False), (True, True), (False, True)])
def test_two_color_write_again(
    tmpdir_factory,
    gb_tiff_file_single,
    gb_tiff_file_multi,
    bg_tiff_file_single,
    align_export,
    align_load,
):
    tmp_dir = tmpdir_factory.mktemp("two_color")
    for filename, reference_image in (gb_tiff_file_single, gb_tiff_file_multi):
        im = ImageStack(filename, align=align_export)
        tmp_file = tmp_dir.join(f"export_{filename.basename}")
        im.export_tiff(tmp_file)

        # The tiff file we imported data was 2 channels.
        assert im._src._tiff_files[0].pages[0].shape[-1] == 2
        im.close()

        # Reload source since we want to be able to test whether we can successfully align
        im_read = ImageStack(tmp_file, align=align_load)
        im = ImageStack(filename, align=align_load)
        np.testing.assert_allclose(im_read.get_image(), im.get_image())

        # Pylake always exports 3 channels, mapped to the correct RGB colors.
        assert im_read._src._tiff_files[0].pages[0].shape[-1] == 3
        assert im_read._src._description.json["Pylake"]["Channel mapping"] == {
            "green": 0,
            "blue": 1,
        }

        im.close()
        im_read.close()
