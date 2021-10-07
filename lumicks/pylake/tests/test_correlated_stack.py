import numpy as np
import json
import tifffile
import itertools
from copy import deepcopy
from pathlib import Path
import pytest
from lumicks.pylake.correlated_stack import CorrelatedStack
from lumicks.pylake.detail.widefield import TiffStack
from lumicks.pylake import channel
import matplotlib as mpl
from matplotlib.testing.decorators import cleanup
from .data.mock_widefield import MockTiffFile
from .data.mock_widefield import make_alignment_image_data, write_tiff_file


@pytest.mark.parametrize("shape", [(3,3), (5,4,3)])
def test_correlated_stack(shape):
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                    times=[["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]),
                        align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)

    assert (stack[0].start == 10)
    assert (stack[1].start == 20)
    assert (stack[-1].start == 60)
    assert (stack[0].num_frames == 1)

    assert (stack[0].stop == 18)
    assert (stack[-1].stop == 68)

    assert (stack[1:2].stop == 28)
    assert (stack[1:3].stop == 38)
    assert (stack[1:2].num_frames == 1)
    assert (stack[1:3].num_frames == 2)

    assert (stack[3:5][0].start == 40)
    assert (stack[3:5][1].start == 50)
    assert (stack[3:5][0].num_frames == 1)

    with pytest.raises(IndexError):
        stack[3:5][2]

    assert(stack[2:5][3:5].num_frames == 0)
    assert(stack[2:5][1:2].start == 40)
    assert(stack[2:5][1:3]._get_frame(1).start == 50)

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


@pytest.mark.parametrize("shape", [(3,3), (5,4,3)])
def test_correlation(shape):
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))

    # Test image stack without dead time
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]]),
                          align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    np.testing.assert_allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]), np.arange(30, 50, 2))

    # Test image stack with dead time
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                       times=[["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]),
                          align_requested=False)
    stack = CorrelatedStack.from_dataset(fake_tiff)

    np.testing.assert_allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]),
                        np.hstack([np.arange(30, 38, 2), np.arange(40, 48, 2)]))

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a stack.
    ch = cc.downsampled_over(stack[0:3].timestamps)
    np.testing.assert_allclose(ch.data, [np.mean(np.arange(10, 18, 2)), np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2))])
    np.testing.assert_allclose(ch.timestamps, [(10 + 16) / 2, (20 + 26) / 2, (30 + 36) / 2])

    ch = cc.downsampled_over(stack[1:4].timestamps)
    np.testing.assert_allclose(ch.data, [np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2)), np.mean(np.arange(40, 48, 2))])
    np.testing.assert_allclose(ch.timestamps, [(20 + 26) / 2, (30 + 36) / 2, (40 + 46) / 2])

    with pytest.raises(TypeError):
        cc.downsampled_over(stack[1:4])

    with pytest.raises(ValueError):
        cc.downsampled_over(stack[1:4].timestamps, where='up')

    with pytest.raises(AssertionError):
        cc["0ns":"20ns"].downsampled_over(stack[3:4].timestamps)

    with pytest.raises(AssertionError):
        cc["40ns":"70ns"].downsampled_over(stack[0:1].timestamps)

    assert (stack[0].raw.start == 10)
    assert (stack[1].raw.start == 20)
    assert (stack[1:3][0].raw.start == 20)
    assert (stack[1:3].raw[0].start == 20)
    assert (stack[1:3].raw[1].start == 30)

    # Regression test downsampled_over losing precision due to reverting to double rather than int64.
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 1588267266006287100, 1000))
    ch = cc.downsampled_over([(1588267266006287100, 1588267266006287120)], where='left')
    assert int(ch.timestamps[0]) == 1588267266006287100


def test_name_change_from_data():
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones((5, 4, 3))], times=[["10", "18"]]), align_requested=False)
    with pytest.deprecated_call():
        CorrelatedStack.from_data(fake_tiff)
        
        
def test_stack_roi():
    first_page = np.arange(60).reshape((6,10))
    data = np.stack([first_page + (j*60) for j in range(3)], axis=2)
    stack_0 = TiffStack(MockTiffFile([data], times=[["10", "20"]]), align_requested=False)

    # recursive cropping
    stack_1 = stack_0.with_roi([1, 7, 3, 6])
    np.testing.assert_equal(
        stack_1.get_frame(0).data,
        data[3:6, 1:7, :]
    )

    stack_2 = stack_1.with_roi([3, 6, 0, 3])
    np.testing.assert_equal(
        stack_2.get_frame(0).data,
        data[3:6, 4:7, :]
    )

    stack_3 = stack_2.with_roi([1, 2, 1, 2])
    np.testing.assert_equal(
        stack_3.get_frame(0).data,
        data[4:5, 5:6, :]
    )

    # negative indices
    with pytest.raises(ValueError):
        stack_4 = stack_0.with_roi([-5, 4, 1, 2])

    # out of bounds
    with pytest.raises(ValueError):
        stack_5 = stack_0.with_roi([0, 11, 1, 2])


@cleanup
def test_plot_correlated():
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2), {"y": "mock", "title": "mock"})

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(MockTiffFile(data=[np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 3))*2,
                                             np.ones((3, 3))*3, np.ones((3, 3))*4, np.ones((3, 3))*5],
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"],
                                              ["60", "70"]]),
                                       align_requested=False)

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
    fake_tiff = TiffStack(MockTiffFile(data=[np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 3))*2,
                                             np.ones((3, 3))*3, np.ones((3, 3))*4, np.ones((3, 3))*5],
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"],
                                              ["60", "70"]]),
                                       align_requested=False)

    # Add test for when there's only a subset in terms of channel data
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 30, 2), {"y": "mock", "title": "mock"})

    with pytest.warns(UserWarning):
        CorrelatedStack.from_dataset(fake_tiff).plot_correlated(cc)

    def mock_click(fig, data_position):
        pos = fig.axes[0].transData.transform(data_position)
        fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, pos[0], pos[1], 1))
        images = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
        assert len(images) == 1
        return images[0].get_array()

    np.testing.assert_allclose(mock_click(mpl.pyplot.gcf(), np.array([0, 40])), np.ones((3, 3)) * 2)
    np.testing.assert_allclose(mock_click(mpl.pyplot.gcf(), np.array([10.1e-9, 40])), np.ones((3, 3)) * 3)


@pytest.fixture(scope="session")
def tiff_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tiffs")


@pytest.fixture(scope="session")
def spot_coordinates():
    return [[20, 30], [50, 50], [120, 30], [50, 70], [150, 60]]


@pytest.fixture(scope="session")
def warp_parameters():
    return {"red_warp_parameters": {"Tx": 20, "Ty": 10, "theta": 3},
            "blue_warp_parameters": {"Tx": 10, "Ty": 20, "theta": -3}
    }


@pytest.fixture(scope="session")
def gray_alignment_image_data(spot_coordinates, warp_parameters):
    return make_alignment_image_data(spot_coordinates, version=1,
                                     bit_depth=8, camera="irm", **warp_parameters)


@pytest.fixture(scope="session", params=[1, 2])
def rgb_alignment_image_data(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(spot_coordinates, version=request.param,
                                     bit_depth=16, camera="wt", **warp_parameters)


@pytest.fixture(scope="session", params=[1,2])
def rgb_alignment_image_data_offset(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(spot_coordinates, version=request.param,
                                     bit_depth=16, offsets=(50, 50), camera="wt", **warp_parameters)


@pytest.fixture(scope="session")
def rgb_tiff_file(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_single.tiff")
    write_tiff_file(rgb_alignment_image_data, n_frames=1, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def rgb_tiff_file_multi(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_multi.tiff")
    write_tiff_file(rgb_alignment_image_data, n_frames=2, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def gray_tiff_file(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_single.tiff")
    write_tiff_file(gray_alignment_image_data, n_frames=1, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def gray_tiff_file_multi(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_multi.tiff")
    write_tiff_file(gray_alignment_image_data, n_frames=2, filename=str(mock_filename))
    return mock_filename


def test_image_reconstruction_grayscale(gray_alignment_image_data):
    reference_image, warped_image, description, bit_depth = gray_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=8),
                          align_requested=True)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert not fr.is_rgb
    assert np.all(fr.data == fr.raw_data)
    np.testing.assert_allclose(fr.raw_data, fr._get_plot_data())


def test_image_reconstruction_rgb(rgb_alignment_image_data, rgb_alignment_image_data_offset):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align_requested=True)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    max_signal = np.max(np.hstack([fr._get_plot_data("green"), fr._get_plot_data("red")]))
    diff = np.abs(fr._get_plot_data('green').astype(float)-fr._get_plot_data("red").astype(float))
    assert np.all(diff/max_signal < 0.05)
    max_signal = np.max(np.hstack([fr._get_plot_data("green"), fr._get_plot_data("blue")]))
    diff = np.abs(fr._get_plot_data('green').astype(float)-fr._get_plot_data("blue").astype(float))
    assert np.all(diff/max_signal < 0.05)

    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)
    np.testing.assert_allclose(original_data / 0.5, fr._get_plot_data(vmax=0.5), atol=0.10)
    max_signal = np.max(np.hstack([reference_image[:, :, 0], fr._get_plot_data("red")]))
    diff = np.abs(reference_image[:, :, 0].astype(float)-fr._get_plot_data("red").astype(float))
    assert np.all(diff/max_signal < 0.05)

    with pytest.raises(ValueError):
        fr._get_plot_data(channel="purple")

    # test that bad alignment matrix gives high error compared to correct matrix
    bad_description = deepcopy(description)
    label = "Alignment red channel" if "Alignment red channel" in description.keys() else "Channel 0 alignment"
    bad_description[label][2] = 25
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(bad_description), bit_depth=16),
                          align_requested=True)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    assert not np.allclose(original_data, fr._get_plot_data(), atol=0.05)

    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data_offset
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align_requested=True)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_multiframe(rgb_alignment_image_data):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image]*6,
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]],
                                       description=json.dumps(description), bit_depth=16),
                          align_requested=True)
    stack = CorrelatedStack.from_dataset(fake_tiff)
    fr = stack._get_frame(2)

    assert fr.is_rgb
    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_missing_metadata(rgb_alignment_image_data):
    # no metadata
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    with pytest.warns(UserWarning, match="File does not contain metadata. Only raw data is available"):
        fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                        description="", bit_depth=16),
                            align_requested=True)

    # missing alignment matrices
    for label in ("Alignment red channel", "Channel 0 alignment"):
        if label in description:
            removed = description.pop(label)
            break
    with pytest.warns(UserWarning, match="File does not contain alignment matrices. Only raw data is available"):
        fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                        description=json.dumps(description), bit_depth=16),
                            align_requested=True)

    description[label] = removed # reset fixture


def test_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    filenames = (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi)
    for filename, align in itertools.product(filenames, (True, False)):
        savename = str(filename.new(purebasename=f"out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename), align)
        stack.export_tiff(savename)
        stack.src._tiff_file.close()
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(str(filename)) as tif0, tifffile.TiffFile(savename) as tif:
            assert len(tif0.pages) == len(tif.pages)
            assert tif0.pages[0].software != tif.pages[0].software
            assert "pylake" in tif.pages[0].software
            if stack._get_frame(0).is_rgb:
                if stack._get_frame(0)._is_aligned:
                    assert "Applied channel 0 alignment" in tif.pages[0].description
                    assert "Channel 0 alignment" not in tif.pages[0].description
                else:
                    assert "Applied channel 0 alignment" not in tif.pages[0].description
                    assert "Channel 0 alignment" in tif.pages[0].description
            for page0, page in zip(tif0.pages, tif.pages):
                assert page0.tags["DateTime"].value == page.tags["DateTime"].value


def test_export_roi(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        with pytest.warns(DeprecationWarning):
            stack.export_tiff(savename, roi=[10, 190, 20, 80])
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60

        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                stack.export_tiff(savename, roi=[-10, 190, 20, 80])

        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                stack.export_tiff(savename, roi=[190, 10, 20, 80])
        stack.src._tiff_file.close()


def test_cropping(rgb_tiff_file, gray_tiff_file):
    for filename in (rgb_tiff_file, gray_tiff_file):
        for align in (True, False):
            stack = CorrelatedStack(filename, align=True)
            cropped = stack.crop_by_pixels(25, 50, 25, 50)
            np.testing.assert_allclose(
                cropped._get_frame(0).data,
                stack._get_frame(0).data[25:50, 25:50],
                err_msg=f"failed on {Path(filename).name}, align={align}, frame.data"
            )
            np.testing.assert_allclose(
                cropped._get_frame(0).raw_data,
                stack._get_frame(0).raw_data[25:50, 25:50],
                err_msg=f"failed on {Path(filename).name}, align={align}, frame.raw_data"
            )
            stack.src._tiff_file.close()


def test_cropping_then_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        stack =stack.crop_by_pixels(10, 190, 20, 80)

        stack.export_tiff(savename)
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60
        stack.src._tiff_file.close()
