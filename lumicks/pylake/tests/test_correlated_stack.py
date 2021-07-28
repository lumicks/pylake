import numpy as np
import json
import tifffile
from copy import deepcopy
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
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)

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
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    np.testing.assert_allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]), np.arange(30, 50, 2))

    # Test image stack with dead time
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                       times=[["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)

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


@cleanup
def test_plot_correlated():
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2), {"y": "mock", "title": "mock"})

    # Regression test for a bug where the start index was added twice. In the regression, this lead to an out of range
    # error.
    fake_tiff = TiffStack(MockTiffFile(data=[np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 3))*2,
                                             np.ones((3, 3))*3, np.ones((3, 3))*4, np.ones((3, 3))*5],
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"],
                                              ["60", "70"]]),
                                       align=True)

    CorrelatedStack.from_data(fake_tiff)[3:5].plot_correlated(cc)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((3, 3)) * 3)

    CorrelatedStack.from_data(fake_tiff)[3:5].plot_correlated(cc, frame=1)
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
                                       align=True)

    # Add test for when there's only a subset in terms of channel data
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 30, 2), {"y": "mock", "title": "mock"})

    with pytest.warns(UserWarning):
        CorrelatedStack.from_data(fake_tiff).plot_correlated(cc)

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
                                     bit_depth=8, **warp_parameters)


@pytest.fixture(scope="session", params=[1, 2])
def rgb_alignment_image_data(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(spot_coordinates, version=request.param,
                                     bit_depth=16, **warp_parameters)


@pytest.fixture(scope="session", params=[1,2])
def rgb_alignment_image_data_offset(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(spot_coordinates, version=request.param,
                                     bit_depth=16, offsets=(50, 50), **warp_parameters)


@pytest.fixture(scope="session")
def rgb_tiff_file(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_single.tiff")
    write_tiff_file(*rgb_alignment_image_data, n_frames=1, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def rgb_tiff_file_multi(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_multi.tiff")
    write_tiff_file(*rgb_alignment_image_data, n_frames=2, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def gray_tiff_file(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_single.tiff")
    write_tiff_file(*gray_alignment_image_data, n_frames=1, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="session")
def gray_tiff_file_multi(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_multi.tiff")
    write_tiff_file(*gray_alignment_image_data, n_frames=2, filename=str(mock_filename))
    return mock_filename


def test_image_reconstruction_grayscale(gray_alignment_image_data):
    reference_image, warped_image, description, bit_depth = gray_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image[:, :, 0]], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=8),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    assert not fr.is_rgb
    assert np.all(fr.data == fr.raw_data)
    np.testing.assert_allclose(fr.raw_data, fr._get_plot_data())


def test_image_reconstruction_rgb(rgb_alignment_image_data, rgb_alignment_image_data_offset):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
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
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    assert not np.allclose(original_data, fr._get_plot_data(), atol=0.05)

    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data_offset
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_multiframe(rgb_alignment_image_data):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image]*6,
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(2)

    assert fr.is_rgb
    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_missing_metadata(rgb_alignment_image_data):
    # no metadata
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description="", bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    with pytest.warns(UserWarning, match="File does not contain metadata. Only raw data is available"):
        fr.data

    # missing alignment matrices
    for label in ("Alignment red channel", "Channel 0 alignment"):
        if label in description:
            removed = description.pop(label)
            break
    fake_tiff = TiffStack(MockTiffFile(data=[warped_image], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    with pytest.warns(UserWarning) as record:
        fr.data
    assert len(record) == 1
    assert record[0].message.args[0] == "File does not contain alignment matrices. Only raw data is available"
    description[label] = removed # reset fixture


def test_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        stack.export_tiff(savename)
        stack.src._src.close()
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(str(filename)) as tif0, tifffile.TiffFile(savename) as tif:
            assert len(tif0.pages) == len(tif.pages)
            assert tif0.pages[0].software != tif.pages[0].software
            assert "pylake" in tif.pages[0].software
            assert "Applied channel 0 alignment" in tif.pages[0].description
            assert "Channel 0 alignment" not in tif.pages[0].description
            for page0, page in zip(tif0.pages, tif.pages):
                assert page0.tags["DateTime"].value == page.tags["DateTime"].value


def test_export_roi(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        stack.export_tiff(savename, roi=[10, 190, 20, 80])
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60

        with pytest.raises(ValueError):
            stack.export_tiff(savename, roi=[-10, 190, 20, 80])

        with pytest.raises(ValueError):
            stack.export_tiff(savename, roi=[190, 10, 20, 80])
        stack.src._src.close()
