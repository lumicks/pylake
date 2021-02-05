import numpy as np
import json
import tifffile
import pytest
from lumicks.pylake.correlated_stack import CorrelatedStack, TiffStack
from lumicks.pylake import channel
import matplotlib as mpl
from matplotlib.testing.decorators import cleanup


# Mock Camera TIFF file
class MockTag():
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class MockTiffPage:
    def __init__(self, data, start_time, end_time, description="", bit_depth=8):
        self._data = data
        bit_depth = bit_depth if data.ndim == 2 else (bit_depth, bit_depth, bit_depth)
        self.tags = {"DateTime": MockTag(f"{start_time}:{end_time}"),
                     "ImageDescription": MockTag(description),
                     "BitsPerSample": MockTag(bit_depth),
                     "SamplesPerPixel": MockTag(1 if (data.ndim==2) else data.shape[2])}

    def asarray(self):
        return self._data.copy()

    @property
    def description(self):
        return self.tags["ImageDescription"].value


class MockTiffFile:
    def __init__(self, data, times, description="", bit_depth=8):
        self.pages = []
        for d, r in zip(data, times):
            self.pages.append(MockTiffPage(d, r[0], r[1], description=description, bit_depth=bit_depth))

    @property
    def num_frames(self):
        return len(self._src.pages)


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
    assert (np.allclose(stack[2:5].start, 30))
    assert (np.allclose(stack[2:5].stop, 58))

    # Test iterations
    assert (np.allclose([x.start for x in stack], [10, 20, 30, 40, 50, 60]))
    assert (np.allclose([x.start for x in stack[1:]], [20, 30, 40, 50, 60]))
    assert (np.allclose([x.start for x in stack[:-1]], [10, 20, 30, 40, 50]))
    assert (np.allclose([x.start for x in stack[2:4]], [30, 40]))
    assert (np.allclose([x.start for x in stack[2]], [30]))


@pytest.mark.parametrize("shape", [(3,3), (5,4,3)])
def test_correlation(shape):
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))

    # Test image stack without dead time
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]]),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    assert (np.allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]), np.arange(30, 50, 2)))

    # Test image stack with dead time
    fake_tiff = TiffStack(MockTiffFile(data=[np.ones(shape)]*6,
                                       times=[["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)

    assert (np.allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]),
                        np.hstack([np.arange(30, 38, 2), np.arange(40, 48, 2)])))

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a stack.
    ch = cc.downsampled_over(stack[0:3].timestamps)
    assert(np.allclose(ch.data, [np.mean(np.arange(10, 18, 2)), np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2))]))
    assert (np.allclose(ch.timestamps, [(10 + 16) / 2, (20 + 26) / 2, (30 + 36) / 2]))

    ch = cc.downsampled_over(stack[1:4].timestamps)
    assert (np.allclose(ch.data, [np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2)), np.mean(np.arange(40, 48, 2))]))
    assert (np.allclose(ch.timestamps, [(20 + 26) / 2, (30 + 36) / 2, (40 + 46) / 2]))

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
    assert np.allclose(imgs[0].get_array(), np.ones((3, 3)) * 3)

    CorrelatedStack.from_data(fake_tiff)[3:5].plot_correlated(cc, frame=1)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    assert np.allclose(imgs[0].get_array(), np.ones((3, 3)) * 4)


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

    assert np.allclose(mock_click(mpl.pyplot.gcf(), np.array([0, 40])), np.ones((3, 3)) * 2)
    assert np.allclose(mock_click(mpl.pyplot.gcf(), np.array([10.1e-9, 40])), np.ones((3, 3)) * 3)


def make_alignment_image_data(spots, Tx_red, Ty_red, theta_red, Tx_blue, Ty_blue, theta_blue, bit_depth,
                              offsets=None):
    def make_transform_matrix(Tx, Ty, theta):
        M = np.eye(3)
        M[0, -1] = Tx
        M[1, -1] = Ty
        theta = np.radians(theta)
        M[0, 0] = np.cos(theta)
        M[0, 1] = -np.sin(theta)
        M[1, 0] = np.sin(theta)
        M[1, 1] = np.cos(theta)
        return M

    def transform_spots(M, spots):
        # apply channel offsets
        # reshape spots into coordinate matrix; [x,y,z] as columns
        N = spots.shape[1]
        spots = np.vstack((spots, np.ones(N)))
        return np.dot(M, spots)[:2]

    def make_image(spots_red, spots_green, spots_blue, bit_depth):
        # RGB image, 2D (normalized) gaussians at spot locations
        sigma = np.eye(2)*5
        X, Y = np.meshgrid(np.arange(0, 200), np.arange(0, 100))
        img = np.zeros((*X.shape, 3))
        for j, pts in enumerate((spots_red.T, spots_green.T, spots_blue.T)):
            for x, y, in pts:
                mu = np.array([x,y])[:,np.newaxis]
                XX = np.vstack((X.ravel(), Y.ravel())) - mu
                quad_form = np.sum(np.dot(XX.T, np.linalg.inv(sigma)) * XX.T, axis=1)
                Z = np.exp(-0.5 * quad_form)
                img[:, :, j] += Z.reshape(X.shape)
            img[:, :, j] = img[:, :, j] / img[:, :, j].max()
        return (img * (2**bit_depth - 1)).astype(f"uint{bit_depth}")

    def make_description(m_red, m_blue, offsets):
        if offsets is None:
            offsets = [0, 0]
        # WARP_INVERSE_MAP flag requires original transformation that resulted in un-aligned image
        return {"Alignment red channel": m_red[:2].ravel().tolist(),
                "Alignment green channel": np.eye(3)[:2].ravel().tolist(),
                "Alignment blue channel": m_blue[:2].ravel().tolist(),
                "Alignment region of interest (x, y, width, height)": [offsets[0], offsets[1], 200, 100],
                "Region of interest (x, y, width, height)": [0, 0, 200, 100]}

    spots = np.array(spots).T # [2 x N]
    img0 = make_image(spots, spots, spots, bit_depth)

    if offsets is not None: # translate origin by offsets
        spots[0] -= offsets[0]
        spots[1] -= offsets[1]
    m_red = make_transform_matrix(Tx_red, Ty_red, theta_red)
    m_blue = make_transform_matrix(Tx_blue, Ty_blue, theta_blue)
    red_spots = transform_spots(m_red, spots)
    blue_spots = transform_spots(m_blue, spots)
    if offsets is not None: # back-translate origin
        for tmp_spots in (red_spots, spots, blue_spots):
            tmp_spots[0] += offsets[0]
            tmp_spots[1] += offsets[1]

    img = make_image(red_spots, spots, blue_spots, bit_depth)
    description = make_description(m_red, m_blue, offsets)

    return img0, img, description


@pytest.fixture(scope="session")
def tiff_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tiffs")


@pytest.fixture(scope="session")
def spot_coordinates():
    return [[20, 30], [50, 50], [120, 30], [50, 70], [150, 60]]


@pytest.fixture(scope="session")
def warp_parameters():
    return {"Tx_red": 20, "Ty_red": 10, "theta_red": 3,
            "Tx_blue": 10, "Ty_blue": 20, "theta_blue": -3}


@pytest.fixture(scope="session")
def gray_alignment_image_data(spot_coordinates, warp_parameters):
    img_args = {"spots": spot_coordinates,
                **warp_parameters,
                "bit_depth": 8}
    img0, img, description = make_alignment_image_data(**img_args)
    return img0, img, description, img_args["bit_depth"]


@pytest.fixture(scope="session")
def rgb_alignment_image_data(spot_coordinates, warp_parameters):
    img_args = {"spots": spot_coordinates,
                **warp_parameters,
                "bit_depth": 16}
    img0, img, description = make_alignment_image_data(**img_args)
    return img0, img, description, img_args["bit_depth"]


@pytest.fixture(scope="session")
def rgb_alignment_image_bad_description(spot_coordinates, warp_parameters):
    img_args = {"spots": spot_coordinates,
                **warp_parameters,
                "bit_depth": 16}
    img0, img, description = make_alignment_image_data(**img_args)
    description["Alignment red channel"][2] = 25
    return img0, img, description, img_args["bit_depth"]


@pytest.fixture(scope="session")
def rgb_alignment_image_data_offset(spot_coordinates, warp_parameters):
    img_args = {"spots": spot_coordinates,
                **warp_parameters,
                "bit_depth": 16,
                "offsets": (50, 50)}
    img0, img, description = make_alignment_image_data(**img_args)
    return img0, img, description, img_args["bit_depth"]


def write_tiff_file(img0, img, description, bit_depth, n_frames, filename):
    if bit_depth == 8:
        img = img[:,:,0]
        channels = 1
    else:
        channels = 3
    movie = np.stack([img for n in range(n_frames)], axis=0)

    tag_orientation = (274, 'H', 1, 1, False) # Orientation = ORIENTATION.TOPLEFT
    tag_sample_format = (339, 'H', channels, (1, )*channels, False) # SampleFormat = SAMPLEFORMAT.UINT

    with tifffile.TiffWriter(filename) as tif:
        for n, frame in enumerate(movie):
            str_datetime = f"{n*10+10}:{n*10+18}"
            tag_datetime = (306, 's', len(str_datetime), str_datetime, False)
            tif.save(frame,
                    description=json.dumps(description, indent=4),
                    software="Bluelake Unknown",
                    metadata=None, contiguous=False,
                    extratags=(tag_orientation, tag_sample_format, tag_datetime))


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
    img0, img, description, bit_depth = gray_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[img[:, :, 0]], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=8),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    assert not fr.is_rgb
    assert np.all(fr.data == fr.raw_data)
    assert np.allclose(fr.raw_data, fr._get_plot_data())


def test_image_reconstruction_rgb(rgb_alignment_image_data, rgb_alignment_image_data_offset,
                                  rgb_alignment_image_bad_description):
    img0, img, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[img], times=[["10", "18"]],
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

    original_data = (img0 / (2**bit_depth - 1)).astype(float)
    assert np.allclose(original_data, fr._get_plot_data(), atol=0.05)
    assert np.allclose(original_data / 0.5, fr._get_plot_data(vmax=0.5), atol=0.10)
    max_signal = np.max(np.hstack([img0[:, :, 0], fr._get_plot_data("red")]))
    diff = np.abs(img0[:, :, 0].astype(float)-fr._get_plot_data("red").astype(float))
    assert np.all(diff/max_signal < 0.05)

    with pytest.raises(ValueError):
        fr._get_plot_data(channel="purple")

    # test that bad alignment matrix gives high error compared to correct matrix
    img0, img, description, bit_depth = rgb_alignment_image_bad_description
    fake_tiff = TiffStack(MockTiffFile(data=[img], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    assert not np.allclose(original_data, fr._get_plot_data(), atol=0.05)

    img0, img, description, bit_depth = rgb_alignment_image_data_offset
    fake_tiff = TiffStack(MockTiffFile(data=[img], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    original_data = (img0 / (2**bit_depth - 1)).astype(float)
    assert np.allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_multiframe(rgb_alignment_image_data):
    img0, img, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[img]*6,
                                       times=[["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(2)

    assert fr.is_rgb
    original_data = (img0 / (2**bit_depth - 1)).astype(float)
    assert np.allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_missing_metadata(rgb_alignment_image_data):
    # no metadata
    img0, img, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(MockTiffFile(data=[img], times=[["10", "18"]],
                                       description="", bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    with pytest.warns(UserWarning) as record:
        fr.data
    assert len(record) == 1
    assert record[0].message.args[0] == "File does not contain metadata. Only raw data is available"

    # missing alignment matrices
    red = description.pop("Alignment red channel")
    fake_tiff = TiffStack(MockTiffFile(data=[img], times=[["10", "18"]],
                                       description=json.dumps(description), bit_depth=16),
                          align=True)
    stack = CorrelatedStack.from_data(fake_tiff)
    fr = stack._get_frame(0)

    with pytest.warns(UserWarning) as record:
        fr.data
    assert len(record) == 1
    assert record[0].message.args[0] == "File does not contain alignment matrices. Only raw data is available"
    description["Alignment red channel"] = red # reset fixture


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
            assert "Applied alignment red channel" in tif.pages[0].description
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
