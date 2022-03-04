import pytest
import numpy as np
from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.detail.image import reconstruct_image, reconstruct_image_sum, reconstruct_num_frames, save_tiff,\
    ImageMetadata, line_timestamps_image, histogram_rows


def test_metadata_from_json():
    json = { 'cereal_class_version': 1,
             'fluorescence': True,
             'force': False,
             'scan count': 0,
             'scan volume': {'center point (um)': {'x': 58.075877109272604,
                                                   'y': 31.978375270573267,
                                                   'z': 0},
                             'cereal_class_version': 1,
                             'pixel time (ms)': 0.5,
                             'scan axes': [{'axis': 0,
                                            'cereal_class_version': 1,
                                            'num of pixels': 240,
                                            'pixel size (nm)': 150,
                                            'scan time (ms)': 0,
                                            'scan width (um)': 36.07468112612217}]}}

    image_metadata = ImageMetadata.from_dataset(json)

    res = image_metadata.resolution
    assert np.isclose(res[0], 1e7 / 150)
    assert np.isclose(res[1], 1e7 / 150)
    assert res[2] == 'CENTIMETER'

    assert np.isclose(image_metadata.metadata['PixelTime'], .0005)
    assert image_metadata.metadata['PixelTimeUnit'] == 's'


@pytest.mark.parametrize("num_lines, pixels_per_line, pad_size", [(5, 3, 3), (5, 4, 2), (4, 7, 5)])
def test_timestamps_image(num_lines, pixels_per_line, pad_size):
    line_info_wave = np.tile(np.array([1, 1, 2], dtype=np.int32), (pixels_per_line, ))
    line_selector = np.zeros(line_info_wave.shape)
    line_selector[0] = True
    pad = np.zeros(pad_size, dtype=np.int32)
    infowave = np.hstack([np.hstack([pad, line_info_wave, pad]) for _ in np.arange(num_lines)])
    start_indices = np.hstack([np.hstack([pad, line_selector, pad]) for _ in np.arange(num_lines)])

    # Generate list of timestamp data.
    # It is important to use timestamps that are large enough such that floating point
    # round-off errors will occur if the data is converted to floating point representation.
    time = np.arange(len(infowave), dtype=np.int64) * int(700e9) + 1623965975045144000

    line_stamps = line_timestamps_image(time, infowave, pixels_per_line)
    assert line_stamps.shape == (sum(start_indices), )
    np.testing.assert_equal(line_stamps, time[start_indices == 1])


def test_reconstruct():
    infowave = np.array([0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 2])
    the_data = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    image = reconstruct_image(the_data, infowave, (5, ))
    assert image.shape == (1, 5)
    assert np.all(image == [4, 8, 12, 0, 0])

    image = reconstruct_image(the_data, infowave, (2, ))
    assert image.shape == (2, 2)
    assert np.all(image == [[4, 8], [12, 0]])

    image = reconstruct_image_sum(the_data, infowave, (5, ))
    assert image.shape == (1, 5)
    assert np.all(image == [4, 8, 12, 0, 0])

    image = reconstruct_image_sum(the_data, infowave, (2, ))
    assert image.shape == (2, 2)
    assert np.all(image == [[4, 8], [12, 0]])


def test_reconstruct_multiframe():
    size = 100
    infowave = np.ones(size)
    infowave[9::10] = 2
    the_data = np.arange(size)

    assert reconstruct_image(the_data, infowave, (5, )).shape == (2, 5)
    assert reconstruct_image(the_data, infowave, (2, )).shape == (5, 2)
    assert reconstruct_image(the_data, infowave, (1, )).shape == (10, 1)
    assert reconstruct_image(the_data, infowave, (2, 2)).shape == (3, 2, 2)
    assert reconstruct_image(the_data, infowave, (3, 2)).shape == (2, 3, 2)
    assert reconstruct_image(the_data, infowave, (5, 2)).shape == (1, 5, 2)

    assert reconstruct_image_sum(the_data, infowave, (5, )).shape == (2, 5)
    assert reconstruct_image_sum(the_data, infowave, (2, )).shape == (5, 2)
    assert reconstruct_image_sum(the_data, infowave, (1, )).shape == (10, 1)
    assert reconstruct_image_sum(the_data, infowave, (2, 2)).shape == (3, 2, 2)
    assert reconstruct_image_sum(the_data, infowave, (3, 2)).shape == (2, 3, 2)
    assert reconstruct_image_sum(the_data, infowave, (5, 2)).shape == (1, 5, 2)

    assert reconstruct_num_frames(infowave, 2, 2) == 3
    assert reconstruct_num_frames(infowave, 2, 3) == 2
    assert reconstruct_num_frames(infowave, 2, 5) == 1


def test_int_tiff(tmpdir):
    def grab_tags(file):
        import tifffile
        from ast import literal_eval

        with tifffile.TiffFile(file) as tif:
            tiff_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                try:
                    tiff_tags[name] = literal_eval(value)
                except (ValueError, SyntaxError):
                    tiff_tags[name] = value

            return tiff_tags

    image16 = np.ones(shape=(10, 10, 3)) * np.iinfo(np.uint16).max
    save_tiff(image16, str(tmpdir.join("1")), dtype=np.uint16, metadata=ImageMetadata(pixel_size_x=1.0, pixel_time=1.0))
    save_tiff(image16, str(tmpdir.join("2")), dtype=np.float32, metadata=ImageMetadata(pixel_size_x=5.0, pixel_time=5.0))
    save_tiff(image16, str(tmpdir.join("3")), dtype=np.uint8, clip=True)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image16, str(tmpdir.join("4")), dtype=np.uint8)
    assert "Can't safely export image with `dtype=uint8` channels" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image16, str(tmpdir.join("5")), dtype=np.float16)
    assert "Can't safely export image with `dtype=float16` channels" in str(excinfo.value)

    tags = grab_tags(str(tmpdir.join("1")))
    assert str(tags['ResolutionUnit']) == "RESUNIT.CENTIMETER"
    np.testing.assert_allclose(tags['ImageDescription']['PixelTime'], 0.001)
    assert tags['ImageDescription']['PixelTimeUnit'] == "s"
    np.testing.assert_allclose(tags['ImageDescription']['shape'], [10, 10, 3])
    np.testing.assert_allclose(tags['XResolution'][0], 10000000)
    np.testing.assert_allclose(tags['YResolution'][0], 10000000)

    tags = grab_tags(str(tmpdir.join("2")))
    assert str(tags['ResolutionUnit']) == "RESUNIT.CENTIMETER"
    np.testing.assert_allclose(tags['ImageDescription']['PixelTime'], 0.005)
    assert tags['ImageDescription']['PixelTimeUnit'] == "s"
    np.testing.assert_allclose(tags['ImageDescription']['shape'], [10, 10, 3])
    np.testing.assert_allclose(tags['XResolution'][0], 2000000)
    np.testing.assert_allclose(tags['YResolution'][0], 2000000)


def test_float_tiff(tmpdir):
    image32 = np.ones(shape=(10, 10, 3)) * np.finfo(np.float32).max
    save_tiff(image32, str(tmpdir.join("1")), dtype=np.float32)
    save_tiff(image32, str(tmpdir.join("2")), dtype=np.float16, clip=True)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image32, str(tmpdir.join("3")), dtype=np.float16)
    assert "Can't safely export image with `dtype=float16` channels" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image32, str(tmpdir.join("1")), dtype=np.uint16)
    assert "Can't safely export image with `dtype=uint16` channels" in str(excinfo.value)


def test_histogram_rows():
    data = np.arange(36).reshape((6,6))

    e, h, w = histogram_rows(data, 1, 0.1)
    np.testing.assert_allclose(e, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.all(np.equal(h, [15, 51, 87, 123, 159, 195]))
    np.testing.assert_allclose(w, 0.1)

    e, h, w = histogram_rows(data, 3, 0.1)
    np.testing.assert_allclose(e, [0.0, 0.3])
    assert np.all(np.equal(h, [153, 477]))
    np.testing.assert_allclose(w, 0.3)

    with pytest.warns(UserWarning):
        e, h, w = histogram_rows(data, 5, 0.1)
        np.testing.assert_allclose(e, [0.0, 0.5])
        assert np.all(np.equal(h, [435, 195]))
        np.testing.assert_allclose(w, [0.5, 0.1])


def test_partial_pixel_image_reconstruction():
    """This function tests whether a partial pixel at the end is fully dropped. This is important,
    since in the timestamp reconstruction, we subtract the minimum value prior to averaging (to
    allow taking averages of larger chunks). Without this functionality, the lowest timestamp to be
    reconstructed can be smaller than the first timestamp. This means that the value subtracted
    from the timestamps prior to summing is smaller. This means that the timestamps more quickly
    get into the range where they are at risk of overflow (and therefore have to be summed in
    smaller blocks). This in turn can lead to unnecessarily long reconstruction times."""
    def size_test(x, axis):
        assert len(x) == 4, "The last pixel should have been dropped since it was partial"
        return np.array([1, 1, 1, 1])

    iw = np.tile([1, 1, 1, 1, 2], (5,))
    iw[-1] = 0
    ts = np.arange(iw.size)
    reconstruct_image(ts, iw, (5, 1), reduce=size_test)


@pytest.mark.parametrize(
    "minimum, maximum, ref_minimum, ref_maximum",
    [
        (1.0, 5.0, [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]),
        ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
        ([1.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0]),
        ([1.0, 2.0, 3.0], [5.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
        ([1.0, 2.0, 3.0], 5.0, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
    ],
)
def test_absolute_ranges(minimum, maximum, ref_minimum, ref_maximum):
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment(minimum, maximum, mode="absolute")
    np.testing.assert_allclose(ca.minimum, ref_minimum)
    np.testing.assert_allclose(ca.maximum, ref_maximum)

    np.testing.assert_allclose(
        ca._get_data_rgb(ref_img), np.clip((ref_img - ca.minimum) / (ca.maximum - ca.minimum), 0, 1)
    )


@pytest.mark.parametrize(
    "minimum, maximum, ref_minimum, ref_maximum",
    [
        (25.0, 75.0, [25.0, 25.0, 25.0], [75.0, 75.0, 75.0]),
        ([25.0, 35.0, 45.0], [75.0, 85.0, 95.0], [25.0, 35.0, 45.0], [75.0, 85.0, 95.0]),
        ([25.0], [55.0, 65.0, 75.0], [25.0, 25.0, 25.0], [55.0, 65.0, 75.0]),
        ([25.0, 35.0, 45.0], [5.0], [25.0, 35.0, 45.0], [5.0, 5.0, 5.0]),
        ([25.0, 35.0, 45.0], 5.0, [25.0, 35.0, 45.0], [5.0, 5.0, 5.0]),
        (1.0, [3.0, 4.0, 5.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0]),
    ],
)
def test_percentile_ranges(minimum, maximum, ref_minimum, ref_maximum):
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment(minimum, maximum, mode="percentile")
    np.testing.assert_allclose(ca.minimum, ref_minimum)
    np.testing.assert_allclose(ca.maximum, ref_maximum)

    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(ref_img, 2, 0), ca.minimum, ca.maximum)
        ]
    )
    minimum, maximum = bounds.T

    np.testing.assert_allclose(
        ca._get_data_rgb(ref_img), np.clip((ref_img - minimum) / (maximum - minimum), 0, 1)
    )


def test_invalid_range():
    with pytest.raises(ValueError, match="Color value bounds should be of length 1 or 3"):
        ColorAdjustment([2.0, 3.0], [1.0, 2.0, 3.0], mode="absolute")

    with pytest.raises(ValueError, match="Mode nust be percentile or absolute"):
        ColorAdjustment(1.0, 1.0, mode="spaghetti")


def test_no_adjust():
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment.nothing()
    np.testing.assert_allclose(ca._get_data_rgb(ref_img), ref_img / ref_img.max())
