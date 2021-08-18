import pytest
import numpy as np
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
