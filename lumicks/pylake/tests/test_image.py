import pytest
import numpy as np

from lumicks.pylake.detail.image import reconstruct_image, reconstruct_num_frames, save_tiff


def test_reconstruct():
    infowave = np.array([0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 2])
    the_data = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    image = reconstruct_image(the_data, infowave, 5)
    assert image.shape == (1, 5)
    assert np.all(image == [4, 8, 12, 0, 0])

    image = reconstruct_image(the_data, infowave, 2)
    assert image.shape == (2, 2)
    assert np.all(image == [[4, 8], [12, 0]])


def test_reconstruct_multiframe():
    size = 100
    infowave = np.ones(size)
    infowave[9::10] = 2
    the_data = np.arange(size)

    assert reconstruct_image(the_data, infowave, 5).shape == (2, 5)
    assert reconstruct_image(the_data, infowave, 2).shape == (5, 2)
    assert reconstruct_image(the_data, infowave, 1).shape == (10, 1)
    assert reconstruct_image(the_data, infowave, 2, 2).shape == (3, 2, 2)
    assert reconstruct_image(the_data, infowave, 2, 3).shape == (2, 3, 2)
    assert reconstruct_image(the_data, infowave, 2, 5).shape == (5, 2)

    assert reconstruct_num_frames(infowave, 2, 2) == 3
    assert reconstruct_num_frames(infowave, 2, 3) == 2
    assert reconstruct_num_frames(infowave, 2, 5) == 1


def test_int_tiff(tmpdir):
    image16 = np.ones(shape=(10, 10, 3)) * np.iinfo(np.uint16).max
    save_tiff(image16, str(tmpdir.join("1")), dtype=np.uint16)
    save_tiff(image16, str(tmpdir.join("2")), dtype=np.float32)
    save_tiff(image16, str(tmpdir.join("3")), dtype=np.uint8, clip=True)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image16, str(tmpdir.join("4")), dtype=np.uint8)
    assert "Can't safely export image with `dtype=uint8` channels" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        save_tiff(image16, str(tmpdir.join("5")), dtype=np.float16)
    assert "Can't safely export image with `dtype=float16` channels" in str(excinfo.value)


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
