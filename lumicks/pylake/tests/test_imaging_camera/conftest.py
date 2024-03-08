import numpy as np
import pytest

from ..data.mock_widefield import write_tiff_file, make_alignment_image_data


@pytest.fixture(scope="module")
def tiff_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tiffs")


@pytest.fixture(scope="module")
def spot_coordinates():
    return [[20, 30], [50, 50], [120, 30], [50, 70], [150, 60]]


@pytest.fixture(scope="module")
def warp_parameters():
    return {
        "red_warp_parameters": {"Tx": 20, "Ty": 10, "theta": 3},
        "blue_warp_parameters": {"Tx": 10, "Ty": 20, "theta": -3},
    }


@pytest.fixture(scope="module")
def gray_alignment_image_data(spot_coordinates, warp_parameters):
    return make_alignment_image_data(
        spot_coordinates, version=1, bit_depth=8, camera="irm", **warp_parameters
    )


@pytest.fixture(scope="module", params=[1, 2])
def rgb_alignment_image_data(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(
        spot_coordinates, version=request.param, bit_depth=16, camera="wt", **warp_parameters
    )


@pytest.fixture(scope="module", params=[1, 2])
def rgb_alignment_image_data_offset(spot_coordinates, warp_parameters, request):
    return make_alignment_image_data(
        spot_coordinates,
        version=request.param,
        bit_depth=16,
        offsets=(50, 50),
        camera="wt",
        **warp_parameters,
    )


@pytest.fixture(scope="module")
def rgb_tiff_file(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_single.tiff")
    _, warped_image, description, _ = rgb_alignment_image_data
    write_tiff_file(warped_image, description, n_frames=1, filename=str(mock_filename))
    return mock_filename


def _create_gb_tiff(mock_filename, spot_coordinates, warp_parameters, num_frames):
    rgb_reference_image, warped_image, description, _ = make_alignment_image_data(
        spot_coordinates, version=2, bit_depth=16, camera="wt", **warp_parameters
    )

    # For two-color tiff files with green and blue, channel 0 refers to the green channel.
    description = {
        key: value for key, value in description.items() if "Red Excitation Laser" not in key
    }
    description["Channel 0 alignment"] = description["Channel 1 alignment"]
    description["Channel 1 alignment"] = description["Channel 2 alignment"]
    description.pop("Channel 2 alignment")

    # RGB -> GB
    warped_image = warped_image[:, :, 1:]
    write_tiff_file(
        warped_image,
        description,
        n_frames=num_frames,
        filename=str(mock_filename),
        planarconfig="contig",  # Allows storing 2-color TIFFs
    )

    reference_img = np.zeros((*rgb_reference_image.shape[:-1], 3), dtype=float)
    reference_img[..., 1:] = rgb_reference_image[..., 1:]

    # Expand reference to multiple frames
    reference_img = np.repeat(reference_img[np.newaxis, ...], num_frames, axis=0).squeeze()

    return mock_filename, reference_img


@pytest.fixture(scope="module")
def gb_tiff_file_single(tiff_dir, spot_coordinates, warp_parameters):
    return _create_gb_tiff(
        tiff_dir.join("single_two_color.tiff"), spot_coordinates, warp_parameters, 1
    )


@pytest.fixture(scope="module")
def gb_tiff_file_multi(tiff_dir, spot_coordinates, warp_parameters):
    return _create_gb_tiff(
        tiff_dir.join("multiple_two_color.tiff"), spot_coordinates, warp_parameters, 5
    )


@pytest.fixture(scope="module")
def rgb_tiff_file_multi(tiff_dir, rgb_alignment_image_data):
    mock_filename = tiff_dir.join("rgb_multi.tiff")
    _, warped_image, description, _ = rgb_alignment_image_data
    write_tiff_file(warped_image, description, n_frames=2, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="module")
def gray_tiff_file(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_single.tiff")
    _, warped_image, description, _ = gray_alignment_image_data
    write_tiff_file(warped_image, description, n_frames=1, filename=str(mock_filename))
    return mock_filename


@pytest.fixture(scope="module")
def gray_tiff_file_multi(tiff_dir, gray_alignment_image_data):
    mock_filename = tiff_dir.join("gray_multi.tiff")
    _, warped_image, description, _ = gray_alignment_image_data
    write_tiff_file(warped_image, description, n_frames=2, filename=str(mock_filename))
    return mock_filename
