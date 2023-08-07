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
