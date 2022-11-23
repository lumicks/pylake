import numpy as np
import pytest
from lumicks.pylake.channel import Continuous, Slice

from ..data.mock_confocal import (
    axes_dict_list,
    confocal_mock_file_h5,
    generate_kymo_with_ref,
    generate_scan_json,
)


# CAVE: If you want to test a cached property, after having modified parameters that change the
# value of the property, ensure to clear the `_cache` attribute before and after testing. To achieve
# both, you can monkeypatch the `_cache` attribute with an empty dict.
@pytest.fixture(scope="module")
def test_kymos():
    return {"standard": generate_kymo_with_ref("standard", np.random.poisson(10, (5, 4, 3)))}


@pytest.fixture
def add_force_channel(monkeypatch):
    def do_add_force_channel(kymo, ref, head=30, tail=10, head_lines=2, channel="force1x"):

        # Force channel that overlaps kymo; step from high to low force
        # Per default two lines of the kymo have a force of 30 and the rest of the lines 10.
        samples_per_line = (
            ref.metadata.pixels_per_line * ref.infowave.samples_per_pixel
            + 2 * ref.infowave.line_padding
        )
        number_of_samples = ref.metadata.lines_per_frame * samples_per_line
        # First line starts after line_padding
        first_pixels = ref.infowave.line_padding + head_lines * samples_per_line
        force = np.r_[
            np.ones(first_pixels) * head, np.ones(number_of_samples - first_pixels) * tail
        ]

        monkeypatch.setattr(
            kymo.file,
            channel,
            Slice(Continuous(force, ref.start, ref.timestamps.dt)),
            raising=False,
        )

        # Return kymo per line force data
        return np.r_[
            np.ones(head_lines) * head, np.ones(ref.metadata.lines_per_frame - head_lines) * tail
        ]

    return do_add_force_channel


def test_kymo_file_h5_interaction(test_kymos, add_force_channel, tmp_path):
    """Test if a kymo works the same with either a MockConfocal or a Bluelake like H5 file as data
    source
    """
    kymo, ref = test_kymos["standard"]
    add_force_channel(kymo, ref, channel="force1x")

    # Create kymo with H5 file as data source
    json_confocal = generate_scan_json(
        axes_dict_list(
            [sa.axis for sa in kymo._metadata.scan_axes],
            [sa.num_pixels for sa in kymo._metadata.scan_axes],
            [sa.pixel_size_um * 1e3 for sa in kymo._metadata.scan_axes],
        )
    )
    name = "standard"
    with confocal_mock_file_h5(
        path=tmp_path,
        name=name,
        scan=False,
        json_confocal=json_confocal,
        infowave=ref.infowave.data.data,
        red_counts=kymo.file.red_photon_count.data,
        green_counts=kymo.file.green_photon_count.data,
        blue_counts=kymo.file.blue_photon_count.data,
        start=ref.start,
        dt=ref.timestamps.dt,
        force_data=kymo.file.force1x.data,
        force_start=ref.start,
        force_channel="1x",
    ) as file:

        kymo_h5 = file.kymos[name]

        # Test if kymos with different data sources work the same
        np.testing.assert_equal(kymo.shape, kymo_h5.shape)
        np.testing.assert_equal(kymo.infowave.start, kymo_h5.infowave.start)
        np.testing.assert_equal(kymo.infowave.stop, kymo_h5.infowave.stop)
        np.testing.assert_equal(kymo.infowave.data, kymo_h5.infowave.data)
        np.testing.assert_equal(kymo.infowave.timestamps, kymo_h5.infowave.timestamps)
        np.testing.assert_equal(kymo.timestamps, kymo_h5.timestamps)
        np.testing.assert_equal(kymo.get_image(), kymo_h5.get_image())
        # Force channel might be used for plotting.
        np.testing.assert_equal(kymo.file.force1x.data, kymo_h5.file.force1x.data)
