import numpy as np
import pytest
from .data.generate_gaussian_data import read_dataset as read_dataset_gaussian
from .data.generate_exponential_data import read_dataset as read_dataset_exponential
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup

@pytest.fixture
def gaussian_1d():
    return read_dataset_gaussian("gaussian_data_1d.npz")


@pytest.fixture
def kymogroups_2lines():
    _, _, photon_count, parameters = read_dataset_gaussian("kymo_data_2lines.npz")
    pixel_size = parameters[0].pixel_size
    centers = [p.center / pixel_size for p in parameters]

    image = CalibratedKymographChannel.from_array(photon_count, pixel_size=pixel_size)
    _, n_frames = image.data.shape

    lines = KymoLineGroup([KymoLine(np.arange(n_frames), np.full(n_frames, c), image) for c in centers])

    # introduce gaps into tracked lines
    use_frames = np.array([0, 1, -2, -1])
    gapped_lines = KymoLineGroup([KymoLine(line.time_idx[use_frames], line.coordinate_idx[use_frames], image)
                                  for line in lines])

    # crop the ends of initial lines and make new set of lines with one cropped and the second full
    truncated_lines = KymoLineGroup([KymoLine(np.arange(1, n_frames-2), np.full(n_frames-3, c), image) for c in centers])
    mixed_lines = KymoLineGroup([truncated_lines[0], lines[1]])

    return lines, gapped_lines, mixed_lines


@pytest.fixture
def exponential_data():
    return read_dataset_exponential("exponential_data.npz")
