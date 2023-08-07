import numpy as np
import pytest

from ..data.mock_file import MockDataFile_v1, MockDataFile_v2


@pytest.fixture(scope="module", params=[MockDataFile_v1, MockDataFile_v2])
def channel_h5_file(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    mock_file.make_continuous_channel("Force HF", "Force 1x", 1, 10, np.arange(5.0))
    mock_file.make_timeseries_channel("Force LF", "Force 1x", [(1, 1.1), (2, 2.1)])
    mock_file.make_timeseries_channel(
        "Force LF variable", "Force 1x", [(1, 1.1), (2, 2.1), (4, 3.1)]
    )

    if mock_class == MockDataFile_v2:
        # fmt: off
        counts = np.uint32([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0,
                            0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 8, 0])
        # fmt: on

        # Generate lines at 1 Hz
        freq = 1e9 / 16
        mock_file.make_continuous_channel("Photon count", "Red", np.int64(20e9), freq, counts)

    return mock_file.file
