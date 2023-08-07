import json

import pytest

from .data.mock_file import MockDataFile_v2
from .data.mock_fdcurve import generate_fdcurve_with_baseline_offset


@pytest.fixture(scope="session")
def fd_h5_file(tmpdir_factory, request):
    mock_class = MockDataFile_v2
    tmpdir = tmpdir_factory.mktemp("fdcurves")
    mock_file = mock_class(tmpdir.join(f"{mock_class.__name__}.h5"))
    mock_file.write_metadata()

    p, data = generate_fdcurve_with_baseline_offset()

    # write data
    fd_metadata = {"Polynomial Coefficients": {f"Corrected Force {n+1}x": p for n in range(2)}}
    fd_attrs = {
        "Start time (ns)": data["LF"]["time"][0],
        "Stop time (ns)": data["LF"]["time"][-1] + 1,
    }
    mock_file.make_fd("fd1", metadata=json.dumps(fd_metadata), attributes=fd_attrs)

    obs_force_lf_data = [datum for datum in zip(data["LF"]["time"], data["LF"]["obs_force"])]
    distance_lf_data = [datum for datum in zip(data["LF"]["time"], data["LF"]["distance"])]
    hf_start_time = data["HF"]["time"][0]
    for n in (1, 2):
        for component in ("x", "y"):
            mock_file.make_timeseries_channel(
                "Force LF", f"Force {n}{component}", obs_force_lf_data
            )
            mock_file.make_continuous_channel(
                "Force HF", f"Force {n}{component}", hf_start_time, 3, data["HF"]["obs_force"]
            )
        mock_file.make_continuous_channel(
            "Force HF", f"Corrected Force {n}x", hf_start_time, 3, data["HF"]["true_force"]
        )
        mock_file.make_timeseries_channel("Distance", f"Distance {n}", distance_lf_data)

    return mock_file.file, (data["LF"]["time"], data["LF"]["true_force"])
