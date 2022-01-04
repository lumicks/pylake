import numpy as np
from lumicks import pylake
import pytest
from textwrap import dedent
from lumicks.pylake.detail.h5_helper import write_h5
from . import test_file_items


def test_attributes(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert type(f.bluelake_version) is str
    assert f.format_version in [1, 2]
    assert type(f.experiment) is str
    assert type(f.description) is str
    assert type(f.guid) is str
    assert np.issubdtype(f.export_time, np.dtype(int).type)


def test_properties(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 1:
        assert f.kymos == {}
    else:
        assert len(f.kymos) == 1
    if f.format_version == 1:
        assert f.scans == {}
    else:
        assert len(f.scans) == 4
    if f.format_version == 1:
        assert f.point_scans == {}
    else:
        assert len(f.point_scans) == 1
    assert f.fdcurves == {}


def test_groups(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 1:
        assert str(f["Force HF"]) == "{'Force 1x', 'Force 1y', 'Force 1z'}"

        for x in range(0, 2):
            t = [name for name in f]
            assert set(t) == set(f)

        for x in range(0, 2):
            t = [name for name in f["Force HF"]]
            assert set(t) == set(["Force 1x", "Force 1y", "Force 1z"])


def test_redirect_list(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            f["Calibration"]

        with pytest.warns(FutureWarning):
            f["Marker"]

        with pytest.warns(FutureWarning):
            f["FD Curve"]

        with pytest.warns(FutureWarning):
            f["Kymograph"]

        with pytest.warns(FutureWarning):
            f["Scan"]


def test_repr_and_str(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert repr(f) == f"lumicks.pylake.File('{h5_file.filename}')"
    if f.format_version == 1:
        assert str(f) == dedent("""\
            File root metadata:
            - Bluelake version: unknown
            - Description: test
            - Experiment: test
            - Export time (ns): -1
            - File format version: 1
            - GUID: invalid

            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
              Force 1z:
              - Data type: float64
              - Size: 5
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1z:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2

            .force1x
            .force1y
            .force1z

            .downsampled_force1x
            .downsampled_force1y
            .downsampled_force1z
        """)
    if f.format_version == 2:
        assert str(f) == dedent("""\
            File root metadata:
            - Bluelake version: unknown
            - Description: test
            - Experiment: test
            - Export time (ns): -1
            - File format version: 2
            - GUID: invalid

            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
              Force 1z:
              - Data type: float64
              - Size: 5
              Force 2x:
              - Data type: float64
              - Size: 70
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1z:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
            Info wave:
              Info wave:
              - Data type: uint8
              - Size: 64
            Photon Time Tags:
              Red:
              - Data type: int64
              - Size: 9
            Photon count:
              Blue:
              - Data type: uint32
              - Size: 64
              Green:
              - Data type: uint32
              - Size: 64
              Red:
              - Data type: uint32
              - Size: 64
            Point Scan:
              PointScan1:
              - Data type: object
              - Size: 1

            .markers
              - force feedback
              - test_marker
              - test_marker2

            .kymos
              - Kymo1

            .scans
              - fast X slow Z multiframe
              - fast Y slow X
              - fast Y slow X multiframe
              - fast Y slow Z multiframe

            .force1x
              .calibration
            .force1y
              .calibration
            .force1z
              .calibration
            .force2x
              .calibration

            .downsampled_force1x
              .calibration
            .downsampled_force1y
              .calibration
            .downsampled_force1z
              .calibration
        """)


def test_invalid_file_format(h5_file_invalid_version):
    with pytest.raises(Exception):
        f = pylake.File.from_h5py(h5_file_invalid_version)


def test_invalid_access(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            m = f["Kymograph"]

        with pytest.raises(IndexError):
            m["Kymo1"]


def test_missing_metadata(h5_file_missing_meta):
    f = pylake.File.from_h5py(h5_file_missing_meta)
    if f.format_version == 2:
        with pytest.warns(UserWarning, match="Scan 'fast Y slow X no meta' is missing metadata and cannot be loaded"):
            scans = f.scans
            assert len(scans) == 1


def _internal_h5_export_api(file, *args, **kwargs):
    return write_h5(file.h5, *args, **kwargs)


def _public_h5_export_api(file, *args, **kwargs):
    return file.save_as(*args, **kwargs)


@pytest.mark.parametrize("save_h5", [_internal_h5_export_api, _public_h5_export_api])
def test_h5_export(tmpdir_factory, h5_file, save_h5):
    f = pylake.File.from_h5py(h5_file)
    tmpdir = tmpdir_factory.mktemp("pylake")

    new_file = f"{tmpdir}/copy.h5"
    save_h5(f, new_file, 5)
    g = pylake.File(new_file)
    assert str(g) == str(f)

    # Verify that all attributes are there and correct
    test_file_items.test_scans(g.h5)
    test_file_items.test_kymos(g.h5)
    test_attributes(g.h5)
    test_file_items.test_channels(g.h5)
    test_file_items.test_calibration(g.h5)
    test_file_items.test_marker(g.h5)
    test_properties(g.h5)

    new_file = f"{tmpdir}/omit_LF1y.h5"
    save_h5(f, new_file, 5, omit_data={"Force LF/Force 1y"})
    omit_lf1y = pylake.File(new_file)

    np.testing.assert_allclose(omit_lf1y["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data)
    np.testing.assert_allclose(omit_lf1y["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data)
    np.testing.assert_allclose(omit_lf1y["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data)
    with pytest.raises(KeyError):
        assert np.any(omit_lf1y["Force LF"]["Force 1y"].data)

    new_file = f"{tmpdir}/omit_1y.h5"
    save_h5(f, new_file, 5, omit_data={"*/Force 1y"})
    omit_1y = pylake.File(new_file)

    np.testing.assert_allclose(omit_1y["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data)
    np.testing.assert_allclose(omit_1y["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_1y["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data)
    with pytest.raises(KeyError):
        assert np.any(omit_1y["Force LF"]["Force 1y"].data)

    new_file = f"{tmpdir}/omit_hf.h5"
    save_h5(f, new_file, 5, omit_data={"Force HF/*"})
    omit_hf = pylake.File(new_file)

    np.testing.assert_allclose(omit_hf["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data)
    np.testing.assert_allclose(omit_hf["Force LF"]["Force 1y"].data, f["Force LF"]["Force 1y"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_hf["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_hf["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data)

    new_file = f"{tmpdir}/omit_two.h5"
    save_h5(f, new_file, 5, omit_data={"Force HF/*", "*/Force 1y"})
    omit_two = pylake.File(new_file)

    np.testing.assert_allclose(omit_two["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_two["Force LF"]["Force 1y"].data, f["Force LF"]["Force 1y"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_two["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(omit_two["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data)
