import pytest
import numpy as np
from matplotlib.testing.decorators import cleanup
from lumicks.pylake.channel import Slice, Continuous, TimeSeries
from lumicks.pylake.piezo_tracking.piezo_tracking import (
    DistanceCalibration,
    PiezoTrackingCalibration,
)
from lumicks.pylake.piezo_tracking.baseline import ForceBaseLine


def trap_pos_camera_distance():
    dt = int(1e9 / 78125)
    trap_pos = Slice(Continuous(np.arange(2.0, 8.0, 0.001), dt=dt, start=1592916040906356300))
    trap_pos_ds = trap_pos.downsampled_by(1000)
    camera_dist = Slice(TimeSeries(2 * trap_pos_ds.data + 1, trap_pos_ds.timestamps + 500 * dt))

    return trap_pos, camera_dist


@pytest.mark.parametrize(
    "sampled_positions",
    [
        Slice(Continuous(np.array([0, 0.5, 1.5, 2.5, 3.5]), dt=1, start=1)),
        Slice(TimeSeries(np.array([0, 0.5, 1.5, 2.5, 3.5]), np.array([0, 0.5, 1.5, 2.5, 3.5]))),
    ],
)
def test_distance_calibration(sampled_positions):
    distance_calibration = DistanceCalibration(*trap_pos_camera_distance(), 1)
    np.testing.assert_allclose(distance_calibration.valid_range(), (3.4995, 7.4995))
    assert str(distance_calibration) == "+ 2.0000 x + 1.0000"
    assert repr(distance_calibration) == "DistanceCalibration(+ 2.0000 x + 1.0000)"

    # Test evaluation of the calibration
    calibrated_slice = distance_calibration(sampled_positions)
    np.testing.assert_allclose(calibrated_slice.data, [1.0, 2.0, 4.0, 6.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(calibrated_slice.timestamps, sampled_positions.timestamps)
    assert calibrated_slice.labels["title"] == "Piezo distance"
    assert calibrated_slice.labels["y"] == "Distance [um]"


def test_lost_tracking():
    trap_pos, camera_dist = trap_pos_camera_distance()
    trap_pos.data[4001] = 1e6  # Put a bad sample here, so we can detect that it gets discarded
    camera_dist.data[4] = 0  # Template lost => this should result in the bad sample being discarded
    with pytest.warns(RuntimeWarning, match="There were frames with missing video tracking"):
        distance_calibration = DistanceCalibration(trap_pos, camera_dist, 1)

    # Test evaluation of the calibration (this should be ok, since we discarded the sample)
    sampled_positions = Slice(Continuous(np.array([0, 0.5, 1.5, 2.5, 3.5]), dt=1, start=1))
    calibrated_slice = distance_calibration(sampled_positions)
    np.testing.assert_allclose(calibrated_slice.data, [1.0, 2.0, 4.0, 6.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(calibrated_slice.timestamps, sampled_positions.timestamps)
    assert calibrated_slice.labels["title"] == "Piezo distance"
    assert calibrated_slice.labels["y"] == "Distance [um]"


def test_from_file():
    trap_pos, camera_dist = trap_pos_camera_distance()

    class MockPiezo:
        def __init__(self):
            self.dict = {"Trap position": {"1X": trap_pos}}

        def __getitem__(self, item):
            return self.dict[item]

        @property
        def distance1(self):
            return camera_dist

    distance_calibration = DistanceCalibration.from_file(MockPiezo())

    # Test evaluation of the calibration
    sampled_positions = Slice(Continuous(np.array([0, 0.5, 1.5, 2.5, 3.5]), dt=1, start=1))
    calibrated_slice = distance_calibration(sampled_positions)
    np.testing.assert_allclose(calibrated_slice.data, [1.0, 2.0, 4.0, 6.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(calibrated_slice.timestamps, sampled_positions.timestamps)
    assert calibrated_slice.labels["title"] == "Piezo distance"
    assert calibrated_slice.labels["y"] == "Distance [um]"


@cleanup
def test_plots():
    distance_calibration = DistanceCalibration(*trap_pos_camera_distance(), 1)
    distance_calibration.plot()
    distance_calibration.plot_residual()


def test_piezo_invalid_signs():
    with pytest.raises(
        ValueError,
        match="Argument `signs` should be a tuple of two floats reflecting the sign for each "
        "channel.",
    ):
        PiezoTrackingCalibration(None, None, None, (1, 1, 1))

    with pytest.raises(ValueError, match="Each sign should be either -1 or 1."):
        PiezoTrackingCalibration(None, None, None, (1, 2))


def test_piezotracking(piezo_tracking_test_data):
    data = piezo_tracking_test_data

    # Calibrate using the trap position
    distance_calibration = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])

    # Estimate the baselines
    baseline_1 = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], data["baseline_force"], degree=2
    )
    baseline_2 = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], data["baseline_force"], degree=2
    )

    # Perform the piezo tracking
    piezo_calibration = PiezoTrackingCalibration(distance_calibration, baseline_1, baseline_2)

    piezo_distance, corrected_force1, corrected_force2 = piezo_calibration.piezo_track(
        data["trap_position"], data["force_1x"], data["force_2x"], trim=False
    )

    np.testing.assert_allclose(corrected_force1.data, data["force_without_baseline"], rtol=1e-6)
    np.testing.assert_allclose(piezo_distance.data, data["correct_distance"], rtol=1e-6)
