import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.channel import Slice, Continuous, TimeSeries
from lumicks.pylake.detail.utilities import downsample
from lumicks.pylake.piezo_tracking.baseline import ForceBaseLine
from lumicks.pylake.piezo_tracking.piezo_tracking import (
    PiezoForceDistance,
    DistanceCalibration,
    PiezoTrackingCalibration,
)


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


def test_plots():
    distance_calibration = DistanceCalibration(*trap_pos_camera_distance(), 1)
    distance_calibration.plot()
    # Needed since otherwise second plot will issue warning about changing layout (since they both
    # call tight_layout()
    plt.close("all")
    distance_calibration.plot_residual()


def test_piezo_invalid_signs(poly_baseline_data, camera_calibration_data):
    baseline_trap_position, baseline_force = poly_baseline_data
    camera_dist, _ = camera_calibration_data
    baseline = ForceBaseLine.polynomial_baseline(baseline_trap_position, baseline_force, 1)
    distance = DistanceCalibration(baseline_trap_position, camera_dist)

    with pytest.raises(
        ValueError,
        match="Argument `signs` should be a tuple of two floats reflecting the sign for each "
        "channel.",
    ):
        PiezoTrackingCalibration(distance, (1, 1, 1))

    with pytest.raises(ValueError, match="Each sign should be either -1 or 1."):
        PiezoTrackingCalibration(distance, (1, 2))


def test_piezotracking(piezo_tracking_test_data):
    data = piezo_tracking_test_data

    # Calibrate using the trap position
    distance_calibration = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])

    # Estimate the baselines
    baseline_1 = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], data["baseline_force"], degree=2
    )
    baseline_2 = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], -data["baseline_force"], degree=2
    )

    # Perform the piezo tracking only
    piezo_calibration = PiezoTrackingCalibration(distance_calibration)

    piezo_distance = piezo_calibration.piezo_track(
        data["trap_position"], data["force_1x"], data["force_2x"]
    )
    np.testing.assert_allclose(piezo_distance.data, data["correct_distance"], rtol=1e-6)

    # Test the full workflow
    fd_generator = PiezoForceDistance(distance_calibration, baseline_1, baseline_2)
    piezo_distance, corrected_force1, corrected_force2 = fd_generator.force_distance(
        data["trap_position"], data["force_1x"], data["force_2x"], trim=False
    )
    np.testing.assert_allclose(corrected_force1.data, data["force_without_baseline"], rtol=1e-6)


def test_piezo_trimming(piezo_tracking_test_data):
    data = piezo_tracking_test_data

    # Calibrate using the trap position
    distance_calibration = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])

    # Baseline range
    min_trap, max_trap = (func(data["baseline_trap_position"].data) for func in (np.min, np.max))

    # Estimate the baselines
    baseline = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], data["baseline_force"], degree=2
    )

    fd_generator = PiezoForceDistance(distance_calibration, baseline, baseline)
    piezo_distance, corrected_force1, corrected_force2 = fd_generator.force_distance(
        data["trap_position"], data["force_1x"], data["force_2x"], trim=True
    )

    mask = np.logical_and(
        min_trap < data["trap_position"].data, data["trap_position"].data < max_trap
    )

    assert np.sum(mask) < len(data["correct_distance"])
    np.testing.assert_allclose(piezo_distance.data, data["correct_distance"][mask], rtol=1e-6)
    np.testing.assert_allclose(
        corrected_force1.data, data["force_without_baseline"][mask], rtol=1e-6
    )


@pytest.mark.parametrize(
    "correct_baseline1, correct_baseline2",
    [(True, False), (False, False), (False, True)],
)
def test_piezo_force_distance_missing_baselines(
    piezo_tracking_test_data,
    correct_baseline1,
    correct_baseline2,
):
    """Test if baseline correction can indeed be omitted"""
    data = piezo_tracking_test_data

    # Calibrate using the trap position
    distance_calibration = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])

    # Estimate the baselines
    baseline1, baseline2 = (
        ForceBaseLine.polynomial_baseline(
            data["baseline_trap_position"], sign * data["baseline_force"], degree=2
        )
        for sign in (1, -1)
    )

    fd_generator = PiezoForceDistance(
        distance_calibration,
        baseline1 if correct_baseline1 else None,
        baseline2 if correct_baseline2 else None,
    )

    piezo_distance, corrected_force1, corrected_force2 = fd_generator.force_distance(
        data["trap_position"], data["force_1x"], data["force_2x"], trim=False
    )

    np.testing.assert_allclose(piezo_distance.data, data["correct_distance"], rtol=1e-6)

    ref_force1 = data["force_without_baseline"] if correct_baseline1 else data["force_1x"].data
    np.testing.assert_allclose(corrected_force1.data, ref_force1, rtol=1e-6)
    ref_force2 = -data["force_without_baseline"] if correct_baseline2 else data["force_2x"].data
    np.testing.assert_allclose(corrected_force2.data, ref_force2, rtol=1e-6)


def test_baseline_range(piezo_tracking_test_data):
    def make_slice(start, stop):
        return Slice(Continuous(np.arange(start, stop), 10000, 10))

    def check_range(calibration, baseline1, baseline2, reference):
        np.testing.assert_allclose(
            PiezoForceDistance(calibration, baseline1, baseline2).valid_range(), reference
        )

    data = piezo_tracking_test_data
    dist_calib = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])
    slice1, slice2 = make_slice(0, 8), make_slice(5, 10)
    b1 = ForceBaseLine.polynomial_baseline(slice1, slice1, degree=1)
    b2 = ForceBaseLine.polynomial_baseline(slice2, slice2, degree=1)

    check_range(dist_calib, b1, None, [0, 7])
    check_range(dist_calib, None, b2, [5, 9])
    check_range(dist_calib, None, None, [-np.inf, np.inf])
    check_range(dist_calib, b1, b2, [5, 7])
    check_range(dist_calib, b2, b1, [5, 7])


def test_piezo_tracking_type_validation(poly_baseline_data, camera_calibration_data):
    baseline_trap_position, baseline_force = poly_baseline_data
    camera_dist, _ = camera_calibration_data
    baseline = ForceBaseLine.polynomial_baseline(baseline_trap_position, baseline_force, 1)
    distance = DistanceCalibration(baseline_trap_position, camera_dist)

    with pytest.raises(
        TypeError,
        match="Expected DistanceCalibration for the first argument, got ForceBaseLine",
    ):
        PiezoForceDistance(baseline, distance, baseline)

    with pytest.raises(
        TypeError,
        match="Expected ForceBaseLine for the second argument, got DistanceCalibration",
    ):
        PiezoForceDistance(distance, distance, baseline)

    with pytest.raises(
        TypeError,
        match="Expected ForceBaseLine for the second argument, got int",
    ):
        PiezoForceDistance(distance, 5, baseline)


def test_downsampling_tracking(piezo_tracking_test_data):
    data = piezo_tracking_test_data

    # Calibrate using the trap position
    distance_calibration = DistanceCalibration(data["baseline_trap_position"], data["camera_dist"])
    baseline = ForceBaseLine.polynomial_baseline(
        data["baseline_trap_position"], data["baseline_force"], degree=2
    )

    fd_generator = PiezoForceDistance(distance_calibration, baseline, baseline)
    piezo_distance, corrected_force1, corrected_force2 = fd_generator.force_distance(
        data["trap_position"],
        data["force_1x"],
        data["force_2x"],
        trim=False,
        downsampling_factor=10,
    )

    np.testing.assert_allclose(
        piezo_distance.data, downsample(data["correct_distance"], 10, np.mean), rtol=1e-6
    )
    np.testing.assert_allclose(
        corrected_force1.data, downsample(data["force_without_baseline"], 10, np.mean), rtol=1e-3
    )
