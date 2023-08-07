import numpy as np
import pytest

from lumicks.pylake.channel import Slice, Continuous, TimeSeries
from lumicks.pylake.calibration import ForceCalibration
from lumicks.pylake.fitting.models import ewlc_odijk_force


def reference_baseline():
    return np.poly1d([22.58134638112967, -601.6397764983628, 4007.686647006411])


@pytest.fixture()
def poly_baseline_data():
    baseline_model = reference_baseline()

    # Baseline correction should be able to deal with non-equidistant arbitrarily sorted points
    # with duplicates.
    trap_position_baseline = np.hstack(
        (
            np.arange(13.35, 12.95, -0.0000025),
            np.arange(13.35, 13.25, -0.0000005),
            np.ones(100000) * 12.95,
        )
    )

    trap = Slice(
        Continuous(trap_position_baseline, 1573123558595351600, int(1e9 / 78125)),
        labels={"title": "Trap position", "y": "y"},
    )
    force = Slice(
        Continuous(
            baseline_model(trap_position_baseline),
            1573123558595351600,
            int(1e9 / 78125),
        ),
        labels={"title": "force", "y": "Force (pN)"},
    )

    return trap, force


@pytest.fixture()
def camera_calibration_data(poly_baseline_data):
    # The "true" camera distance is given by trap position - reference point.
    ds_factor = 10
    trap2_ref = 9.15
    baseline_trap_position, baseline_force = poly_baseline_data
    distance_ds = baseline_trap_position.downsampled_by(ds_factor) - trap2_ref
    old_dt = baseline_trap_position.timestamps[1] - baseline_trap_position.timestamps[0]
    return (
        Slice(TimeSeries(distance_ds.data, distance_ds.timestamps + (ds_factor // 2) * old_dt)),
        trap2_ref,
    )


@pytest.fixture()
def piezo_tracking_test_data(poly_baseline_data, camera_calibration_data):
    baseline = reference_baseline()
    baseline_trap_position, baseline_force = poly_baseline_data
    camera_dist, trap2_ref = camera_calibration_data

    # Tether experiment data
    sample_rate = 78
    dt = int(1e9 / sample_rate)
    tether_length_um = np.hstack(
        (np.arange(0.65, 0.7, 0.08 / sample_rate), np.arange(0.7, 0.785, 0.02 / sample_rate))
    )
    wlc_force = ewlc_odijk_force("tether")(
        tether_length_um, {"tether/Lp": 60, "tether/Lc": 0.75, "tether/St": 1400, "kT": 4.11}
    )

    stiffness = 0.15
    stiffness_um = stiffness * 1e3  # pN/um (0.15 pN/nm)

    """
    If we assume that the baseline force leads to a real displacement, then our function for the
    trap position becomes implicit, since the displacement depends on the baseline which in turn
    depends on the trap position:
    
        trap_trap_distance = tether_length + 2 * bead_radius + 2 * displacement_um
    
        And displacement_um is given by (wlc_force + baseline(trap_position)) / stiffness

    So we solve the following to obtain the trap position:
    
        displacement = 2 * (wlc_force + baseline(trap_position)) / stiffness
        0 = tether_length + 2 * bead_radius + displacement - (trap_position - trap2_ref)
    """
    from scipy.optimize import minimize_scalar

    bead_radius = 1  # 1 micron beads
    trap_position = []
    for tether_dist, force in zip(tether_length_um, wlc_force):

        def implicit_trap_position_equation(x):
            trap_trap_dist = x - trap2_ref
            displacement = 2 * (force + baseline(x)) / stiffness_um
            return (tether_dist + 2 * bead_radius + displacement - trap_trap_dist) ** 2

        trap_position.append(minimize_scalar(implicit_trap_position_equation, [12.95, 13.35]).x)

    trap_position = Slice(Continuous(np.array(trap_position), 0, dt))

    # Add our baseline force (assumption is that the baseline force leads to a real displacement)
    force_pn = wlc_force + baseline(trap_position.data)

    calibration = ForceCalibration(
        "Stop time (ns)", [{"Stop time (ns)": 1, "kappa (pN/nm)": stiffness}]
    )

    force_1x = Slice(Continuous(force_pn, 0, dt), calibration=calibration)
    force_2x = Slice(Continuous(-force_pn, 0, dt), calibration=calibration)

    return {
        "correct_distance": tether_length_um,
        "trap_position": trap_position,
        "force_without_baseline": wlc_force,
        "force_1x": force_1x,
        "force_2x": force_2x,
        "baseline_trap_position": baseline_trap_position,
        "baseline_force": baseline_force,
        "camera_dist": camera_dist - 2 * bead_radius,  # Bluelake subtracts the radii already
    }
