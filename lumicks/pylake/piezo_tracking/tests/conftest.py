import pytest
import numpy as np
from lumicks.pylake.channel import Continuous, Slice


@pytest.fixture()
def poly_baseline_data():
    true_coefficients = [22.58134638112967, -601.6397764983628, 4007.686647006411]
    baseline_model = np.poly1d(true_coefficients)

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
