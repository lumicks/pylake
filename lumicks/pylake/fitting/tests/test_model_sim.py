import pytest
import numpy as np
from lumicks.pylake.fitting.models import odijk
from lumicks.pylake.fitting.parameters import Params, Parameter


def test_simulation_api():
    dna = odijk("DNA")
    force = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(
        dna(force, {"DNA/Lp": 50.0, "DNA/Lc": 16.0, "DNA/St": 1500.0, "kT": 4.11}),
        [8.74792941, 10.8733908, 11.81559925],
    )

    np.testing.assert_allclose(
        dna(
            [0.1, 0.2, 0.3],
            Params(
                **{
                    "DNA/Lp": Parameter(50.0),
                    "DNA/Lc": Parameter(16.0),
                    "DNA/St": Parameter(1500.0),
                    "kT": Parameter(4.11),
                }
            ),
        ),
        [8.74792941, 10.8733908, 11.81559925],
    )


def test_simulation_api_wrong_par():
    dna = odijk("DNA")

    with pytest.raises(KeyError):
        dna([1], {"DNA/Lp": 50.0, "DNA/Lc": 16.0, "DN/St": 1500.0, "kT": 4.11})
