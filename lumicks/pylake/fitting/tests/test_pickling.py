import pickle

import numpy as np

from lumicks.pylake.fitting.fit import FdFit
from lumicks.pylake.fitting.models import ewlc_odijk_force


def test_pickle(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pylake")

    model = ewlc_odijk_force("DNA")
    fit = FdFit(model)
    x = np.arange(1, 20, 5)
    y = model(np.arange(1, 20, 5), params={"DNA/Lp": 50, "DNA/Lc": 24, "DNA/St": 1000, "kT": 4.12})
    fit._add_data("test x", x, y)
    fit["kT"].value = 4.12

    fn = f"{tmpdir}/test.pkl"
    with open(fn, "wb") as f:
        pickle.dump(fit, f)
        pickle.dump(model, f)

    # Verify that we can open the model again
    with open(fn, "rb") as f:
        pickled_fit = pickle.load(f)
        pickled_model = pickle.load(f)

    # Test whether we can still fit
    pickled_fit.fit()

    # Test whether data is there and whether we can slice by the pickled model (should have its
    # UUID stored).
    np.testing.assert_allclose(x, fit[pickled_model].data["test x"].x)
    np.testing.assert_allclose(y, fit[pickled_model].data["test x"].y)

    # Verify fit result
    np.testing.assert_allclose(
        [x.value for x in dict(pickled_fit.params).values()],
        [49.996726625805394, 24.00234442122973, 1415.93306, 4.12],
    )
