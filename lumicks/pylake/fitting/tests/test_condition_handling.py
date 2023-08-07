from collections import OrderedDict

import numpy as np
import pytest

from lumicks.pylake.fitting.fitdata import FitData, Condition
from lumicks.pylake.fitting.detail.utilities import parse_transformation
from lumicks.pylake.fitting.detail.link_functions import generate_conditions


def test_build_conditions():
    param_names = ["a", "b", "c"]
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    d1 = FitData("name1", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d2 = FitData("name2", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d3 = FitData("name3", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))

    assert generate_conditions(
        {"name1": d1, "name2": d2, "name3": d3}, parameter_lookup, param_names
    )

    # Tests whether we pick up when a parameter that's generated in a transformation doesn't
    # actually exist in the combined model
    d4 = FitData(
        "name4",
        [1, 2, 3],
        [1, 2, 3],
        parse_transformation(param_names, {"c": "i_should_not_exist"}),
    )
    with pytest.raises(
        RuntimeError,
        match="Parameter transformations of data_sets contain transformed parameter names that are "
        "not in the combined parameter list parameter_lookup",
    ):
        generate_conditions({"name1": d1, "name2": d2, "name4": d4}, parameter_lookup, param_names)

    # Tests whether we pick up on when a parameter exists in the model, but there's no
    # transformation for it.
    d5 = FitData("name5", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    param_names = ["a", "b", "c", "i_am_new"]
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    with pytest.raises(
        RuntimeError,
        match="Source parameters in the data parameter transformations of data_sets are "
        "incompatible with the specified model parameters in model_params",
    ):
        assert generate_conditions(
            {"name1": d1, "name2": d2, "name5": d5}, parameter_lookup, param_names
        )

    # Verify that the data gets linked up to the correct conditions
    d1 = FitData("name1", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d2 = FitData("name2", [1, 2, 3], [1, 2, 3], parse_transformation(param_names))
    d6 = FitData(
        "name6", [1, 2, 3], [1, 2, 3], parse_transformation(param_names, {"c": "i_am_new"})
    )
    conditions, data_link = generate_conditions(
        {"name1": d1, "name2": d2, "name3": d6}, parameter_lookup, param_names
    )
    assert np.all(data_link[0] == [d1, d2])
    assert np.all(data_link[1] == [d6])

    # Test whether a parameter transformation to a value doesn't lead to an error
    d4 = FitData("name4", [1, 2, 3], [1, 2, 3], parse_transformation(param_names, {"c": 5}))
    assert generate_conditions(
        {"name1": d1, "name2": d2, "name3": d4}, parameter_lookup, param_names
    )


def test_condition_struct():
    param_names = ["gamma", "alpha", "beta", "delta", "gamma_specific", "beta_specific", "zeta"]
    parameter_lookup = OrderedDict(zip(param_names, np.arange(len(param_names))))
    param_trafos = parse_transformation(
        ["gamma", "alpha", "beta", "delta", "zeta"],
        {"gamma": "gamma_specific", "delta": 5, "beta": "beta_specific"},
    )
    param_vector = np.array([2, 4, 6, 8, 10, 12, 14])

    c = Condition(param_trafos, parameter_lookup)
    assert np.all(c.p_local == [None, None, None, 5, None])
    np.testing.assert_allclose(param_vector[c.p_indices], [10, 4, 12, 14])
    assert np.all(c.p_external == np.array([0, 1, 2, 4]))
    assert list(c.transformed) == ["gamma_specific", "alpha", "beta_specific", 5, "zeta"]
    np.testing.assert_allclose(c.get_local_params(param_vector), [10, 4, 12, 5, 14])
