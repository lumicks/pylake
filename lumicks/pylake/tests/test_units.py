import pytest
from lumicks.pylake.detail.units import determine_unit, dimensionless


def test_determine_unit():
    assert determine_unit("pN", "pN", "add") == "pN"
    assert determine_unit("pN", "pN", "sub") == "pN"
    assert determine_unit("pN", "pN", "div") == dimensionless
    with pytest.raises(NotImplementedError):
        assert determine_unit("pN", "pN", "mul")

    with pytest.raises(TypeError):
        determine_unit("pN", "apples", "add")
    with pytest.raises(TypeError):
        determine_unit("pN", "apples", "sub")
    with pytest.raises(NotImplementedError):
        determine_unit("pN", "apples", "div")
    with pytest.raises(NotImplementedError):
        determine_unit("pN", "apples", "mul")

    with pytest.raises(TypeError):
        determine_unit("apples", "pN", "add")
    with pytest.raises(TypeError):
        determine_unit("apples", "pN", "sub")
    with pytest.raises(NotImplementedError):
        determine_unit("apples", "pN", "div")
    with pytest.raises(NotImplementedError):
        determine_unit("apples", "pN", "mul")

    with pytest.raises(TypeError):
        determine_unit("pN", dimensionless, "add")
    with pytest.raises(TypeError):
        determine_unit("pN", dimensionless, "sub")
    assert determine_unit("pN", dimensionless, "div") == "pN"
    assert determine_unit("pN", dimensionless, "mul") == "pN"

    with pytest.raises(TypeError):
        determine_unit(dimensionless, "pN", "add")
    with pytest.raises(TypeError):
        determine_unit(dimensionless, "pN", "sub")
    with pytest.raises(NotImplementedError):
        determine_unit(dimensionless, "pN", "div")
    assert determine_unit("pN", dimensionless, "mul") == "pN"

    assert determine_unit("pN", None, "add") == "pN"
    assert determine_unit("pN", None, "sub") == "pN"
    assert not determine_unit("pN", None, "div")
    assert not determine_unit("pN", None, "mul")

    assert determine_unit(None, "pN", "add") == "pN"
    assert determine_unit(None, "pN", "sub") == "pN"
    assert not determine_unit(None, "pN", "div")
    assert not determine_unit(None, "pN", "mul")
