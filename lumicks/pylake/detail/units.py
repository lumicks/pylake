
dimensionless = "dimensionless"


def determine_unit(unit1, unit2, operation):
    """Compares units. Raises exception when units are not easily converted or incompatible. None is considered an
    unknown unit.

    Parameters
    ----------
    unit1, unit2 : str
        A unit specification such as pN.
    operation: str
        Operation to be performed on the unit.

    Returns
    -------
    str
        A unit specification such as pN.
    """
    if operation == "add" or operation == "sub":
        if unit1 == unit2:
            return unit1
        # Since addition and subtraction only make sense for compatible units, it is assumed that operations using
        # data types which have no units associated with them result in the unit which is already known.
        elif not unit1:
            return unit2
        elif not unit2:
            return unit1
        else:
            raise TypeError(f"Addition / Subtraction not valid between different units {unit1} and {unit2}")
    elif operation == "div":
        if unit2 == dimensionless:
            return unit1
        elif unit1 == unit2:
            return dimensionless
        elif not unit1 or not unit2:
            return None
        else:
            raise NotImplementedError(f"Division not implemented between units {unit1} and {unit2}")
    elif operation == "mul":
        if unit2 == dimensionless:
            return unit1
        elif unit1 == dimensionless:
            return unit2
        elif not unit1 or not unit2:
            return None
        else:
            raise NotImplementedError(f"Multiplication not implemented between units {unit1} and {unit2}")

    raise RuntimeError(f"Unknown operation {operation} between {unit1} and {unit2}")
