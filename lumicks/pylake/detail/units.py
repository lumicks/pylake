
au = "dimensionless"


def determine_unit(unit1, unit2, operation):
    """Compares units. Raises exception when units are not easily converted or incompatible.

    Parameters
    ----------
    unit1 : string
        A unit specification such as pN.
    unit2 : string
        A unit specification such as pN.
    operation: string
        Operation to be performed on the unit.
    """
    if operation == "add" or operation == "sub":
        if unit1 == unit2:
            return unit1
        else:
            raise TypeError(f"Addition / Subtraction not valid between different units {unit1} and {unit2}")
    elif operation == "div":
        if unit2 == au:
            return unit1
        elif unit1 == unit2:
            return au
        else:
            raise NotImplementedError(f"Division not implemented between units {unit1} and {unit2}")
    elif operation == "mul":
        if unit2 == au:
            return unit1
        elif unit1 == au:
            return unit2
        else:
            raise NotImplementedError(f"Multiplication not implemented between units {unit1} and {unit2}")

    raise RuntimeError(f"Unknown operation {operation} between {unit1} and {unit2}")
