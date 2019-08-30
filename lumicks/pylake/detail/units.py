
dimensionless = "dimensionless"


def determine_unit(unit1, unit2, operation):
    """Determines the return unit of an operation. Raises exception when units are not easily converted or incompatible.

    None is considered an unknown unit. Data types with unknown units will default to unit type None. For addition and
    subtraction the unit type of None can be inferred as being the unit type of the other operand (as this is the only
    valid option). For other operations, the unit cannot be inferred when one of the operands is unknown. As such,
    these operands will return None (unknown unit).

    Dimensionless is a special unit that indicates that a property has no units. For dimensionless quantities the rules
    are clear as it just behaves like a regular unit. This special case allows us to incorporate some additional options
    for operands, but will become superfluous if an external unit bookkeeping library is used.

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
