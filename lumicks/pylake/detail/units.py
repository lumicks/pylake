
dimensionless = "dimensionless"


def determine_unit(unit1, unit2, operation):
    """Determines the return unit of an operation. Raises exception when units are not easily converted or incompatible.

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
        else:
            raise TypeError(f"Addition / Subtraction not valid between different units {unit1} and {unit2}")
    elif operation == "div":
        if unit2 == dimensionless:
            return unit1
        elif unit1 == unit2:
            return dimensionless
        else:
            raise NotImplementedError(f"Division not implemented between units {unit1} and {unit2}")
    elif operation == "mul":
        if unit2 == dimensionless:
            return unit1
        elif unit1 == dimensionless:
            return unit2
        else:
            raise NotImplementedError(f"Multiplication not implemented between units {unit1} and {unit2}")

    raise RuntimeError(f"Unknown operation {operation} between {unit1} and {unit2}")


class WithUnit:
    def __init__(self, src, unit):
        """
        Minimal wrapper to wrap a class with a unit.

        Parameters
        ----------
        src: object to be wrapped
            A numerical class which needs to be endowed with a unit.
        unit: str
            Unit to attach to this class.
        """
        self._src = src
        self.unit = unit

    @staticmethod
    def _unpack_other(other):
        """Unpack raw data from wrapper

        Parameters
        ----------
        other: array_like or wrapper class with _src or _src.data
            A numerical object on which an operation can be performed

        Returns
        -------
        array_like
        """
        if isinstance(other, WithUnit):
            if hasattr(other._src, "data"):
                return other._src.data
            else:
                return other._src
        else:
            return other

    @staticmethod
    def _determine_unit(lhs, rhs, operation):
        """Attempt to determine the unit after an operation

        Parameters
        ----------
        lhs: array_like or class derived off of WithUnit
            A numerical object on which an operation can be performed
        rhs: array_like or class derived off of WithUnit
            A numerical object on which an operation can be performed
        operation: str
            A string referring to the operation type

        Returns
        -------
        str
        """
        lhs_unit = getattr(lhs, "unit", dimensionless)
        rhs_unit = getattr(rhs, "unit", dimensionless)
        return determine_unit(lhs_unit, rhs_unit, operation)
