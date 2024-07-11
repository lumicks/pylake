import math

import numpy as np


class ValueMixin:
    """Mixin class to emulate numeric types. Requires a `value` attribute on the subclass.

    Follows the description for Emulating Numeric Types:
    (https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)

    The returned type is always a primitive type, since the metadata stored in subclasses of this
    mixin cannot be guaranteed to still be accurate after any operation on the value.

    Note: cannot be used in combination with dataclass, which overrides the  `==` operator.

    Operators will work for `str` and `bool` types stored in the `value` attribute;
    however full behavior/methods for these types are not tested or guaranteed to work as expected.
    """

    def __init__(self, value):
        self.value = value

    @staticmethod
    def extract_value(method):
        def wrapper(self, other):
            if isinstance(other, ValueMixin):
                other = other.value

            return method(self, other)

        return wrapper

    @extract_value
    def __add__(self, other):
        return self.value + other

    @extract_value
    def __sub__(self, other):
        return self.value - other

    @extract_value
    def __mul__(self, other):
        return self.value * other

    @extract_value
    def __truediv__(self, other):
        return self.value / other

    @extract_value
    def __floordiv__(self, other):
        return self.value // other

    @extract_value
    def __mod__(self, other):
        return self.value % other

    @extract_value
    def __divmod__(self, other):
        return divmod(self.value, other)

    @extract_value
    def __pow__(self, other):
        return self.value**other

    @extract_value
    def __and__(self, other):
        return self.value & other

    @extract_value
    def __xor__(self, other):
        return self.value ^ other

    @extract_value
    def __or__(self, other):
        return self.value | other

    @extract_value
    def __radd__(self, other):
        return other + self.value

    @extract_value
    def __rsub__(self, other):
        return other - self.value

    @extract_value
    def __rmul__(self, other):
        return other * self.value

    @extract_value
    def __rtruediv__(self, other):
        return other / self.value

    @extract_value
    def __rfloordiv__(self, other):
        return other // self.value

    @extract_value
    def __rmod__(self, other):
        return other % self.value

    @extract_value
    def __rdivmod__(self, other):
        return divmod(other, self.value)

    @extract_value
    def __rpow__(self, other):
        return other**self.value

    @extract_value
    def __rand__(self, other):
        return other & self.value

    @extract_value
    def __rxor__(self, other):
        return other ^ self.value

    @extract_value
    def __ror__(self, other):
        return other | self.value

    def __array__(self, *args, **kwargs):
        return np.array(self.value, *args, **kwargs)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __abs__(self):
        return abs(self.value)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __trunc__(self):
        return math.trunc(self.value)

    def __ceil__(self):
        return math.ceil(self.value)

    def __floor__(self):
        return math.floor(self.value)

    @extract_value
    def __eq__(self, other):
        return self.value == other

    @extract_value
    def __lt__(self, other):
        return self.value < other

    @extract_value
    def __gt__(self, other):
        return self.value > other

    @extract_value
    def __le__(self, other):
        return self.value <= other

    @extract_value
    def __ge__(self, other):
        return self.value >= other
