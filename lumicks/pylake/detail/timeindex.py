import re

__all__ = ["Timeindex", "to_timestamp"]

# It's impossible to see, but this regex matches a floating point number and suffix
regex_template = r"((?P<{suffix}>\d*\.?\d+)\s*{suffix})?"

# Day, hour, minute, second, millisecond, microsecond, nanosecond
units = ["d", "h", "m", "s", "ms", "us", "ns"]

# Matches strings like "1s 216ms", "-1m 30s", "-1.4s", "2.7h"
regex = re.compile(
    "^(?P<sign>-?)" + r"\s*".join(regex_template.format(suffix=u) for u in units) + "$"
)

ns = 1
us = 1000 * ns
ms = 1000 * us
s = 1000 * ms
m = 60 * s
h = 60 * m
d = 24 * h

# Ratio of nanosecond to `unit`
ratios = {unit: globals()[unit] for unit in units}


class Timeindex:
    """A helper class for indexing in time units

    It translates strings like "1.4s", "-2s 36ms", "1h 7m" to nanosecond timestamps.

    Parameters
    ----------
    timestring : str

    """
    def __init__(self, timestring):
        match = regex.match(timestring)
        if not match:
            raise RuntimeError(f"Invalid time string '{timestring}'")

        self.sign = -1 if match.group("sign") else 1

        self.d = 0
        self.h = 0
        self.m = 0
        self.s = 0
        self.ms = 0
        self.us = 0
        self.ns = 0
        unit_matches = {unit: match.group(unit) for unit in units}
        self.__dict__.update({unit: float(value) for unit, value in unit_matches.items() if value})

        self.total_ns = self.sign * sum(int(self.__dict__[unit] * ratios[unit]) for unit in units)

    def __int__(self):
        return self.total_ns


def to_timestamp(value, first, after_last):
    """Convert `value` to a timestamp (ns) or return it unchanged if it's already a timestamp

    Parameters
    ----------
    value : Union[str, int]
        Either a timestring to be processed by `Timeindex` or a nanosecond timestamp
    first, after_last : int
        Nanosecond timestamps of the first and past-the-end elements of a range
    """
    try:
        idx = Timeindex(value).total_ns
        if idx >= 0:
            return first + idx
        else:
            return after_last + idx
    except TypeError:
        return value
