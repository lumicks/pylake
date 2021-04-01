import numpy as np
import json


def generate_scan_json(axes):
    """Generate a mock JSON for a Scan or Kymo.

    Parameters
    ----------
    axes : List[Dict]
        List of dictionaries with an element for each axis. These dictionaries need the following
        fields:
        "axis" : int
            Axis order.
        "num of pixels" : int
            Number of pixels along this axis.
        "pixel size (nm)" : float
            Pixel size along this axis.
    """
    enc = json.JSONEncoder()

    axes_metadata = [
        {
            "axis": axis["axis"],
            "cereal_class_version": 1,
            "num of pixels": axis["num of pixels"],
            "pixel size (nm)": axis["pixel size (nm)"],
            "scan time (ms)": 0,
            "scan width (um)": axis["pixel size (nm)"] * axis["num of pixels"] / 1000.0
            + 0.5,
        }
        for axis in axes
    ]

    return enc.encode(
        {
            "value0": {
                "cereal_class_version": 1,
                "fluorescence": True,
                "force": False,
                "scan count": 0,
                "scan volume": {
                    "center point (um)": {"x": 58.075877109272604, "y": 31.978375270573267, "z": 0},
                    "cereal_class_version": 1,
                    "pixel time (ms)": 0.2,
                    "scan axes": axes_metadata,
                },
            }
        }
    )
