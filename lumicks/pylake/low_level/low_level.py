import numpy as np

from lumicks.pylake.kymo import Kymo
from lumicks.pylake.scan import Scan
from lumicks.pylake.channel import Slice, Continuous, empty_slice
from lumicks.pylake.point_scan import PointScan
from lumicks.pylake.detail.confocal import ScanMetaData, ConfocalFileProxy
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP


def create_confocal_object(
    name,
    infowave,
    json_metadata,
    red_channel=empty_slice,
    green_channel=empty_slice,
    blue_channel=empty_slice,
) -> Kymo | Scan | PointScan:
    """Create a confocal object from slices and json metadata

    Parameters
    ----------
    name : str
        Name of this object
    infowave : lumicks.pylake.Slice
        Info wave that encodes how the photon counts should be assembled into an image.
    json_metadata : str
        json metadata generated by Bluelake.
    red_channel, green_channel, blue_channel : lumicks.pylake.Slice
        Photon counts
    """
    metadata = ScanMetaData.from_json(json_metadata)
    file = ConfocalFileProxy(infowave, red_channel, green_channel, blue_channel)
    confocal_cls = {0: PointScan, 1: Kymo, 2: Scan}
    return confocal_cls[metadata.num_axes](
        name, file, infowave.start, infowave.stop, metadata, location=None
    )


def make_continuous_slice(data, start, dt, y_label="y", name="") -> Slice:
    """Make a continuous slice of data

    Converts a raw `array_like` to a pylake `Slice`.

    Parameters
    ----------
    data : array_like
        Source of data
    start : int
        Start timestamp in nanoseconds since epoch
    dt : int
        Timestep in nanoseconds
    y_label : str
        Label to show on the y-axis.
    name : str
        Name of the slice (used on plot titles).
    """
    if start < _FIRST_TIMESTAMP:
        raise ValueError(
            f"Starting timestamp must be larger than {_FIRST_TIMESTAMP}. You provided: {start}."
        )

    return Slice(Continuous(np.asarray(data), start, dt), labels={"title": name, "y": y_label})
