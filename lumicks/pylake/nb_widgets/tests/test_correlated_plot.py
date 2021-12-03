from lumicks.pylake.tests.data.mock_confocal import generate_scan
from lumicks.pylake.channel import Slice, Continuous
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.testing.decorators import cleanup


@cleanup
def test_plot_correlated():
    start = 1592916040906356300
    dt = int(1e9)
    cc = Slice(Continuous(np.arange(0, 1000, 2), start, dt), {"y": "mock", "title": "mock"})

    img = np.array([np.ones((5, 4)) * idx for idx in np.arange(3)])
    scan = generate_scan(
        "test",
        img,
        [1, 1],
        start=start,
        dt=dt,
        samples_per_pixel=4,
        line_padding=4,
    )

    scan.plot_correlated(cc, channel="red")
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), np.zeros((5, 4)))

    scan.plot_correlated(cc, channel="red", frame=1)
    imgs = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.image.AxesImage)]
    np.testing.assert_allclose(imgs[0].get_array(), np.ones((5, 4)))

    # When no data overlaps, we need to raise.
    with pytest.raises(AssertionError, match="No overlap between range and selected channel"):
        scan.plot_correlated(cc["500s":], channel="red", frame=1)
