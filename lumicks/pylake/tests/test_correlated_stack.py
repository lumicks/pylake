import numpy as np
import pytest
from lumicks.pylake.correlated_stack import CorrelatedStack, TiffStack
from lumicks.pylake import channel


# Mock Camera TIFF file
class TiffPage:
    def __init__(self, start_time, end_time):
        class Tag:
            @property
            def value(self):
                return f"{start_time}:{end_time}"

        self.tags = {"DateTime": Tag()}


class MockTiff:
    def __init__(self, times):
        self.pages = []
        for r in times:
            self.pages.append(TiffPage(r[0], r[1]))

    @property
    def num_frames(self):
        return len(self._src.pages)


def test_correlated_stack():
    fake_tiff = TiffStack(MockTiff([["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]))
    stack = CorrelatedStack.from_data(fake_tiff)

    assert (stack[0].start == 10)
    assert (stack[1].start == 20)
    assert (stack[-1].start == 60)
    assert (stack[0].num_frames == 1)

    assert (stack[0].stop == 18)
    assert (stack[-1].stop == 68)

    assert (stack[1:2].stop == 28)
    assert (stack[1:3].stop == 38)
    assert (stack[1:2].num_frames == 1)
    assert (stack[1:3].num_frames == 2)

    assert (stack[3:5][0].start == 40)
    assert (stack[3:5][1].start == 50)
    assert (stack[3:5][0].num_frames == 1)

    with pytest.raises(IndexError):
        stack[3:5][2]

    assert(stack[2:5][3:5].num_frames == 0)
    assert(stack[2:5][1:2].start == 40)
    assert(stack[2:5][1:3]._get_frame(1).start == 50)

    with pytest.raises(IndexError):
        stack[::2]

    with pytest.raises(IndexError):
        stack[1:2]._get_frame(1).stop

    # Integration test whether slicing from the stack object actually provides you with correct slices
    assert (np.allclose(stack[2:5].start, 30))
    assert (np.allclose(stack[2:5].stop, 58))

    # Test iterations
    assert (np.allclose([x.start for x in stack], [10, 20, 30, 40, 50, 60]))
    assert (np.allclose([x.start for x in stack[1:]], [20, 30, 40, 50, 60]))
    assert (np.allclose([x.start for x in stack[:-1]], [10, 20, 30, 40, 50]))
    assert (np.allclose([x.start for x in stack[2:4]], [30, 40]))
    assert (np.allclose([x.start for x in stack[2]], [30]))


def test_correlation():
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))

    # Test image stack without dead time
    fake_tiff = TiffStack(MockTiff([["10", "20"], ["20", "30"], ["30", "40"], ["40", "50"], ["50", "60"], ["60", "70"]]))
    stack = CorrelatedStack.from_data(fake_tiff)
    assert (np.allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]), np.arange(30, 50, 2)))

    # Test image stack with dead time
    fake_tiff = TiffStack(MockTiff([["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]))
    stack = CorrelatedStack.from_data(fake_tiff)

    assert (np.allclose(np.hstack([cc[x.start:x.stop].data for x in stack[2:4]]),
                        np.hstack([np.arange(30, 38, 2), np.arange(40, 48, 2)])))

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a stack.
    ch = cc.downsampled_over(stack[0:3].timestamps)
    assert(np.allclose(ch.data, [np.mean(np.arange(10, 18, 2)), np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2))]))
    assert (np.allclose(ch.timestamps, [(10 + 18) / 2, (20 + 28) / 2, (30 + 38) / 2]))

    ch = cc.downsampled_over(stack[1:4].timestamps)
    assert (np.allclose(ch.data, [np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2)), np.mean(np.arange(40, 48, 2))]))
    assert (np.allclose(ch.timestamps, [(20 + 28) / 2, (30 + 38) / 2, (40 + 48) / 2]))

    with pytest.raises(TypeError):
        cc.downsampled_over(stack[1:4])

    with pytest.raises(ValueError):
        cc.downsampled_over(stack[1:4].timestamps, where='up')

    with pytest.raises(AssertionError):
        cc["0ns":"20ns"].downsampled_over(stack[3:4].timestamps)

    with pytest.raises(AssertionError):
        cc["40ns":"70ns"].downsampled_over(stack[0:1].timestamps)

    assert (stack[0].raw.start == 10)
    assert (stack[1].raw.start == 20)
    assert (stack[1:3][0].raw.start == 20)
    assert (stack[1:3].raw[0].start == 20)
    assert (stack[1:3].raw[1].start == 30)

    # Regression test downsampled_over losing precision due to reverting to double rather than int64.
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 1588267266006287100, 1000))
    ch = cc.downsampled_over([(1588267266006287100, 2588267266006287100)], where='left')
    assert int(ch.timestamps[0]) == 1588267266006287100
