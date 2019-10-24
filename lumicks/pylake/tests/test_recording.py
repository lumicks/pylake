import numpy as np
import pytest
from lumicks.pylake.recording import Recording, TiffStack
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


def test_recording():
    fake_tiff = TiffStack(MockTiff([["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]))
    recording = Recording(fake_tiff)

    assert (recording[0].get_frame().start == 10)
    assert (recording[1].get_frame().start == 20)
    assert (recording[-1].get_frame().start == 60)
    assert (recording[0].num_frames == 1)

    assert (recording[0].get_frame().stop == 18)
    assert (recording[-1].get_frame().stop == 68)

    assert (recording[1:2].get_frame().stop == 28)
    assert (recording[1:3].get_frame().stop == 28)
    assert (recording[1:2].num_frames == 1)
    assert (recording[1:3].num_frames == 2)

    assert (recording[3:5][0].get_frame().start == 40)
    assert (recording[3:5][1].get_frame().start == 50)
    assert (recording[3:5][0].num_frames == 1)

    with pytest.raises(IndexError):
        recording[3:5][2]

    assert(recording[2:5][3:5].num_frames == 0)
    assert(recording[2:5][1:2].get_frame().start == 40)
    assert(recording[2:5][1:3].get_frame(1).start == 50)

    with pytest.raises(IndexError):
        recording[1:2].get_frame(1).stop


def test_correlation():
    cc = channel.Slice(channel.Continuous(np.arange(10, 80, 2), 10, 2))
    fake_tiff = TiffStack(MockTiff([["10", "18"], ["20", "28"], ["30", "38"], ["40", "48"], ["50", "58"], ["60", "68"]]))
    recording = Recording(fake_tiff)

    # Integration test whether slicing from the recording object actually provides you with correct slices
    assert(np.allclose(cc[recording[2:5].time_slice].timestamps, np.arange(30, 58, 2)))
    assert(np.allclose(cc[recording[0].time_slice].timestamps, np.arange(10, 18, 2)))
    assert(np.allclose(cc[recording[2:5][0].time_slice].timestamps, np.arange(30, 38, 2)))
    assert(np.allclose(cc[recording[2:5][1].time_slice].timestamps, np.arange(40, 48, 2)))

    # Unit test which tests whether we obtain an appropriately downsampled time series when ask for downsampling of a
    # slice based on a recording.
    ch = recording[0:3].downsample_channel(cc)
    assert(np.allclose(ch.data, [np.mean(np.arange(10, 18, 2)), np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2))]))
    assert (np.allclose(ch.timestamps, [(10 + 18) / 2, (20 + 28) / 2, (30 + 38) / 2]))

    ch = recording[1:4].downsample_channel(cc)
    assert (np.allclose(ch.data, [np.mean(np.arange(20, 28, 2)), np.mean(np.arange(30, 38, 2)), np.mean(np.arange(40, 48, 2))]))
    assert (np.allclose(ch.timestamps, [(20 + 28) / 2, (30 + 38) / 2, (40 + 48) / 2]))