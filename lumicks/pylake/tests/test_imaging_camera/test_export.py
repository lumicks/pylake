import itertools

import numpy as np
import pytest
import tifffile

from lumicks.pylake import ImageStack
from lumicks.pylake.channel import Slice, Continuous


def test_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    filenames = (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi)
    for filename, align in itertools.product(filenames, (True, False)):
        savename = str(filename.new(purebasename=f"out_{filename.purebasename}"))
        stack = ImageStack(str(filename), align=align)
        stack.export_tiff(savename)
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(str(filename)) as tif_in, tifffile.TiffFile(savename) as tif_out:
            # Check `_tiff_frames()` and `_tiff_timestamp_ranges()`:
            assert len(tif_in.pages) == len(tif_out.pages)
            # Check `_tiff_writer_kwargs()`:
            assert (
                tif_in.pages[0].software in tif_out.pages[0].software
                and tif_in.pages[0].software != tif_out.pages[0].software
            )
            assert "Pylake" in tif_out.pages[0].software
            # Check `_tiff_image_metadata()`:
            if stack._get_frame(0).is_rgb:
                if stack._get_frame(0)._is_aligned:
                    assert "Applied channel 0 alignment" in tif_out.pages[0].description
                    assert "Channel 0 alignment" not in tif_out.pages[0].description
                else:
                    assert "Applied channel 0 alignment" not in tif_out.pages[0].description
                    assert "Channel 0 alignment" in tif_out.pages[0].description
            # Check `_tiff_timestamp_ranges()`
            for page0, page in zip(tif_in.pages, tif_out.pages):
                assert page0.tags["DateTime"].value == page.tags["DateTime"].value

        stack.close()


def test_export_roi(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = ImageStack(str(filename))[:, 20:80, 10:190]

        stack.export_tiff(savename)
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60

        stack.close()


@pytest.mark.parametrize("vertical, correlated", [(False, False), (False, True), (True, True)])
def test_stack_movie_export(
    tmpdir_factory,
    rgb_tiff_file,
    rgb_tiff_file_multi,
    gray_tiff_file,
    gray_tiff_file_multi,
    vertical,
    correlated,
):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    for idx, filename in enumerate((rgb_tiff_file_multi, gray_tiff_file_multi)):
        stack = ImageStack(str(filename))

        dt_stack = stack.frame_timestamp_ranges()[1][0] - stack.frame_timestamp_ranges()[0][0]
        corr_data = (
            Slice(
                Continuous(np.arange(100), stack.start, dt_stack // 2),
                labels={"title": "title", "y": "y"},
            )
            if correlated
            else None
        )

        fn = f"{tmpdir}/cstack{idx}.gif"
        stack.export_video(
            "red", fn, start_frame=0, stop_frame=2, channel_slice=corr_data, vertical=vertical
        )
        assert stat(fn).st_size > 0

        with pytest.raises(
            ValueError,
            match=(
                "channel must be 'red', 'green', 'blue' or a combination of 'r', 'g', "
                "and/or 'b', got 'gray'."
            ),
        ):
            stack.export_video("gray", "dummy.gif")  # Gray is not a color!

        stack.close()


@pytest.mark.parametrize(
    "start, stop, ref, correlated",
    [
        (None, None, [0, 1], True),
        (1, None, [1], True),
        (0, None, [0, 1], True),
        (0, 2, [0, 1], True),
        (None, 1, [0], True),
        (None, None, [0, 1], False),
        (1, None, [1], False),
        (0, None, [0, 1], False),
        (0, 2, [0, 1], False),
        (None, 1, [0], False),
    ],
)
def test_movie_export_range(
    monkeypatch, tmpdir_factory, rgb_tiff_file_multi, start, stop, ref, correlated
):
    tmpdir = tmpdir_factory.mktemp("pylake")
    stack = ImageStack(str(rgb_tiff_file_multi))

    dt_stack = stack.frame_timestamp_ranges()[1][0] - stack.frame_timestamp_ranges()[0][0]
    corr_data = (
        Slice(
            Continuous(np.arange(100), stack.start, dt_stack // 2),
            labels={"title": "title", "y": "y"},
        )
        if correlated
        else None
    )

    fn = f"{tmpdir}/test.gif"

    with monkeypatch.context() as m:
        frames = []

        if correlated:
            # Plot correlated returns a frame setter that export uses to change the frame
            def faux_set_frame(frame):
                frames.append(frame)

            def faux_plot_correlated(*_, **__):
                return faux_set_frame

            m.setattr("lumicks.pylake.image_stack.ImageStack.plot_correlated", faux_plot_correlated)
        else:

            def faux_plot(*_, frame, **__):
                frames.append(frame)

            m.setattr("lumicks.pylake.image_stack.ImageStack.plot", faux_plot)

        stack.export_video("red", fn, start_frame=start, stop_frame=stop, channel_slice=corr_data)
        stack.close()

        # We take the unique frames, since plots are called a few times to "prepare" for rendering
        np.testing.assert_equal(np.unique(frames), ref)
