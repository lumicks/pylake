import itertools

import pytest
import tifffile

from lumicks.pylake import ImageStack


def test_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    filenames = (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi)
    for filename, align in itertools.product(filenames, (True, False)):
        savename = str(filename.new(purebasename=f"out_{filename.purebasename}"))
        stack = ImageStack(str(filename), align=align)
        stack.export_tiff(savename)
        stack._src.close()
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

        stack._src.close()


def test_stack_movie_export(
    tmpdir_factory, rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi
):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    for idx, filename in enumerate((rgb_tiff_file_multi, gray_tiff_file_multi)):
        stack = ImageStack(str(filename))
        fn = f"{tmpdir}/cstack{idx}.gif"
        stack.export_video("red", fn, start_frame=0, stop_frame=2)
        assert stat(fn).st_size > 0

        with pytest.raises(ValueError, match="Channel should be red, green, blue or rgb"):
            stack.export_video("gray", "dummy.gif")  # Gray is not a color!

        stack._src.close()
