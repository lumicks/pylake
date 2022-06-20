import pytest
import tifffile
import itertools
from lumicks.pylake import CorrelatedStack


def test_export(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    filenames = (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi)
    for filename, align in itertools.product(filenames, (True, False)):
        savename = str(filename.new(purebasename=f"out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename), align)
        stack.export_tiff(savename)
        stack.src.close()
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(str(filename)) as tif0, tifffile.TiffFile(savename) as tif:
            assert len(tif0.pages) == len(tif.pages)
            assert tif0.pages[0].software != tif.pages[0].software
            assert "pylake" in tif.pages[0].software
            if stack._get_frame(0).is_rgb:
                if stack._get_frame(0)._is_aligned:
                    assert "Applied channel 0 alignment" in tif.pages[0].description
                    assert "Channel 0 alignment" not in tif.pages[0].description
                else:
                    assert "Applied channel 0 alignment" not in tif.pages[0].description
                    assert "Channel 0 alignment" in tif.pages[0].description
            for page0, page in zip(tif0.pages, tif.pages):
                assert page0.tags["DateTime"].value == page.tags["DateTime"].value


def test_export_roi(rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
    from os import stat

    for filename in (rgb_tiff_file, rgb_tiff_file_multi, gray_tiff_file, gray_tiff_file_multi):
        savename = str(filename.new(purebasename=f"roi_out_{filename.purebasename}"))
        stack = CorrelatedStack(str(filename))
        with pytest.warns(DeprecationWarning):
            stack.export_tiff(savename, roi=[10, 190, 20, 80])
        assert stat(savename).st_size > 0

        with tifffile.TiffFile(savename) as tif:
            assert tif.pages[0].tags["ImageWidth"].value == 180
            assert tif.pages[0].tags["ImageLength"].value == 60

        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                stack.export_tiff(savename, roi=[-10, 190, 20, 80])

        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                stack.export_tiff(savename, roi=[190, 10, 20, 80])
        stack.src.close()
