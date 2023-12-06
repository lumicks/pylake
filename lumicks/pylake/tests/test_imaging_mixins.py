import numpy as np
import pytest
from tifffile import TiffFile

from lumicks.pylake.detail.imaging_mixins import TiffExport


def tiffexport_factory(
    images,
    frames=True,
    image_metadata=True,
    timestamp_ranges=True,
    writer_kwargs=True,
    start_time=0,
    exposure_time=5,
    frame_time=10,
):
    def tiff_frames_factory(frames):
        return lambda iterator: (frame for frame in frames) if iterator else frames

    def tiff_image_metadata():
        return {"Writing": "tests is awesome!"}

    def tiff_timestamp_ranges_factory(number_of_frames):
        def tiff_timestamp_ranges(include_dead_time):
            return [
                np.array([i, i + exposure_time])
                for i in range(
                    start_time, start_time + frame_time * number_of_frames, start_time + frame_time
                )
            ]

        return tiff_timestamp_ranges

    def tiff_writer_kwargs():
        return {"software": "Pylake Test", "photometric": "rgb"}

    export = TiffExport()
    if frames:
        export._tiff_frames = tiff_frames_factory(images)
    if image_metadata:
        export._tiff_image_metadata = tiff_image_metadata
    if timestamp_ranges:
        export._tiff_timestamp_ranges = tiff_timestamp_ranges_factory(len(images))
    if writer_kwargs:
        export._tiff_writer_kwargs = tiff_writer_kwargs
    return export


def test_export_tiff_insufficient_implementation(tmp_path):
    images = np.ones(shape=(2, 10, 10, 3))  # (n, h, w, c)

    export_bare = TiffExport()
    export_missing_frames = tiffexport_factory(images, frames=False)
    export_missing_meta = tiffexport_factory(images, image_metadata=False)
    export_missing_time = tiffexport_factory(images, timestamp_ranges=False)
    export_missing_kwargs = tiffexport_factory(images, writer_kwargs=False)
    filename = tmp_path / "insufficient.tiff"

    classmodule = TiffExport.__module__
    classname = TiffExport.__name__
    message_frames = f"`{classmodule}.{classname}` does not implement `_tiff_frames\\(\\)`."
    message_meta = f"`{classmodule}.{classname}` does not implement `_tiff_image_metadata\\(\\)`."
    message_time = f"`{classmodule}.{classname}` does not implement `_tiff_timestamp_ranges\\(\\)`."
    message_kwargs = f"`{classmodule}.{classname}` does not implement `_tiff_writer_kwargs\\(\\)`."

    # Test existence of interface methods and throwing of `NotImplementedError`
    with pytest.raises(NotImplementedError, match=message_frames):
        export_bare._tiff_frames()
    with pytest.raises(NotImplementedError, match=message_meta):
        export_bare._tiff_image_metadata()
    with pytest.raises(NotImplementedError, match=message_time):
        export_bare._tiff_timestamp_ranges(include_dead_time=False)
    with pytest.raises(NotImplementedError, match=message_kwargs):
        export_bare._tiff_writer_kwargs()
    # Test if `export_tiff()` requires all `_tiff_*()` methods
    with pytest.raises(NotImplementedError, match=message_frames):
        export_missing_frames.export_tiff(filename)
    with pytest.raises(NotImplementedError, match=message_meta):
        export_missing_meta.export_tiff(filename)
    with pytest.raises(NotImplementedError, match=message_time):
        export_missing_time.export_tiff(filename)
    with pytest.raises(NotImplementedError, match=message_kwargs):
        export_missing_kwargs.export_tiff(filename)


def test_export_tiff_empty(tmp_path):
    export_empty = tiffexport_factory(np.array([]))
    filename = tmp_path / "empty.tiff"

    # Test empty `export_tiff` array
    with pytest.raises(RuntimeError, match="Can't export TIFF if there are no images."):
        export_empty.export_tiff(filename)

    # Test empty `export_tiff` iterator
    with pytest.raises(RuntimeError, match="Can't export TIFF if there are no images."):
        export_empty.export_tiff(filename, dtype=np.float32)


@pytest.mark.parametrize(
    "output_dtype, data_dtype",
    [
        (np.float32, np.float32),
        (np.float32, np.float16),
        (np.float32, np.uint32),
        (np.float16, np.float16),
        (np.uint32, np.uint16),
        (np.uint32, np.uint8),
        (np.uint16, np.uint16),
        (np.uint16, np.uint8),
    ],
)
def test_export_tiff_sufficient_range(tmp_path, output_dtype, data_dtype):
    """Test TIFF export when the data type has sufficient range for the data stored"""
    info = np.finfo(data_dtype) if np.dtype(data_dtype).kind == "f" else np.iinfo(data_dtype)
    images = np.ones(shape=(2, 10, 10, 3)) * info.max  # (n, h, w, c)
    images[0, 0, 0, 0] = info.min
    export_image = tiffexport_factory(images)
    export_image.export_tiff(tmp_path / "tmp", dtype=output_dtype)

    # Validate data
    with TiffFile(tmp_path / "tmp") as t:
        np.testing.assert_allclose(t.asarray(), images)


@pytest.mark.parametrize(
    "output_dtype, data_dtype, test_max",
    [
        (np.float16, np.float32, True),
        (np.uint8, np.float32, True),
        (np.uint16, np.float32, True),
        (np.uint32, np.float32, True),
        (np.uint16, np.uint32, True),
        (np.uint8, np.uint16, True),
        (np.uint8, np.uint32, True),
        (np.float16, np.float32, False),
        (np.uint8, np.float32, False),
    ],
)
def test_export_tiff_insufficient_range(tmp_path, output_dtype, data_dtype, test_max):
    """Test TIFF export when the data type has insufficient range for the data stored"""
    info = np.finfo(data_dtype) if np.dtype(data_dtype).kind == "f" else np.iinfo(data_dtype)
    images = np.ones(shape=(2, 10, 10, 3)) * (info.max if test_max else info.min)  # (n, h, w, c)
    export_image = tiffexport_factory(images)

    file_name = tmp_path / output_dtype.__name__
    export_image.export_tiff(file_name, dtype=output_dtype, clip=True)
    out_info = (
        np.finfo(output_dtype) if np.dtype(output_dtype).kind == "f" else np.iinfo(output_dtype)
    )
    with TiffFile(file_name) as t:
        np.testing.assert_allclose(
            t.asarray(),
            np.ones(shape=(2, 10, 10, 3)) * (out_info.max if test_max else out_info.min),
        )

    # Raise because unsafe
    with pytest.raises(
        RuntimeError,
        match=f"Can't safely export image with `dtype={output_dtype.__name__}` channels",
    ):
        export_image.export_tiff(file_name, dtype=output_dtype)


@pytest.mark.parametrize(
    "output_dtype, number_of_bits, sampleformat",
    [
        (None, 64, 3),
        (np.uint16, 16, None),
        (np.float32, 32, 3),
    ],
)
def test_export_tiff_tags(tmp_path, output_dtype, number_of_bits, sampleformat):
    """Test proper export of tiff data format and tags

    Parameters
    ----------
    sampleformat : int or None
        From the TIFF specification:
        1 = unsigned integer data, (default, if not defined)
        2 = two’s complement signed integer data
        3 = IEEE floating point data [IEEE]
        4 = undefined data format
    """

    def grab_tags(file):
        from ast import literal_eval

        import tifffile

        tiff_tags = []
        with tifffile.TiffFile(file) as tif:
            for page in tif.pages:
                page_tags = {}
                for tag in page.tags.values():
                    name, value = tag.name, tag.value
                    try:
                        page_tags[name] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        page_tags[name] = value
                tiff_tags.append(page_tags)
        return tiff_tags

    # Create tifffile with 2 frames
    images = np.ones(shape=(2, 5, 10, 3))  # (n, h, w, c)
    exposure_time = 5
    frame_time = 10
    export_images = tiffexport_factory(images, exposure_time=exposure_time, frame_time=frame_time)
    filename = tmp_path / "testimages"
    export_images.export_tiff(filename, dtype=output_dtype)

    # Check if tags were properly stored for both frames, i.e. test functionality of
    # `_tiff_image_metadata()`, `_tiff_timestamp_ranges()` and `_tiff_writer_kwargs()` (see
    # `tiffexport_factory()`)
    tiff_tags = grab_tags(filename)
    assert len(tiff_tags) == len(images)
    for frame_idx, tags in enumerate(tiff_tags):
        assert tags["ImageDescription"]["Writing"] == "tests is awesome!"
        assert (
            tags["DateTime"] == f"{frame_idx * frame_time}:{frame_idx * frame_time + exposure_time}"
        )
        assert tags["Software"] == "Pylake Test"
        assert tags["ImageWidth"] == 10
        assert tags["ImageLength"] == 5
        assert tags["BitsPerSample"] == (number_of_bits,) * 3
        if sampleformat:
            assert tags["SampleFormat"] == (sampleformat,) * 3
