import pytest
import numpy as np

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
        def tiff_timestamp_ranges():
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
        export_bare._tiff_timestamp_ranges()
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


def test_export_tiff_int(tmp_path):
    images = np.ones(shape=(2, 10, 10, 3))  # (n, h, w, c)
    export_image16 = tiffexport_factory(images * np.iinfo(np.uint16).max)

    # Sufficient bit-depth or forced clipping
    export_image16.export_tiff(tmp_path / "uint16", dtype=np.uint16)
    export_image16.export_tiff(tmp_path / "float32", dtype=np.float32)
    export_image16.export_tiff(tmp_path / "clipped", dtype=np.uint8, clip=True)

    # Raise because unsafe
    with pytest.raises(RuntimeError, match="Can't safely export image with `dtype=uint8` channels"):
        export_image16.export_tiff(tmp_path / "uint8", dtype=np.uint8)

    with pytest.raises(
        RuntimeError, match="Can't safely export image with `dtype=float16` channels"
    ):
        export_image16.export_tiff(tmp_path / "float16", dtype=np.float16)


@pytest.mark.filterwarnings(
    # Numpy 1.24 raises a RuntimeWarning upon overflow during type cast (see call with `clip=True`)
    "ignore:overflow encountered in cast:RuntimeWarning:lumicks.pylake.detail.imaging_mixins:48"
)
def test_export_tiff_float(tmp_path):
    images = np.ones(shape=(2, 10, 10, 3))  # (n, h, w, c)
    export_image32 = tiffexport_factory(images * np.finfo(np.float32).max)

    # Sufficient bit-depth or forced clipping
    export_image32.export_tiff(tmp_path / "float32", dtype=np.float32)
    export_image32.export_tiff(tmp_path / "clipped", dtype=np.float16, clip=True)

    # Raise because unsafe
    with pytest.raises(
        RuntimeError, match="Can't safely export image with `dtype=float16` channels"
    ):
        export_image32.export_tiff(tmp_path / "float16", dtype=np.float16)

    with pytest.raises(
        RuntimeError, match="Can't safely export image with `dtype=uint16` channels"
    ):
        export_image32.export_tiff(tmp_path / "uint16", dtype=np.uint16)


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
        import tifffile
        from ast import literal_eval

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
