import json
from copy import deepcopy

import numpy as np
import pytest

from lumicks.pylake import ImageStack
from lumicks.pylake.detail.widefield import TiffStack

from ..data.mock_widefield import MockTiffFile, make_frame_times


def test_image_reconstruction_grayscale(gray_alignment_image_data):
    reference_image, warped_image, description, bit_depth = gray_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image],
                times=make_frame_times(1),
                description=description,
                bit_depth=8,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert not fr.is_rgb
    assert np.all(fr.data == fr.raw_data)
    np.testing.assert_allclose(fr.raw_data, fr._get_plot_data())


def test_image_reconstruction_rgb(rgb_alignment_image_data, rgb_alignment_image_data_offset):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image],
                times=make_frame_times(1),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    max_signal = np.max(np.hstack([fr._get_plot_data("green"), fr._get_plot_data("red")]))
    diff = np.abs(fr._get_plot_data("green").astype(float) - fr._get_plot_data("red").astype(float))
    assert np.all(diff / max_signal < 0.05)
    max_signal = np.max(np.hstack([fr._get_plot_data("green"), fr._get_plot_data("blue")]))
    diff = np.abs(
        fr._get_plot_data("green").astype(float) - fr._get_plot_data("blue").astype(float)
    )
    assert np.all(diff / max_signal < 0.05)

    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)
    max_signal = np.max(np.hstack([reference_image[:, :, 0], fr._get_plot_data("red")]))
    diff = np.abs(reference_image[:, :, 0].astype(float) - fr._get_plot_data("red").astype(float))
    assert np.all(diff / max_signal < 0.05)

    with pytest.raises(ValueError):
        fr._get_plot_data(channel="purple")

    # test that bad alignment matrix gives high error compared to correct matrix
    bad_description = deepcopy(description)
    label = (
        "Alignment red channel"
        if "Alignment red channel" in description.keys()
        else "Channel 0 alignment"
    )
    bad_description[label][2] = 25
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image],
                times=make_frame_times(1),
                description=bad_description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    assert fr.is_rgb
    assert not np.allclose(original_data, fr._get_plot_data(), atol=0.05)

    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data_offset
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image],
                times=make_frame_times(1),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)

    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_no_alignment_requested(rgb_alignment_image_data):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image],
                times=make_frame_times(1),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=False,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(0)
    np.testing.assert_allclose(fr.data, warped_image)


def test_image_reconstruction_rgb_multiframe(rgb_alignment_image_data):
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    fake_tiff = TiffStack(
        [
            MockTiffFile(
                data=[warped_image] * 6,
                times=make_frame_times(6, step=10),
                description=description,
                bit_depth=16,
            )
        ],
        align_requested=True,
    )
    stack = ImageStack.from_dataset(fake_tiff)
    fr = stack._get_frame(2)

    assert fr.is_rgb
    original_data = (reference_image / (2**bit_depth - 1)).astype(float)
    np.testing.assert_allclose(original_data, fr._get_plot_data(), atol=0.05)


def test_image_reconstruction_rgb_missing_metadata(rgb_alignment_image_data):
    # no metadata
    reference_image, warped_image, description, bit_depth = rgb_alignment_image_data
    with pytest.warns(
        UserWarning, match="File does not contain metadata. Only raw data is available"
    ):
        fake_tiff = TiffStack(
            [
                MockTiffFile(
                    data=[warped_image],
                    times=make_frame_times(1),
                    bit_depth=16,
                    no_metadata=True,
                )
            ],
            align_requested=True,
        )

    # missing alignment matrices
    for label in ("Alignment red channel", "Channel 0 alignment"):
        if label in description:
            removed = description.pop(label)
            break
    with pytest.warns(
        UserWarning, match="File does not contain alignment matrices. Only raw data is available"
    ):
        fake_tiff = TiffStack(
            [
                MockTiffFile(
                    data=[warped_image],
                    times=make_frame_times(1),
                    description=description,
                    bit_depth=16,
                )
            ],
            align_requested=True,
        )

    description[label] = removed  # reset fixture
