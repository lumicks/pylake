Correlated stacks
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from the camera's.
These videos can be opened and sliced using `CorrelatedStack`::

    stack = lk.CorrelatedStack("example.tiff")  # Loading a stack.
    stack_slice = stack[2:10]  # Grab frame 2 to 9

You can also spatially crop to select a smaller region of interest::

    stack_roi = stack.crop_by_pixels(10, 50, 20, 80)  # pixel coordinates as x_max, x_min, y_max, y_min

This can be useful, for instance, after applying color alignment to RGB images as the edges
can become corrupted due to interpolation artifacts.

Full color RGB images are automatically reconstructed using the alignment matrices
from Bluelake if available. This functionality can be turned off with the optional
`align` keyword::

    stack2 = lk.CorrelatedStack("example2.tiff", align=False)

You can obtain the image stack data as a `numpy` array using the `get_image()` method::

    red_data = stack.get_image(channel="red") # shape = [n_frames, y_pixels, x_pixels]
    rgb_data = stack.get_image(channel="rgb") # shape = [n_frames, y_pixels, x_pixels, 3 channels]

If the `channel` argument is not provided, the default behavior is `"rgb"` for 3-color images. For single-color
images, this argument is ignored as there is only one channel available.

Quite often, it is interesting to correlate events on the camera's to `channel` data.
To quickly explore the correlation between images in a `CorrelatedStack` and channel data
you can use the following function::

    # Making a plot where force is correlated to images in the stack.
    stack.plot_correlated(file.force1x)

.. image:: correlatedstack.png

In some cases, additional processing may be needed, and we desire to have the data
downsampled over the video frames. This can be done using the function `Slice.downsampled_over`
using timestamps obtained from the `CorrelatedStack`::

    # Determine the force trace averaged over frame 2...9.
    file.force1x.downsampled_over(stack[2:10].timestamps)

The aligned image stack can also be exported to tiff format::

    stack.export_tiff("aligned_stack.tiff")
    stack[5:20].export_tiff("aligned_short_stack.tiff") # export a slice of the CorrelatedStack
