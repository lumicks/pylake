Correlated stacks
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from the camera's.
These videos can be opened and sliced using `CorrelatedStack`::

    stack = lk.CorrelatedStack("cas9_wf.tiff")  # Loading a stack.
    stack_slice = stack[2:10]  # Grab frame 2 to 9

You can quickly plot an individual frame using the `plot()` method::

    stack.plot(frame=0, channel="rgb")

.. image:: correlatedstack_aligned.png

Sometimes a few bright pixels can dominate the image.
When this is the case, it may be beneficial to manually set the color limits for each of the channels.
This can be accomplished by providing a :class:`~lumicks.pylake.ColorAdjustment` to plotting or export functions::

    stack.plot(channel="red", adjustment=lk.ColorAdjustment([50, 50, 50], [100, 250, 196]))


By default the limits should be provided in absolute values, although percentiles can be used instead for convenience::

    stack.plot(channel="red", adjustment=lk.ColorAdjustment([5, 5, 5], [95, 95, 95], mode="percentile"))



To define the location of the tether between beads, supply the `(x, y)` pixel coordinates of the end points
to the `define_tether()` method::

    stack = stack.define_tether((126, 193), (341, 200))
    stack.plot()
    stack.plot_tether(lw=0.7)

.. image:: correlatedstack_tether.png

Note, after defining a tether location the image is rotated such that the tether is horizontal in the field of view.
You can also plot the overlay of the tether location using `plot_tether(**kwargs)`, which also accepts keyword
arguments that are passed to `plt.plot()`.

You can also spatially crop to select a smaller region of interest::

    stack_roi = stack.crop_by_pixels(45, 420, 150, 245)  # pixel coordinates as x_min, x_max, y_min, y_max
    stack_roi.plot()  # note: the default channel is "rgb"

.. image:: correlatedstack_cropped.png

This can be useful, for instance, after applying color alignment to RGB images as the edges
can become corrupted due to interpolation artifacts.

You can also plot only a single color channel. Note that here we pass some additional formatting arguments, which are
forwarded to `plt.imshow()`::

    stack_roi.plot(channel="red", cmap="magma", adjustment=lk.ColorAdjustment(550, 800))

.. image:: correlatedstack_red.png

Full color RGB images are automatically reconstructed using the alignment matrices
from Bluelake if available. This functionality can be turned off with the optional
`align` keyword::

    stack2 = lk.CorrelatedStack("cas9_wf.tiff", align=False)
    stack2.plot()

.. image:: correlatedstack_raw.png

You can obtain the image stack data as a `numpy` array using the `get_image()` method::

    red_data = stack.get_image(channel="red") # shape = [n_frames, y_pixels, x_pixels]
    rgb_data = stack.get_image(channel="rgb") # shape = [n_frames, y_pixels, x_pixels, 3 channels]

If the `channel` argument is not provided, the default behavior is `"rgb"` for 3-color images. For single-color
images, this argument is ignored as there is only one channel available.


Finally, the aligned image stack can also be exported to TIFF format::

    stack.export_tiff("aligned_stack.tiff")
    stack[5:20].export_tiff("aligned_short_stack.tiff") # export a slice of the CorrelatedStack

Correlating force with the image stack
--------------------------------------

Quite often, it is interesting to correlate events on the camera's to `channel` data.
To quickly explore the correlation between images in a `CorrelatedStack` and channel data
you can use the following function::

    # Making a plot where force is correlated to images in the stack.
    stack = lk.CorrelatedStack("example.tiff)
    stack.plot_correlated(file.force1x)

.. image:: correlatedstack.png

If the plot is interactive (for example, when `%matplotlib notebook` is used in a Jupyter notebook), you can click
on the left graph to select a particular force. The corresponding video frame will then automatically appear on the right.

In some cases, additional processing may be needed, and we wish to have the data
downsampled over the video frames. This can be done using the function `Slice.downsampled_over`
using timestamps obtained from the `CorrelatedStack`::

    # Determine the force trace averaged over frame 2...9.
    file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges)