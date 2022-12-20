Image Stacks
============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from the camera's.
These videos can be opened and sliced using :class:`~lumicks.pylake.ImageStack`::

    stack = lk.ImageStack("wf.tiff")  # Loading a stack.
    stack_slice = stack[2:10]  # Grab frame 2 to 9

You can also slice by time::

    stack_slice = stack["2s":"10s"]  # Slice from 2 to 10 seconds from beginning of the stack

You can easily load multiple TIFF files by simply listing them consecutively::

    stack = lk.ImageStack("wf.tiff", "wf2.tiff")  # Loading two tiff files in a single stack.

You can quickly plot an individual frame using the
:meth:`~lumicks.pylake.ImageStack.plot()` method::

    stack.plot(frame=0, channel="rgb")

.. image:: figures/imagestack/imagestack_aligned.png

Sometimes a few bright pixels can dominate the image.
When this is the case, it may be beneficial to manually set the color limits for each of the channels.
This can be accomplished by providing a :class:`~lumicks.pylake.ColorAdjustment` to plotting or export functions::

    stack.plot(channel="red", adjustment=lk.ColorAdjustment([50, 50, 50], [100, 250, 196]))


There are also a number of custom colormaps for plotting single channel images. These are available from :data:`~lumicks.pylake.colormaps`; the available colormaps are:
`.red`, `.green`, `.blue`, `.magenta`, `.yellow`, and `.cyan`. For example, we can plot the blue channel image with the cyan colormap::

    stack.plot(channel="blue", cmap=lk.colormaps.cyan)

By default the limits should be provided in absolute values, although percentiles can be used instead for convenience::

    stack.plot(channel="red", adjustment=lk.ColorAdjustment([5, 5, 5], [95, 95, 95], mode="percentile"))

Gamma adjustments can be applied in addition to the bounds by supplying an extra argument named `gamma`.
For example, a gamma adjustment of `2` to the red channel can be applied as follows::

    stack.plot(channel="red", adjustment=lk.ColorAdjustment([5, 5, 5], [95, 95, 95], mode="percentile", gamma=[2, 1, 1]))

To define the location of the tether between beads, supply the `(x, y)` pixel coordinates of the end points
to the :func:`~lumicks.pylake.ImageStack.define_tether()` method::

    stack = stack.define_tether((126, 193), (341, 200))
    stack.plot()
    stack.plot_tether(lw=0.7)

.. image:: figures/imagestack/imagestack_tether.png

Note, after defining a tether location the image is rotated such that the tether is horizontal in
the field of view. You can also plot the overlay of the tether location using
:func:`plot_tether(**kwargs) <lumicks.pylake.ImageStack.plot_tether()>`,
which also accepts keyword arguments that are passed to :func:`plt.plot()
<matplotlib.pyplot.plot()>`.

You can also spatially crop to select a smaller region of interest::

    stack_roi = stack.crop_by_pixels(45, 420, 150, 245)  # pixel coordinates as x_min, x_max, y_min, y_max
    stack_roi.plot()  # note: the default channel is "rgb"

.. image:: figures/imagestack/imagestack_cropped.png

Alternatively, you can crop directly by slicing the stack::

    stack_roi = stack[:, 150:245, 45:420]

Here the first index can be used to select a subset of frames and the second and third indices
perform a cropping operation. Note how the axes are switched when compared to
:meth:`~lumicks.pylake.ImageStack.crop_by_pixels()` to follow the numpy
convention (rows and then columns).

Cropping can be useful, for instance, after applying color alignment to RGB images as the edges
can become corrupted due to interpolation artifacts.

You can also plot only a single color channel. Note that here we pass some additional formatting
arguments, which are forwarded to :func:`plt.imshow() <matplotlib.pyplot.imshow()>`::

    stack_roi.plot(channel="red", cmap="magma", adjustment=lk.ColorAdjustment(550, 800))

.. image:: figures/imagestack/imagestack_red.png

Full color RGB images are automatically reconstructed using the alignment matrices
from Bluelake if available. This functionality can be turned off with the optional
`align` keyword. Note that the align parameter has to be provided as a keyworded argument (i.e. `align=False`)::

    stack2 = lk.ImageStack("wf.tiff", align=False)
    stack2.plot()

.. image:: figures/imagestack/imagestack_raw.png

You can obtain the image stack data as a :class:`numpy <numpy.ndarray>` array using the
:meth:`~lumicks.pylake.ImageStack.get_image()` method::

    red_data = stack.get_image(channel="red") # shape = [n_frames, y_pixels, x_pixels]
    rgb_data = stack.get_image(channel="rgb") # shape = [n_frames, y_pixels, x_pixels, 3 channels]

If the `channel` argument is not provided, the default behavior is `"rgb"` for 3-color images. For single-color
images, this argument is ignored as there is only one channel available.


Finally, the aligned image stack can also be exported to TIFF format::

    stack.export_tiff("aligned_stack.tiff")
    stack[5:20].export_tiff("aligned_short_stack.tiff") # export a slice of the ImageStack

Correlating force with the image stack
--------------------------------------

Quite often, it is interesting to correlate events on the camera's to `channel` data.
To quickly explore the correlation between images in a :class:`~lumicks.pylake.ImageStack` and channel data
you can use the following function::

    # Making a plot where force is correlated to images in the stack.
    stack = lk.ImageStack("example.tiff")
    stack.plot_correlated(file.force1x)

.. image:: figures/imagestack/imagestack.png

If the plot is interactive (for example, when `%matplotlib notebook` is used in a Jupyter notebook), you can click
on the left graph to select a particular force. The corresponding video frame will then automatically appear on the right.

In some cases, additional processing may be needed, and we wish to have the data
downsampled over the video frames. This can be done using the :meth:`~lumicks.pylake.channel.Slice.downsampled_over`
method with timestamps obtained from the :class:`~lumicks.pylake.ImageStack`::

    # Determine the force trace averaged over frame 2...9.
    file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges())

By default, this averages only over the exposure time of the images in the stack.
If you wish to average over the full time range from the start of the scan to the next scan, pass the extra parameter `include_dead_time=True`::

    file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges(include_dead_time=True))
