Image Stacks
============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from different cameras as TIF files.
These videos can be opened using :class:`~lumicks.pylake.ImageStack`::

    stack = lk.ImageStack("stack.tiff")  # Loading a stack.

Long videos are exported as a set of multiple files due to the size constraint on the TIF format.
In this case, you can supply the filenames consecutively (i.e. `lk.ImageStack("tiff1.tiff", "tiff2.tiff")`)
to treat them as a single object.

Full color RGB fluorescence images recorded with the widefield or TIRF functionality
are automatically reconstructed using the alignment matrices from Bluelake, if available. This functionality can be
turned off with the optional `align` keyword. Note that the `align` parameter has to be provided as a keyworded argument::

    raw_stack = lk.ImageStack("stack.tiff", align=False)


Slicing and cropping
--------------------

We can slice the stack by frame index::

    stack_slice = stack[2:10]  # Grab frame 3 to 9

or by time::

    stack["2s":"10s"]  # Slice from 2 to 10 seconds from beginning of the stack

You can also spatially crop to select a smaller region of interest::

    stack_roi = stack_slice.crop_by_pixels(10, 400, 10, 120)  # pixel coordinates as x_min, x_max, y_min, y_max

Cropping can be useful, for instance, after applying color alignment to RGB images since many
of the pixels near the edges don't have data. We can see this by plotting just the `"red"` channel
(details of the plotting method are explained in the next section)::

    plt.figure()
    plt.subplot(211)
    stack_slice.plot("red")
    plt.subplot(212)
    stack_roi.plot("red")
    plt.show()

.. image:: figures/imagestack/imagestack_red_cropped.png

Alternatively, you can crop directly by slicing the stack::

    # slice with [frames, rows (y), columns (x)]
    # equivalent to stack[2:10].crop_by_pixels(10, 400, 10, 120)
    stack[2:10, 10:120, 10:400]

Here the first index can be used to select a subset of frames and the second and third indices
perform a cropping operation. Note how the axes are switched when compared to
:meth:`~lumicks.pylake.ImageStack.crop_by_pixels()` to follow the numpy
convention (rows and then columns).

You can also obtain the image stack data as a :class:`numpy array <numpy.ndarray>` using the
:meth:`~lumicks.pylake.ImageStack.get_image()` method::

    red_data = stack.get_image(channel="red") # shape = [n_frames, y_pixels, x_pixels]
    rgb_data = stack.get_image(channel="rgb") # shape = [n_frames, y_pixels, x_pixels, 3 channels]

Plotting and exporting
----------------------

You can quickly plot an individual frame using the :meth:`~lumicks.pylake.ImageStack.plot()` method::

    plt.figure()
    stack_roi.plot("rgb")
    plt.show()

.. image:: figures/imagestack/imagestack_cropped.png

The first argument is the color channel that you wish to plot. All additional arguments must be supplied as keyword arguments.

You can also plot only a single color channel. Note that you can also pass additional formatting
arguments (here, the `"magma"` colormap), which are forwarded to :func:`plt.imshow() <matplotlib.pyplot.imshow()>`::

    plt.figure()
    stack_roi.plot(channel="red", cmap="magma")
    plt.show()

.. image:: figures/imagestack/imagestack_red.png

There are also a number of custom colormaps for plotting single channel images. These are available from :data:`~lumicks.pylake.colormaps`;
the available colormaps are: `.red`, `.green`, `.blue`, `.magenta`, `.yellow`, and `.cyan`.

If the `channel` argument is not provided, the default behavior is `"rgb"` for 3-color images. For single-color
images, this argument is ignored as there is only one channel available.

Sometimes a few bright pixels can dominate the image. When this is the case, it may be beneficial to manually set the color limits
for each of the channels. This can be accomplished by providing a :class:`~lumicks.pylake.ColorAdjustment` to plotting or export functions::

    plt.figure()
    stack_roi.plot("rgb", adjustment=lk.ColorAdjustment([100, 100, 100], [185, 200, 200]))
    plt.show()

.. image:: figures/imagestack/imagestack_adjust_absolute.png

By default the limits should be provided in absolute values, although percentiles can be used instead for convenience::

    plt.figure()
    stack_roi.plot("rgb", adjustment=lk.ColorAdjustment(20, 99, mode="percentile"))
    plt.show()

.. image:: figures/imagestack/imagestack_adjust_percentile.png

Finally, the aligned image stack can also be exported to TIFF format::

    stack.export_tiff("aligned_stack.tiff")
    stack[5:20].export_tiff("aligned_short_stack.tiff")  # export a slice of the ImageStack

Defining a tether
-----------------

To define the location of the tether between beads, supply the `(x, y)` pixel coordinates of the end points
to the :func:`~lumicks.pylake.ImageStack.define_tether()` method::

    stack_tether = stack_roi.define_tether((97, 59), (286, 57))

    plt.figure()
    stack_tether.plot("green", adjustment=lk.ColorAdjustment(0, 99, mode="percentile"), cmap=lk.colormaps.green)
    stack_tether.plot_tether(lw=0.7)
    plt.show()

.. image:: figures/imagestack/imagestack_tether.png

Note, after defining a tether location the image is rotated such that the tether is horizontal in
the field of view. You can also plot the overlay of the tether location using
:func:`plot_tether(**kwargs) <lumicks.pylake.ImageStack.plot_tether()>`,
which also accepts keyword arguments that are passed to :func:`plt.plot()
<matplotlib.pyplot.plot()>`.

You can also define a tether interactively using the :meth:`~lumicks.pylake.ImageStack.crop_and_rotate` method. See the
:ref:`Notebook widgets<crop_and_rotate>` tutorial for more information.


Correlating force with the image stack
--------------------------------------

Quite often, it is interesting to correlate events on the camera's to `channel` data.
To quickly explore the correlation between images in a :class:`~lumicks.pylake.ImageStack` and channel data
you can use the following function::

    # Making a plot where force is correlated to images in the stack.
    file = lk.File("stack.h5")  # Loading a stack.
    stack[2:, 10:120, 10:400].plot_correlated(file.force1x, channel="rgb", frame=208, adjustment=lk.ColorAdjustment(20, [98, 99.9, 100], mode="percentile"))

.. image:: figures/imagestack/imagestack_correlated.png

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