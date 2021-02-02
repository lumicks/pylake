Correlated stacks
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from the camera's.
These videos can be opened and sliced using `CorrelatedStack`::

    stack = lk.CorrelatedStack("example.tiff")  # Loading a stack.
    stack_slice = stack[2:10]  # Grab frame 2 to 9

Full color RGB images are automatically reconstructed using the alignment matrices
from Bluelake if available. This functionality can be turned off with the optional
`align` keyword::

    stack2 = lk.CorrelatedStack("example2.tiff", align=False)

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

Generally, the edges of an aligned image can become corrupted due interopolation artefacts. 
In this case, we can export a cropped region of interest by supplying the `roi` kwarg in the form
`[min_x_pixel, max_x_pixel, min_y_pixel, max_y_pixel]`::

    stack.export_tiff("aligned_cropped_stack.tiff", roi=[20, 280, 20, 180])
