Correlated stacks
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Bluelake has the ability to export videos from the camera's.
These videos can be opened and sliced using `CorrelatedStack`::

    stack = pylake.CorrelatedStack("example.tiff")  # Loading a stack.
    stack_slice = stack[2:10]  # Grab frame 2 to 9

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
