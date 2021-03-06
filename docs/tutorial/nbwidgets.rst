Notebook widgets
================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

When analyzing notebooks, it can be helpful to make use of interactive widgets. For this, we provide some widgets
to help you analyze your data. To enable such widgets, start the notebook with::

    %matplotlib widget

Channel slicing
---------------

Let's say we want to do some analyses on slices of channel data. It would be nice to just quickly visually select some
regions using a widget. Let's load the file and run the widget::

    file = lk.File("file.h5")
    channel = file["Force LF"]["Force 1x"]
    selector = channel.range_selector()

.. image:: slice_widget.png

You can use the left mouse button to select time ranges (by clicking the left and then the right boundary of the region
you wish to select). The right mouse button can be used to remove previous selections. We can access the selected
timestamps of the ranges we selected by invoking
:attr:`~lumicks.pylake.nb_widgets.fd_selector.SliceRangeSelector.ranges`::

    >>> selector.ranges
    [array([1572279165841737600, 1572279191523516800], dtype=int64),
    array([1572279201850211200, 1572279224153072000], dtype=int64)]

And the actual slices from :attr:`~lumicks.pylake.nb_widgets.fd_selector.SliceRangeSelector.slices`. If we want to
plot all of our selections in separate plots for instance, we can do the following::

    for data_slice in selector.slices:
        plt.figure()
        data_slice.plot()

.. image:: slice_widget2.png


F,d selection
-------------

Range selection by time
^^^^^^^^^^^^^^^^^^^^^^^

Assume we have an F,d curve we want to analyze. We know that this file contains one F,d curve which should be split up
into two segments that we should be analyzing separately::

    fdcurves = file.fdcurves
    selector = lk.FdRangeSelector(fdcurves)

This opens up a little widget, where you can use the left mouse button to select time ranges and the right mouse
button to remove previous selections.

.. image:: fd_widget.png

Once we've selected some time ranges, we can output the timestamps::

    >>> selector.ranges

    {'Fd pull #6': [array([1572278052698057600, 1572278078909667200], dtype=int64), array([1572278086737161600, 1572278099133193600], dtype=int64)]}

These timestamps can directly be used to extract the relevant data::

    for t_start, t_stop in selector.ranges["Fd pull #6"]:
        plt.figure()
        plt.plot(fdcurves["Fd pull #6"].f[t_start:t_stop].data)

.. image:: fd_widget2.png

This produces a separate plot for each selection. There's also a more direct way to get these plots, namely through
`FdRangeSelector.fdcurves`. This gives you an `FdCurve` for each section you selected::

    for fdcurve in selector.fdcurves["Fd pull #6"]:
        plt.figure()
        fdcurve.plot_scatter()

.. image:: fd_widget3.png

Processing multiple files
^^^^^^^^^^^^^^^^^^^^^^^^^

Now let's say our experiment is split up over multiple files, each containing a few F,d curves. We would like to load
these curves all at once and make our selections. We can do this using automatically using `glob`. With `glob.glob`
we grab a list of all `.h5` files in the directory `my_directory`. We then iterate over this list and open each file.
Then, for all those files, we add each individual curves to our variable `fdcurves`::

    import glob

    fdcurves = {}
    for filename in glob.glob('my_directory/*.h5'):
        file = lk.File(filename)
        for key, curve in file.fdcurves.items():
            fdcurves[key] = curve

Using this dictionary, we can open our widget and see all the data at once::

    selector = lk.FdRangeSelector(fdcurves)

Plotting the curves can be done similarly as before. Here `.values()` indicates that we want the values from the
dictionary of curve sets, and not the keys (which in our case are the curve names)::

    for curve_set in selector.fdcurves.values():
        for fdcurve in curve_set:
            plt.figure()
            fdcurve.plot_scatter()

Range selection by distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to select a portion of an F,d curve based on distance::

    selector = lk.FdDistanceRangeSelector(fdcurves)

.. image:: fd_dist_widget.png    

Again, we can retrieve the selected data just as with `FdRangeSelector`::

    original = fdcurves["Fd pull #6"]
    sliced = selector.fdcurves["Fd pull #6"][0]

    plt.figure()

    plt.subplot(2, 1, 1)
    original.plot_scatter(label="original")
    sliced.plot_scatter(label="sliced")
    plt.legend()

    plt.subplot(2, 1, 2)
    original.f.plot()
    sliced.f.plot(start=original.start)

.. image::  fd_dist_widget2.png

The returned F,d curves correspond to the longest contiguous (in time) stretch of data that falls 
within the distance thresholds. However, noise in the distance measurement can lead to short gaps of the time
trace falling slightly outside of the thresholds, as illustrated below:

.. image:: fd_dist_widget3a.png

To avoid premature truncation caused by this noise, there is an additional `max_gap` keyword argument 
to `FdDistanceRangeSelector` that can be used to adjust the acceptable length of noise gaps. The default values
is zero, such that all data points are guaranteed to fall within the selected distance range. The effect of this 
argument is shown below for an F,d curve sliced with the same distance thresholds:

.. image:: fd_dist_widget3.png

Range selection of single curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The selector widgets can also be easily accessed from single F,d curve instances::

    fdcurve = fdcurves["Fd pull #6"]
    t_selector = fdcurve.range_selector()
    d_selector = fdcurve.distance_range_selector(max_gap=3)
