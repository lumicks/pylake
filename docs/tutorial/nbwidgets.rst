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
    selector = channel.range_selector

.. image:: slice_widget.png

You can use the left mouse button to select time ranges (by clicking the left and then the right boundary of the region
you wish to select). The right mouse button can be used to remove previous selections. We can access the selected
timestamps of the ranges we selected by invoking
:attr:`~lumicks.pylake.nb_widgets.fd_selector.SliceRangeSelectorWidget.ranges`::

    >>> selector.ranges
    [array([1572279165841737600, 1572279191523516800], dtype=int64),
    array([1572279201850211200, 1572279224153072000], dtype=int64)]

And the actual slices from :attr:`~lumicks.pylake.nb_widgets.fd_selector.SliceRangeSelectorWidget.slices`. If we want to
plot all of our selections in separate plots for instance, we can do the following::

    for data_slice in selector.slices:
        plt.figure()
        data_slice.plot()

.. image:: slice_widget2.png


F,d selection
-------------

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
