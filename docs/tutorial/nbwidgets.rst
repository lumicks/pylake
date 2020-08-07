Notebook widgets
================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

When analyzing notebooks, it can be helpful to make use of interactive widgets. For this, we provide To enable such
widgets, start the notebook with::

    %matplotlib widget

F,d selection
-------------

Assume we have an F,d curve we want to analyze we want to analyze. We know that this file contains two segments that we
should be analyzing separately. Let's load the file and run the widget::

    file = lk.File("file.h5")
    fdcurves = file.fdcurves
    selector = lk.FdRangeSelector(fdcurves)

This opens up a little widget, where you can use the left mouse button to select time ranges.

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

Now let's say we want to load a number of files that contain one F,d curve each. We can do this using `glob`. Using
`glob` we grab a list of all `.h5` files in the directory `my_directory`. We iterate over this list and open each file
after which we add each Fd curve to a dictionary of Fd curves::

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
