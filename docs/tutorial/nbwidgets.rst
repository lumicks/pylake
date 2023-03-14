Notebook Widgets
================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

When analyzing notebooks, it can be helpful to make use of interactive widgets. For this, we provide some widgets
to help you analyze your data. To enable such widgets, start the notebook with::

    %matplotlib widget  # enable this line if you are using jupyter lab
    #%matplotlib notebook  # enable this line if you are using jupyter notebook

We can download the data needed for this tutorial directly from Zenodo using Pylake.
Since we don't want it in our working folder, we'll put it in a folder called `"test_data"`::

    kymo_filenames = lk.download_from_doi("10.5281/zenodo.7729525", "test_data")
    stack_filenames = lk.download_from_doi("10.5281/zenodo.7729700", "test_data")
    fd_filenames = lk.download_from_doi("10.5281/zenodo.7729929", "test_data")


Channel slicing
---------------

Let's say we want to do some analyses on slices of channel data.
It would be nice to just quickly visually select some regions using a widget.
Let's load the file and open an interactive plot using the :meth:`~lumicks.pylake.channel.Slice.range_selector()` method::

    file = lk.File("test_data/kymo.h5")
    channel = file["Force HF"]["Force 1x"]

    plt.figure()
    selector = channel.range_selector()

.. image:: figures/nbwidgets/slice_widget.png

This returns a :class:`~lumicks.pylake.nb_widgets.range_selector.SliceRangeSelectorWidget`.
You can use the left mouse button to select time ranges (by clicking the left and then the right
boundary of the region you wish to select).
The right mouse button can be used to remove previous selections.
We can access the selected timestamps of the ranges we selected by accessing the :attr:`~lumicks.pylake.nb_widgets.range_selector.SliceRangeSelectorWidget.ranges` property::

    >>> selector.ranges
    array([[1638534508099192400, 1638534546751480400],
           [1638534592854827600, 1638534628712978000]], dtype=int64)

And the actual slices from :attr:`~lumicks.pylake.nb_widgets.range_selector.SliceRangeSelectorWidget.slices`.
If we want to plot all of our selections in separate plots for instance, we can do the following::

    plt.figure(figsize=(3, 6))
    for idx, data_slice in enumerate(selector.slices):
        plt.subplot(len(selector.slices), 1, idx + 1)
        data_slice.plot()

    plt.tight_layout()

.. image:: figures/nbwidgets/slice_widget2.png


F,d selection
-------------

Range selection by time
^^^^^^^^^^^^^^^^^^^^^^^

Assume we have an F,d curve we want to analyze.
We know that this file contains one F,d curve which should be split up into three segments that should be analyzed separately.
Let's open the :class:`~lumicks.pylake.FdRangeSelector` and make a few selections::

    file = lk.File("test_data/fd_multiple_Lc.h5")
    fdcurves = file.fdcurves
    selector = lk.FdRangeSelector(fdcurves)

This opens up a little widget, where you can use the left mouse button to select time ranges and the right mouse button to remove previous selections.

.. image:: figures/nbwidgets/fd_widget.png

After making a few selections, the properties :attr:`~lumicks.pylake.FdRangeSelector.ranges` and :attr:`~lumicks.pylake.FdRangeSelector.fdcurves` contain the time ranges and force extension curves corrseponding to our selection.
Once we've selected some time ranges, we can output the timestamps using the :attr:`~lumicks.pylake.FdRangeSelector.ranges` property::

    >>> selector.ranges

    {'40': array([[1588263182203865800, 1588263189671475400], [1588263189981376200, 1588263190821107400], [1588263191131008200, 1588263195189709000]], dtype=int64)}

These timestamps can directly be used to extract the relevant data::

    plt.figure(figsize=(8, 2.5))
    for idx, (t_start, t_stop) in enumerate(selector.ranges["40"]):
        plt.subplot(1, len(selector.ranges["40"]), idx + 1)
        plt.scatter(
            fdcurves["40"].d[t_start:t_stop].data,
            fdcurves["40"].f[t_start:t_stop].data,
            s=2,  # Use a smaller marker size
        )
        plt.xlabel("Distance [$\mu$m]")
        plt.ylabel("Force [pN]")

    plt.tight_layout()

.. image:: figures/nbwidgets/fd_widget2.png

This produces a separate plot for each selection.
There's also a more direct way to get these plots, namely through :attr:`~lumicks.pylake.FdRangeSelector.fdcurves`.
This gives you an :class:`~lumicks.pylake.fdcurve.FdCurve` for each section you selected::

    plt.figure()
    for fdcurve in selector.fdcurves["40"]:
        fdcurve.plot_scatter()

.. image:: figures/nbwidgets/fd_widget3.png

Processing multiple files
^^^^^^^^^^^^^^^^^^^^^^^^^

Now let's say our experiment is split up over multiple files, each containing a few F,d curves. We would like to load
these curves all at once and make our selections. We can do this using automatically using `glob`. With `glob.glob`
we grab a list of all `.h5` files in the directory `my_directory`. We then iterate over this list and open each file.
Then, for all those files, we add each individual curves to our variable `fdcurves`::

    import glob

    fdcurves = {}
    for filename in glob.glob('*.h5'):
        file = lk.File(filename)
        for key, curve in file.fdcurves.items():
            fdcurves[key] = curve

Using this dictionary, we can open our widget and see all the data at once::

    selector = lk.FdRangeSelector(fdcurves)

.. image:: figures/nbwidgets/fd_widget4.png

Plotting the curves can be done similarly as before. Here `.values()` indicates that we want the values from the
dictionary of curve sets, and not the keys (which in our case are the curve names)::

    for curve_set in selector.fdcurves.values():
        if curve_set:
            # Open a figure only if we selected regions in this dataset
            plt.figure()
        for fdcurve in curve_set:
            fdcurve.plot_scatter()

.. image:: figures/nbwidgets/fd_widget5.png
.. image:: figures/nbwidgets/fd_widget6.png

Range selection by distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to select a portion of an F,d curve based on distance using the :class:`~lumicks.pylake.FdDistanceRangeSelector`::

    selector = lk.FdDistanceRangeSelector(fdcurves)

.. image:: figures/nbwidgets/fd_dist_widget.png

Again, we can retrieve the selected data from :attr:`~lumicks.pylake.FdDistanceRangeSelector.ranges` and :attr:`~lumicks.pylake.FdDistanceRangeSelector.fdcurves` just as with :class:`~lumicks.pylake.FdRangeSelector`::

    original = fdcurves["40"]
    sliced = selector.fdcurves["40"][0]

    plt.figure()

    plt.subplot(2, 1, 1)
    original.plot_scatter(label="original")
    sliced.plot_scatter(label="sliced")
    plt.legend()

    plt.subplot(2, 1, 2)
    original.f.plot()
    sliced.f.plot(start=original.start)

    plt.tight_layout()

.. image:: figures/nbwidgets/fd_dist_widget2.png

The returned F,d curves correspond to the longest contiguous (in time) stretch of data that falls
within the distance thresholds.

Range selection of single curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The selector widgets can also be easily accessed from single :class:`~lumicks.pylake.fdcurve.FdCurve` instances using :meth:`~lumicks.pylake.fdcurve.FdCurve.range_selector`::

    plt.figure()
    fdcurve = fdcurves["40"]
    t_selector = fdcurve.range_selector()

.. image:: figures/nbwidgets/single_curve_widget1.png

And for the distance-based selector we can use :meth:`~lumicks.pylake.fdcurve.FdCurve.distance_range_selector`::

    d_selector = fdcurve.distance_range_selector()

.. image:: figures/nbwidgets/single_curve_widget2.png

.. _crop_and_rotate:

Cropping and Rotating Image Stacks
----------------------------------

You can interactively define the location of a tether for a :class:`~lumicks.pylake.ImageStack` by using :meth:`~lumicks.pylake.ImageStack.crop_and_rotate`::

    stack = lk.ImageStack("test_data/tether.tiff")
    editor = stack.crop_and_rotate()
    plt.show()

Simply left-click on the start of the tether

.. image:: figures/nbwidgets/widget_stack_editor_1.png
  :nbattach:

and then on the end of the tether

.. image:: figures/nbwidgets/widget_stack_editor_2.png
  :nbattach:

After a tether is defined, the view will update showing the location of the tether and the
image rotated such that the tether is horizontal.

To crop an image, right-click and drag a rectangle around the region of interest. Once the rectangle is defined,
you can edit the shape by right-clicking and dragging the various handles.

.. image:: figures/nbwidgets/widget_stack_editor_3.png
  :nbattach:

You can also use the mouse wheel to scroll through the individual frames (if using Jupyter Lab, hold `Shift` while scrolling).

*Note that* :meth:`~lumicks.pylake.ImageStack.crop_and_rotate` *accepts all of the arguments
that can be used for* :meth:`~lumicks.pylake.ImageStack.plot()`.

To obtain a copy of the edited :class:`~lumicks.pylake.ImageStack` object, use::

    plt.figure()
    new_stack = editor.image
    new_stack.plot()
    new_stack.plot_tether()

.. image:: figures/nbwidgets/widget_stack_editor_4.png

.. _kymotracker_widget:

Kymotracking
------------

.. note::
    For details of the tracking algorithms and downstream analyses see the :doc:`/tutorial/kymotracking` tutorial.

For tracking binding events on a kymograph, using the :func:`~lumicks.pylake.track_greedy` algorithm purely by function calls can be challenging if not all parts
of the kymograph look the same or when the signal to noise ratio is somewhat low.
To help with this, we included a kymotracking widget that can help you track subsections of the kymograph and iteratively tweak the algorithm parameters as you do so.
You can open this widget by creating a :class:`~lumicks.pylake.KymoWidgetGreedy` as follows::

    file = lk.File("test_data/kymo.h5")
    kymo = file.kymos["16"]
    kymowidget = lk.KymoWidgetGreedy(kymo, "green", axis_aspect_ratio=2)

.. image:: figures/nbwidgets/kymotracker.png

Here we see the optional `axis_aspect_ratio` argument that allows us to control the aspect ratio of the plot and how much data is visible at a given time.
You can easily pan horizontally by clicking and dragging left or right.

You can optionally also pass algorithm parameters when opening the widget::

    lk.KymoWidgetGreedy(kymo, "green", axis_aspect_ratio=2, min_length=4, pixel_threshold=9, window=6, sigma=0.2)

.. image:: figures/nbwidgets/kymotracker2.png

You can also change the range of each of the algorithm parameter sliders. To do this, simply pass a dictionary where the key indicates the algorithm
parameter and the value contains its desired range in the form `(minimum bound, maximum bound)`. For example::

    widget = lk.KymoWidgetGreedy(
        kymo,
        "green",
        axis_aspect_ratio=2,
        min_length=4,
        pixel_threshold=9,
        slider_ranges={"window": (0, 20)},
        window=20,
    )

.. image:: figures/nbwidgets/kymotracker3.png

You can perform tracking by clicking the `Track all` button.

.. image:: figures/nbwidgets/kymotracker4.png

Detected tracks are accessible through the :attr:`~lumicks.pylake.KymoWidgetGreedy.tracks` property::

    >>> print(kymowidget.tracks)
    KymoTrackGroup(N=132)

For more information on its use, please see the example :ref:`cas9_kymotracking`.
