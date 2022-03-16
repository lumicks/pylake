.. warning::
    This is beta functionality. While usable, this has not yet been tested in a large
    number of different scenarios. The API may also still be subject to change.

Cas9 kymotracking
=================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _cas9_kymotracking:

Analyzing Cas9 binding to DNA
-----------------------------

In this notebook we will analyze some measurements of Cas9 activity obtained while stretching the DNA tether at
different forces.

Choosing an interactive backend allows us to interact with the plots. Note that depending on whether you are using
Jupyter notebook or Jupyter lab, you should be using a different interactive backend. For Jupyter lab that means also
installing `ipympl <https://github.com/matplotlib/ipympl>`_ as an extra dependency. Let's begin by importing the
required Python modules and choosing an interactive backend for `matplotlib`::

    import matplotlib.pyplot as plt
    import numpy as np

    # Use notebook if you're in Jupyter Notebook
    # %matplotlib notebook

    # Use widget (depends on ipympl) if you're using Jupyter lab
    %matplotlib widget

Plotting the kymograph
----------------------

Let’s load our Bluelake data and have a look at what the kymograph looks like. We can easily grab the kymo by calling
`popitem()` on the list of kymos, which returns the first kymograph::

    file = lk.File('test_data/cas9_kymo_compressed.h5')
    _, kymo = file.kymos.popitem()

In this experiment, force was measured alongside the kymograph. Let’s plot them together to get a feel for what the
data looks like::

    plt.figure(figsize=(7, 4))

    # Plot the kymograph
    ax1 = plt.subplot(2, 1, 1)

    # We use aspect="auto" because otherwise the kymograph would be very long and thin
    kymo.plot("green", adjustment=lk.ColorAdjustment(0, 4), aspect="auto")

    # Plot the force
    ax2 = plt.subplot(2, 1, 2, sharex = ax1)
    plt.xlim(ax1.get_xlim())
    file["Force LF"]["Force 1x"].plot()
    plt.tight_layout()

    plt.show()

.. image:: kymo_force.png

Note how color adjustment is specified for the kymograph. Any photon count outside the range provided to :class:`~lumicks.pylake.ColorAdjustment` will be clipped to the nearest color.

What we can observe in this data is that as more force is applied, we get an increased binding activity. Let’s see
if we can put the kymotracker to some good use and quantify these.

Computing the background
------------------------
First, we select a small region without traces to determine the background signal::

    background = kymo["100s":"200s"].crop_by_distance(28, 31)
    green_background_per_pixel = np.mean(background.get_image("green"))

Downsampling the kymograph
--------------------------

To make it a bit easier to tweak the algorithm parameters, we will make use of a notebook widget.
While we could work on the full time resolution data, we can make things a little easier for the kymotracking algorithm by downsampling the data a little bit.
We downsample the data by a factor of `2`::

    kymo_ds = kymo.downsampled_by(2)

Performing the kymotracking
---------------------------

Now that we’ve loaded some data, we can begin tracing lines on it. For this, we open the widget.
This will provide us with a view of the kymograph and some dials to tune in algorithm settings. We open with a
custom `axis_aspect_ratio` which determines our field of view. This input is not necessary, but provides a better
view of our data.

We will be using the greedy algorithm. The `threshold` should typically be chosen somewhere between the expected
baseline photon count and the photon count of a true line (note that you can see the local photon count between square
brackets while hovering over the kymograph). The `line width` should roughly be set to the expected line width of a
line. The `window` should be chosen such that small gaps in a trace can be overcome, but not so large that spurious
points may be strung together as a trace. `Sigma` controls how much the location can fluctuate from one point to the
next, while the `min length` determines how many peak points should be in a trace for it to be considered a valid
trace.

Holding down the left mouse button and dragging pans the view, while the right mouse button allows us to drag a region
where we should trace lines. Any line which overlaps with the selected area will be removed before tracing new ones.

The icon with the little square can be used to toggle zoom mode, which will allow you to zoom in one subsection of the
kymograph. Clicking it again brings us back out of zoom mode. You can zoom out again by clicking the home button. Quite
often, it is beneficial to find some adequate settings for track all, and then fine-tune the results using the manual
rectangle selection. It’s not mandatory to use the same settings throughout the kymograph. For example, if you see a
particular event where two lines are disconnected that should be connected, temporarily increase the window size and
just drag a rectangle over that particular line while having the option `Track lines` enabled.

Now, let’s track some traces. There are two ways to approach this analysis. The first is to just use the rectangle
selection, which can be quite time intensive. Alternatively, you can use `Track all` to simply track all lines found
in the kymograph, and then remove spurious detections by hand. This can be good to get a feel for the parameters as
well. If we select the `Remove lines` mode we will start removing lines without grabbing new ones. This
functionality can be used to remove spurious detections.

Finally, if you wish to connect two lines in the kymograph manually, you can switch to the `Connect Lines` mode.
In this mode you can click the end of one kymoline with the right mouse button and connect it to another by dragging to the start of the line you wish to connect it to.

Note that in this data for example, there are some regions where fluorescence starts building up on the surface of the
bead. This binding should be omitted from the analysis::

    kymowidget = lk.KymoWidgetGreedy(kymo_ds, "green", axis_aspect_ratio=2, min_length=4, pixel_threshold=3, window=6, sigma=0.14, vmax=8)

.. image:: kymowidget.png

One last thing to note is that we assigned the `KymoWidgetGreedy` to the variable `kymowidget`. That means that from
this point on, we can interact with it through the handle name `kymowidget`.

Exporting from the widget results in a file that contains the line coordinates in pixels and real units.
If we also want to export the photon counts in a region around the traced line, we can include a `sampling_width`.
This sums the photon counts from `pixel_position - sampling_width` to (and including) `pixel_position + sampling_width`::

    kymowidget.save_lines("kymotracks_calibrated.txt", sampling_width=3)

Analyzing the results
---------------------

Once traced, the lines are available in `kymowidget.lines`. Lines have a `position` list and a `time` list. Let’s grab
the longest line we found, and have a look at its position over time::

    lengths = [len(line) for line in kymowidget.lines]

    # Get the index of the longest kymo line
    longest_index = np.argmax(lengths)

    # Select the longest line
    longest_line = kymowidget.lines[longest_index]

    plt.figure(figsize=(5, 3))
    plt.plot(longest_line.seconds, longest_line.position)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [$\mu$m]')
    plt.tight_layout()
    plt.show()

.. image:: kymo_position_over_time.png

The line coordinates can be used to sample the photon counts in the image. The example below demonstrates how to obtain the sum of the photon counts in a pixel region around the line from -3 to 3 (a line with a width of 7 pixels). The background per pixel as computed earlier is subtracted from the photon counts. Since the kymograph was downsampled by a factor 2 after computing the background, the background per pixel is multiplied by 2::

    window = 3
    bg_corrected = longest_line.sample_from_image(window) - (2 * window + 1) * 2 * green_background_per_pixel

    plt.figure()
    plt.plot(longest_line.seconds, bg_corrected)
    plt.ylabel('Photon count')
    plt.xlabel('Time [s]')
    plt.title('Photon counts along the longest line')
    plt.tight_layout()
    plt.show()

.. image:: photon_counts_longest.png

Since we are interested in how the binding events are affected by the applied force, let’s have a look how long the line
segments are when we compare them to the force::

    plt.figure(figsize=(6, 3))
    ax1 = plt.subplot(1, 1, 1)
    time = file["Force LF"]["Force 1x"].seconds
    force = file["Force LF"]["Force 1x"].data
    plt.plot(time, force)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [pN]')

    ax2 = ax1.twinx()
    line_start_times = np.array([line.seconds[0] for line in kymowidget.lines])
    line_stop_times = np.array([line.seconds[-1] for line in kymowidget.lines])
    line_durations = line_stop_times - line_start_times
    [plt.plot(line_start_times, line_durations, 'k.') for line in kymowidget.lines]
    plt.ylabel('Trace Duration [s]')
    plt.xlabel('Start time [s]')
    plt.tight_layout()

.. image:: line_duration_vs_force.png

However, what we wanted to know was how the force affects initiation. To determine this, we will need to know the force
at which events were started. To do this, we compare the `line_start_time` we just computed to the time in the force
channel. What we want is the index with the smallest distance to our line start time. We can use `np.argmin()` for
this, which will return the index of the minimum value in a list. Once we have the index, we can quickly look up the
force for each line start position::

    force_index = [np.argmin(np.abs(time - line_start_time)) for line_start_time in line_start_times]
    line_forces = force[force_index]

We can look at the number of events started at each force by making a histogram of these start events. Let's make a
`10` bin histogram for forces from `10` to `60`::

    events_started, edges = np.histogram(line_forces, 10, range=(10, 60))

Since we didn’t spend an equal amount of time in each force bin, we should normalize by the time spent in each force
bin. We can also compute this with a histogram::

    samples_spent_at_force, edges = np.histogram(force, 10, range=(10, 60))

And that gives us sufficient information to make the plot::

    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure()
    plt.plot(centers, events_started / samples_spent_at_force)
    plt.xlabel('Force [pN]')
    plt.ylabel('Average # binding events / # force samples')

.. image:: binding_vs_force.png
