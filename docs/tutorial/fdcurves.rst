FD curves
=========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

The following code loads an HDF5 file and lists all of the FD curves inside of it::

    import lumicks.pylake as lk

    file = lk.File("example.h5")
    list(file.fdcurves)  # e.g. shows: "['baseline', '1', '2']"

To visualizes an FD curve, you can use the built-in `.plot_scatter()` function::

    # Pick a single FD curve
    fd = file.fdcurves["baseline"]
    fd.plot_scatter()

Here, `.fdcurves` is a standard Python dictionary, so we can do standard `dict` thing with it.
For example, we can iterate over all the FD curve in a file and plot them::

    for name, fd in file.fdcurves.items():
        fd.plot_scatter()
        plt.savefig(name)

By default, the FD channel pair is `downsampled_force2` and `distance1`.
This assumes that the force extension was done by moving trap 1, which is the most common.
In that situation the force measured by trap 2 is more precise because that trap is static.
The channels can be switched with the following code::

    alt_fd = fd.with_channels(force='1x', distance='2')
    alt_fd.plot_scatter()

    # or as quick one-liner for plotting
    fd.with_channels(force='2y', distance='2').plot_scatter()

The raw data can be accessed as well::

    # Access the raw data: default force and distance channels
    force = fd.f
    distance = fd.d

    # Access the raw data: specific channels
    force = fd.downsampled_force1y
    distance = fd.distance2

    # Plot manually: FD curve
    plt.scatter(distance.data, force.data)
    # Plot manually: force timetrace
    plt.plot(force.timestamps, force.data)

FD Ensembles
------------

It's also possible to work with multiple FD curves as an ensemble::

    fd_ensemble = lk.FdEnsemble(file.fdcurves)

We can align the FD curves using the align function::

    fd_ensemble.align_linear(distance_range_low=0.02, distance_range_high=0.02)

This aligns all the curves to the first and estimates an offset in force and distance, which is subtracted from the
data. Force is aligned by taking the mean of the lowest distances, while distance is aligned by considering the last
segment of each FD curve and regressing linear lines there, from which the offset is computed. Note that this requires
the ends of the aligned F,d curves to be in a comparably folded state and obtained in the elastic range of the force,
distance curve. If any of these assumptions are not met, this method should not be applied. We can obtain the force
and distance from such an ensemble using::

    f = fd_ensemble.f
    d = fd_ensemble.d
