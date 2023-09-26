Confocal Scans
==============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

We can download the data needed for this tutorial directly from Zenodo using Pylake.
Since we don't want it in our working folder, we'll put it in a folder called `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.7729636", "test_data")

The following code uses scans as an example.
Kymographs work the same way -- just substitute :attr:`file.scans <lumicks.pylake.File.scans>` with
:attr:`file.kymos <lumicks.pylake.File.kymos>`. To load an HDF5 file and list all of the scans
inside of it, run::

    import lumicks.pylake as lk

    file = lk.File("test_data/scan.h5")
    list(file.scans)  # e.g. shows: "['reference', 'bleach', 'imaging']"

:attr:`.scans <lumicks.pylake.File.scans>` is a regular Python dictionary so we can iterate over it::

    # Plot all scans in a file
    plt.figure()
    for name, scan in file.scans.items():
        scan.plot(channel="rgb")
        plt.savefig(name)
    plt.show()

Or just pick a single one by providing the name of the scan as ``scan=file.scans["name"]``::

    scan = file.scans["41"]
    plt.figure()
    scan.plot("red")
    plt.show()

.. _confocal_plotting:

Plotting and Exporting
----------------------

Pylake provides a convenience :meth:`plot()<lumicks.pylake.scan.Scan.plot>` method to quickly
visualize your data. For details and examples see the :doc:`plotting_images` section.

The scan can also be exported to TIFF format::

    scan.export_tiff("image.tiff")

Scans can also be exported to video formats. Exporting the red channel of a multi-scan GIF can be
done as follows::

    multiframe_scan.export_video(
        "red",
        "test_red.gif",
        adjustment=lk.ColorAdjustment([0], [4])
    )

Or if we want to export a subset of frames (the first frame being 2, and the last frame being 15) of all three channels
at a frame rate of 2 frames per second, we can do this::

    multiframe_scan.export_video(
        "rgb",
        "test_rgb.gif",
        start_frame=2,
        stop_frame=15,
        fps=2,
        adjustment=lk.ColorAdjustment([0, 0, 0], [4, 4, 4])
    )

For other video formats such as `.mp4` or `.avi`, ffmpeg must be installed. See
:ref:`installation instructions <ffmpeg_installation>` for more information on this.

The images contain pixel data where each pixel represents summed photon counts.
The photon count per pixel can be accessed as follows::

    photons = scan.red_photon_count
    plt.figure()
    plt.plot(photons.timestamps, photons.data)
    plt.show()

Scan metadeta
--------------
There are several properties available for convenient access to the scan metadata:

* :attr:`scan.center_point_um <lumicks.pylake.scan.Scan.center_point_um>` provides a dictionary of
  the central x, y, and z coordinates of the scan in micrometers relative to the brightfield field
  of view
* :attr:`scan.size_um <lumicks.pylake.scan.Scan.size_um>` provides the scan size in
  micrometers along the axes of the scan
* :attr:`scan.pixelsize_um <lumicks.pylake.scan.Scan.pixelsize_um>` provides the pixel size in
  micrometers
* :attr:`scan.lines_per_frame <lumicks.pylake.scan.Scan.lines_per_frame>` provides the number
  scanned lines in each frame (number of rows in the raw data array)
* :attr:`scan.pixels_per_line <lumicks.pylake.scan.Scan.pixels_per_line>` provides the number of
  pixels in each line of the scan (number of columns in the raw data array)
* :attr:`scan.fast_axis <lumicks.pylake.scan.Scan.fast_axis>` provides the fastest axis that was
  scanned (x or y)
* :attr:`scan.num_frames <lumicks.pylake.scan.Scan.num_frames>` provides the number of frames
  available
* :attr:`kymo.pixel_time_seconds <lumicks.pylake.scan.Scan.pixel_time_seconds>` provides the pixel
  dwell time.

Raw data and data selection
----------------------------

You can access the raw image data directly. For a :class:`~lumicks.pylake.scan.Scan` with only a single frame::

    rgb = scan.get_image("rgb")  # matrix with `shape == (h, w, 3)`
    blue = scan.get_image("blue")  # single color so `shape == (h, w)`

    # Plot manually
    plt.figure()
    plt.imshow(rgb)
    plt.show()

For scans with multiple frames::

    # returned data has `shape == (n_frames, h, w, 3)`
    rgb = multiframe_scan.get_image("rgb")
    # returned data has `shape == (n_frames, h, w)`
    blue = multiframe_scan.get_image("blue")

    # Manually plot the RGB image of the first frame.
    plt.figure()
    plt.imshow(rgb[0, :, :, :])
    plt.show()

We can also slice out a subset of frames from an image stack::

    sliced_scan = multiframe_scan[5:10]

This will return a new :class:`~lumicks.pylake.scan.Scan` containing data equivalent to::

    multiframe_scan.get_image("rgb")[5:10, :, :, :]

We can also slice the frames by time::

    # get frames corresponding to the time range 30 through 90 seconds
    sliced_scan = multiframe_scan["30s":"90s"]

Or directly using timestamps::

    # get frames that fall between the start and stop of a force channel
    multiframe_scan[multiframe_file.force1x.start:multiframe_file.force1x.stop]

Correlating a multiframe scan with data channels
-------------------------------------------------
The frames of a multiframe scan can be correlated to the force or other data channels.
Downsample channel data according to the frames in a scan using :func:`~lumicks.pylake.scan.Scan.frame_timestamp_ranges()`::

    frame_timestamp_ranges = multiframe_scan.frame_timestamp_ranges()

You can choose to add the flag `include_dead_time = True` if you want to include the dead time at the end of each frame (default is `False`). This returns a list of start and stop timestamps that can be passed directly to :func:`~lumicks.pylake.channel.Slice.downsampled_over`, which will then return a :class:`~lumicks.pylake.channel.Slice` with a datapoint per frame::

    downsampled = multiframe_file.force1x.downsampled_over(frame_timestamp_ranges)

The multi-frame confocal scans can also be correlated with a channel :class:`~lumicks.pylake.channel.Slice` using an interactive plot.  ::

    multiframe_scan.plot_correlated(multiframe_file.force1x, adjustment=lk.ColorAdjustment([0, 0, 0], [4, 4, 4]))
    plt.show()

Note that you need an interactive backend for this plot to work; instead of running ``%matplotlib inline`` at the top of the notebook, run ``%matplotlib notebook``. If some cells were already executed, you will need to restart the kernel as well.
