Confocal images
===============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

The following code uses scans as an example.
Kymographs work the same way -- just substitute `file.scans` with `file.kymos`.
To load an HDF5 file and lists all of the scans inside of it, run::

    import lumicks.pylake as lk

    file = lk.File("example.h5")
    list(file.scans)  # e.g. shows: "['reference', 'bleach', 'imaging']"

Once again, `.scans` is a regular Python dictionary so we can easily iterate over it::

    # Plot all scans in a file
    for name, scan in file.scans.items():
        scan.plot(channel="rgb")
        plt.savefig(name)

Or just pick a single one::

    scan = file.scans["name"]
    scan.plot("red")

Scan data and details
---------------------

You can access the raw image data directly. For a `Scan` with only a single frame::

    rgb = scan.rgb_image  # matrix with `shape == (h, w, 3)`
    blue = scan.blue_image  # single color so `shape == (h, w)`

    # Plot manually
    plt.imshow(rgb)

For scans with multiple frames::

    # returned data has `shape == (n_frames, h, w, 3)`
    rgb = multiframe_scan.rgb_image
    # returned data has `shape == (n_frames, h, w)`
    blue = multiframe_scan.blue_image

    # Manually plot the RGB image of the first frame.
    plt.imshow(rgb[0, :, :, :])

We can also slice out a subset of frames from an image stack::

    sliced_scan = multiframe_scan[5:10]

This will return a new `Scan` containing data equivalent to:

    multiframe_scan.rgb_image[5:10, :, :, :]

The images contain pixel data where each pixel represents summed photon counts.
For an even lower-level look at data, the raw photon count samples can be accessed::

    photons = scan.red_photons
    plt.plot(photons.timestamps, photons.data)

There are also several properties available for convenient access to the scan metadata:

* `scan.center_point_um` provides a dictionary of the central x, y, and z coordinates of the scan in micrometers relative to the brightfield field of view
* `scan.size_um` provides a list of scan sizes in micrometers along the axes of the scan
* `scan.pixelsize_um` provides the pixel size in micrometers
* `scan.lines_per_frame` provides the number scanned lines in each frame (number of rows in the raw data array)
* `scan.pixels_per_line` provides the number of pixels in each line of the scan (number of columns in the raw data array)
* `scan.fast_axis` provides the fastest axis that was scanned (x or y)
* `scan.num_frames` provides the number of frames available


Plotting and Exporting
----------------------

As shown above, there are convenience functions for plotting either the full RGB image or a single color channel.
If a few pixels dominate the image, one might want to set the scale by hand. We can pass an extra argument to `plot_red`
named `vmax` to accomplish this. This parameter gets forwarded to :func:`matplotlib.pyplot.imshow`::

    scan.plot(channel="red", vmax=5)

Multi-frame scans are also supported::

    print(scan.num_frames)
    print(scan.blue_image.shape)  # (self.num_frames, h, w) -> single color channel
    print(scan.rgb_image.shape)  # (self.num_frames, h, w, 3) -> three color channels

    scan.plot("green", frame=3)  # plot the third frame -- defaults to the first frame if no argument is given

The images can also be exported in the TIFF format::

    scan.save_tiff("image.tiff")

Scans can also be exported to video formats.
Exporting the red channel of a multi-scan GIF can be done as follows for example::

    scan.export_video_red("test_red.gif")

Or if we want to export a subset of frames (the first frame being 10, and the last frame being 40) of all three channels
at a frame rate of 40 frames per second, we can do this::

    scan.export_video_rgb("test_rgb.gif", start_frame=10, end_frame=40, fps=40)

For other video formats such as `.mp4` or `.avi`, ffmpeg must be installed. See
:ref:`installation instructions <ffmpeg_installation>` for more information on this.


Correlating scans
-----------------

We can downsample a scan according to the frames in a scan. We can use :func:`~lumicks.pylake.scan.Scan.frame_timestamp_ranges()` for this::

    frame_timestamp_ranges = scan.frame_timestamp_ranges()

This returns a list of start and stop timestamps that can be passed directly to :func:`~lumicks.pylake.channel.Slice.downsampled_to`, which will then return a :class:`~lumicks.pylake.channel.Slice` with a datapoint per frame::

    downsampled = f.force1x.downsampled_over(frame_timestamp_ranges)

We can also correlate multi-frame confocal scans with a channel :class:`~lumicks.pylake.channel.Slice` using a small interactive plot::

    scan.plot_correlated(f.force1x)
