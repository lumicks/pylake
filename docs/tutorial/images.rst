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
        scan.plot_rgb()
        plt.savefig(name)

Or just pick a single one::

    scan = file.scans["name"]
    scan.plot_red()

If a few pixels dominate the image, one might want to set the scale by hand. We can pass an extra argument to `plot_red`
named `vmax` to accomplish this. This parameter gets forwarded to :func:`matplotlib.pyplot.imshow`::

    scan.plot_red(vmax=5)

Access the raw image data::

    rgb = scan.rgb_image  # matrix with `shape == (h, w, 3)`
    blue = scan.blue_image  # single color so `shape == (h, w)`

    # Plot manually
    plt.imshow(rgb)

The images contain pixel data where each pixel is made up a multiple photon count samples collected by the scanner.
For an even lower-level look at data, the raw photon count samples can be accessed::

    photons = scan.red_photons
    plt.plot(photons.timestamps, photons.data)

The images can also be exported in the TIFF format::

    scan.save_tiff("image.tiff")

Multi-frame scans are also supported::

    print(scan.num_frames)
    print(scan.blue_image.shape)  # (self.num_frames, h, w) -> single color channel
    print(scan.rgb_image.shape)  # (self.num_frames, h, w, 3) -> three color channels

    scan.plot(frame=3)  # plot the third frame -- defaults to the first frame if no argument is given

Scans can also be exported to video formats.
Exporting the red channel of a multi-scan GIF can be done as follows for example::

    scan.export_video_red("test_red.gif")

Or if we want to export a subset of frames (the first frame being 10, and the last frame being 40) of all three channels
at a frame rate of 40 frames per second, we can do this::

    scan.export_video_rgb("test_rgb.gif", start_frame=10, end_frame=40, fps=40)

For other video formats such as `.mp4` or `.avi`, ffmpeg must be installed. See
:ref:`installation instructions <ffmpeg_installation>` for more information on this.
