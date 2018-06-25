Confocal images
===============

The following code uses scans as an example.
Kymographs work the same way -- just substitute `file.scans` with `file.kymos`.
To load an HDF5 file and lists all of the scans inside of it, run::

    from lumicks import pylake

    file = pylake.File("example.h5")
    list(file.scans)  # e.g. shows: "['reference', 'bleach', 'imaging']"

Once again, `.scans` is a regular Python dictionary so we can easily iterate over it::

    # Plot all scans in a file
    for name, scan in file.scans.items():
        scan.plot_rgb()
        plt.savefig(name)

Or just pick a single one::

    scan = file.scans["name"]
    scan.plot_red()

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
