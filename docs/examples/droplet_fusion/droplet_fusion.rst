.. warning::
    Disclaimer: This analysis has not yet been tested in a large number of different scenarios.

Droplet fusion
==============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _droplet_fusion:

Analyzing a droplet fusion event
--------------------------------

The data in this notebook were acquired by moving two fluorescently labeled RNA droplets together at a constant speed.
The droplets were held by optical tweezers and the right tweezers is moving while the left one is static.
As the droplets come close together, they fuse to form one, larger droplet.
The relaxation time of the fusion event, 𝜏, together with the radii of the droplets can reveal something about the material properties of the droplets;
when plotting 𝜏 vs the average droplet radius for many droplets, the slope is given by 𝜂/𝛾, viscosity (Pa*s) /surface tension (N/m), assuming a Newtonian fluid [1]_. 
The ratio 𝜂/𝛾 is also known as the inverse capillary velocity.

Both the PSD signal (usually used to determine force exerted by the optical tweezers) and a scan were recorded during the experiment.
The PSD signal has a much higher time resolution than the images, therefore it is best to use PSD signal to determine the relaxation time of the fusion event. 
In this Notebook, we will first obtain the relaxation time from the PSD signal and then estimate the size of the droplets from the scan.

Download the droplet fusion data
--------------------------------

The droplet fusion data are stored on zenodo.org.
We can download the data directly from Zenodo using the function :func:`~lumicks.pylake.download_from_doi`.
The data will be stored in the folder called `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.12772709", "test_data")

Relaxation time of fusion event
-------------------------------

First, plot the PSD signal using Pylake. 
Since the assumptions underlying force calibration are not met during the fusion event, the absolute value of the force is not reliable, and we label the y-axis as 'PSD signal'::

    f = lk.File("test_data/Droplet_fusion_data.h5")
    plt.figure()
    f["Force HF"]["Force 2x"].plot()  
    plt.ylabel("PSD signal (a.u.)")
    plt.show()

.. image:: force_signal.png

The jump in the signal after 5 seconds shows the typical exponential relaxation for a droplet fusion event.

Select data for fit
^^^^^^^^^^^^^^^^^^^

Below we are selecting the force and trap data at the fusion event. When fitting the fusion relaxation time, 
it is important that the traps holding the droplets are either both static, or one of the traps is moving at a constant speed. 
We plot the trap position over time to check which of these conditions is met::

    start = "5.414s"
    stop = "5.9s"

    force_selection = f.force2x[start:stop]
    trap_selection = f["Trap position"]["1X"][start:stop]

    plt.figure()
    plt.subplot(2, 1, 1)
    force_selection.plot()
    plt.ylabel("PSD signal (a.u.)")
    plt.subplot(2, 1, 2)
    trap_selection.plot()
    plt.ylabel(r"x-coordinate ($\mu$m)")
    plt.tight_layout()
    plt.show()

.. image:: selected_data.png

Model for fusion
^^^^^^^^^^^^^^^^

The force data during the fusion event is fitted with the following equation: :math:`f(t) = ae^{-t/\tau}+bt+c`

The term :math:`bt` accounts for the movement of the trap, assuming a constant trap speed ([2]_, [3]_). 
(When both traps are static, the term :math:`bt` should be removed from the model.)
The parameter of interest is :math:`𝜏`, the relaxation time scale of the fusion event::

    from scipy.optimize import curve_fit

    def relaxation_model(t, tau, a, b, c):
        return a * np.exp(-t / tau) + b * t + c

Fit the data and plot the result::

    time = force_selection.seconds
    force = force_selection.data

    popt, pcov = curve_fit(relaxation_model, time, force, [0.1, force[0], 0, 0])
    plt.figure()
    plt.plot(time, force)
    plt.plot(time, relaxation_model(time,*popt), label=fr"$\tau$ = {popt[0]:0.4f}s")
    plt.ylabel(r"PSD signal (a.u.)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

.. image:: fit.png

The array :math:`popt` contains all the fitted parameters::

    >>> print(popt)

    [ 0.0557409   3.78890848 -4.19208305 -6.46343689]

The first parameter in :math:`popt` is :math:`𝜏` and the other 3 parameters are :math:`a`, :math:`b` and :math:`c` respectively, as defined in the model above.
The matrix :math:`pcov` is the covariance matrix and the standard deviation errors in the fitted parameters can be obtained as::

    >>> np.sqrt(np.diag(pcov))

    [0.00035864, 0.01059128, 0.02548404, 0.00926486]

The relaxation time obtained from the fit is 0.0557 +- 0.0004 seconds. 

In practice, the obtained relaxation time also depends on the data selection. 
It is recommended to repeat the fit for multiple time intervals, and determine the uncertainty in the relaxation time accordingly.

Now, we will proceed to determine the size of the droplets before the fusion event.

Droplet size
------------

First load the scan and print the relevant metadata::

    >>> for name, scan in f.scans.items():
    ...        print(f"num frames: {scan.num_frames}")
    ...        frame_duration = (scan.frame_timestamp_ranges()[0][1]-scan.frame_timestamp_ranges()[0][0])/1e9
    ...        print(f"frame duration: {frame_duration} s")

    num frames: 27
    frame duration: 0.4977664 s

Plot a frame before the fusion event::

    framenr = 2
    plt.figure()
    scan.plot(channel="green", frame=framenr)
    plt.show()

.. image:: frame2.png

Plot a frame after the fusion event::

    framenr = 6
    plt.figure()
    scan.plot(channel="green", frame=framenr)
    plt.show()

.. image:: frame6.png

If the droplets are in focus, the size of the droplet can be estimated from the 2D scan. 
The estimate has limited precision because the sphere edges in the scanned images are not very sharp.
For experimental data such as the one used in this notebook, we would expect an error on the order of ~10%.

The first step, is to use image segmentation to identify the two droplets in the image. 
The threshold may need to be optimized for your data::

    from skimage.measure import label, regionprops

    framenr = 2  # Choose a frame before the fusion event on which you want to identify and measure droplets

    image = scan.get_image(channel="red")[framenr]
    image = image / np.max(image)
    threshold = 0.5
    blobs = image > threshold
    label_img = label(blobs)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("original, normalized image")
    plt.imshow(image)
    plt.subplot(2, 1, 2)
    plt.title("Identified objects")
    plt.imshow(label_img)
    plt.tight_layout()
    plt.show()

    if (blobs := len(np.unique(label_img))) != 3:
        raise RuntimeError(f"Expected 2 blobs, found {blobs - 1} instead! Maybe adjust the threshold?")

.. image:: image_segmentation.png

For this scan, the fast axis is along the horizontal coordinate (you can check the direction of the fast axis by typing :attr:`scan.fast_axis <lumicks.pylake.scan.Scan.fast_axis>`). 
Therefore, we estimate the size of the droplets by looking at the width of the identified objects:::

    def get_center_and_width(scan, mask, axis):
        """Grabs the center and width along the fast scanning axis"""
        widths = np.sum(mask, axis=axis)
        max_width = np.max(widths)

        # Grab the position
        coordinate_weighted_mask = np.indices(mask.shape)[axis] * mask
        centers = np.sum(coordinate_weighted_mask, axis=axis) / np.clip(np.sum(mask, axis=axis), 1, np.inf)
        
        # Since some scanlines can have the same width, we'd want the vertical position to be the average of these
        max_scanline = int(np.mean(np.nonzero(max_width == widths)[0]))

        if axis:
            center = (centers[max_scanline], max_scanline)
        else:
            center = (max_scanline, centers[max_scanline])
        
        return center, max_width


    def plot_width(scan, center, width, axis):
        plt.plot(center[0], center[1], "ko")
        if axis:
            plt.plot([center[0] - 0.5 * width, center[0] + 0.5 * width], [center[1], center[1]])
        else:
            plt.plot([center[0], center[0]], [center[1] - 0.5 * width, center[1] + 0.5 * width])

    droplet_radii = np.array([]) 

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    axis = 1 if scan.fast_axis == "X" else 0
    center, width = get_center_and_width(scan, label_img == 1, axis)
    droplet_radii = np.append(droplet_radii, 0.5 * width)
    plot_width(scan, center, width, axis)
    center, width = get_center_and_width(scan, label_img == 2, axis)
    droplet_radii = np.append(droplet_radii, 0.5 * width)
    plot_width(scan, center, width, axis)

    plt.title("Width along the fast axis")
    plt.show()

.. image:: droplet_size_pixels.png

The array `droplet_radii` contains the radii of both droplets in the image, in pixels. 
Below we are multiplying this array by the pixel size in micron to obtain the radii in micron::

    droplet_radii_um = droplet_radii * scan.pixelsize_um[0]

The radii for the droplets in micrometers are::

    >>> print(droplet_radii_um)

    [1.15 1.1]

We now determined the relaxation time as well as the droplet radii. The next step is to measure these two quantities for many different fusion events, plot 𝜏 vs average radius and determine the slope.


.. [1]  Brangwynne C.P. *et al*, Germline P Granules Are Liquid Droplets That Localize by Controlled Dissolution/Condensation, Science (2009)
.. [2]  Patel A. *et al*, A Liquid-to-Solid Phase Transition of the ALS Protein FUS Accelerated by Disease Mutation, Cell (2015)
.. [3]  Kaur T. *et al*, Molecular Crowding Tunes Material States of Ribonucleoprotein Condensates, Biomolecules (2019) 
