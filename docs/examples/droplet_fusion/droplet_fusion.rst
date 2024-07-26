
Droplet fusion
==============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _droplet_fusion:

Analyzing a droplet fusion event
--------------------------------

The data in this notebook were acquired during an experiment where two protein droplets are brought together at a constant speed using optical tweezers. 
As the droplets come close together, they fuse to form one, larger droplet.
The relaxation time of the fusion event, together with the size of the droplets can reveal something about the material properties of the droplet.
In this Notebook, will obtain the relaxation time from the force signal and estimate the size of the droplets from the scan.

Download the droplet fusion data
--------------------------------

The droplet fusion data are stored on zenodo.org.
We can download the data directly from Zenodo using the function :func:`~lumicks.pylake.download_from_doi`.
The data will be stored in the folder called `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.12772709", "test_data")

Relaxation time of fusion event
-------------------------------

First, plot the force signal using Pylake. 
Since the assumptions underlying force calibration are not met during the fusion event, the absolute value of the force is not reliable, and we label the y-axis as 'laser signal'::

    f = lk.File("test_data/Droplet_fusion_data.h5")
    plt.figure()
    f["Force HF"]["Force 2x"].plot()  
    plt.ylabel("Laser signal (a.u.)")
    plt.show()

.. image:: force_signal.png

The jump in the signal after 5 seconds shows the typical exponential relaxation for a droplet fusion event.

Select data for fit
^^^^^^^^^^^^^^^^^^^
Below we are selecting the force and trap data at the fusion event. When fitting the fusion relaxation time, 
it is important that the traps holding the droplets are either both static, or one of the traps is moving at a constant speed. 
We plot the trap position over time to make sure one of these conditions is met::

    start = "5.4s"
    stop = "5.9s"

    force_selection = f.force2x[start:stop]
    trap_selection = f["Trap position"]["1X"][start:stop]

    plt.figure()
    plt.subplot(2, 1, 1)
    force_selection.plot()
    plt.ylabel("Laser signal (a.u.)")
    plt.subplot(2, 1, 2)
    trap_selection.plot()
    plt.ylabel("x-coordinate ($\mu$m)")
    plt.tight_layout()
    plt.show()

.. image:: selected_data.png

Model for fusion
^^^^^^^^^^^^^^^^

We will be fitting the force data during the fusion event with the following equation: :math:`f(t) = ae^{-t/\tau}+bt+c`

The term $bt$ accounts for the movement of the trap, assuming a constant trap speed [1]_. The parameter of interest is $\tau$, the relaxation time schale of the fusion event::

    from scipy.optimize import curve_fit

    def relaxation_model(t, tau, a, b, c):
        return a * np.exp(-t / tau) + b * t + c

Fit the data and plot the result::

    time = force_selection.seconds
    force = force_selection.data

    popt, pcov = curve_fit(relaxation_model, time, force, [0.1,force[0],0,0])
    print(popt)
    plt.figure()
    plt.plot(time,force)
    plt.plot(time,relaxation_model(time,*popt),label=f'$\\tau$ = {popt[0]:0.2f}s')
    plt.legend()
    plt.show()

.. image:: fit.png

The relaxation time obtained from the fit is 0.05 seconds. 

Now we will proceed to determine the size of the droplets before the fusion event.

Droplet size
------------

First load the scan and print the relevant metadata::

    >>> for name, scan in f.scans.items():
    >>>        print(f"num frames: {scan.num_frames}")
    >>>        frame_duration = (scan.frame_timestamp_ranges()[0][1]-scan.frame_timestamp_ranges()[0][0])/1e9
    >>>        print(f"frame duration: {frame_duration} s")

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
The estimate will not be extremely precise, because the right droplet is moving as the scan is recorded, and the boarders of the scanned images are not sharp.
The estimate in the size will therefore have an error of ~10%.

The first step, is to use image segmentation to identify the two droplets in the image. 
The threshold may need to be optimized for your data::

    import math
    from skimage.measure import label, regionprops

    framenr = 2  # Choose a frame before the fusion event on which you want to identify and measure droplets

    image = scan.get_image(channel='red')[framenr]
    image = image / np.max(image)
    threshold = 0.5
    blobs = image > threshold
    label_img = label(blobs)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("original, normalized image")
    plt.imshow(image)
    plt.subplot(2, 1, 2)
    plt.title(f"Identified objects")
    plt.imshow(label_img)
    plt.tight_layout()
    plt.show()

.. image:: image_segmentation.png

Estimate the size of the droplet by measuring the length of the longest axis and the shortest axis and taking the average::

    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    average_droplet_radii = np.array([])  # List of arrays to collect all droplet radii

    for n, props in enumerate(regions):
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), "-r", linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), "-r", linewidth=2.5)
        ax.plot(x0, y0, ".g", markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, "-b", linewidth=2.5)
       
        average_droplet_radii = np.append(
            average_droplet_radii, 0.25 * (props.minor_axis_length + props.major_axis_length)
        )
    plt.title("The short and long axis of the identified objects")
    plt.show()

.. image:: droplet_size_pixels.png

The array `average_droplet_radii` contains the average of the large and small radius for all the droplets in the image, in pixels. 
Below we are multiplying this array by the pixel size in micron to obtain the radii in micron::

    average_droplet_radii_um = average_droplet_radii * scan.pixelsize_um[0]

The average radii for the droplets in micrometers are::

    >>> print(average_droplet_radii)

    [11.05117733 10.73977514]

We now determined the relaxation time as well as the droplet radii. The next step is to measure these two quantities for many different fusion events and plotting 𝜏 vs average radius.
For a Newtonian fluid, the slope of this plot is given by 𝜂/𝛾, viscosity (Pa*s) /surface tension (N/m), also known as the inverse capillary velocity [2]_. 

.. [1]  Patel A. *et* al, A Liquid-to-Solid Phase Transition of the ALS Protein FUS Accelerated by Disease Mutation, Cell (2015).
.. [2]  Brangwynne C.P. *et al*, Germline P Granules Are Liquid Droplets That Localize by Controlled Dissolution/Condensation, Science (2009).
