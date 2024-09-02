
Kymograph with markers
======================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _kymo_markers:

Convert kymograph coordinates to basepairs using markers
--------------------------------------------------------

In this notebook we use two red markers with known coordinates on a kymograph to convert coordinates from micrometers to base pair.
The workflow is as follows:

- Determine the location of the red markers using a peak detection algorithm on the red channel 
- Use the known coordinates of the markers to convert coordinates to basepairs

We then compare the binding profile, with base pair coordinates, to the expected target sites for the protein.

This Notebook requires the `peakutils` package. Run the following cell to make sure it is installed:

Download the data file
----------------------

The kymograph is stored on zenodo.org.
We can download the data directly from Zenodo using the function :func:`~lumicks.pylake.download_from_doi`.
The data will be stored in the folder called `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.7729525", "test_data")

Plot the kymograph
------------------

Before starting the analysis on the high frequency data, let's look at the complete kymograph::

    file = lk.File("test_data/kymo.h5")
    kymo = file.kymos["16"]
    plt.figure()
    kymo.plot("rgb", adjustment=lk.ColorAdjustment(0, 98, mode="percentile"))

.. image:: kymo_complete.png

The above image shows the full kymograph. 

Markers and target sites
------------------------

Provide the binding coordinates of the red markers and the two expected target sites for the gren protein, on lambda DNA::

    marker1 = 33.786  # The location of the markers in kbp
    marker2 = 44.826
    dna_kbp = 48.502  # length of DNA in kbp
    target_sites = [18.084, 30.533]  # Target binding sites


Detect markers on kymograph
---------------------------

First, indicate the location of the bead edges by looking at the image of the kymograph. These coordinates are used to crop the kymograph::

    top_bead_edge = 9  # Rough location of bead edges in um
    bottom_bead_edge = 24.56

The following analysis requires the force to be constant, therefore we only choose the time interval from 10 seconds to 35 seconds.
Compute the average red profile for the selected area of the kymograph::

    kymo_calibration = kymo["10s":"35s"].crop_by_distance(top_bead_edge, bottom_bead_edge) 
    profile_red = np.mean(kymo_calibration.get_image("red"), axis=1)/np.max(np.mean(kymo_calibration.get_image('red'), axis=1))
    x = np.arange(len(profile_red))*kymo_calibration.pixelsize_um

    plt.figure()
    plt.plot(x,profile_red, 'r')
    plt.title("Red profile")
    plt.xlabel("Position (um)")
    plt.ylabel("Normalized intensity (a.u.)")
    plt.savefig("profile_red.png")

.. image:: profile_red.png

The two highest peaks correspond to the locations of the markers. Use a peak finding algorithm to determined the precise location of these peaks::

    indexes = peakutils.indexes(profile_red, thres=0.4, min_dist=30)
    print(indexes) 
    peaks_x = peakutils.interpolate(x, profile_red, ind=indexes, width = 2)
    print(peaks_x)

    plt.figure()
    plt.plot(x,profile_red, 'r')
    plt.vlines(peaks_x, ymin=0, ymax = 1)
    plt.title("Identified peaks")
    plt.xlabel("Position (um)")
    plt.ylabel("Normalized intensity (a.u.)")

.. image:: profile_red_peaks.png

If the above analysis results in less or more than two identified peaks, the values of `thres` or the `min_dist` may have to be adjusted.

Functions for conversion to basepairs and vice versa
----------------------------------------------------

The following functions use the peak coordinates to convert from micron to base pairs, or vice versa.
The functions also check whether kymograph has to be flipped::

    def um_to_kbp(coord, maxx, peak1, peak2, coord1 =  marker1, coord2 = marker2):
    """Convert coordinates along the kymo in micron to kbp
    
        Parameters
        -----------
        coord: coordinate in um to be converted
        maxx: Max x-coordinate assuming that the coordinates are from one bead ege to the next. 
        This value is used to determine whether the coordinates have to be flipped
        peak1: coordinate of first peak um
        peak2: coordinate of second peak in um
        coord1: coordinate of first reference dye in kbp
        coord2: coordinate of second reference dye in kbp
        
        Typical use: um_to_kbp(coord, maxx = np.max(x), peak1 = peaks_x[0], peak2 = peaks_x[1], coord1 =  marker1, coord2 = marker2)
        
        returns:
        coordinate x converted to kbp
        """
        if maxx - peak1 - peak2 < 0:
            a = (coord2 - coord1)/(peak2 - peak1)
            b = coord1 - a*peak1
            c = 0
        else: # Flip coordinates if peaks are in the top half of the kymo
            a = -(coord2 - coord1)/(peak2 - peak1)
            b = coord1 - a*peak1 
            c = coord2 - coord1
        return a*coord + b + c


    def kbp_to_um(coord_kbp, maxx, peak1, peak2, coord1 =  marker1, coord2 = marker2 ):
        """Conver coordinates along the kymo in micron to kbp
        
        Parameters
        -----------
        coord_kbp: coordinate in kbp to be converted
        maxx: Max x-coordinate assuming that the coordinates are from one bead ege to the next. 
        This value is used to determine whether the coordinates have to be flipped
        peak1: coordinate of first peak um
        peak2: coordinate of second peak in um
        coord1: coordinate of first reference dye in kbp
        coord2: coordinate of second reference dye in kbp
        
        returns:
        coordinate x converted to kbp
        """
        if maxx - peak1 - peak2 < 0:
            a = (coord2 - coord1)/(peak2 - peak1)
            b = coord1 - a*peak1
            c = 0
        else: # Flip coordinates if peaks are in the top half of the kymo
            a = -(coord2 - coord1)/(peak2 - peak1)
            b = coord1 - a*peak1
            c = peak2 - peak1
        return (coord_kbp - b)/a + c

Green profile with base pair coordinates
----------------------------------------

Using the identified peaks and the functions above, we can now convert the coordinates on the kymograph to basepairs::

    profile = np.mean(kymo_calibration.get_image('green'),axis=1)/np.max(np.mean(kymo_calibration.get_image("green"),axis=1))
    um_coords = np.arange(len(profile))*kymo_calibration.pixelsize_um
    kbp_coords = um_to_kbp(um_coords, maxx = np.max(x), peak1 = peaks_x[0], peak2 = peaks_x[1], coord1 =  marker1, coord2 = marker2)

    plt.figure()
    plt.plot(kbp_coords, profile, "lightgreen")
    for i in target_sites:
        plt.vlines(i,ymin=np.min(profile), ymax=1, color = "k")
    plt.xlabel("Position (kbp)")
    plt.ylabel("Normalized intensity (a.u.)")
    plt.title("Profile with target coordinates in kbp")
   
.. image:: green_profile.png

Kymograph with target sites overlaid
------------------------------------

We can also use the markers to convert the coordinates of the target sites from base pairs to micrometers, and overlay them with the kymograph.
Below, the coordinates of the markers are added as well, as a control::

    maxt=kymo_calibration.duration-1
    plt.figure(figsize=(8,8))
    kymo_calibration.plot(channel='rgb', aspect = 1, adjustment=lk.ColorAdjustment([0], [8]))
    for i in target_sites:
        plt.hlines(kbp_to_um(i, maxx = np.max(x), peak1 = peaks_x[0], peak2 = peaks_x[1]), xmin = 0, xmax=maxt, color = "yellow", linestyle = "dashed", linewidth = 0.5)
    plt.hlines(kbp_to_um(marker1, maxx = np.max(x), peak1 = peaks_x[0], peak2 = peaks_x[1]), xmin = 0, xmax=maxt, color = "white", linestyle = "dashed", linewidth = 0.5)
    plt.hlines(kbp_to_um(marker2, maxx = np.max(x), peak1 = peaks_x[0], peak2 = peaks_x[1]), xmin = 0, xmax=maxt, color = "white", linestyle = "dashed", linewidth = 0.5)

.. image:: kymo_target_sites.png

Quantify target binding
-----------------------

The profiles above can reveal overall enrichment on a target binding site. 
Uncertainty? How to choose the width around the target site? 

Further quantify target binding [1]_
- Track binding events using the Kymotracker, bin them, and determine which % is on/off target
- Compare the duration of binding on- vs off-target

.. [1] M. D. Newton, DNA stretching induces Cas9 off-target activity, NSMB, 7016-7018 (2019).