
Rate of binding
===============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _kon:

Determine the rate of binding from a kymograph
----------------------------------------------

In this Notebook, we will determine the binding time of a fluorescently labeled protein binding to DNA. The binding time is relates to the effective on rate of protein as binding time = 1/:math:`k_{on}*[P]`  
The binding rate :math:`k_{on}` relates tho the dissociation constant as.

.. math::

    K_{off} = \frac{k_{off}}{k_{on}}

For this example, we don't know the protein concentration and can therefore not determine :math:`k_{on}` . 
We will determine the binding time and refer to the inverse of the binding time, :math:`k_{on}*[P]` , as the *effective binding rate*.

To determine the effective binding rate, we look at a protein binding to a target site. We track the binding events, and then determine the time intervals *between* the binding events:

.. image:: kon.png

These time intervals tell you how long it takes for a protein to bind to an empy target site. 
In this experiment, we used a construct with a repeat sequence. Each repeat sequence contains 1 target site. Hence, we have multiple locations with target binding on the same kymograph.

Load and plot the kymographs
----------------------------

The kymograph and corresponding tracks that are used in this tutorial are stored on zenodo.org.
The following line of code downloads the data and stores the data in the folder `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.14198300", "test_data")

Load the first file and plot the kymograph::

    file1 = lk.File("test_data/kymo1.h5")
    _, kymo1 = file1.kymos.popitem()
    
    plt.figure()
    kymo1.plot("g", aspect = 5, adjustment=lk.ColorAdjustment([0], [5]))

.. image:: kymo1.png

Load the tracks
---------------

For this tutorial, the binding events have already been tracked in Pylake.
Load the tracks into Pylake::

    tracks1 = lk.load_tracks("test_data/tracks1.csv",  kymo1.crop_by_distance(4.9,13.8), "green")

Note that the kymograph passed to `lk.load_tracks` is cropped, because tracking was performed on a cropped kymograph, see :ref:`tracking`.

Use the same approach as above to load the tracks exported from Lakeview, except that the part :func:`Kymo.crop_by_distance() <lumicks.pylake.kymo.Kymo.crop_by_distance>` has to be removed.

Select target location
----------------------

First, plot the tracks::

    plt.figure()
    tracks1.plot()
    plt.show()

.. image:: tracks1.png

Next, select the coordinates of the target binding site, for which you would like to determine the on-rate.
Often, the location of a target site is identified using, for example, fluorescent markers.
On this kymograph, all binding events were on a target sequence. So we can select the target locations manually.

First, we select the following region::

    plt.figure()
    tracks1.plot()
    plt.hlines(y=5.4, xmin=0,xmax=320)
    plt.hlines(y=6.2, xmin=0, xmax=320)

.. image:: track_selection1.png

Select all tracks that are on average within the two coordinates as indicated in the above image::

    track_selection1 = tracks1[[5.4 < np.mean(track.position) < 6.2 for track in tracks1]]

Since we are using a repeat sequence and all observed binding events were on-target, we select more regions on the same kymograph::

    track_selection2 = tracks1[[1.6 < np.mean(track.position) < 2  for track in tracks1]]
    track_selection3 = tracks1[[6.8 < np.mean(track.position) < 8  for track in tracks1]]
    track_selection4 = tracks1[[8.4 < np.mean(track.position) < 10 for track in tracks1]]
    track_selection5 = tracks1[[6   < np.mean(track.position) < 7  for track in tracks1]]

Compute binding intervals
-------------------------

For each of the selections above, compute the time between binding events, which corresponds to the time of binding to an empty target site::

    def time_intervals(tracks):
        intervals =  [tracks[x+1].seconds[0]-tracks[x].seconds[-1] for x in range(len(tracks)-1)]
        return intervals

    track_intervals1 = time_intervals(track_selection1)
    track_intervals2 = time_intervals(track_selection2)
    track_intervals3 = time_intervals(track_selection3)
    track_intervals4 = time_intervals(track_selection4)
    track_intervals5 = time_intervals(track_selection5)

Combine all the intervals::

    intervals_total = track_intervals1 + track_intervals2 + track_intervals3 + track_intervals4 + track_intervals5

Check how many intervals we have in total::

    >>> len(intervals_total)
    32

Determine kon
-------------

Binding times are typically exponentially distributed. The distribution can be expressed in terms of the effective on-rate, :math:`k_{on}`, or in terms of the binding lifetime, :math:`\tau_{on}`:

.. math::

    P(t) = k_{on}e^{-k_{on}t} = \frac{1}{\tau_{on}} e^{-t/\tau_{on}}

Pylake has build-in function to fit an exponential ditribution::

    single_exponential_fit = lk.DwelltimeModel(np.array(intervals_total), n_components=1)

    plt.figure()
    single_exponential_fit.hist(bin_spacing="log")
    plt.show()

.. image:: hist_fit.png

The fitted binding time is 32 seconds, which is equivalent to an effective rate :math:`k_{on} = 1/32 = 0.031 s^{-1}``.

The confidence intervals can be determined using Bootstrapping::

    bootstrap = single_exponential_fit.calculate_bootstrap(iterations=10000)

    plt.figure()
    bootstrap.hist(alpha=0.05)
    plt.show()

.. image:: bootstrap.png

Conclusion and Outlook
----------------------

The binding time is 32 seconds with a 95% confidence interval of (21,46).

Compare value between experiments with the same protein concentration in the flow cell.
If you would like to compute the dissociation constant and compare to bulk experiments, the concentration has to be determined. 
