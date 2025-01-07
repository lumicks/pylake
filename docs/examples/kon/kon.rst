
Rate of binding
===============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _kon:

Determine the rate of binding
-----------------------------

In this Notebook, we will determine the binding time of a fluorescently labeled protein binding to DNA. 
The protein binds and unbinds to target sites on DNA and the result is recorded as a kymograph.  
We track the binding events, and then determine the time intervals *between* the binding events:

.. image:: kon.png

These time intervals tell you how long it takes for a protein to bind to an empy target site.

The binding time, :math:`\tau_{on}` relates to the on rate of protein, :math:`k_{on}`, as :math:`\tau_{on}=1/ (k_{on}[P])` . 
The binding rate :math:`k_{on}` relates to the dissociation constant as.

.. math::

    K_{off} = \frac{k_{off}}{k_{on}}

For this example, we don't know the protein concentration and can therefore not determine :math:`k_{on}` . 
We will determine the binding time and refer to the inverse of the binding time, as the *effective binding rate*, :math:`k'_{on} = k_{on}[P]` .
Further, we assume that the *bleaching time* for the dye is much longer than the binding time, such that we can ignore the effect of bleaching.

Load and plot the kymographs
----------------------------

The kymograph and corresponding tracks that are used in this tutorial are stored on zenodo.org.
The following line of code downloads the data and stores the data in the folder `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.14198300", "test_data")

Load and plot the kymograph::

    file1 = lk.File("test_data/kymo1.h5")
    _, kymo1 = file1.kymos.popitem()
    
    plt.figure()
    kymo1.plot("g", aspect = 5, adjustment=lk.ColorAdjustment([0], [5]))

.. image:: kymo1.png

Load the tracks
---------------

For this tutorial, the binding events have already been tracked in Pylake.
Load the tracks as follows::

    tracks1 = lk.load_tracks("test_data/tracks1.csv",  kymo1.crop_by_distance(4.9,13.8), "green")

Note that the kymograph passed to `lk.load_tracks` is cropped, because tracking was performed on a cropped kymograph, see :ref:`tracking`.

Use the same approach as above to load the tracks exported from Lakeview, except that the part :func:`Kymo.crop_by_distance() <lumicks.pylake.kymo.Kymo.crop_by_distance>` has to be removed.

Select target location
----------------------

Plot the tracks::

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
    plt.hlines(y=8.4, xmin=0,xmax=320)
    plt.hlines(y=9, xmin=0, xmax=320)

.. image:: track_selection1.png

Select all tracks that are on average within the two coordinates indicated in the above image::

    track_selection1 = tracks1[[8.4 < np.mean(track.position) < 9 for track in tracks1]]

Plot the final selection of tracks::

    plt.figure(figsize = (9,1))
    track_selection1.plot()

.. image:: track_selection1_plot.png

Since we are using a repeat sequence and all observed binding events were on-target, we select multiple regions on the same kymograph::

    coordinates = [(8.4,9),(7,7.6),(6.2,6.8),(5.5,6.1),(4.8,5.4),(4.1,4.7),(3.3,3.9),(2.6,3.2),(1.9,2.5),(1.2,1.8),(0.5,1.1)]

Using the above coordinates, we can select the corresponding region from the kymograph, and compute the time intervals between the tracked binding events::

    def time_intervals(tracks):
        """Compute the time intervals between all tracks in a given selection"""
        intervals =  [tracks[x+1].seconds[0]-tracks[x].seconds[-1] for x in range(len(tracks)-1)]
        return intervals

    intervals_total = []

    for coordinate in coordinates:
        bot, top = coordinate
        track_selection =  tracks1[[bot < np.mean(track.position) < top for track in tracks1]]
        intervals = time_intervals(track_selection)
        intervals_total += intervals

All the time intervals between binding events are stored in the list `intervals_total`. Check how many intervals we have in total::

    >>> len(intervals_total)
    46

Determine kon
-------------

Binding times are typically exponentially distributed. The distribution can be expressed in terms of the effective on-rate, :math:`k'_{on}`, or in terms of the binding lifetime, :math:`\tau_{on}`:

.. math::

    P(t) = k'_{on}e^{-k'_{on}t} = \frac{1}{\tau_{on}} e^{-t/\tau_{on}}

Fit an exponential ditribution to the distribution of time intervals using Pylake::

    single_exponential_fit = lk.DwelltimeModel(np.array(intervals_total), n_components=1)

    plt.figure()
    single_exponential_fit.hist()
    plt.show()

.. image:: hist_fit.png

The fitted binding time is 36 seconds, which is equivalent to an effective rate :math:`k'_{on} = 1/36 = 0.028 s^{-1}`.

The confidence intervals can be determined using Bootstrapping::

    bootstrap = single_exponential_fit.calculate_bootstrap(iterations=10000)

    plt.figure()
    bootstrap.hist(alpha=0.05)
    plt.show()

.. image:: bootstrap.png

Conclusion and Outlook
----------------------

The binding time is 36 seconds with a 95% confidence interval of (24,50).

As mentioned in the introduction, the obtained binding time depends on the protein concentration. 
Since we don't know the protein concentration, this value can only be compared to measurements with the same protein concentration in the flow cell.
If you would like to compute the dissociation constant and compare to bulk experiments, the concentration has to be determined [1]_. 

.. [1] Schaich *et al*, Single-molecule analysis of DNA-binding proteins from nuclear extracts (SMADNE), NAR (2023)
