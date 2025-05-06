
Rate of binding
===============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _kon:

Determine the rate of binding
-----------------------------

In this Notebook, we will determine the time between binding events for a fluorescently labeled protein binding to DNA. 
The protein binds and unbinds to target sites on DNA and the result is recorded as a kymograph.  
We track the binding events, and then determine the time intervals *between* the binding events:

.. image:: kon.png

These time intervals tell you how long it takes for a protein to bind to an empy target site.

The time between binding events, :math:`\tau_{on}` relates to the on rate of protein, :math:`k_{on}`, as :math:`\tau_{on}=1/ (k_{on}[P])` . 
The binding rate :math:`k_{on}` relates to the dissociation constant as.

.. math::

    K_{D} = \frac{k_{off}}{k_{on}}

For this example, we don't know the protein concentration and can therefore not determine :math:`k_{on}` . 
We will determine the time between binding events and refer to the inverse of that, as the *effective binding rate*, :math:`k'_{on} = k_{on}[P]` .
Further, we assume that the *bleaching time* for the dye is much longer than the time between binding, such that we can ignore the effect of bleaching.

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

Below, we use the above coordinates to select the corresponding region from the kymograph.
We check that non of the events overlap in time (as we cannot compute kon for overlapping events) and proceed to compute the time intervals between events::

    def check_any_overlap(tracks):
    # Iterate over tracked binding events to check for overlap
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if check_range_overlap(tracks[i], tracks[j]):
                raise Exception("Two or more binding events overlap in time! Remove the overlapping events before continuing the analysis.")

    def check_range_overlap(track1, track2):
        # Find the minimum and maximum values in each array
        min1, max1 = np.min(track1.seconds), np.max(track1.seconds)
        min2, max2 = np.min(track2.seconds), np.max(track2.seconds)
        
        # Check if the ranges overlap
        if (min1 <= max2 and min1 >= min2) or (min2 <= max1 and min2 >= min1):
            return True
        else:
            return False
    
    def time_intervals(tracks):
        """Compute the time intervals between all tracks in a given selection"""
        intervals =  [tracks[x+1].seconds[0]-tracks[x].seconds[-1] for x in range(len(tracks)-1)]
        return intervals

    intervals_total = []

    for coordinate in coordinates:
        bot, top = coordinate
        track_selection =  tracks1[[bot < np.mean(track.position) < top for track in tracks1]]
        check_any_overlap(track_selection)
        intervals = time_intervals(track_selection)
        intervals_total += intervals

All the time intervals between binding events are stored in the list `intervals_total`. Check how many intervals we have in total::

    >>> len(intervals_total)
    46

Determine kon
-------------

The time intervals between binding are typically exponentially distributed. The distribution can be expressed in terms of the effective on-rate, :math:`k'_{on}`, or in terms of the binding lifetime, :math:`\tau_{on}`:

.. math::

    P(t) = k'_{on}e^{-k'_{on}t} = \frac{1}{\tau_{on}} e^{-t/\tau_{on}}

Below, we fit an exponential function to the distribution of time intervals using Pylake. The parameter `discretization_timestep` accounts for the discrete nature of the data: all time intervals are a multiple of the kymo line time. 
For this dataset, we could ignore this parameter, because the average time interval is much larger than the kymo line time. When the observed time intervals are close to the kymo line time, it is important to include this parameter for a good fit. 
We cannot observe time intervals smaller than the line time, which is accounted for by adding the parameter `min_observation_time`. ::

    single_exponential_fit = lk.DwelltimeModel(np.array(intervals_total), n_components=1, , discretization_timestep = kymo1.line_time_seconds, min_observation_time = kymo1.line_time_seconds)

    plt.figure()
    single_exponential_fit.hist()
    plt.show()

.. image:: hist_fit.png

The fitted average time between binding events is 35 seconds, corresponding to an effective rate :math:`k'_{on} = 1/35 = 0.029 s^{-1}`.

The confidence intervals can be determined using Bootstrapping::

    bootstrap = single_exponential_fit.calculate_bootstrap(iterations=10000)

    plt.figure()
    bootstrap.hist(alpha=0.05)
    plt.show()

.. image:: bootstrap.png

Conclusion and Outlook
----------------------

The average time between binding events is 35 seconds with a 95% confidence interval of (24,50).

As mentioned in the introduction, the obtained time depends on the protein concentration. 
Since we don't know the protein concentration, this value can only be compared to measurements with the same protein concentration in the flow cell.
If you would like to compute the dissociation constant and compare to bulk experiments, the concentration has to be determined [1]_. 

.. [1] Schaich *et al*, Single-molecule analysis of DNA-binding proteins from nuclear extracts (SMADNE), NAR (2023)
