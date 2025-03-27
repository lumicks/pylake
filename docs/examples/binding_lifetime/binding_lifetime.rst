
Binding lifetime analysis
=========================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

.. _binding_lifetime:

Determine the binding lifetime from a kymograph
-----------------------------------------------

In this Notebook, we will determine the binding lifetime of a fluorescently labeled protein binding to DNA. The binding lifetime is also referred to as the binding duration
and relates to the off rate as :math:`k_{off}` = 1/binding lifetime.

First, we will track the binding events. Then we collect the binding durations and use maximum likelihood fitting to fit an exponential function to the data.
We will also demonstrate how to select the most suitable model for fitting the data.

In this Notebook, we use Pylake features that require an interactive backend, which allows you to interact with plots. Run the cell below to activate the interactive backend::

    %matplotlib widget

Load and plot the kymographs
----------------------------

The two kymographs that are used in this tutorial are stored on zenodo.org.
The following line of code downloads the data and stores the data in the folder `"test_data"`::

    filenames = lk.download_from_doi("10.5281/zenodo.14198300", "test_data")

Load the first file and plot the kymograph::

    file1 = lk.File("test_data/kymo1.h5")
    _, kymo1 = file1.kymos.popitem()

    plt.figure()
    kymo1.plot("g", aspect = 5, adjustment=lk.ColorAdjustment([0], [5]))

.. image:: kymo1.png

Load and plot the second kymograph::

    file2 = lk.File("test_data/kymo2.h5")
    _, kymo2 = file2.kymos.popitem()

    plt.figure()
    kymo2.plot("g", aspect = 5, adjustment=lk.ColorAdjustment([0], [5]))

.. image:: kymo2.png

Track the binding events
------------------------

Tracking in Lakeview
^^^^^^^^^^^^^^^^^^^^

Tracking can be performed in Lakeview and tracks can be exported and then loaded into Pylake for further analysis. If tracking was performed in Lakeview, go to Section :ref:`save_and_load_tracks`.

Tracking using Pylake
^^^^^^^^^^^^^^^^^^^^^

Select the region that you would like to track and load the selection into the kymotracker, :class:`~lumicks.pylake.KymoWidgetGreedy`. In this example, we crop the beads from the kymograph using :meth:`~lumicks.pylake.kymo.Kymo.crop_by_distance()`::

    kymo1_selection = kymo1.crop_by_distance(4.9,13.8)
    kymotracker1 = lk.KymoWidgetGreedy(kymo1_selection, "green", axis_aspect_ratio=2, pixel_threshold=6, min_length=4, track_width=0.4, vmax=10)

.. image:: kymotracker.png

The kymotracker allows for manual adjustements; you can delete, stitch or cut tracks. Click `Track All` and inspect the tracks. Perform manual adjustments if needed.

Track the binding events on the second kymograph::

    kymo2_selection = kymo2.crop_by_distance(4.9,13.8)
    kymotracker2 = lk.KymoWidgetGreedy(kymo2_selection, "green", axis_aspect_ratio=2, pixel_threshold=6, min_length=4, track_width=0.4, vmax=10)

If the the kymotracking parameters have been optimized and if manual adjustments are not needed it is also possible to track using :func:`~lumicks.pylake.track_greedy`, for example as `tracks1 = lk.track_greedy(kymo1_selection, "green")`.

.. _save_and_load_tracks:

Save and load tracks
--------------------

The coordinates and intensities of the tracks can be saved as csv: `kymotracker1.tracks.save("tracks1.csv", sampling_width=3, correct_origin=True)`.

For this example, the tracking was already performed and included in the downloaded data set.
Load the tracks into Pylake::

    tracks1 = lk.load_tracks("test_data/tracks1.csv",  kymo1.crop_by_distance(4.9,13.8), "green")
    tracks2 = lk.load_tracks("test_data/tracks2.csv",  kymo2.crop_by_distance(4.9,13.8), "green")

Use the same approach as above to load the tracks from Lakeview, except that the part :func:`Kymo.crop_by_distance() <lumicks.pylake.kymo.Kymo.crop_by_distance>` has to be removed.

The (loaded) tracks can be plotted on top of the original kymograph to visualize the result of the tracking::

    plt.figure()
    kymo1_selection.plot("g", aspect=5, adjustment=lk.ColorAdjustment(0, 5))
    tracks1.plot()

.. image:: tracks1.png

Note that two tracks at t=0 were manually removed, because the starting points of these tracks are not visible. This means that we cannot determine the duration of these tracks.
The length of each track corresponds to the duration of a binding event. As can be seen from the above image, there is a large variation in track lengths.
By collecting all these track durations into a 'binding lifetime distribution', we can analyze the binding lifetime in more detail.

Combine tracks
--------------

Tracks from multiple kymographs can be combined by adding them together. Note that imaging settings such as the line time and pixel time should be the same as further explained in :ref:`global_analysis`::

    tracks_total = tracks1 + tracks2

The total number of tracks is::

    >>> print(len(tracks_total))
    134

Fit an exponential distribution
-------------------------------

Single exponential fit
^^^^^^^^^^^^^^^^^^^^^^

Binding lifetimes are typically exponentially distributed. The distribution can be expressed in terms of the rate, :math:`k_{off}`, or in terms of the binding lifetime, :math:`\tau`:

.. math::

    P(t) = k_{off}e^{-k_{off}t} = \frac{1}{\tau} e^{-t/\tau}

Fit a single exponential to the dwell times and plot the result::

    single_exponential_fit = tracks_total.fit_binding_times(n_components = 1, observed_minimum = False, discrete_model = True)

    plt.figure()
    single_exponential_fit.hist()

.. image:: exponential_fit.png

The fitted lifetime :math:`\tau = 4` seconds.

The parameter `n_components` indicates the number of exponential time scales in the fit, as further explained below.
The parameters `observed_minimum` and `discrete_model` are further explained in :ref:`dwelltime_analysis`.

Double exponential fit
^^^^^^^^^^^^^^^^^^^^^^

Sometimes, the distribution can best be fit by multiple exponential time scales.
These exponential time scales reveal something about the underlying mechanism of binding.
For example, the protein of interest binds with higher affinity to the target site, while it binds more transiently to off-target sites.
Such behavior has been observed for various proteins such as Cas9 [1]_.

In binding lifetime analysis, it is therefore important to test which number of exponentials optimally fits the data.

The binding lifetime distributions with 2 exponential time scales is given by:

.. math::

    P(t) = \frac{a_1}{\tau_1} e^{-t/\tau_1} + \frac{a_2}{\tau_2} e^{-t/\tau_2}

Fit a double exponential distribution to the binding lifetimes by setting `n_components = 2`::

    double_exponential_fit = tracks_total.fit_binding_times(n_components = 2, observed_minimum = False, discrete_model = True)

    plt.figure()
    double_exponential_fit.hist()

.. _double_exponential_fit:
.. image:: double_exponential_fit.png

The component :math:`a_1=0.94` with lifetime :math:`\tau_1 = 3` seconds, while component :math:`a_2=0.059` with lifetime :math:`\tau_2 = 18` seconds.

Next we have to select which is the optimal model: 1 or 2 exponential time scales.
There are various methods for model selection. We will discuss 3 of them below.

Confidence intervals and model comparison
-----------------------------------------

Profile likelihood
^^^^^^^^^^^^^^^^^^

The :ref:`pop_confidence_intervals`, can be used to judge how precisely we can estimate the model parameters and helps to decide which model is optimal::

    profile_1 = single_exponential_fit.profile_likelihood()

    plt.figure()
    profile_1.plot()
    plt.tight_layout()
    plt.show()

.. image:: profile1.png

The parameter to be fitted is given on the x-axis of the plots and the optimal value is where the curve is at its minimum.
The lower the :math:`\chi^2` value at the minimum, the better the fit.
The point where the profile crosses the dashed horizontal line is an indication for the 95% confidence interval.

The profile likelihood for the single exponent looks parabolic and is almost symmetric, which indicates that the estimate of the lifetime is precise.

The likelihood profile for the double exponential fit::

    profile_2 = double_exponential_fit.profile_likelihood()

    plt.figure()
    profile_2.plot()
    plt.tight_layout()
    plt.show()

.. image:: profile2.png

For the double exponential fit, the profiles look more skewed. The values of :math:`\chi^2` at the minimum are lower, which indicates a better fit.
When looking at the likelihood profiles, the double exponential fit therefore seems optimal.
Note however that the lower boundaries of the confidence intervals for `amplitude 1` and `lifetime 1` are almost zero and that the confidence interval for `lifetime 1` is very wide. Gathering more data may help to reduce the confidence intervals and get a better estimate of the components and lifetimes. 

Bootstrapping
^^^^^^^^^^^^^

Bootstrapping can be used to select the most suitable model and is a good method for determining the confidence intervals for the fitted parameters.
During bootstrapping, a random sample is taken from the original dataset and fitted. The fitted parameters are gathered in the bootstrapping distribution.
In the example below, we perform 10000 iterations, which means that 10000 times we take a sample from the data and fit the sample with a single exponential distribution.
The resulting 10000 binding lifetimes are plotted in the histogram.

Compute and plot the bootstrapping the distribution for the single exponential fit. This will take a while...::

    bootstrap1 = single_exponential_fit.calculate_bootstrap(iterations=10000)

    plt.figure()
    bootstrap1.hist(alpha=0.05)

.. image:: bootstrap1.png

The bootstrapping distribution for the single exponential fit is unimodal and almost symmetric, which indicates that we have well defined parameter estimates. The width of the distribution gives the confidence intervals:
The fitted binding lifetime is 4 seconds has a 95% confidence interval of (3,5.2) seconds.

Compute and plot the bootstrapping the distribution for the double exponential fit::

    bootstrap2 = double_exponential_fit.calculate_bootstrap(iterations=10000)

    plt.figure()
    bootstrap2.hist(alpha=0.05)

.. image:: bootstrap2.png

The bootstrapping distribution for the double exponential fit is sometimes bimodal and the component :math:`a_2` has a peak close to zero.
This indicates that for many of the bootstrap samples, the fraction associated with the second lifetime was really small and that the parameters of this second lifetime cannot be estimated reliably from the data.

According to the bootstrapping distributions, the single exponential fit is better suitable for the data.

Bayesian Information Criterion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically adding more parameters (components) to a model, will make the fit better. However, having too many parameters can lead to overfitting.
The Bayesian information Criterion (BIC) quantifies the quality of the fit by looking at the value of the likelihood function and penalizes the addition of parameters.

The BIC for the single and double exponential fit are respectively given by::

    >>> print(single_exponential_fit.bic)
    >>> print(double_exponential_fit.bic)

    1317.4882114486545
    1309.1566330421654

The BIC value for the double exponential fit is minimal, but the difference is smaller than 10, so the evidence is not super strong.

Conclusion and Outlook
----------------------

We fitted a single exponential and double exponential to the distribution of binding lifetimes.
Then, we used the likelihood profile, bootstrapping and BIC to determine the most suitable model.
The likelihood profile and bootstrapping indicated that when using a two-component model, the second lifetime has very wide confidence intervals and the fraction of events that have this lifetime associated with them is very small.
The BIC indicated that a double exponential is more suitable, but the difference between the small and large model is not very large.

Looking at Figure with the :ref:`double exponential fit <double_exponential_fit>`, there are only a few data points larger than 20 seconds that support the second exponential time scale.
Therefore, the data set is likely too small to support a second exponential time scale (Fitting two exponentials without overfitting, typically requires a few hundred data points).

With the current dataset, we conclude that the most suitable model is a single exponential as it gives us the most precise estimates. The fitted lifetime is :math:`\tau = 4` seconds with a 95% confidence interval of (3,5.2) seconds as determined by bootstrapping.
However, given that we do see a hint that there may be a second lifetime involved, it would be worthwhile to gather more data in this case.

Splitting tracks by position
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the target sites of the protein are known, the binding lifetimes can also be split by position and analyzed separately [1]_.
For example, to select all tracks from `kymo1_selection` that have an average position larger than 8 micron, type::

    track_selection = tracks1[[np.mean(track.position) > 8 for track in tracks1]]

Similarly, we can have a two-sided interval. For example, tracks with a position between 5.5 and 6.2 micron can be obtained by::

    track_selection = tracks1[[5.5 < np.mean(track.position) < 6.2 for track in tracks1]]

Note that the position coordinates for the cropped kymograph `kymo1_selection` are not the same as for `kymo1`!
By analyzing on-target and off-target events separately, the effect of target binding on the binding lifetime can be studied in more detail.

.. [1] Newton, DNA stretching induces Cas9 off-target activity, NSMB (2019)
