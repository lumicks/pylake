.. warning::
    This is early access alpha functionality. While usable, this has not yet been tested in a large number of different
    scenarios. The API is still be subject to change *without any prior deprecation notice*! If you use this
    functionality keep a close eye on the changelog for any changes that may affect your analysis.

Kymotracking
============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


Kymographs are a vital tool in understanding binding events on DNA.
They are generated using one-dimensional scans over time.
Moving particles can then be seen moving along the spatial coordinate in time.
It is often desired to track the position of such particles in time.
Pylake comes with a few features that enable you to do so.


Performing the tracking
-----------------------

Performing Kymotracking is usually performed in two steps. Typically, there is an image processing step, following by
a tracing step, where the actual traces are found.


Using the greedy algorithm
--------------------------

First, we need to get an image to perform the tracing on. Let's grab a kymograph from a Bluelake file::

    file = lk.File('kymograph.h5')
    name, kymo = file.kymos.popitem()

Let's have a look at what it looks like::

    kymo.plot_green(aspect="auto", vmax=5)

.. image:: kymo_unscaled.png

What we also see is that we can clearly see the bead edges in the kymograph. Let's only take a subsection of the
kymograph, to make sure that we don't get any spurious line detections that come from the bead edges. Let's only
use the region from 7 to 22 micrometer. Let's have a look at what that looks like::

    >>> kymo.plot_green(aspect="auto", vmax=5)
    >>> plt.ylim([22, 7])

If we zoom in a bit, we can see that our traces in this image are about 0.3 microns wide, so let's set this as our
`line_width`. Note that we invoked `plt.colorbar()` to add a little color legend here.

.. image:: kymo_zoom.png

The peaks have an intensity of about 3-7 photon counts, whereas the background fluctuates around 0-2. Let's set our
`pixel_threshold` to 4. We also see that sometimes, a particle momentarily disappears. To still connect these in a
single trace, we want to allow for some gaps in the connection step. Let's use a `window` size of 9 in this test.
Considering that we only want to run the kymotracker on part of the image, we also pass the rect argument, which defines a rectangle over which to track peaks.
In this case, we track particles between 0 and 350 seconds, and 7 and 22 micron.
Running the algorithm is easy using the function :func:`~lumicks.pylake.track_greedy`::

    traces = lk.track_greedy(kymo, "green", line_width=0.3, pixel_threshold=3, window=6, rect=[[0, 7], [350, 22]])

The result of tracking is a list of kymograph traces::

    >>> print(len(traces))  # the number of traces found in the kymo
    7417

Sometimes, we can have very short spurious traces. To remove these from the list of detected traces we can use
:func:`~lumicks.pylake.filter_lines`. To omit all traces with fewer than 4 detected points, we
can invoke::

    >>> traces = lk.filter_lines(traces, 4)

We can get the determined time and positions of these traces from the `seconds` and `position`
attributes. Let's plot the first trace::

    plt.plot(traces[0].seconds, traces[0].position)

Plotting all the detected traces is also quite easy by iterating over the list of traces::

    kymo.plot_green(aspect="auto")
    for trace in traces:
        plt.plot(trace.seconds, trace.position)

.. image:: kymotracked.png

Localization refinement
-----------------------

Centroid
^^^^^^^^

Once we are happy with the traces found by the algorithm, we may still want to refine them. Since the algorithm finds
traces by determining local peaks and stringing these together, it is possible that some scan lines in the kymograph
don't have an explicit point on the trace associated with them. Using :func:`~lumicks.pylake.refine_lines_centroid` we
can refine the traces found by the algorithm. This function interpolates the lines such that each time point gets its
own point on the trace. Subsequently, these points are then refined using a brightness weighted centroid. Let's perform
line refinement and plot the longest trace::

    longest_trace_idx = np.argmax([len(trace) for trace in traces])  # Get the longest trace

    refined = lk.refine_lines_centroid(traces, line_width=4)

    plt.plot(refined[longest_trace_idx].seconds, refined[longest_trace_idx].position, '.')
    plt.plot(traces[longest_trace_idx].seconds, traces[longest_trace_idx].position, '.')
    plt.legend(["Post refinement", "Pre-refinement"])
    plt.ylabel('Position [um]')
    plt.xlabel('Time [s]')

.. image:: kymo_refine.png

We can see now that a few points were added post refinement (shown in blue). The others remain unchanged, since we used
the same `line_width`.

Fortunately, the signal to noise level in this kymograph is quite good. In practice, when the signal to noise is lower,
one will have to resort to some fine tuning of the algorithm parameters over different regions of the kymograph to get
an acceptable result.

Maximum Likelihood Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function `refine_lines_gaussian()` instead uses an MLE optimization of a Poisson likelihood with a Gaussian expectation
to characterize both the expected peak shape and photonic noise of the observed signal, adapted from :cite:`mortensen2010gauloc`.
For each frame in the kymograph, we fit a small region around the tracked peak to the data by maximizing the following likelihood function:

.. math::

    \mathcal{L(\theta)} = \prod_i^M e^{-E_i(\theta)} \frac{E_i(\theta)^{n_i}}{n_i!}

where :math:`\theta` represents the parameters to be fitted, :math:`M` is the number of pixels to be fit and :math:`n_i` and :math:`E_i(\theta)` are the observed photon count and expectation value
for pixel :math:`i`. The shape of the peak is described with a Gaussian expectation function

.. math::

    E_i(\theta) = \frac{N a}{\sqrt{2 \pi \sigma^2}} \exp \left[ \frac{-(x_i-\mu)^2}{2 \sigma^2} \right] + b

Here :math:`N` is the total photons emitted in the fitted image (line), :math:`a` is the pixel size, :math:`\mu` is the peak center,
:math:`x_i` is the pixel center position, :math:`\sigma^2` is the variance, and :math:`b` is the background level in
photons/pixel.

This function is called in a similar manner as the centroid refinement::

    refined = lk.refine_lines_centroid(traces, window=3, refine_missing_frames=True, overlap_strategy="skip")

The number of pixels to be included in the fit is determined by the `window` argument, with a total size of `2*window+1` pixels.
The exact value of this parameter is dependent on the quality of the data and should be balanced between including enough pixels to fully
capture the peak lineshape while avoiding overlap with other traces or spurious high-photon count pixels due to noise or background.
The effect of different window sizes are demonstrated in the following figure:

.. image:: kymo_gau_window.png

As noted in the above section, there may be intermediate frames which were not tracked in the original trace. As with the centroid fitting,
we can optionally interpolate an initial guess for these frames before the Gaussian refinement by setting the argument
`refine_missing_frames=True`. It should be noted, however, that frames with low photons counts (for instance due to fluorophore blinking)
may not be well fit by this algorithm.

Additionally, the presence of a nearby trace wherein the sampled pixels of the two traces overlap may interfere with the
refinement algorithm. How the algorithm handles this situation is determined by the `overlap_strategy` argument. Setting `overlap_strategy="ignore"`
simply ignores the situation and fits the data. A problem with the refinement in this case will manifest as the peak of the second trace
is found rather than that of the current trace. Sometimes this can be avoided by decreasing the size of the `window` argument such that overlap no longer occurs.
Alternatively we can simply ignore these frames by using `overlap_strategy="skip"`, in which case these frames
are simply dropped from the trace.

There is also an optional keyword argument `initial_sigma` that can be used to pass an initial guess for :math:`\sigma` in micrometers
in the above expectation equation to the optimizer. The default value is `1.1*pixel_size`.


Using the kymotracker widget
----------------------------

Using the algorithm purely by function calls can be challenging if not all parts of the kymograph look the same or
when the signal to noise ratio is somewhat low. To help with this, we included a kymotracking widget that can help you
track subsections of the kymograph and iteratively tweak the algorithm parameters as you do so. You can open this widget
by invoking the following command::

    kymowidget = lk.KymoWidgetGreedy(kymo, "green")

You can optionally also pass algorithm parameters when opening the widget::

    KymoWidgetGreedy(kymo, "green", axis_aspect_ratio=2, min_length=4, pixel_threshold=3, window=6, sigma=0.14)

Traced lines are accessible through the `.lines` property::

    >>> lines = kymowidget.lines
    KymoLineGroup(N=199)

For more information on its use, please see the example :ref:`cas9_kymotracking`.

Using the lines algorithm
-------------------------

The second algorithm present is an algorithm that works purely on signal derivative information. It works by blurring
the image, and then performing sub-pixel accurate line detection. It can be a bit more robust to low signal levels,
but is generally less temporally and spatially accurate due to the blurring involved::

    traces = lk.track_lines(kymo, "green", line_width=0.3, max_lines=50)

The interface is mostly the same, aside from an extra required parameter named `max_lines` which indicates the maximum
number of lines we want to detect.


Extracting summed intensities
-----------------------------

Sometimes, it can be desirable to extract pixel intensities in a region around our kymograph trace. We can quite easily
extract these using the method :func:`~lumicks.pylake.kymotracker.kymoline.KymoLine.sample_from_image`. For instance,
if we want to sum the pixels in a 9 pixel area around the longest kymograph trace, we can invoke::

    plt.figure()
    longest_trace_idx = np.argmax([len(trace) for trace in traces])
    longest_trace = traces[longest_trace_idx]
    plt.plot(longest_trace.seconds, longest_trace.sample_from_image(num_pixels=5))
    plt.xlabel('Time [s]')
    plt.ylabel('Summed signal')

Here `num_pixels` is the number of pixels to sum on either side of the trace.

.. image:: kymo_sumcounts.png


Exporting kymograph traces
--------------------------

Exporting kymograph traces to `csv` files is easy. Just invoke `save` on the returned value::

    traces.save("traces.csv")

We can also save photon counts by passing a width in pixels to sum counts over::

    traces.save("traces_calibrated.csv", sampling_width=3)


How the algorithms work
-----------------------
:func:`~lumicks.pylake.track_greedy`

The first method implemented for performing such a tracking is based on :cite:`sbalzarini2005feature,mangeol2016kymographclear`.
It starts by performing peak detection, performing a grey dilation on the image, and detection which pixels remain
unchanged. Peaks that fall below a certain intensity threshold are discarded. Since this peak detection operates at a
pixel granularity, it is followed up by a refinement step to attain subpixel accuracy. This refinement is performed by
computing an offset from a brightness-weighted centroid in a small neighborhood `w` around the pixel.

.. math::

    offset = \frac{1}{m} \sum_{i^2 < w^2} i I(x + i)

Where m is given by:

.. math::

    m = \sum_{i^2 < w^2} I(x + i)

After peak detection the feature points are linked together using a forward search analogous to
:cite:`mangeol2016kymographclear`. This is in contrast with the linking algorithm in :cite:`sbalzarini2005feature`
which uses a graph-based optimization approach. This linking step traverses the kymograph, tracing particles starting
from each frame.

- The algorithm starts at time frame one (the first pixel column).

- It selects the peak with the highest pixel intensity and initiates the first trace.

- Next, it evaluates the subsequent frame, and computes a connection score for each peak in the next frame (to be specified in more detail later).

- If a peak is found with an acceptable score, the peak is added to the trace.

- When no more candidates are available we look in the next `window` frames to see if we can find an acceptable peak there, following the same procedure.

- Once no more candidates are found in the next `window` frames, the trace is terminated and we proceed by initiating a new trace from the peak which is now the highest.

- Once there are no more peaks in the frame from which we are currently initiating traces, we start initiating traces from the next frame. This process is continued until there are no more peaks left to trace.

The score function is based on a prediction of where we expect future peaks. Based on the peak location of the tip of
the trace `x` and a velocity `v`, it computes a predicted position over time. The score function assumes a Gaussian
uncertainty around that prediction, placing the mean of that uncertainty on the predicted extrapolation. The width of
this uncertainty is given by a base width (provided as sigma) and a growing uncertainty over time given by a diffusion
rate. This results in the following model for the connection score.

.. math::

    S(x, t) = N\left(x + v t, \sigma_{base} + \sigma_{diffusion} \sqrt{t}\right).

Here `N` refers to a normal distribution. In addition to the model, we also have to set a cutoff, after which we deem
peaks to be so unlikely to be connected that they shouldn't be. By default, this cutoff is set at two sigma. Scores
outside this cutoff are set to zero which means they will not be accepted as a new point.


:func:`~lumicks.pylake.track_lines`

The second algorithm is an algorithm that looks for curvilinear structures in an image. This method is based on sections
1, 2 and 3 from :cite:`steger1998unbiased`. This method attempts to find lines purely based on the derivatives of the
image. It blurs the image based with a user specified line width and then attempts to find curvilinear sections.

Based on the second derivatives of the blurred image, a Hessian matrix is constructed. This Hessian matrix is
decomposed using an eigenvector decomposition to obtain the perpendicular and tangent directions to the line. To attain
subpixel accuracy, the maximum is computed perpendicular to the line using a local Taylor expansion. This expansion
provides an offset on the pixel position. When this offset falls within the pixel, then this point is considered to
be part of a line. If it falls outside the pixel, then it is not a line.

This provides a narrow mask, which can be traced. Whenever ambiguity arises on which point to connect next, a score
comprised of the distance to the next subpixel minimum and angle between the successive normal vectors is computed.
The candidate with the lowest score is then selected.

Since this algorithm is specifically looking for curvilinear structures, it can have issues with structures that are
more blob-like (such as short-lived fluorescent events) or diffusive traces, where the particle moves randomly rather
than in a uniform direction.


Studying diffusion processes
----------------------------

To study diffusive processes we can make use of the Mean Squared Displacement (MSD).
There are multiple ways to estimate this quantity.
We use the following estimator:

.. math::

    \hat{\rho}[n] = \frac{1}{N - n} \sum_{i=1}^{N-n}\left(x_{i+n} - x_{i}\right)^2

where :math:`\hat{\rho}[n]` corresponds to the estimate of the MSD for lag :math:`n`, :math:`N` is the number of time points in the tracked line, and :math:`x_i` is the trace position at time frame :math:`i`.

What we can see in this definition is that it uses the same data points several times, thereby resulting in a well averaged estimate.
However, the downside of this estimator is that the calculated values are highly correlated :cite:`qian1991single,michalet2010mean,michalet2012optimal` which needs to be accounted for in subsequent analyses.

In the following, we'll use three simulated :class:`~lumicks.pylake.kymotracker.kymoline.KymoLine` instances of length 62 based on a diffusion constant of `10.0`.

With Pylake, we can calculate the MSD from a :class:`~lumicks.pylake.kymotracker.kymoline.KymoLine` with a single command::

    kymolines[0].msd()

This returns a tuple of lags and MSD estimates. If we only wish MSDs up to a certain lag, we can provide a `max_lag` argument::

    >>> kymolines[0].msd(mag_lag = 5)
    (array([0.16, 0.32, 0.48, 0.64, 0.8 ]), array([12.48593965, 16.34844311, 17.21359513, 27.25210869, 32.34473104]))

MSDs are typically used to calculate diffusion constants.
With pure diffusive motion (a complete absence of drift) in an isotropic medium, 1-dimensional MSDs can be fitted by the following relation:

.. math::

    \rho[n] = 2 D t + offset

where :math:`D` is the diffusion constant in :math:`um^2/s`, :math:`t` is time, and the offset is determined by the localization accuracy:

.. math::

    offset = 2 \sigma^2 - 4 R D \Delta t

where :math:`\sigma` is the static localization accuracy, :math:`R` is a motion blur constant and :math:`\Delta t` represents the time step.

While it may be tempting to use a large number of MSDs in the diffusion estimation procedure, this actually produces poor estimates of the diffusion constant :cite:`qian1991single,michalet2010mean,michalet2012optimal`.
There exists an optimal number of lags to fit such that the estimation error is minimal.
This optimal number of lags depends on the ratio between the diffusion constant and the dynamic localization accuracy:

.. math::

    \epsilon_{localization} = \frac{offset}{slope} = \frac{2 \sigma^2 - 4 R D \Delta t}{2 D \Delta t} = \frac{\sigma^2}{D \Delta t} - 2 R

When localization is infinitely accurate, the optimal number of points is two :cite:`michalet2010mean`.
At the optimal number of lags, it doesn't matter whether we use a weighted or unweighted least squares algorithm to fit the curve :cite:`michalet2010mean`, and therefore we opt for the latter, analogously to :cite:`michalet2012optimal`.
With Pylake, you can obtain an estimated diffusion constant by invoking::

    >>> kymolines[0].estimate_diffusion_ols()
    7.386961688211464

Let's get diffusion constants for all three :class:`~lumicks.pylake.kymotracker.kymoline.KymoLine` instances::

    >>> [kymoline.estimate_diffusion_ols() for kymoline in kymolines]
    [7.386961688211464, 9.052904296390873, 11.551531668363802]

We can see that there is considerable variation in the estimates, which is unfortunately typical for diffusion coefficient estimates.
By default, `estimate_diffusion_ols` will use the optimal number of lags as specified in :cite:`michalet2012optimal`. You can however, override this optimal number of lags, by specifying a `max_lag` parameter::

    >>> [kymoline.estimate_diffusion_ols(max_lag=30) for kymoline in kymolines]
    [3.064522711727202, 1.7564518650402365, 1.9175754533226452]

Note however, that this will likely degrade your estimate.
We can also plot the MSD estimates::

    [kymoline.plot_msd(marker='.') for kymoline in kymolines]

.. image:: msdplot_default_lags.png

By default, this will use the optimal number of lags (which in this case seems to be around 3-4), but once again a `max_lag` parameter can be specified to plot a larger number of lags::

    [kymoline.plot_msd(max_lag=100, marker='.') for kymoline in kymolines]

.. image:: msdplot_100_lags.png

It's not hard to see from this graph why taking too many lags results in unacceptably large variances.


Dwelltime analysis
------------------------

The lifetime of the bound state(s) can be determined using `KymoLineGroup.fit_binding_times()`. This method defines
the bound dwelltime as the length of each tracked line (in seconds). The lifetimes are then determined using
Maximum Likelihood Estimation (MLE) :cite:`kaur2019dwell,woody2016memlet`. The number of exponential components
to be used for the fit is chosen with the `n_components` argument.

The likelihood :math:`\mathcal{L}` is defined for a mixture of exponential distributions as:

.. math::

    \mathcal{L} = \prod_j^T \left[ \frac{1}{N} \sum_i^M \frac{a_i}{\tau_i} \exp{\left( \frac{-t_j}{\tau_i} \right)} \right]

where :math:`T` is the number of observed dwell times, :math:`M` is the number of exponential components, :math:`t` is time,
:math:`\tau_i` is the lifetime of component :math:`i`, and :math:`a_i` is the fractional contribution of component :math:`i`
under the constraint of :math:`\sum_i^M a_i = 1`. The normalization constant :math:`N` is defined as:

.. math::

    N = \sum_i^M a_i \left[
    \exp{ \left( \frac{-t_{min}}{\tau_i} \right)} -
    \exp{ \left( \frac{-t_{max}}{\tau_i} \right)}
    \right]

where :math:`t_{min}` and :math:`t_{max}` are the minimum and maximum possible observation times.

The normalization constant takes into account the minimum and maximum possible observation times of
a track. Note that when :math:`t_{min}=0` and :math:`t_{max}=\infty`, :math:`N=1`. However, for real experimental data,
there are physical limitations on the measurement times (such as pixel integration time). Additionally, the minimum length of tracked lines
here is dependent on the specific input parameters used for the tracking algorithm. Therefore, in order to estimate these bounds,
the method uses the shortest track time and the length of the experiment, respectively.


First let's try to fit the dwelltime distribution to a single exponential (the simplest case). To do this we
simply call::

    dwell_1 = traces.fit_binding_times(n_components=1)

This returns a `BindingDwelltimes` object which contains information about the optimized model.
We can visually inspect the result with::

    dwell_1.plot(bin_spacing="log")

.. image:: kymo_bind_dwell_1.png

The `bin_spacing` argument can be either `"log"` or `"linear"` and controls the spacing of the bin edges.
The scale of the x- and y-axes can be controlled with the optional `xscale` and `yscale` arguments; if they are not specified
the default visualization is `lin-lin` for `bin_spacing="linear"` and `lin-log` for `bin_spacing="log"`.

You can optionally pass the number of bins to be plotted as `n_bins`. Note that the number of bins
is purely for visualization purposes; the model is optimized directly on the unbinned dwelltimes. This is the main
advantage of the MLE method over analyses that use a least squares fitting to binned data, where the bin widths and number
of bins can drastically affect the optimized parameters.

We can clearly see that this distribution is not fit well by a single exponential decay.
Let's now see what a double exponential distribution looks like::

    dwell_2 = traces.fit_binding_times(n_components=2)
    dwell_2.plot(bin_spacing="log")

.. image:: kymo_bind_dwell_2.png

Here we see that the double exponential fit visually looks better and the log likelihood is also higher than that
for the single exponential fit. However, the log likelihood does not take into account model complexity, and will
always increase for a model with more degrees of freedom. Instead, we can look at the Bayesian Information Criterion
or Akaike Information Criterion to determine which model is better::

    >>> print(dwell_1.bic, dwell_1.aic)
    532.3299315589168  529.0366267341923

    >>> print(dwell_2.bic, dwell_2.aic)
    520.4562630650156  510.5763485908421

These information criterion values weight the log likelihood with the model complexity, and as such are more useful for
model selection. In general, the model with the lowest value is optimal. We can see that both values are lower for the double exponential model,
indicating that it is a better fit to the data.

We can see this effect if we purposely overfit the data. The following plot shows the result of fitting simulated data (randomly sampled
from a single exponential distribution) with either a one- or two-component model. In the figure legends we see that the log likelihood
decreases slightly for the two-component model because of the larger degrees of freedom. However, the BIC for the one-component model is
indeed lower, as expected:

.. image:: kymo_bic_compare.png

Going back to our experimental data, we can next attempt to estimate confidence intervals (CI) for the parameters using bootstrapping.
Here, a random dataset with the same size as the original is sampled (with replacement) from the original dataset. This sampled dataset
is then fit using the MLE method, just as for the original dataset. The fit results in a new estimate for the model parameters.
This process is repeated many times, and the distribution of the resulting parameters can be analyzed to estimate certain statistics about the them::

    dwell_2.run_bootstrap(iterations=1000)
    dwell_2.bootstrap.plot(alpha=0.05)

.. image:: kymo_bind_bootstrap_2.png

Here we see the distributions of the bootstrapped parameters. The vertical lines indicate the
means of the distributions, while the red area indicates the estimated confidence intervals. The `alpha` argument determines
the CI that is estimated as `100*(1-alpha)` % CI; in this case we're showing the estimate for the 95% CI. The values for the
lower and upper bounds are the `100*(alpha/2)` and `100*(1-alpha/2)` percentiles of the distributions.

Note, however, that while the means correspond well with the optimized model parameters, the distributions are not symmetric.
In such a case, the simple method of using percentiles as CI values may not be appropriate. For more advanced analysis,
the distribution values are directly available through the properties `BindingDwelltimes.bootstrap.amplitude_distributions` and
`BindingDwelltimes.bootstrap.lifetime_distributions` which return the data as a `numpy` array with
shape `[# components, # bootstrap samples]`.
