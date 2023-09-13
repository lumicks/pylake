Introduction
============

This chapter gives an introduction to the analysis of protein binding events.
Methods for analyzing binding are introduced, as well as how to choose between different models.
This chapter does not focus on math or syntax, but rather on the underlying idea, providing an introduction to :ref:`this section on dwell time analysis <dwelltime-analysis>`.

!!!!! binding duration vs dwell time vs unbinding time, choose one.

Dwell time analysis more general and be applied to many different systems, eg protein dynamics along DNA, state switching. 
In this article, we focus on the duration of , but much of the theory can be applied more generally.

Binding duration
----------------

Cellular processes are regulated by protein binding to DNA to eg activate a gene. 
Proteins binding to do DNA do so with varying dynamics.
For example, the duration of binding depends on whether the protein bound to its target on DNA or not. 
Proteins that bind off-target typically bind more transiently. If off and on target are combined, you see multiple time scales :cite:`Newton2019`. 
Sometimes, a protein binds first and then can change to a more stable conformation. Resulting in different time scales of unbinding from a single location (CITE).

The distribution of binding durations of a protein thus tells a lot about the (mechanical) properties of this protein and how it interacts with DNA.
In this tutorial, we explain how to unravel the mysteries of the binding times distribution.

Binding dynamics of individual proteins can directly be visualized by holding a force-stretched DNA between optically trapped beads
and observing a fluorescently labelled protein bind and unbind from the DNA.

IMAGE OF Kymograph

The first step is to track the binding events and access the lengths of the tracks, which is described in the :doc:`Kymotracking tutorial</tutorial/kymotracking>` under 'dwell time analysis'.


Inspecting the distribution
---------------------------

Unbinding typically follows an exponential process with a rate, k: P(t)=ke^-kt (CITE).
An unbinding time distribution with a single unbinding rate could like as follows:

HISTOGRAM 


A histogram with multiple time scales could look as follows:

IMAGE OF HISTOGRAM, Lin-lin time scale, or log lin


How many exponentials would be optimal?
Directly fitting to the histogram is not optimal, as the answer will largely be influenced by how the bin sizes are chosen.
(:cite:`kaur2019dwell` compares least squares with ML fitting). 
Instead, we will use a method called 'maximum likelihood estimation'.

Fitting the distribution of binding times
-----------------------------------------

Maximum likelihood estimation 

normalized by mix and max time

The minimum has to be closer than the binding lifetime. The maximum shorter than the kymotrack duration. Pylake does renormalize the distribution to account for this min an max.
probability distribution not from 0 to infinity, but from min to max observation time.
Typically, the minimum is a few kymolines.


Confidence intervals
--------------------

After fitting, we can get the confidence intervals. Pylake has two different approaches.

Bootstrapping
^^^^^^^^^^^^^

Imagine having a data set with N dwellt times.
Boostrapping works as follows:

1) Take N data points at random (the same point can be picked multiple times)
2) Fit the likelihood distribution to get the unbinding times
3) Repeat many times. Typically, 500 or 1000 times.
4) Collect all the resulting binding times in a histogram

The width of the resulting histogram is an indication of the uncertainty of the estimate.
If you get a multi modal histogram, this could indicate that your model is not optimal, which is further discussed below (or just ref to the doc page).

The name 'bootstrapping' comes from the saying 'pulling one up from its own bootstraps' as in the past, it was said that this method generates data out of nothing (CITE).
Now it is a well accepted approach in statistical analysis.

Profile likelihood confidence intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bootstrapping


Picking the right model
-----------------------
Bayesion Information Criterion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By eye, a fit with two exponentials looks good. Often, a fit will get better and better when you add more parameters.
However, having too many parameters can also lead to overfitting: there are more parameters than scritly supported by the data.
To test the optimal number of parameters, the Bayesion Information Criterion (BIC) and Akaike Information Criterion (AIC) are often used.
These two methods look at the quality of the fit, but also give a penalty for every added parameters. 
In general, the model with the lowest BIC and AIC value is optimal. If Δ BIC is less than 2, the test is inconclusive. 
If Δ BIC is between 2 and 6, the evidence against the other model is positive. If it’s between 6 and 10, the evidence for the best model and against the weaker model is strong.

If the BIC and AIC are not conclusive, having more data could help. 

Boostrapping
^^^^^^^^^^^^
In addition to BIC and AIK, the bootstrapping distribution also can indicate whether to many parameters have been used.


Miscellaneous
-------------

Do I have enough data?
^^^^^^^^^^^^^^^^^^^^^^
Width of the boostrapping distribution will be larger for smaller data sets.
Typical values:
If you want to fit multiple exponential time scales, more data points are needed. To distinguish time scales that are very close together, 
you will need more data points than when they are far apart.
