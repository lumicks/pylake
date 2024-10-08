This page lists references used for implementing specific functionality into Pylake. Please see below for more
information on these topics.

Force Calibration
-----------------

The passive force calibration method was based on a number of publications by the Flyvbjerg group :cite:`berg2004power,tolic2004matlab,hansen2006tweezercalib,berg2003unintended,berg2006power,norrelykke2010power`.
Corrections for the drag coefficient of lateral and axial force when approaching the surface were based on :cite:`schaffer2007surface,brenner1961slow`.
The active calibration method was based on :cite:`tolic2006calibration`. The corrections for dual trap calibration were based on work presented in :cite:`stimson1926motion,goldman1966slow` and numerically compared to simulations using the software :cite:`the_fenics_project_developers_2023_10432590`.

F,d Fitting
-----------

Several polymer models from literature were included. These include Odijk's extensible worm-like chain model
:cite:`odijk1995stiff,wang1997stretching`, the Freely Jointed Chain model
:cite:`smith1996overstretching,wang1997stretching`, the Marko-Siggia interpolation model :cite:`marko1995stretching`
and the twistable worm-like chain model :cite:`broekmans2016dna,gross2011quantifying`.

We used :cite:`broekmans2016dna` as reference material throughout the implementation of the F,d fitting routines, both
as a starting point for the model implementations as well as for the procedure on how to reliably fit the twistable
worm-like chain model (via its inverted form). For the twistable worm-like chain, we perform model inversion via
interpolation rather than point-wise inversion through optimization analogously to :cite:`broekmans2016dna`. Parameter
estimation is performed via Maximum Likelihood Estimation :cite:`maiwald2016driving,raue2009structural`. We also
implemented several asymptotic information criteria for ranking model fits :cite:`cavanaugh1997unifying`.

Kymotracker
-----------

The line algorithm was based on sections 1, 2 and 3 from :cite:`steger1998unbiased` to find line centers based on local
geometric considerations. The greedy algorithm was based on two papers. It initially detects feature points based on
:cite:`sbalzarini2005feature`, followed by line tracing inspired by :cite:`mangeol2016kymographclear`. Instead of
subtracting the background and thresholding the values below zero to zero, we perform a bias correction for background
by symmetrizing the used window (:cite:`berglund2008fast`).

Diffusion constant estimation
-----------------------------

The unweighted ordinary least squares estimation method (including the optimal number of lag computation) is implemented from :cite:`michalet2012optimal`.
The generalized least squares method was based on :cite:`bullerjahn2020optimal`
The CVE method is based on :cite:`vestergaard2014optimal,vestergaard2015estimation,vestergaard2016optimizing`.

References
----------

.. bibliography:: refs.bib
   :style: alpha
