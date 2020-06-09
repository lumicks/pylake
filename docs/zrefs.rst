This page lists references used for implementing specific functionality into Pylake. Please see below for more
information on these topics.

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

References
----------

.. bibliography:: refs.bib
   :style: alpha
