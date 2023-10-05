Rates & Rate Constants
======================

Chemical kinetics is the study of the rates of chemical reactions. An example of such a reaction
could be the dissociation of a protein of interest from a DNA tether. The standard representation
of this reaction is in the form of a chemical equation:

.. math::

    DP \rightarrow D + P

where :math:`DP` represents the bound DNA/Protein complex, :math:`D` is the DNA tether
and :math:`P` is the protein of interest. The arrow indicates a chemical reaction, in this case
dissociation of the reactant complex on the lefthand side of the equation into the two products
on the righthand side of the equation.

We are interested in the *rate* of this reaction; in other words, how does the concentration of the
bound complex change over time? Mathematically, we can represent this with a differential equation:

.. math::

    -\frac{\mathrm{d}[DP]}{\mathrm{d}t} = k_\mathrm{off}[DP]

Here the brackets :math:`[DP]` are used to denote "concentration of :math:`DP`".
Therefore, the lefthand side of the equation is the change in the concentration of the bound
complex with respect to time. This is the *rate* of the reaction. The value is negative, since,
as the reaction proceeds the concentration decreases as the protein falls off of the tether.

The righthand side states that this rate is linear with respect to the concentration of the reactant
:math:`DP` with a coefficient :math:`k_\mathrm{off}`, known as the *rate constant*. Because this rate is
only dependent on the concentration of a single chemical species (the bound complex), it is said
to obey *first order kinetics*.

The term "constant" is somewhat a misnomer; the value is actually dependent on temperature.
Therefore, this sometimes the term *rate coefficient* is also used.

Let's now consider the opposition reaction of a protein binding to a DNA tether:

.. math::

    D + P \rightarrow DP


In a similar fashion, we can write the rate equation for the formation of the bound product as:

.. math::

    \frac{\mathrm{d}[DP]}{\mathrm{d}t} = k_\mathrm{on}[D][P]

Note here the lefthand side is positive as the complex is formed as the reaction proceeds and is
characterized by a rate constant :math:`k_\mathrm{on}`. However, this time the rate is dependent on the
concentration of two different chemical species: both the DNA and the protein. This is reaction
therefore follows *second order kinetics*.
