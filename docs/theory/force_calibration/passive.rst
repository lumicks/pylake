Passive calibration
-------------------

Passive calibration is also often referred to as thermal calibration and involves calibration without
moving the trap or stage. In passive calibration, the Brownian motion of the bead in the trap is
analyzed in order to find calibration factors for both the positional detection as well as the force.

To fit passive calibration data, we will use a model based on a number of publications by the
Flyvbjerg group :cite:`berg2004power,tolic2004matlab,hansen2006tweezercalib,berg2006power`.
The Pylake implementation of this model is :class:`~.PassiveCalibrationModel`.
The most basic form of passive calibration starts by fitting the following equation to the power spectrum:

.. math::

    P(f) = \frac{D_\mathrm{measured}}{\pi ^ 2 \left(f^2 + f_c ^ 2\right)} g(f, f_\mathrm{diode}, \alpha) \tag{$\mathrm{V^2/Hz}$}

where :math:`D_\mathrm{measured}` corresponds to the diffusion constant (in :math:`V^2/s`), :math:`f`
the frequency and :math:`f_c` the corner frequency. The first part of the equation is the same as
before and represents the part of the spectrum that originates from the physical motion of the bead.
The second term :math:`g` takes into account the slower response of the position detection system and is given by:

.. math::

    g(f, f_\mathrm{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_\mathrm{diode})^2} \tag{$-$}

Here :math:`\alpha` corresponds to the fraction of the signal response that is instantaneous, while
:math:`f_\mathrm{diode}` characterizes the frequency response of the diode.
Note that not all sensor types require this second term.

To convert the parameters obtained from this spectral fit to a trap stiffness, the following is computed:

.. math::

    \kappa = 2 \pi \gamma_0 f_c \tag{$\mathrm{N/m}$}

Here :math:`\kappa` then represents the estimated trap stiffness.

We can calibrate the position by considering the diffusion of the bead:

.. math::

    D_\mathrm{physical} = \frac{k_B T}{\gamma_0} \tag{$\mathrm{m^2/s}$}

Here :math:`k_B` is the Boltzmann constant and :math:`T` is the local temperature in Kelvin.
Comparing this to its measured counterpart in Volts squared per second provides us with the desired
calibration factor:

.. math::

    R_d = \sqrt{\frac{D_\mathrm{physical}}{D_\mathrm{measured}}} \tag{$\mathrm{m/V}$}

The force response :math:`R_f` can then be computed as:

.. math::

    R_f = \kappa R_d \tag{$\mathrm{N/V}$}

All three of these quantities depend on the parameter :math:`\gamma_0`, which corresponds to the
drag coefficient of a sphere and is given by:

.. math::

    \gamma_0 = 3 \pi \eta d \tag{$\mathrm{kg/s}$}

where :math:`\eta` corresponds to the dynamic viscosity [Pa*s] and :math:`d` is the bead diameter [m].
