.. _diode_theory:

Position sensitive detector
---------------------------

The previous section introduced the origin of the frequency spectrum of a bead in an optical trap.
In reality, our measurement is affected by two processes:

1. The motion of the bead in the trap.
2. The response of the detector to the incident light.

.. image:: figures/diode_filtering.png
  :nbattach:

This second factor depends on the type of measurement device being used.
Typical position sensitive detectors are made of silicon.
Such a detector has a very high bandwidth for visible light (in the MHz range).
Unfortunately, the bandwidth is markedly reduced for the near infra-red light of the trapping laser
:cite:`berg2003unintended,berg2006power`.
This makes it less sensitive to changes in signal at high frequencies.

Why is the bandwidth limited?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The high bandwidth of visible light detection of a silicon photodiode is achieved when incoming photons
are absorbed in the so-called depletion layer of the diode.
Unfortunately, silicon has an increased transparency at the near infra-red wavelength of the trapping laser.
The result of this is that light penetrates deeper into the substrate of the diode, where it generates
charge carriers in a different region of the diode.
These charge carriers then have to diffuse back to the depletion layer, which takes time.
As a result, a fraction of the signal is detected with a much lower bandwidth.

.. image:: figures/diode.png
  :nbattach:

This effect is often referred to as the parasitic filtering effect and is frequently modelled as a first order lowpass filter.
This model is characterized by two numbers whose values depend on the incident laser power :cite:`berg2003unintended`:

- A frequency `f_diode`, given in Hertz.
- A unit-less relaxation factor `alpha` which reflects the fraction of light that is transmitted instantaneously.

.. _high_corner_freq:

High corner frequencies
^^^^^^^^^^^^^^^^^^^^^^^

In literature, the diode parameters are frequently estimated simultaneously with the calibration data
:cite:`berg2003unintended,hansen2006tweezercalib,berg2006power,tolic2006calibration,tolic2004matlab,berg2004power`.
Unfortunately, this can cause issues when calibrating at high powers.

Recall that the physical spectrum is characterized by a corner frequency `fc`, and diffusion constant `D`.
The corner frequency depends on the laser power and bead size (smaller beads resulting in higher corner frequencies).
The parasitic filtering effect also has a corner frequency (`f_diode`) and depends on the incident intensity :cite:`berg2003unintended`.

When these two frequencies are similar, they cannot be estimated from the power spectrum reliably anymore.
The reason for this is that the effects that these parameters have on the power spectrum becomes very similar.
When working with small beads or at high laser powers, it is important to verify that the corner frequency `fc`
does not approach the frequency of the filtering effect `f_diode`.

Sometimes, the parameters of this diode have been characterized independently.
In that case, the arguments `fixed_diode` and `fixed_alpha` can be passed to :func:`~lumicks.pylake.calibrate_force()`
to fix these parameters to their predetermined values, resolving this issue.
For more information on how to achieve this with Pylake, please refer to the :ref:`diode calibration tutorial<diode_tutorial>`.

Mathematical background
^^^^^^^^^^^^^^^^^^^^^^^

In literature, it is frequently modelled up to good accuracy with a first order approximation :cite:`berg2003unintended,tolic2006calibration,berg2006power`.

.. math::

    g(f, f_\mathrm{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_\mathrm{diode})^2} \tag{$-$}
