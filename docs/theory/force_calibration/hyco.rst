Hydrodynamically correct model
------------------------------

While the idealized model discussed in the previous section is sometimes sufficiently accurate,
there are scenarios where more detailed models are necessary.

The frictional forces applied by the viscous environment to the bead are proportional to the bead's
velocity. The idealized model is based on the assumption that the bead's velocity is constant, which,
for a stochastic process such as Brownian motion, is not an accurate assumption. In addition, the
bead and the surrounding fluid have their own mass and inertia, which are also neglected in the idealized model.
Together, the non-constant speed and the inertial effects result in frequency-dependent frictional
forces that a more accurate hydrodynamically correct model takes into account.
These effects are strongest at higher frequencies, and for larger bead diameters.

The following equation accounts for a frequency dependent drag :cite:`tolic2006calibration`:

.. math::

    P_\mathrm{hydro}(f) = \frac{D \mathrm{Re}(\gamma / \gamma_0)}{\pi^2 \left(\left(f_{c,0} +
    f \mathrm{Im}(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f \mathrm{Re}(\gamma / \gamma_0)\right)^2\right)}
    \tag{$\mathrm{m^2/Hz}$}

where the corner frequency is given by:

.. math::

    f_{c, 0} = \frac{\kappa}{2 \pi \gamma_0} \tag{$\mathrm{Hz}$}

and :math:`f_{m, 0}` parameterizes the time it takes for friction to dissipate the kinetic energy of the bead:

.. math::

    f_{m, 0} = \frac{\gamma_0}{2 \pi m} \tag{$\mathrm{Hz}$}

with :math:`m` the mass of the bead.
Finally, :math:`\gamma` corresponds to the frequency dependent drag.
For measurements in bulk, far away from a surface, :math:`\gamma` = :math:`\gamma_\mathrm{stokes}`,
where :math:`\gamma_\mathrm{stokes}` is given by:

.. math::

    \gamma_\mathrm{stokes} = \gamma_0 \left(1 + (1 - i)\sqrt{\frac{f}{f_{\nu}}} - \frac{2}{9}\frac{f}{f_{\nu}} i\right)
    \tag{$\mathrm{kg/s}$}

Here :math:`f_{\nu}` is the frequency at which the penetration depth equals the radius of the bead,
:math:`4 \nu/(\pi d^2)` with :math:`\nu` the kinematic viscosity.

This approximation is reasonable, when the bead is far from the surface.

When approaching the surface, the drag experienced by the bead depends on the distance between the
bead and the surface of the flow cell. An approximate expression for the frequency dependent drag is
then given by :cite:`tolic2006calibration`:

.. math::

    \gamma(f, R/l) = \frac{\gamma_\mathrm{stokes}(f)}{1 - \frac{9}{16}\frac{R}{l}
    \left(1 - \left((1 - i)/3\right)\sqrt{\frac{f}{f_{\nu}}} + \frac{2}{9}\frac{f}{f_{\nu}}i -
    \frac{4}{3}(1 - e^{-(1-i)(2l-R)/\delta})\right)} \tag{$\mathrm{kg/s}$}

Where :math:`\delta = R \sqrt{\frac{f_{\nu}}{f}}` represents the aforementioned penetration depth,
:math:`R` corresponds to the bead radius and :math:`l` to the distance from the bead center to the nearest surface.

While these models may look daunting, they are all available in Pylake and can be used by simply
providing a few additional arguments to the :class:`~.PassiveCalibrationModel`. It is recommended to
use these equations when less than 10% systematic error is desired :cite:`tolic2006calibration`.
No general statement can be made regarding the accuracy that can be achieved with the simple Lorentzian
model, nor the direction of the systematic error, as it depends on several physical parameters involved
in calibration :cite:`tolic2006calibration,berg2006power`.

The figure below shows the difference between the hydrodynamically correct model (solid lines) and the
idealized Lorentzian model (dashed lines) for various bead sizes. It can be seen that for large bead
sizes and higher trap powers the differences can be substantial.

.. image:: figures/hydro.png
  :nbattach:

.. note::

    One thing to note is that when considering the surface in the calibration procedure, the drag
    coefficient returned from the model corresponds to the drag coefficient extrapolated back to its
    bulk value.

Faxen's law
^^^^^^^^^^^

The hydrodynamically correct model presented in the previous section works well when the bead center
is at least 1.5 times the radius above the surface. When moving closer than this limit, we fall back
to a model that more accurately describes the change in drag at low frequencies, but neglects the
frequency dependent effects.

To understand why, let's introduce Faxen's approximation for drag on a sphere near a surface under
creeping flow conditions. This model is used for lateral calibration very close to a surface
:cite:`schaffer2007surface` and is given by the following equation:

.. math::

    \gamma_\mathrm{faxen}(R/l) = \frac{\gamma_0}{
        1 - \frac{9R}{16l} + \frac{1R^3}{8l^3} - \frac{45R^4}{256l^4} - \frac{1R^5}{16l^5}
    } \tag{$\mathrm{kg/s}$}

At frequency zero, the frequency dependent model used in the previous section reproduces this model
up to and including its second order term in :math:`R/l`. It is, however, a lower order model and the
accuracy decreases rapidly as the distance between the bead and surface become very small.
The figure below shows how the model predictions at frequency zero deviate strongly from the higher order model:

.. image:: figures/freq_dependent_drag_zero.png
  :nbattach:

In addition, the deviation from a Lorentzian due to the frequency dependence of the drag is reduced
upon approaching a surface :cite:`schaffer2007surface`.

.. image:: figures/freq_dependence_near.png
  :nbattach:

These two aspects make using Faxen's law in combination with a Lorentzian a more suitable model for
situations where we have to calibrate extremely close to the surface.

Axial Calibration
^^^^^^^^^^^^^^^^^

For calibration in the axial direction, no hydrodynamically correct theory exists.

Similarly as for the lateral component, we will fall back to a model that describes the change in
drag at low frequencies. However, while we had a simple expression for the lateral drag as a function
of distance, no simple closed-form equation exists for the axial dimension. Brenner et al provide an
exact infinite series solution :cite:`brenner1961slow`. Based on this solution :cite:`schaffer2007surface`
derived a simple equation which approximates the distance dependence of the axial drag coefficient.

.. math::

    \gamma_\mathrm{axial}(R/l) = \frac{\gamma_0}{
        1.0
        - \frac{9R}{8l}
        + \frac{1R^3}{2l^3}
        - \frac{57R^4}{100l^4}
        + \frac{1R^5}{5l^5}
        + \frac{7R^{11}}{200l^{11}}
        - \frac{1R^{12}}{25l^{12}}
    } \tag{$\mathrm{kg/s}$}

This model deviates less than 0.1% from Brenner's exact formula for :math:`l/R >= 1.1` and less than
0.3% over the entire range of :math:`l` :cite:`schaffer2007surface`.
Plotting these reveals that there is a larger effect of the surface in the axial than lateral direction.

.. image:: figures/drag_coefficient.png
  :nbattach:
