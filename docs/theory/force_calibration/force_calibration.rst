Force Calibration
=================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Why is force calibration necessary?
-----------------------------------

Optical tweezers typically measure forces and displacements by detecting deflections of a trapping laser by a trapped bead.
These deflections are measured at the back focal plane of the beam using position sensitive detectors (PSDs).

.. image:: figures/back_focal.png
  :nbattach:

For small displacements :math:`x` of the bead from the center of the trap, the relation between the force :math:`F` pulling the bead back towards the center and the displacement can be assumed linear:

.. math::

    x = R_d V

and

.. math::

    F = R_f V

Where :math:`V` is the position-dependent voltage signal from the PSD and :math:`R_d` and :math:`R_f` are the displacement and force sensitivity proportionality constants, respectively.
Force calibration refers to computing these conversion factors.

Several methods exist to calibrate optical traps based on sensor signals.
In this section, we will provide an overview of the physical background of power spectral calibration.

Why does the power spectrum look the way it does?
-------------------------------------------------

Consider a small bead freely diffusing in a medium (no optical trapping taking place).
Neglecting hydrodynamic and inertial effects (more on this later), we obtain the following equation of motion:

.. math::

    \dot{x} = \frac{1}{\gamma} F_{thermal}(t)

where :math:`x` is the time-dependent position of the bead, :math:`\dot{x}` is the first derivative with respect to time, :math:`\gamma`  is the drag coefficient and :math:`F_{thermal}(t)` is the thermal force driving the diffusion.
This thermal force is assumed to have the statistical properties of uncorrelated white noise:

.. math::

    F_{thermal}(t) = \sqrt{2 \gamma k_B T} \xi(t)

where :math:`k_B` is the Boltzmann constant, :math:`T` is the absolute temperature, and :math:`\xi(t)` is normalized white noise.

We can rewrite this in terms of the diffusion constant by using Einstein's relation :math:`D = k_B T / \gamma`:

.. math::

    \dot{x} = \frac{1}{\gamma} \sqrt{2 \gamma^2 D} \xi(t) = \sqrt{2D} \xi (t) \tag{$\mathrm{m/s}$}

with the diffusion constant :math:`D`.
Note how this constant depends on the drag coefficient.

The expected power spectrum of such free diffusion is given by the following equation:

.. math::

    P_\mathrm{diffusion}(f) = \frac{D}{\pi^2 f^2} \tag{$\mathrm{m^2/Hz}$}

We can plot this spectrum for different diffusion constants::

    f = np.arange(100, 23000)
    for diffusion in 10**np.arange(1, 4):
        plt.loglog(f, diffusion / (np.pi * f**2), label=f"D={diffusion} $\mu m^2/s$")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [$\mu m^2$/Hz]");
    plt.legend()

.. image:: figures/diffusion_spectra.png

Here, the vertical axis represents a displacement amplitude while the horizontal axis represents frequency.
Observe how, for free diffusion, the low frequencies have larger amplitudes than the high frequencies.

Next, consider the effect of the optical trap on this diffusion.
The optical trap will pull the bead back towards the focus of the beam.
As such, the trap will constrain motion, limiting in particular the high amplitudes.
In other words, we expect this effect to be more prominent at low frequencies as they have larger amplitudes.
Compared to the previous spectra, this should result in a plateau at the low frequency end of the spectrum.
Still neglecting hydrodynamic and inertial effects, one can write down the differential equation for a trapped bead.

.. math::

    \dot{x} + \frac{\kappa}{\gamma} x = \sqrt{2D} \xi (t) \tag{$\mathrm{m/s}$}

Note that we now have a few additional parameters, namely the drag coefficient :math:`\gamma` and the trap stiffness :math:`\kappa`.
From this, the following power spectrum can be derived:

.. math::

    P_{\mathrm{diffusion}}(f) = \frac{D}{\pi^2 \left(f^2 + \left(\frac{\kappa}{2 \pi \gamma}\right)^2\right)} = \frac{D}{\pi^2 \left(f^2 + f_c^2\right) } \tag{$\mathrm{m^2/Hz}$}

Note how we've defined a corner frequency :math:`f_c` from the trap stiffness and the drag coefficient.
When plotting this equation for various values of the corner frequency, we see the expected plateau::

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    diffusion, corner_freq = 1000, 1000
    for diffusion in 10**np.arange(1, 4):
        plt.loglog(f, diffusion / (np.pi * (f**2 + corner_freq**2)), label=f"D={diffusion} $\mu m^2/s$")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [$\mu m^2$/Hz]");
    plt.legend()

    plt.subplot(1, 2, 2)
    diffusion, corner_freq = 1000, 1000
    for corner_freq in [1000, 5000, 10000]:
        line, = plt.loglog(
            f, diffusion / (np.pi * (f**2 + corner_freq**2)), label=f"$f_c$={corner_freq} Hz"
        )
        plt.axvline(corner_freq, color=line.get_color(), linestyle="--")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [$\mu m^2$/Hz]");
    plt.legend()

.. image:: figures/lorentzians.png

The simple model plotted here is known as the Lorentzian model and it is only a good approximation for small beads (more on that later).
We see from the plot that a stiffer trap constrains diffusion more strongly (leading to a wider plateau) and a higher corner frequency.
In practice, we wish to fit this spectrum in order to determine the corner frequency which in turn provides information on the trap stiffness once the drag coefficient is known.

Fitting a power spectrum
------------------------

In the previous section, the physical origin of the power spectrum was introduced.
However, there are some practical aspects to consider.
So far, we have only considered the expectation value of the power spectrum.
In reality, power spectral values follow a distribution.

The real and imaginary part of the frequency spectrum are normally distributed.
As a consequence, the squared magnitude of the power spectrum is exponentially distributed.
This has two consequences:

- Fitting the power spectral values directly using a simple least squares fitting routine, we would get very biased estimates. These estimates would overestimate the plateau and corner frequency, resulting in overestimated trap stiffness and force response and an underestimated distance response.
- The signal to noise ratio is poor (equal to one :cite:`norrelykke2010power`).

A commonly used method for dealing with this involves data averaging, which trades resolution for an improved signal to noise ratio.
In addition, by virtue of the central limit theorem, data averaging leads to a more symmetric data distribution (more amenable to standard least-squares fitting procedures).

There are two ways to perform such averaging:

- The first is to split the time series into windows of equal length, compute the power spectrum for each chunk of data and averaging these. This procedure is referred to as *windowing*.
- The second is to calculate the spectrum for the full dataset followed by downsampling in the spectral domain by averaging adjacent bins according to :cite:`berg2004power`. This procedure is referred to as *blocking*.

We use the blocking method for spectral averaging, since this allows us to reject noise peaks at high resolution prior to averaging.
Note however, that the error incurred by this blocking procedure depends on :math:`n_b`, the number of points per block, :math:`\Delta f`, the spectral resolution and inversely on the corner frequency :cite:`berg2004power`.

Setting the number of points per block too low would result in a bias from insufficient averaging :cite:`berg2004power`.
Insufficient averaging would result in an overestimation of the force response (:math:`R_f`) and an underestimation of the distance response (:math:`R_d`).
In practice, one should use a high number of points per block (:math:`n_b \gg 100`), unless a very low corner frequency precludes this.
In such cases, it is preferable to increase the measurement time.

Passive calibration
-------------------

Passive calibration is also often referred to as thermal calibration and involves calibration without moving the trap or stage.
In passive calibration, the Brownian motion of the bead in the trap is analyzed in order to find calibration factors for both the positional detection as well as the force.

To fit passive calibration data, we will use a model based on a number of publications by the Flyvbjerg group :cite:`berg2004power,tolic2004matlab,hansen2006tweezercalib,berg2006power`.
The Pylake implementation of this model is :class:`~.PassiveCalibrationModel`.
The most basic form of passive calibration starts by fitting the following equation to the power spectrum:

.. math::

    P(f) = \frac{D_\mathrm{measured}}{\pi ^ 2 \left(f^2 + f_c ^ 2\right)} g(f, f_\mathrm{diode}, \alpha) \tag{$\mathrm{V^2/Hz}$}

where :math:`D_\mathrm{measured}` corresponds to the diffusion constant (in :math:`V^2/s`), :math:`f` the frequency and :math:`f_c` the corner frequency.
The first part of the equation is the same as before and represents the part of the spectrum that originates from the physical motion of the bead.
The second term :math:`g` takes into account the slower response of the position detection system and is given by:

.. math::

    g(f, f_\mathrm{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_\mathrm{diode})^2} \tag{$-$}

Here :math:`\alpha` corresponds to the fraction of the signal response that is instantaneous, while :math:`f_\mathrm{diode}` characterizes the frequency response of the diode.
Note that not all sensor types require this second term.

To convert the parameters obtained from this spectral fit to a trap stiffness, the following is computed:

.. math::

    \kappa = 2 \pi \gamma_0 f_c \tag{$\mathrm{N/m}$}

Here :math:`\kappa` then represents the estimated trap stiffness.

We can calibrate the position by considering the diffusion of the bead:

.. math::

    D_\mathrm{physical} = \frac{k_B T}{\gamma_0} \tag{$\mathrm{m^2/s}$}

Here :math:`k_B` is the Boltzmann constant and :math:`T` is the local temperature in Kelvin. Comparing this to its measured counterpart in Volts squared per second provides us with the desired calibration factor:

.. math::

    R_d = \sqrt{\frac{D_\mathrm{physical}}{D_\mathrm{measured}}} \tag{$\mathrm{m/V}$}

The force response :math:`R_f` can then be computed as:

.. math::

    R_f = \kappa R_d \tag{$\mathrm{N/V}$}

All three of these quantities depend on the parameter :math:`\gamma_0`, which corresponds to the drag coefficient of a sphere and is given by:

.. math::

    \gamma_0 = 3 \pi \eta d \tag{$\mathrm{kg/s}$}

where :math:`\eta` corresponds to the dynamic viscosity [Pa*s] and :math:`d` is the bead diameter [m].

Hydrodynamically correct model
------------------------------

While the idealized model discussed in the previous section is sometimes sufficiently accurate, there are scenarios where more detailed models are necessary.

The frictional forces applied by the viscous environment to the bead are proportional to the bead's velocity.
The idealized model is based on the assumption that the bead's velocity is constant, which, for a stochastic process such as Brownian motion, is not an accurate assumption.
In addition, the bead and the surrounding fluid have their own mass and inertia, which are also neglected in the idealized model.
Together, the non-constant speed and the inertial effects result in frequency-dependent frictional forces that a more accurate hydrodynamically correct model takes into account.
These effects are strongest at higher frequencies, and for larger bead diameters.

The following equation accounts for a frequency dependent drag :cite:`tolic2006calibration`:

.. math::

    P_\mathrm{hydro}(f) = \frac{D \mathrm{Re}(\gamma / \gamma_0)}{\pi^2 \left(\left(f_{c,0} + f \mathrm{Im}(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f \mathrm{Re}(\gamma / \gamma_0)\right)^2\right)} \tag{$\mathrm{m^2/Hz}$}

where the corner frequency is given by:

.. math::

    f_{c, 0} = \frac{\kappa}{2 \pi \gamma_0} \tag{$\mathrm{Hz}$}

and :math:`f_{m, 0}` parameterizes the time it takes for friction to dissipate the kinetic energy of the bead:

.. math::

    f_{m, 0} = \frac{\gamma_0}{2 \pi m} \tag{$\mathrm{Hz}$}

with :math:`m` the mass of the bead.
Finally, :math:`\gamma` corresponds to the frequency dependent drag.
For measurements in bulk, far away from a surface, :math:`\gamma` = :math:`\gamma_\mathrm{stokes}`, where :math:`\gamma_\mathrm{stokes}` is given by:

.. math::

    \gamma_\mathrm{stokes} = \gamma_0 \left(1 + (1 - i)\sqrt{\frac{f}{f_{\nu}}} - \frac{2}{9}\frac{f}{f_{\nu}} i\right) \tag{$\mathrm{kg/s}$}

Here :math:`f_{\nu}` is the frequency at which the penetration depth equals the radius of the bead, :math:`4 \nu/(\pi d^2)` with :math:`\nu` the kinematic viscosity.

This approximation is reasonable, when the bead is far from the surface.

When approaching the surface, the drag experienced by the bead depends on the distance between the bead and the surface of the flow cell.
An approximate expression for the frequency dependent drag is then given by :cite:`tolic2006calibration`:

.. math::

    \gamma(f, R/l) = \frac{\gamma_\mathrm{stokes}(f)}{1 - \frac{9}{16}\frac{R}{l}\left(1 - \left((1 - i)/3\right)\sqrt{\frac{f}{f_{\nu}}} + \frac{2}{9}\frac{f}{f_{\nu}}i - \frac{4}{3}(1 - e^{-(1-i)(2l-R)/\delta})\right)} \tag{$\mathrm{kg/s}$}

Where :math:`\delta = R \sqrt{\frac{f_{\nu}}{f}}` represents the aforementioned penetration depth, :math:`R` corresponds to the bead radius and :math:`l` to the distance from the bead center to the nearest surface.

While these models may look daunting, they are all available in Pylake and can be used by simply providing a few additional arguments to the :class:`~.PassiveCalibrationModel`.
It is recommended to use these equations when less than 10% systematic error is desired :cite:`tolic2006calibration`.
No general statement can be made regarding the accuracy that can be achieved with the simple Lorentzian model, nor the direction of the systematic error, as it depends on several physical parameters involved in calibration :cite:`tolic2006calibration,berg2006power`.

The figure below shows the difference between the hydrodynamically correct model (solid lines) and the idealized Lorentzian model (dashed lines) for various bead sizes.
It can be seen that for large bead sizes and higher trap powers the differences can be substantial.

.. image:: figures/hydro.png
  :nbattach:

.. note::

    One thing to note is that when considering the surface in the calibration procedure, the drag coefficient returned from the model corresponds to the drag coefficient extrapolated back to its bulk value.

Faxen's law
-----------

The hydrodynamically correct model presented in the previous section works well when the bead center is at least 1.5 times the radius above the surface.
When moving closer than this limit, we fall back to a model that more accurately describes the change in drag at low frequencies, but neglects the frequency dependent effects.

To understand why, let's introduce Faxen's approximation for drag on a sphere near a surface under creeping flow conditions.
This model is used for lateral calibration very close to a surface :cite:`schaffer2007surface` and is given by the following equation:

.. math::

    \gamma_\mathrm{faxen}(R/l) = \frac{\gamma_0}{
        1 - \frac{9R}{16l} + \frac{1R^3}{8l^3} - \frac{45R^4}{256l^4} - \frac{1R^5}{16l^5}
    } \tag{$\mathrm{kg/s}$}

At frequency zero, the frequency dependent model used in the previous section reproduces this model up to and including its second order term in :math:`R/l`.
It is, however, a lower order model and the accuracy decreases rapidly as the distance between the bead and surface become very small.
The figure below shows how the model predictions at frequency zero deviate strongly from the higher order model:

.. image:: figures/freq_dependent_drag_zero.png
  :nbattach:

In addition, the deviation from a Lorentzian due to the frequency dependence of the drag is reduced upon approaching a surface :cite:`schaffer2007surface`.

.. image:: figures/freq_dependence_near.png
  :nbattach:

These two aspects make using Faxen's law in combination with a Lorentzian a more suitable model for situations where we have to calibrate extremely close to the surface.

Axial Calibration
-----------------

For calibration in the axial direction, no hydrodynamically correct theory exists.

Similarly as for the lateral component, we will fall back to a model that describes the change in drag at low frequencies.
However, while we had a simple expression for the lateral drag as a function of distance, no simple closed-form equation exists for the axial dimension.
Brenner et al provide an exact infinite series solution :cite:`brenner1961slow`.
Based on this solution :cite:`schaffer2007surface` derived a simple equation which approximates the distance dependence of the axial drag coefficient.

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

This model deviates less than 0.1% from Brenner's exact formula for :math:`l/R >= 1.1` and less than 0.3% over the entire range of :math:`l` :cite:`schaffer2007surface`.
Plotting these reveals that there is a larger effect of the surface in the axial than lateral direction.

.. image:: figures/drag_coefficient.png
  :nbattach:

Active Calibration
------------------

For certain applications, passive force calibration, as described above, is not sufficiently accurate.
Using active calibration, the accuracy of the calibration can be improved.
The reason for this is that active calibration uses fewer assumptions than passive calibration.

When performing passive calibration, we base our calculations on a theoretical drag coefficient.
This theoretical drag coefficient depends on parameters that are only known with limited precision:

- The diameter of the bead :math:`d` in microns.
- The dynamic viscosity :math:`\eta` in Pascal seconds.
- The distance to the surface :math:`h` in microns.

This viscosity in turn depends strongly on the local temperature around the bead, which depends on several physical parameters (e.g. the power of the trapping laser, the buffer medium, the bead size and material) and is typically poorly known.

During active calibration, the trap or nanostage is oscillated sinusoidally.
These oscillations result in a driving peak in the force spectrum.
Using power spectral analysis, the force can then be calibrated without prior knowledge of the drag coefficient.

When the power spectrum is computed from an integer number of oscillations, the driving peak is visible at a single data point at :math:`f_\mathrm{drive}`.

.. image:: figures/driving_input.png
  :nbattach:

The physical spectrum is then given by a thermal part (like before):

.. math::

    P^\mathrm{thermal}(f) = \frac{D}{\pi ^ 2 \left(f^2 + f_c^2\right)} \tag{$\mathrm{m^2/Hz}$}

And an active part:

.. math::

    P^\mathrm{active}(f) = \frac{A^2}{2\left(1 + \frac{f_c^2}{f_\mathrm{drive}^2}\right)} \delta(f - f_\mathrm{drive}) \tag{$\mathrm{m^2/Hz}$}

Here :math:`A` refers to the driving amplitude. Added together, these give rise to the full power spectrum:

.. math::

    P^\mathrm{total}(f) = P^\mathrm{thermal}(f) + P^\mathrm{active}(f) \tag{$\mathrm{m^2/Hz}$}

Since we know the driving amplitude, we know how the bead reacts to the driving motion and we can observe this response in the power spectrum, we can use this relation to determine the positional calibration.

If we use the basic Lorentzian model, then the theoretical power (integral over the delta spike) corresponding to the driving input is given by :cite:`tolic2006calibration`:

.. math::

    W_\mathrm{physical} = \frac{A^2}{2\left(1 + \frac{f_c^2}{f_\mathrm{drive}^2}\right)} \tag{$\mathrm{m^2}$}

Subtracting the thermal part of the spectrum, we can determine the same quantity experimentally.

.. math::

    W_\mathrm{measured} = \left(P_\mathrm{measured}^\mathrm{total}(f_\mathrm{drive}) - P_\mathrm{measured}^\mathrm{thermal}(f_\mathrm{drive})\right) \Delta f \tag{$\mathrm{V^2}$}

where :math:`\Delta f` refers to the width of one spectral bin.
Here the thermal contribution that needs to be subtracted is obtained from fitting the thermal part of the spectrum using the passive calibration procedure from before.
The desired positional calibration is then:

.. math::

    R_d = \sqrt{\frac{W_\mathrm{physical}}{W_\mathrm{measured}}} \tag{$\mathrm{m/V}$}

Note how this time around, we did not rely on assumptions on the viscosity of the medium or the bead size.

As a side effect of this calibration, we actually obtain an experimental estimate of the drag coefficient:

.. math::

    \gamma_\mathrm{measured} = \frac{k_B T}{R_d^2 D_\mathrm{measured}} \tag{$\mathrm{kg/s}$}

Analogously to passive calibration, there is also a hydrodynamically correct theory for active calibration which should be used when inertial forces cannot be neglected.
This involves fitting the thermal spectrum with the hydrodynamically correct power spectrum discussed earlier, but also requires using a hydrodynamically correct model for the peak:

.. math::

    P_\mathrm{hydro}^\mathrm{active}(f) = \frac{\left(A f_\mathrm{drive} \left|\gamma / \gamma_0\right|\right)^2 \delta \left(f - f_\mathrm{drive}\right)}{2 \left(\left(f_{c,0} + f \mathrm{Im}(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f \mathrm{Re}(\gamma / \gamma_0)\right)^2\right)} \tag{$\mathrm{m^2/Hz}$}

We can also include a distance to the surface like before.
This results in an expression for the drag coefficient :math:`\gamma` that depends on the distance to the surface which is given by the same equations as listed in the section on the hydrodynamically correct model.
