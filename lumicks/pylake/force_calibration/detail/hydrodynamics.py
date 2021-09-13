import numpy as np

"""
References
----------
.. [1] Berg-Sørensen, K. & Flyvbjerg, H. Power spectrum analysis for optical tweezers. Rev. Sci.
       Instrum. 75, 594 (2004).
.. [2] Tolić-Nørrelykke, I. M., Berg-Sørensen, K. & Flyvbjerg, H. MatLab program for precision
       calibration of optical tweezers. Comput. Phys. Commun. 159, 225–240 (2004).
.. [3] Hansen, P. M., Tolic-Nørrelykke, I. M., Flyvbjerg, H. & Berg-Sørensen, K.
       tweezercalib 2.1: Faster version of MatLab package for precise calibration of optical
       tweezers. Comput. Phys. Commun. 175, 572–573 (2006).
.. [4] Berg-Sørensen, K., Peterman, E. J. G., Weber, T., Schmidt, C. F. & Flyvbjerg, H. Power
       spectrum analysis for optical tweezers. II: Laser wavelength dependence of parasitic
       filtering, and how to achieve high bandwidth. Rev. Sci. Instrum. 77, 063106 (2006).
.. [5] Tolić-Nørrelykke, S. F, and Flyvbjerg, H, "Power spectrum analysis with least-squares
       fitting: amplitude bias and its elimination, with application to optical tweezers and
       atomic force microscope cantilevers." Review of Scientific Instruments 81.7 (2010)
.. [6] Tolić-Nørrelykke S. F, Schäffer E, Howard J, Pavone F. S, Jülicher F and Flyvbjerg, H.
       Calibration of optical tweezers with positional detection in the back focal plane,
       Review of scientific instruments 77, 103101 (2006).
"""


def calculate_dissipation_frequency(gamma0, bead_radius, rho_bead):
    """frequency_m parametrizes the time it takes for friction to dissipate the kinetic energy of
    the sphere.

    Parameters
    ----------
    gamma0 : float
        drag coefficient [mPas]
    bead_radius : float
        radius of the bead [m]
    rho_bead : float
        density of the bead [kg/m^3]

    Note that the mass term has lead to some confusion in literature. The expression used for m
    in the dissipation frequency described in [1] uses:

      m_star = bead_mass + (2.0 * np.pi * rho_sample * radius ** 3.0) / 3.0

    However, [6] specifically mentions in reference 30 (in a footnote) that this term only includes
    the bead mass whereas [1] contains a typographical error. Note that in the code that implements
    [6] (ref [3]), they did include it however. We decided to follow the paper and omit the second
    term. The effects of this difference were very small for all simulations tested."""
    bead_mass = (4.0 / 3.0) * np.pi * bead_radius ** 3 * rho_bead
    return gamma0 / (2.0 * np.pi * bead_mass)


def calculate_complex_drag(f, gamma0, rho_sample, bead_radius, distance_to_surface=None):
    """Calculates the complex drag coefficient which takes into account surface effects
    analogously to [1].

    Parameters
    ----------
    f : float
        Frequency [Hz]
    gamma0 : float
        Drag coefficient [mPas]
    bead_radius : float
        Bead radius [m]
    rho_sample : float
        Sample mass density [kg/m^3]
    distance_to_surface : float
        Distance from bead center to nearest surface [m]
    """
    # frequency_nu is the frequency at which the penetration depth in the liquid of the
    # bead’s linear harmonic motion equals the radius of the bead. It parametrizes the
    # flow pattern established around a sphere undergoing linear harmonic oscillations in
    # an incompressible fluid and it is unrelated to the mass of particle.
    #
    # The equation given in [6] is:
    #   nu / (pi * radius**2)
    #
    # Where nu is the kinematic viscosity. In the paper gamma0 is defined as:
    #
    #   gamma0 = 6 * pi * rho_sample * nu * radius
    #
    # Hence, we can obtain nu (kinematic viscosity) as:
    #
    #   gamma0 / (6 * pi * rho_sample * radius)
    nu = gamma0 / (6.0 * np.pi * rho_sample * bead_radius)
    frequency_nu = nu / (np.pi * bead_radius ** 2)

    freq_ratio = f / frequency_nu
    sqrt_freq_ratio = np.sqrt(freq_ratio)

    # Stokes contribution to the drag coefficient (gamma / gamma0 from the paper)
    # Equation D4 from [6]
    real_stokes = 1 + sqrt_freq_ratio
    imag_stokes = -sqrt_freq_ratio - 2 / 9 * freq_ratio

    # If we have no distance specified, we assume we are very far from the surface, in which case
    # Stokes drag applies.
    if distance_to_surface is None:
        return real_stokes, imag_stokes

    epsilon = (2 * distance_to_surface - bead_radius) * sqrt_freq_ratio / bead_radius
    r_over_l = bead_radius / distance_to_surface

    # Effect of the surface (depends on depth) e.g. denominator from Eq D6 in [6] which has been
    # re-written with Euler's formula to resemble the form used in [1].
    # fmt: off
    real_surf = 1 - 9 / 16 * r_over_l * (1 - sqrt_freq_ratio / 3 - 4 / 3 * (1 - np.exp(-epsilon) * np.cos(epsilon)))
    imag_surf = - 9 / 16 * r_over_l * (sqrt_freq_ratio / 3 + 2 / 9 * freq_ratio + 4 / 3 * np.exp(-epsilon) * np.sin(epsilon))
    # fmt: on

    # Perform the complex division in Eq. D6 [6] producing gamma / gamma0
    norm = real_surf * real_surf + imag_surf * imag_surf
    real_drag = (real_stokes * real_surf + imag_stokes * imag_surf) / norm
    imag_drag = (imag_stokes * real_surf - real_stokes * imag_surf) / norm

    return real_drag, imag_drag


def passive_power_spectrum_model_hydro(
    f,
    fc,
    diffusion_constant,
    gamma0,
    bead_radius,
    rho_sample,
    rho_bead,
    distance_to_surface,
):
    """Theoretical model for the hydrodynamically correct power spectrum.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency values, in Hz.
    fc : float
        Corner frequency, in Hz.
    diffusion_constant : float
        Diffusion constant, in (a.u.)^2/s
    gamma0 : float
        Drag coefficient, in mPas.
    bead_radius : float
        Bead radius, in m.
    rho_sample : float
        Sample mass density, in kg/m^3
    rho_bead : float
        Bead mass density, in kg/m^3
    distance_to_surface : float
        Distance to nearest surface, in m
    """
    re_drag, im_drag = calculate_complex_drag(
        f, gamma0, rho_sample, bead_radius, distance_to_surface
    )
    frequency_m = calculate_dissipation_frequency(gamma0, bead_radius, rho_bead)
    denominator = (fc + f * im_drag - f ** 2 / frequency_m) ** 2 + (f * re_drag) ** 2
    power_spectrum = diffusion_constant / (np.pi ** 2) * re_drag / denominator  # Equation D2 [6]
    return power_spectrum


def theoretical_driving_power_hydrodynamics(
    fc,
    driving_frequency,
    driving_amplitude,
    gamma0,
    bead_radius,
    rho_sample,
    rho_bead,
    distance_to_surface,
):
    """Theoretical equation for the hydrodynamically correct response peak amplitude.

    Parameters
    ----------
    fc : float
        Corner frequency [Hz]
    driving_frequency : float
        Driving frequency [Hz]
    driving_amplitude : float
        Driving amplitude [m]
    gamma0 : float
        Drag coefficient [mPas]
    bead_radius : float
        Bead radius [m]
    rho_sample : float
        Sample density
    rho_bead : float
        Bead density
    distance_to_surface : float
        Distance from the bead center to the surface
    """
    re_drag, im_drag = calculate_complex_drag(
        driving_frequency, gamma0, rho_sample, bead_radius, distance_to_surface
    )
    frequency_m = calculate_dissipation_frequency(gamma0, bead_radius, rho_bead)

    # fmt: off
    denominator = 2 * ((fc + driving_frequency * im_drag - driving_frequency ** 2 / frequency_m) ** 2 + (driving_frequency * re_drag) ** 2)
    # Equation D3 from [6]
    power_theory = (driving_amplitude * driving_frequency) ** 2 * (re_drag ** 2 + im_drag ** 2) / denominator
    # fmt: on

    return power_theory
