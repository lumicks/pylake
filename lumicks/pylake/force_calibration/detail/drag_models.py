import warnings

import numpy as np


def faxen_factor(distance_to_surface_m, radius_m):
    """Faxen factor for lateral drag coefficient.

    This factor provides a correction to the drag force for a nearby wall.

    [6] Schäffer, E., Nørrelykke, S. F., & Howard, J. "Surface forces and drag coefficients of
    microspheres near a plane surface measured with optical tweezers." Langmuir, 23(7), 3654-3665
    (2007).

    Parameters
    ----------
    distance_to_surface_m : float
        Distance from the center of the bead to the surface [m]
    radius_m : float
        Radius of the bead [m]
    """
    height_ratio = radius_m / distance_to_surface_m
    denominator = (
        1
        - 9 / 16 * height_ratio
        + 1 / 8 * height_ratio**3
        - 45 / 256 * height_ratio**4
        - 1 / 16 * height_ratio**5
    )
    return 1.0 / denominator


def brenner_axial(distance_to_surface_m, radius_m):
    """Brenner factor for lateral drag coefficient.

    This factor provides a correction to the drag force for a nearby wall.

    [6] Schäffer, E., Nørrelykke, S. F., & Howard, J. "Surface forces and drag coefficients of
    microspheres near a plane surface measured with optical tweezers." Langmuir, 23(7), 3654-3665
    (2007).

    Parameters
    ----------
    distance_to_surface_m : float
        Distance from the center of the bead to the surface [m]
    radius_m : float
        Radius of the bead [m]
    """
    height_ratio = radius_m / distance_to_surface_m
    denominator = (
        1.0
        - (9 / 8) * height_ratio
        + 0.5 * height_ratio**3
        - (57 / 100) * height_ratio**4
        + (1 / 5) * height_ratio**5
        + (7 / 200) * height_ratio**11
        - (1 / 25) * height_ratio**12
    )
    return 1.0 / denominator


def coth(x):
    return np.cosh(x) / np.sinh(x)


def cosech(x):
    return 1.0 / np.sinh(x)


def to_curvilinear_coordinates(r1, r2, distance):
    r"""Transform bead radii to curvilinear coordinates.

    Transforms the radii and distance to a coordinate system with the origin at the radical plane
    between the beads. The new coordinates are given as:

    .. math::

        r_1 = a \mathrm{cosech} \alpha
        d_1 = a \mathrm{coth} \alpha
        r_2 = - a \mathrm{cosech} \beta
        d_2 = - a \mathrm{coth} \beta

    Here :math:`\alpha`, :math:`\beta` and :math:`a` are the variables in the new coordinate system.

    Parameters
    ----------
    r1, r2 : float
        Bead radii.
    distance : float
        Distance between the beads.

    References
    ----------
    .. [1] Stimson, M., & Jeffery, G. B. (1926). The motion of two spheres in a viscous fluid.
           Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical
           and Physical Character, 111(757), 110-116 (2007).
    """
    # Point has to be radical and thus fulfill d1**2 - r1**2 = d2**2 - r2**2
    # Substituting r2 by D - d1 yields: d1 = (r1**2 - r2**2 + D**2) / (2*D)
    if r1 + r2 > distance:
        raise ValueError(
            f"Distance between beads {distance} has to be bigger than their summed radii {r1} + "
            f"{r2}."
        )

    d1 = (r1**2 - r2**2 + distance**2) / (2 * distance)
    d2 = distance - d1
    d1_over_r1 = d1 / r1
    d2_over_r2 = d2 / r2

    alpha = np.arccosh(d1_over_r1)
    beta = -np.arccosh(d2_over_r2)
    a = distance / (d1_over_r1 / np.sinh(alpha) - d2_over_r2 / np.sinh(beta))

    return a, alpha, beta


def calculate_delta(n, alpha, beta):
    """Eqn. 27 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    alpha, beta : float
        Curvilinear coordinates obtained with `to_curvilinear_coordinates`.
    """
    return (
        4.0 * np.sinh((n + 0.5) * (alpha - beta)) ** 2
        - (2.0 * n + 1.0) ** 2 * np.sinh(alpha - beta) ** 2
    )


def calculate_k(n, a):
    """Eqn. 25 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    a : float
        Curvilinear coordinate obtained with `to_curvilinear_coordinates`.
    """
    return (a**2 * n * (n + 1)) / (np.sqrt(2) * (2 * n - 1) * (2 * n + 1) * (2 * n + 3))


def calculate_an(n, k, alpha, beta, delta):
    """Eqn. 28 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    k : float
        Term obtained with `calculate_k`.
    alpha, beta : float
        Curvilinear coordinate obtained with `to_curvilinear_coordinates`.
    delta : float
        Term obtained with `calculate_delta`.
    """
    mul = (2 * n + 3) * k
    term1 = 4 * np.exp(-(n + 0.5) * (alpha - beta)) * np.sinh((n + 0.5) * (alpha - beta))
    term2 = (2 * n + 1) ** 2 * np.exp(alpha - beta) * np.sinh(alpha - beta)
    term3 = (
        2 * (2 * n - 1) * np.sinh((n + 0.5) * (alpha - beta)) * np.cosh((n + 0.5) * (alpha + beta))
    )
    term4 = (
        -2 * (2 * n + 1) * np.sinh((n + 1.5) * (alpha - beta)) * np.cosh((n - 0.5) * (alpha + beta))
    )
    term5 = -(2 * n + 1) * (2 * n - 1) * np.sinh(alpha - beta) * np.cosh(alpha + beta)
    return mul * (term1 + term2 + term3 + term4 + term5) / delta


def calculate_bn(n, k, alpha, beta, delta):
    """Eqn. 29 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    k : float
        Term obtained with `calculate_k`.
    alpha, beta : float
        Curvilinear coordinate obtained with `to_curvilinear_coordinates`.
    delta : float
        Term obtained with `calculate_delta`.
    """
    mul = -(2 * n + 3) * k
    term1 = (
        2 * (2 * n - 1) * np.sinh((n + 0.5) * (alpha - beta)) * np.sinh((n + 0.5) * (alpha + beta))
    )
    term2 = (
        -2 * (2 * n + 1) * np.sinh((n + 1.5) * (alpha - beta)) * np.sinh((n - 0.5) * (alpha + beta))
    )
    term3 = (2 * n + 1) * (2 * n - 1) * np.sinh(alpha - beta) * np.sinh(alpha + beta)
    return mul * (term1 + term2 + term3) / delta


def calculate_cn(n, k, alpha, beta, delta):
    """Eqn. 30 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    k : float
        Term obtained with `calculate_k`.
    alpha, beta : float
        Curvilinear coordinate obtained with `to_curvilinear_coordinates`.
    delta : float
        Term obtained with `calculate_delta`.
    """
    mul = -(2 * n - 1) * k
    term1 = 4 * np.exp(-(n + 0.5) * (alpha - beta)) * np.sinh((n + 0.5) * (alpha - beta))
    term2 = -((2 * n + 1) ** 2) * np.exp(-(alpha - beta)) * np.sinh(alpha - beta)
    term3 = (
        2 * (2 * n + 1) * np.sinh((n - 0.5) * (alpha - beta)) * np.cosh((n + 1.5) * (alpha + beta))
    )
    term4 = (
        -2 * (2 * n + 3) * np.sinh((n + 0.5) * (alpha - beta)) * np.cosh((n + 0.5) * (alpha + beta))
    )
    term5 = (2 * n + 1) * (2 * n + 3) * np.sinh(alpha - beta) * np.cosh(alpha + beta)
    return mul * (term1 + term2 + term3 + term4 + term5) / delta


def calculate_dn(n, k, alpha, beta, delta):
    """Eqn. 31 from Stimson et al.

    Parameters
    ----------
    n : int
        iteration
    k : float
        Term obtained with `calculate_k`.
    alpha, beta : float
        Curvilinear coordinate obtained with `to_curvilinear_coordinates`.
    delta : float
        Term obtained with `calculate_delta`.
    """
    mul = (2 * n - 1) * k
    term1 = (
        2 * (2 * n + 1) * np.sinh((n - 0.5) * (alpha - beta)) * np.sinh((n + 1.5) * (alpha + beta))
    )
    term2 = (
        -2 * (2 * n + 3) * np.sinh((n + 0.5) * (alpha - beta)) * np.sinh((n + 0.5) * (alpha + beta))
    )
    term3 = (2 * n + 1) * (2 * n + 3) * np.sinh(alpha - beta) * np.sinh(alpha + beta)
    return mul * (term1 + term2 + term3) / delta


def coupling_correction_factor_stimson(
    radius1, radius2, distance, *, max_summands=100000, tol=1e-10
):
    r"""Calculate the bead correction factors.

    In the calibration of a dual-trap using active calibration, a bead is excited only by a reduced
    driving flow field. The reason for this is that the other bead slows the fluid down. This
    leads to a lower amplitude response than expected (and hence a lower peak power on the PSD).
    This leads to a higher sensitivity than expected. Using the correction factor :math:`c`
    calculated from this function, we can correct the displacement sensitivity :math:`R_d`,
    force sensitivity :math:`R_f`, and stiffness :math:`\kappa` as follows:

    .. math::

        R_{d, corrected} = c R_d
        R_{f, corrected} = \frac{R_f}{c}
        \kappa_{corrected} = \frac{\kappa}{c^2}

    Note that this function assumes the beads to be aligned along the axis in which the oscillation
    is taking place.

    Parameters
    ----------
    radius1, radius2 : float
        Bead radii
    distance : float
        Distance between the bead centers
    max_summands : int
        How many summands to use maximally. More leads to more accuracy.
    tol : float
        Termination tolerance (when to stop summing), lower leads to more accuracy.

    References
    ----------
    .. [1] Stimson, M., & Jeffery, G. B. (1926). The motion of two spheres in a viscous fluid.
           Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical
           and Physical Character, 111(757), 110-116 (2007).
    """
    a, alpha, beta = to_curvilinear_coordinates(radius1, radius2, distance)
    # In the paper we have:
    #   F = 2 sqrt(2) pi nu V / a * summation, whereas the stokes force was given by 6 pi nu r V
    # We are interested in the factor that translates from the regular flow velocity, to the
    # effective flow velocity when coupling is present. We divide the force we get by the
    # one we'd get in the absence of taking it into account.
    pre_factor = -(1.0 / 3.0) * np.sqrt(2) / a
    tol1 = tol / abs(pre_factor / radius1)
    tol2 = tol / abs(pre_factor / radius2)
    coupling1, coupling2 = 0, 0

    for n in np.arange(1, max_summands + 1):
        k = calculate_k(n, a)

        # When summing for beads that are very close, the sinh and cosh functions can result in overflows for the
        # delta constant, which forms denominator of an, bn, cn and dn. The limit of those coefficients in that case
        # is zero, so we can safely silence this error and simply check whether delta is infinite (in which case
        # we set the coefficients to zero).
        with np.errstate(over="ignore"):
            delta = calculate_delta(n, alpha, beta)

        if np.isfinite(delta):
            an = calculate_an(n, k, alpha, beta, delta)
            bn = calculate_bn(n, k, alpha, beta, delta)
            cn = calculate_cn(n, k, alpha, beta, delta)
            dn = calculate_dn(n, k, alpha, beta, delta)
        else:
            an, bn, cn, dn = 0, 0, 0, 0

        d_coupling1 = (2 * n + 1) * (an + bn + cn + dn)
        d_coupling2 = (2 * n + 1) * (an - bn + cn - dn)
        coupling1 += d_coupling1
        coupling2 += d_coupling2

        # Converged?
        if abs(d_coupling1) < tol1 and abs(d_coupling2) < tol2:
            break
    else:
        warnings.warn(
            RuntimeWarning(
                "Warning, maximum summations exceeded. Coupling factor may be inaccurate"
            )
        )

    return pre_factor * coupling1 / radius1, pre_factor * coupling2 / radius2


def coupling_correction_factor_goldmann(radius, distance, allow_rotation=True):
    r"""Calculates a coupling correction factor for oscillating perpendicular to the
    center-to-center axes of two beads.

    In the calibration of a dual-trap using active calibration, a bead is excited only by a reduced
    driving flow field. The reason for this is that the other bead slows the fluid down. This
    leads to a lower amplitude response than expected (and hence a lower peak power on the PSD).
    This leads to a higher sensitivity than expected. Using the correction factor :math:`c`
    calculated from this function, we can correct the displacement sensitivity :math:`R_d`,
    force sensitivity :math:`R_f`, and stiffness :math:`\kappa` as follows:

    .. math::

        R_{d, corrected} = c R_d
        R_{f, corrected} = \frac{R_f}{c}
        \kappa_{corrected} = \frac{\kappa}{c^2}

    Note that this function assumes the beads to be aligned along the axis perpendicular to the
    axis in which the oscillation is taking place. Both approximation models were obtained from
    [3]_ but were originally presented in [1]_ (spheres prevented to rotate) and [2]_ (spheres
    allowed to rotate).

    Parameters
    ----------
    radius : float
        Bead radius
    distance : float
        Distance between the bead centers
    allow_rotation : float
        Provide the solution for when spheres are allowed to rotate

    References
    ----------
    .. [1] Happel, J., & Brenner, H. (1983). Low Reynolds number hydrodynamics: with special
           applications to particulate media (Vol. 1). Springer Science & Business Media.
    .. [2] Wakiya, S. (1967). Slow motions of a viscous fluid around two spheres. Journal of the
           Physical Society of Japan, 22(4), 1101-1109.
    .. [3] Goldman, A. J., Cox, R. G., & Brenner, H. (1966). The slow motion of two identical
           arbitrarily oriented spheres through a viscous fluid. Chemical Engineering Science,
           21(12), 1151-1170.
    """
    r_over_d = radius / distance

    if allow_rotation:
        # Solution for when spheres allowed to rotate. Eqn 5.8 in [3]. Originally from [2].
        factors = [1, -3 / 4, 9 / 16, -59 / 64, 273 / 256, -1107 / 1024, 1 / (1 + r_over_d)]
    else:
        # Solution for when spheres are prevented to rotate. Eqn 3.59 in [3]. Originally from [1].
        factors = [1, -3 / 4, 9 / 16, -59 / 64, 465 / 256, -15813 / 7168, 2 / (1 + r_over_d)]

    powers = np.arange(len(factors))

    return np.sum(factors * r_over_d**powers)
