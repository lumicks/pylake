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
