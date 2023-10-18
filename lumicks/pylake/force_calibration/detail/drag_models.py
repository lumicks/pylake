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
