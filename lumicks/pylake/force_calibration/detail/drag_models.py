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
