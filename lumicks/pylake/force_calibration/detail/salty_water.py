"""Helper functions for the salty water model from [1]_.

References
----------
.. [1] Kestin, J., Khalifa, H. E., & Correia, R. J. (1981). Tables of the dynamic and
   kinematic viscosity of aqueous NaCl solutions in the temperature range 20–150 C and
   the pressure range 0.1–35 MPa. Journal of physical and chemical reference data, 10(1),
   71-88.
"""

import numpy as np


def _poly(variable, powers, coefficients):
    variable_tilde = np.tile(variable, (len(powers), 1)).T
    return np.sum(
        np.asarray(coefficients) * (variable_tilde ** np.asarray(powers)), axis=1
    ).squeeze()


def zero_pressure_viscosity(temperature, molality):
    """Zero-pressure viscosity (equation 2).

    Parameters
    ----------
    temperature : array_like
        Temperature [C].
    molality : float
        Molality of NaCl [mol/kg].
    """
    uw_20 = 1002.0  # Viscosity of water at 20C [Pa s]

    def mu_w(t):
        # Equation 3 from the paper (viscosity of water)
        polynomial = _poly(20 - t, np.arange(1, 5), [1.2378, -1.303e-3, 3.06e-6, 2.55e-8])
        return uw_20 * 10 ** (polynomial / (96 + t))

    an = _poly(molality, np.arange(1, 4), [3.324e-2, 3.624e-3, -1.879e-4])  # Eqn 4
    bn = _poly(molality, np.arange(1, 4), [-3.96e-2, 1.02e-2, -7.02e-4])  # Eqn 5

    viscosity_water = mu_w(temperature)
    return viscosity_water * 10 ** (an + bn * np.log10(viscosity_water / uw_20))


def pressure_factor(temperature, molality):
    """Pressure factor

    Parameters
    ----------
    temperature : array_like
        Temperature [C].
    molality : float
        Molality of NaCl [mol/kg].
    """

    def beta_w(t):
        return _poly(t, np.arange(0, 5), [-1.297, 5.74e-2, -6.97e-4, 4.47e-6, -1.05e-8])

    def beta_s_e(t):
        """Excess pressure coefficient at saturation (Eqn. 8)"""
        gamma0 = 0.545
        gamma1 = 2.8e-3
        return gamma0 + gamma1 * t - beta_w(t)

    def ms(t):
        """Concentration at NaCl saturation (Eqn 9)"""
        return _poly(t, np.arange(0, 3), [6.044, 2.8e-3, 3.6e-5])

    def beta_star(m, t):
        """Reduced excess pressure coefficient (Eqn 10)"""
        return _poly(m / ms(t), np.arange(1, 4), [2.5, -2.0, 0.5])

    return beta_s_e(temperature) * beta_star(molality, temperature) + beta_w(temperature)


def _check_salt_model_validity(which, temperature, pressure, molality):
    if not np.all(np.logical_and(temperature >= 20, temperature < 150)):
        raise ValueError(
            f"{which} function is only valid for 20°C <= T < 150°C, you provided {temperature}"
        )

    if not np.all(pressure <= 35):
        raise ValueError(f"{which} function is only valid for p <= 35 MPa, you provided {pressure}")

    if not np.all(molality <= 6.0):
        raise ValueError(
            f"{which} function is only valid for m <= 6.0 mol/kg NaCl, you provided {molality}"
        )


def _density_of_salt_solution(temperature, molality, pressure):
    """Determine the density of water with NaCl.

    This model is based on [1]_.

    Parameters
    ----------
    temperature : array_like
        Temperature in C
    molality : float
        Molality NaCl [mol/kg]
    pressure : float, optional
        Pressure (default: 0.101325) [MPa].

    Raises
    ------
    ValueError
        When the provided values are outside the valid range of this model. The valid ranges are:
        Temperature (20°C <= T < 150°C), pressure <= 35 MPa and molality <= 6 mol/kg.

    References
    ----------
    .. [1] Kestin, J., Khalifa, H. E., & Correia, R. J. (1981). Tables of the dynamic and
       kinematic viscosity of aqueous NaCl solutions in the temperature range 20–150 C and
       the pressure range 0.1–35 MPa. Journal of physical and chemical reference data, 10(1),
       71-88.
    """
    import scipy.constants

    _check_salt_model_validity("Density", temperature, pressure, molality)

    temperature_k = scipy.constants.convert_temperature(temperature, "C", "K")
    mass_nacl = molality * 58.4428 / 1000  # kg NaCl
    mass_fraction = mass_nacl / (1 + mass_nacl)  # mass solute / mass total

    powers_ab = np.arange(-2, 3, 1)
    powers_other = np.arange(0, 3, 1)

    # All polynomials only depend on temperature
    def poly(powers, coefficients):
        return _poly(temperature_k, powers, coefficients)

    # See page 73 of [1].
    a_coeff = [1.006741e2, -1.127522, 5.916365e-3, -1.035794e-5, 9.270048e-9]
    b_coeff = [1.042948, -1.1933677e-2, 5.307535e-5, -1.0688768e-7, 8.492739e-11]

    c_coeff = [1.23268e-9, -6.861928e-12, 0]
    d_coeff = [-2.5166e-3, 1.11766e-5, -1.70552e-8]
    e_coeff = [2.84851e-3, -1.54305e-5, 2.23982e-8]
    f_coeff = [-1.5106e-5, 8.4605e-8, -1.2715e-10]
    g_coeff = [2.7676e-5, -1.5694e-7, 2.3102e-10]
    h_coeff = [6.4633e-8, -4.1671e-10, 6.8599e-13]

    terms = [
        poly(powers_ab, a_coeff),
        -poly(powers_ab, b_coeff) * pressure,
        -poly(powers_other, c_coeff) * pressure**2,
        mass_fraction * poly(powers_other, d_coeff),
        (mass_fraction**2) * poly(powers_other, e_coeff),
        -mass_fraction * poly(powers_other, f_coeff) * pressure,
        -(mass_fraction**2) * poly(powers_other, g_coeff) * pressure,
        -0.5 * poly(powers_other, h_coeff) * pressure**2,
    ]

    return 1.0 / np.sum(terms, axis=0)


def molarity_to_molality(molarity, temperature, pressure, molecular_weight):
    """Convert molarity (mol solute/L solution) to molality (mol solute/kg solvent)

    Parameters
    ----------
    molarity : float
        Molarity of the solution [mol solute/L solution].
    temperature : float
        Temperature of the solution in Celsius.
    pressure : float
        Pressure of the solution [MPa].
    molecular_weight : float
        Molecular weight of the solute [g/mol].

    Raises
    ------
    ValueError
        When the provided values are outside the valid range of this model. The valid ranges are:
        Temperature (20°C <= T < 150°C), pressure <= 35 MPa and molality <= 6 mol/kg.
    """

    from scipy.optimize import brentq

    def implicit_equation(molality):
        mol_solute = molarity
        volume_solution = 1.0e-3  # 1 L in m^3
        mass_solution = _density_of_salt_solution(temperature, molality, pressure) * volume_solution
        mass_salt = molecular_weight * mol_solute * 1e-3
        return mol_solute / (mass_solution - mass_salt) - molality

    # For the root to lie in the interval, the rhs must change sign.
    if np.sign(implicit_equation(0.0)) * np.sign(implicit_equation(6.0)) > 0:
        raise ValueError(
            f"Cannot convert molarity to molality because the requested molarity {molarity} is "
            f"outside the valid range of the solution density model ("
            f"{molality_to_molarity(0.0, temperature, pressure, molecular_weight=58.4428)}, "
            f"{molality_to_molarity(6.0, temperature, pressure, molecular_weight=58.4428)})."
        )

    # Bounds are based on the valid bounds for the density model
    return brentq(implicit_equation, 0.0, 6.0)


def molality_to_molarity(molality, temperature, pressure, molecular_weight):
    """Convert molality (mol solute/kg solvent) to molarity (mol solute/L solution)

    Parameters
    ----------
    molality : float
        Molality of the solution [mol solute/kg solvent].
    temperature : float
        Temperature of the solution in Celsius.
    pressure : float
        Pressure of the solution [MPa].
    molecular_weight : float
        Molecular weight of the solute [g/mol].
    """
    mol_solute = molality
    mass_salt = molecular_weight * mol_solute * 1e-3
    mass_solution = 1.0 + mass_salt  # solvent (1.0) + mass of the salt
    volume_solution = (
        1e3 * mass_solution / _density_of_salt_solution(temperature, molality, pressure)
    )

    return mol_solute / volume_solution
