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
