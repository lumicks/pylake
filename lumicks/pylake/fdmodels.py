import numpy as np


def WLC(F, Lp, Lc, S, kT = 4.11):
    """
    Odijk's Worm-like Chain model

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    F : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    S : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (1.0 - 1.0/2.0*np.sqrt(kT/(F*Lp)) + F/S)


def tWLC(F, Lp, Lc, S, C, g0, g1, Fc, kT=4.11):
    """
    Twistable Worm-like Chain model

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).

    Parameters
    ----------
    F : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    S : float
        stretching modulus [pN]
    C : float
        twist rigidity [pN nm^2]
    g0 : float
        twist stretch coupling [pN Nm]
    g1 : float
        twist stretch coupling [nm]
    Fc : float
        critical force for twist stretch coupling [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    g = np.zeros(np.size(F))
    g[F < Fc] = g0 + g1 * Fc
    g[F >= Fc] = g0 + g1 * F(F >= Fc)

    return Lc * (1.0 - 1.0 / 2.0 * np.sqrt(kT / (F * Lp)) + (C / (-g ** 2.0 + S * C)) * F)


def FJC(F, Lp, Lc, S, kT=4.11):
    """
    Freely-Jointed Chain

    References:
       1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
          Elastic Response of Individual Double-Stranded and Single-Stranded
          DNA Molecules, Science 271, 795-799 (1996).
       2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
          DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    F : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    S : float
        elastic modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (np.coth(2.0*F * Lp / kT) - kT / (2.0 * F * Lp)) * (1.0 + F/S)


def invWLC(d, Lp, Lc, S, kT = 4.11):
    """
    Inverted Odijk's Worm-like Chain model

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    d : array_like
        extension [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    S : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return ((2.0 * (Lp * Lc * S * d - Lp * S * Lc ** 2.0)) / (3.0 * Lp * Lc ** 2.0) -
             (-16.0 * Lp ** 2.0 * S ** 2.0 * (d * d) * Lc ** 2.0 +
            32.0 * Lp ** 2.0 * S ** 2.0 * d * Lc ** 3.0 - 16.0 * Lp ** 2.0 * S ** 2.0 * Lc ** 4.0) /
            (24.0 * Lp * Lc ** 2.0 * (-8.0 * Lp ** 3.0 * S ** 3.0 * (d * d * d) * Lc ** 3.0 +
            24.0 * Lp ** 3.0 * S ** 3.0 * (d * d) * Lc ** 4.0 -
            24.0 * Lp ** 3.0 * S ** 3.0 * d * Lc ** 5.0 +
            27.0 * kT * Lp ** 2.0 * S ** 2.0 * Lc ** 6.0 + 8.0 * Lp ** 3.0 * S ** 3.0 * Lc ** 6.0 +
            3.0 * np.sqrt(3.0) *
            np.sqrt(-16.0 * kT * Lp ** 5.0 * S ** 5.0 * (d * d * d) * Lc ** 9.0 +
            48.0 * kT * Lp ** 5. * S ** 5. * (d * d) * Lc ** 10.0 -
            48.0 * kT * Lp ** 5. * S ** 5. * d * Lc ** 11.0 +
            27.0 * kT ** 2. * Lp ** 4. * S ** 4.0 * Lc ** 12.0 +
            16.0 * kT * Lp ** 5. * S ** 5. * Lc ** 12)) ** (1.0 / 3.0)) +
            (1.0 / (6. * Lp * Lc * Lc)) *
            (-8.0 * Lp ** 3. * S ** 3. * (d * d * d) * Lc * Lc * Lc +
            24.0 * Lp ** 3. * S ** 3. * (d * d) * Lc * Lc * Lc * Lc -
            24.0 * Lp ** 3. * S ** 3. * d * Lc * Lc * Lc * Lc * Lc +
            27.0 * kT * Lp ** 2. * S ** 2.0 * Lc * Lc * Lc * Lc * Lc * Lc +
            8.0 * Lp ** 3. * S ** 3. * Lc * Lc * Lc * Lc * Lc * Lc +
            3.0 * np.sqrt(3) *
            np.sqrt(-16. * kT * Lp ** 5. * S ** 5. * (d * d * d) * Lc ** 9.0 +
            48.0 * kT * Lp ** 5. * S ** 5. * (d * d) * Lc ** 10.0 -
            48.0 * kT * Lp ** 5. * S ** 5. * d * Lc ** 11.0 +
            27.0 * kT ** 2. * Lp ** 4. * S ** 4. * Lc ** 12.0 +
            16.0 * kT * Lp ** 5. * S ** 5. * Lc ** 12.0)) ** (1. / 3.0)
            )

