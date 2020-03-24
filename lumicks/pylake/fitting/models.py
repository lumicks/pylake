import numpy as np
from .parameters import Parameter

"""
Available models:
    Models with distance as dependent parameter
    - WLC
        Odijk's Extensible Worm-Like Chain model with distance as dependent parameter (useful for 10 pN < F < 30 pN)
    - Marko_Siggia_eWLC_distance
        Margo Siggia's Worm-like Chain model with distance as dependent parameter (useful for F < 10 pN).
    - tWLC
        Twistable Worm-Like Chain model with distance as dependent parameter (useful for 10 pN < F)
    - FJC
        Freely Jointed Chain model with distance as dependent parameter

    Models with Force as dependent parameter
    - Marko_Siggia
        Simplified Marko Siggia's Worm-like Chain model with force as dependent parameter (useful for F << 10 pN).
    - Marko_Siggia_eWLC_force
        Marko Siggia's Worm-like Chain model with force as dependent parameter (useful for F < 10 pN).
    - invWLC
        Inverted Extensible Worm-Like Chain model with force as dependent parameter (useful for 10 pN < F < 30 pN)
    - invtWLC
        Inverted Twistable Worm-Like Chain model with force as dependent parameter (useful for 10 pN < F)
    - invFJC
        Inverted Freely Joint Chain model with force as dependent parameter
"""


def offset(name):
    """Offset on the the model output."""
    from .model import Model
    from .detail.model_implementation import offset_model, offset_model_jac, offset_model_derivative, Defaults

    return Model(
        name,
        offset_model,
        offset_model_jac,
        derivative=offset_model_derivative,
        offset=Parameter(value=0.01, lb=0, ub=np.inf),
    )


def marko_siggia_ewlc_force(name):
    """
    Marko Siggia's Worm-like Chain model with force as dependent parameter.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).
    """
    from .model import Model
    from .detail.model_implementation import (
        marko_sigga_ewlc_solve_force,
        marko_sigga_ewlc_solve_force_jac,
        marko_sigga_ewlc_solve_force_derivative,
        Defaults,
    )

    return Model(
        name,
        marko_sigga_ewlc_solve_force,
        marko_sigga_ewlc_solve_force_jac,
        derivative=marko_sigga_ewlc_solve_force_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def marko_siggia_ewlc_distance(name):
    """
    Marko Siggia's Worm-like Chain model with distance as dependent parameter.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).
    """
    from .model import Model
    from .detail.model_implementation import (
        marko_sigga_ewlc_solve_distance,
        marko_sigga_ewlc_solve_distance_jac,
        marko_sigga_ewlc_solve_distance_derivative,
        Defaults,
    )

    return Model(
        name,
        marko_sigga_ewlc_solve_distance,
        marko_sigga_ewlc_solve_distance_jac,
        derivative=marko_sigga_ewlc_solve_distance_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def marko_siggia_simplified(name):
    """
    Markov Siggia's Worm-like Chain model based on only entropic contributions (valid for F << 10 pN).

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).
    """
    from .model import Model
    from .detail.model_implementation import (
        Marko_Siggia,
        Marko_Siggia_jac,
        Marko_Siggia_derivative,
        Defaults,
    )

    return Model(
        name,
        Marko_Siggia,
        Marko_Siggia_jac,
        derivative=Marko_Siggia_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def odijk(name):
    """
    Odijk's Extensible Worm-Like Chain model with distance as dependent variable (useful for 10 pN < F < 30 pN).

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import WLC, WLC_jac, WLC_derivative, Defaults

    return Model(
        name,
        WLC,
        WLC_jac,
        derivative=WLC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def inverted_odijk(name):
    """
    Odijk's Extensible Worm-Like Chain model with force as dependent variable (useful for 10 pN < F < 30 pN).

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import invWLC, invWLC_jac, invWLC_derivative, Defaults

    return Model(
        name,
        invWLC,
        invWLC_jac,
        derivative=invWLC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def freely_jointed_chain(name):
    """
    Freely-Jointed Chain with distance as dependent parameter.

    References:
        1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
           Elastic Response of Individual Double-Stranded and Single-Stranded
           DNA Molecules, Science 271, 795-799 (1996).
        2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
           DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import FJC, FJC_jac, FJC_derivative, Defaults

    return Model(
        name,
        FJC,
        FJC_jac,
        derivative=FJC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def inverted_freely_jointed_chain(name):
    """
    Inverted Freely-Jointed Chain with force as dependent parameter.

    References:
        1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
           Elastic Response of Individual Double-Stranded and Single-Stranded
           DNA Molecules, Science 271, 795-799 (1996).
        2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
           DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import InverseModel
    return InverseModel(freely_jointed_chain(name))


def twistable_wlc(name):
    """
    Twistable Worm-like Chain model. With distance as dependent variable.

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).
       2. Broekmans, Onno D., et al. DNA twist stability changes with
          magnesium (2+) concentration, Physical review letters 116.25,
          258102 (2016).
    """
    from .model import Model
    from .detail.model_implementation import tWLC, tWLC_jac, tWLC_derivative, Defaults

    return Model(
        name,
        tWLC,
        tWLC_jac,
        derivative=tWLC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lb=0.0, ub=50000.0, unit="pN"),
        C=Parameter(value=440.0, lb=0.0, ub=50000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lb=-50000.0, ub=50000.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0, unit="nm"),
    )


def inverted_twistable_wlc(name):
    """
    Twistable Worm-like Chain model. With force as dependent variable.

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).
       2. Broekmans, Onno D., et al. DNA twist stability changes with
          magnesium (2+) concentration, Physical review letters 116.25,
          258102 (2016).
    """
    from .model import Model
    from .detail.model_implementation import invtWLC, invtWLC_jac, Defaults

    return Model(
        name,
        invtWLC,
        invtWLC_jac,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lb=0.0, ub=100.0, unit="pN"),
        C=Parameter(value=440.0, lb=0.0, ub=50000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lb=-50000.0, ub=50000.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0, unit="nm"),
    )
