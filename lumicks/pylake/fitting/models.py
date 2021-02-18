from .parameters import Parameter


def force_offset(name):
    """Offset on the the model output.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        force_offset_model,
        offset_model_jac,
        offset_model_derivative,
        offset_equation,
        offset_equation_tex,
    )

    return Model(
        name,
        force_offset_model,
        dependent="f",
        jacobian=offset_model_jac,
        eqn=offset_equation,
        eqn_tex=offset_equation_tex,
        derivative=offset_model_derivative,
        f_offset=Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="pN"),
    )


def distance_offset(name):
    """Offset on the the model output.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        distance_offset_model,
        offset_model_jac,
        offset_model_derivative,
        offset_equation,
        offset_equation_tex,
    )

    return Model(
        name,
        distance_offset_model,
        dependent="d",
        jacobian=offset_model_jac,
        eqn=offset_equation,
        eqn_tex=offset_equation_tex,
        derivative=offset_model_derivative,
        d_offset=Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="micron"),
    )


def marko_siggia_ewlc_force(name):
    """Marko Siggia's Worm-like Chain model with force as dependent parameter.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        marko_siggia_ewlc_solve_force,
        marko_siggia_ewlc_solve_force_jac,
        marko_siggia_ewlc_solve_force_derivative,
        marko_siggia_ewlc_solve_force_equation,
        marko_siggia_ewlc_solve_force_equation_tex,
        Defaults,
    )

    return Model(
        name,
        marko_siggia_ewlc_solve_force,
        dependent="f",
        jacobian=marko_siggia_ewlc_solve_force_jac,
        derivative=marko_siggia_ewlc_solve_force_derivative,
        eqn=marko_siggia_ewlc_solve_force_equation,
        eqn_tex=marko_siggia_ewlc_solve_force_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def marko_siggia_ewlc_distance(name):
    """Marko Siggia's Worm-like Chain model with distance as dependent parameter.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        marko_siggia_ewlc_solve_distance,
        marko_siggia_ewlc_solve_distance_jac,
        marko_siggia_ewlc_solve_distance_derivative,
        marko_siggia_ewlc_solve_distance_equation,
        marko_siggia_ewlc_solve_distance_equation_tex,
        Defaults,
    )

    return Model(
        name,
        marko_siggia_ewlc_solve_distance,
        dependent="d",
        jacobian=marko_siggia_ewlc_solve_distance_jac,
        derivative=marko_siggia_ewlc_solve_distance_derivative,
        eqn=marko_siggia_ewlc_solve_distance_equation,
        eqn_tex=marko_siggia_ewlc_solve_distance_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def marko_siggia_simplified(name):
    """Marko Siggia's Worm-like Chain model based on only entropic contributions (valid for F << 10 pN). This model
    has force as a dependent variable.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        marko_siggia_simplified,
        marko_siggia_simplified_jac,
        marko_siggia_simplified_derivative,
        marko_siggia_simplified_equation,
        marko_siggia_simplified_equation_tex,
        Defaults,
    )

    return Model(
        name,
        marko_siggia_simplified,
        dependent="f",
        jacobian=marko_siggia_simplified_jac,
        derivative=marko_siggia_simplified_derivative,
        eqn=marko_siggia_simplified_equation,
        eqn_tex=marko_siggia_simplified_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def inverted_marko_siggia_simplified(name):
    """Marko Siggia's Worm-like Chain model based on only entropic contributions (valid for F << 10 pN). This model
    has distance as a dependent variable.

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        inverted_marko_siggia_simplified,
        inverted_marko_siggia_simplified_jac,
        inverted_marko_siggia_simplified_derivative,
        inverted_marko_siggia_simplified_equation,
        inverted_marko_siggia_simplified_equation_tex,
        Defaults,
    )

    return Model(
        name,
        inverted_marko_siggia_simplified,
        dependent="d",
        jacobian=inverted_marko_siggia_simplified_jac,
        derivative=inverted_marko_siggia_simplified_derivative,
        eqn=inverted_marko_siggia_simplified_equation,
        eqn_tex=inverted_marko_siggia_simplified_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def odijk(name):
    """Odijk's Extensible Worm-Like Chain model with distance as dependent variable (useful for 10 pN < F < 30 pN).

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        WLC,
        WLC_jac,
        WLC_derivative,
        WLC_equation,
        WLC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        WLC,
        dependent="d",
        jacobian=WLC_jac,
        derivative=WLC_derivative,
        eqn=WLC_equation,
        eqn_tex=WLC_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def inverted_odijk(name):
    """Odijk's Extensible Worm-Like Chain model with force as dependent variable (useful for 10 pN < F < 30 pN).

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        invWLC,
        invWLC_jac,
        invWLC_derivative,
        invWLC_equation,
        invWLC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        invWLC,
        dependent="f",
        jacobian=invWLC_jac,
        derivative=invWLC_derivative,
        eqn=invWLC_equation,
        eqn_tex=invWLC_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def freely_jointed_chain(name):
    """Freely-Jointed Chain with distance as dependent parameter.

    References:
        1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
           Elastic Response of Individual Double-Stranded and Single-Stranded
           DNA Molecules, Science 271, 795-799 (1996).
        2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
           DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        FJC,
        FJC_jac,
        FJC_derivative,
        FJC_equation,
        FJC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        FJC,
        dependent="d",
        jacobian=FJC_jac,
        eqn=FJC_equation,
        eqn_tex=FJC_equation_tex,
        derivative=FJC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def inverted_freely_jointed_chain(name):
    """Inverted Freely-Jointed Chain with force as dependent parameter.

    References:
        1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
           Elastic Response of Individual Double-Stranded and Single-Stranded
           DNA Molecules, Science 271, 795-799 (1996).
        2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
           DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import InverseModel

    return InverseModel(freely_jointed_chain(name))


def twistable_wlc(name):
    """Twistable Worm-like Chain model. With distance as dependent variable.

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).
       2. Broekmans, Onno D., et al. DNA twist stability changes with
          magnesium (2+) concentration, Physical review letters 116.25,
          258102 (2016).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        tWLC,
        tWLC_jac,
        tWLC_derivative,
        tWLC_equation,
        tWLC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        tWLC,
        dependent="d",
        jacobian=tWLC_jac,
        derivative=tWLC_derivative,
        eqn=tWLC_equation,
        eqn_tex=tWLC_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lower_bound=0.0, upper_bound=50.0, unit="pN"),
        C=Parameter(value=440.0, lower_bound=0.0, upper_bound=5000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lower_bound=-5000.0, upper_bound=0.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lower_bound=-100.0, upper_bound=1000.0, unit="nm"),
    )


def inverted_twistable_wlc(name):
    """Twistable Worm-like Chain model. With force as dependent variable. This model uses a more performant implementation
    for inverting the model. It inverts the model by interpolating the forward curve and using this interpolant to
    invert the function.

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).
       2. Broekmans, Onno D., et al. DNA twist stability changes with
          magnesium (2+) concentration, Physical review letters 116.25,
          258102 (2016).

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        invtWLC,
        invtWLC_jac,
        invtWLC_equation,
        invtWLC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        invtWLC,
        dependent="f",
        jacobian=invtWLC_jac,
        eqn=invtWLC_equation,
        eqn_tex=invtWLC_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lower_bound=0.0, upper_bound=50.0, unit="pN"),
        C=Parameter(value=440.0, lower_bound=0.0, upper_bound=5000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lower_bound=-5000.0, upper_bound=0.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lower_bound=0.0, upper_bound=1000.0, unit="nm"),
    )
