__all__ = [
    "force_offset",
    "distance_offset",
    "ewlc_marko_siggia_force",
    "ewlc_marko_siggia_distance",
    "wlc_marko_siggia_force",
    "wlc_marko_siggia_distance",
    "ewlc_odijk_distance",
    "dsdna_ewlc_odijk_distance",
    "ewlc_odijk_force",
    "efjc_distance",
    "ssdna_efjc_distance",
    "efjc_force",
    "twlc_distance",
    "twlc_force",
    "marko_siggia_ewlc_force",
    "marko_siggia_ewlc_distance",
    "marko_siggia_simplified",
    "inverted_marko_siggia_simplified",
    "odijk",
    "dsdna_odijk",
    "freely_jointed_chain",
    "ssdna_fjc",
    "inverted_freely_jointed_chain",
    "inverted_odijk",
    "twistable_wlc",
    "inverted_twistable_wlc",
]

from deprecated import deprecated

from .parameters import Parameter

force_model_vars = {
    "dependent": "f",
    "dependent_unit": "pN",
    "independent": "d",
    "independent_unit": "micron",
}
distance_model_vars = {
    "independent": "f",
    "independent_unit": "pN",
    "dependent": "d",
    "dependent_unit": "micron",
}


def force_offset(name):
    """Offset on the model output.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        offset_equation,
        offset_model_jac,
        force_offset_model,
        offset_equation_tex,
        offset_model_derivative,
    )

    return Model(
        name,
        force_offset_model,
        **force_model_vars,
        jacobian=offset_model_jac,
        eqn=offset_equation,
        eqn_tex=offset_equation_tex,
        derivative=offset_model_derivative,
        f_offset=Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="pN"),
    )


def distance_offset(name):
    """Offset on the model output.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    """
    from .model import Model
    from .detail.model_implementation import (
        offset_equation,
        offset_model_jac,
        offset_equation_tex,
        distance_offset_model,
        offset_model_derivative,
    )

    return Model(
        name,
        distance_offset_model,
        **distance_model_vars,
        jacobian=offset_model_jac,
        eqn=offset_equation,
        eqn_tex=offset_equation_tex,
        derivative=offset_model_derivative,
        d_offset=Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="micron"),
    )


def ewlc_marko_siggia_force(name):
    """Marko Siggia's Worm-like Chain model with force as the dependent variable.

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near F = 0.1 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] Wang, M. D., Yin, H., Landick, R., Gelles, J., & Block, S. M. (1997). Stretching DNA
           with optical tweezers. Biophysical journal, 72(3), 1335-1346.
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        ewlc_marko_siggia_force,
        ewlc_marko_siggia_force_jac,
        ewlc_marko_siggia_force_equation,
        ewlc_marko_siggia_force_derivative,
        ewlc_marko_siggia_force_equation_tex,
    )

    return Model(
        name,
        ewlc_marko_siggia_force,
        **force_model_vars,
        jacobian=ewlc_marko_siggia_force_jac,
        derivative=ewlc_marko_siggia_force_derivative,
        eqn=ewlc_marko_siggia_force_equation,
        eqn_tex=ewlc_marko_siggia_force_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def ewlc_marko_siggia_distance(name):
    """Marko Siggia's Worm-like Chain model with distance as the dependent variable

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near F = 0.1 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] Wang, M. D., Yin, H., Landick, R., Gelles, J., & Block, S. M. (1997). Stretching DNA
           with optical tweezers. Biophysical journal, 72(3), 1335-1346.
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        ewlc_marko_siggia_distance,
        ewlc_marko_siggia_distance_jac,
        ewlc_marko_siggia_distance_equation,
        ewlc_marko_siggia_distance_derivative,
        ewlc_marko_siggia_distance_equation_tex,
    )

    return Model(
        name,
        ewlc_marko_siggia_distance,
        **distance_model_vars,
        jacobian=ewlc_marko_siggia_distance_jac,
        derivative=ewlc_marko_siggia_distance_derivative,
        eqn=ewlc_marko_siggia_distance_equation,
        eqn_tex=ewlc_marko_siggia_distance_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def wlc_marko_siggia_force(name):
    r"""Marko Siggia's Worm-like Chain (WLC) model.

    This model [1]_ is based on only entropic contributions. This model is valid at low forces [2]_:

    .. math::

        F << \frac{1}{4} \left(k_B T S_t^2 / L_p\right)^\frac{1}{3}

    Where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature, :math:`S_t` is the
    stretch modulus and :math:`L_p` is the persistence length. At higher forces an extensible WLC
    model (which takes into account enthalpic stretching) should be used.

    This model has force as the dependent variable. Differs from exact WLC solution by up to -10%
    near F=0.1 pN. Approaches exact WLC solution at lower and higher forces [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        wlc_marko_siggia_force,
        wlc_marko_siggia_force_jac,
        wlc_marko_siggia_force_equation,
        wlc_marko_siggia_force_derivative,
        wlc_marko_siggia_force_equation_tex,
    )

    return Model(
        name,
        wlc_marko_siggia_force,
        **force_model_vars,
        jacobian=wlc_marko_siggia_force_jac,
        derivative=wlc_marko_siggia_force_derivative,
        eqn=wlc_marko_siggia_force_equation,
        eqn_tex=wlc_marko_siggia_force_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def wlc_marko_siggia_distance(name):
    r"""Marko Siggia's Worm-like Chain (WLC) model.

    This model [1]_ is based on only entropic contributions. This model is valid at low forces [2]_:

    .. math::

        F << \frac{1}{4} \left(k_B T S_t^2 / L_p\right)^\frac{1}{3}

    Where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature, :math:`S_t` is the
    stretch modulus and :math:`L_p` is the persistence length. At higher forces an extensible WLC
    model (which takes into account enthalpic stretching) should be used.

    This model has distance as the dependent variable. Differs from exact WLC solution by up to -10%
    near F=0.1 pN. Approaches exact WLC solution at lower and higher forces [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        wlc_marko_siggia_distance,
        wlc_marko_siggia_distance_jac,
        wlc_marko_siggia_distance_equation,
        wlc_marko_siggia_distance_derivative,
        wlc_marko_siggia_distance_equation_tex,
    )

    return Model(
        name,
        wlc_marko_siggia_distance,
        **distance_model_vars,
        jacobian=wlc_marko_siggia_distance_jac,
        derivative=wlc_marko_siggia_distance_derivative,
        eqn=wlc_marko_siggia_distance_equation,
        eqn_tex=wlc_marko_siggia_distance_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def ewlc_odijk_distance(name):
    """Odijk's Extensible Worm-Like Chain model with distance as the dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ is useful for 10 pN < F < 30 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        ewlc_odijk_distance,
        ewlc_odijk_distance_jac,
        ewlc_odijk_distance_equation,
        ewlc_odijk_distance_derivative,
        ewlc_odijk_distance_equation_tex,
    )

    return Model(
        name,
        ewlc_odijk_distance,
        **distance_model_vars,
        jacobian=ewlc_odijk_distance_jac,
        derivative=ewlc_odijk_distance_derivative,
        eqn=ewlc_odijk_distance_equation,
        eqn_tex=ewlc_odijk_distance_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def dsdna_ewlc_odijk_distance(name, dna_length_kbp, um_per_kbp=0.34, temperature=24.53608821):
    """Model for dsDNA with distance as the dependent variable.

    Odijk's Extensible Worm-Like Chain model [1]_ [2]_ with distance as the dependent
    variable using user-specified kilobase-pairs (useful for 10 pN < F < 30 pN). Default model
    parameters were obtained from [3]_ [4]_ and [5]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    dna_length_kbp: integer
        The length of the dna in tether/construct measured in kilobase-pairs
    um_per_kbp: float
        The number of kilobase pairs evaluating to 1 um. This is used to convert the length in kbp
        to um as applied in the fit function [6]_.
    temperature: float
        The temperature in celsius. This is used to calculate the boltzmann * temperature (kT) value

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    .. [3] Manning, G. S. (2006). The persistence length of DNA is reached from the persistence
           length of its null isomer through an internal electrostatic stretching force.
           Biophysical journal, 91(10), 3607-3616.
    .. [4] Liu, J. H., Xi, K., Zhang, X., Bao, L., Zhang, X., & Tan, Z. J. (2019). Structural
           flexibility of DNA-RNA hybrid duplex: stretching and twist-stretch coupling. Biophysical
           journal, 117(1), 74-86.
    .. [5] Herrero-Galán, E., Fuentes-Perez, M. E., Carrasco, C., Valpuesta, J. M.,
           Carrascosa, J. L., Moreno-Herrero, F., & Arias-Gonzalez, J. R. (2013). Mechanical
           identities of RNA and DNA double helices unveiled at the single-molecule level. Journal
           of the American Chemical Society, 135(1), 122-131.
    .. [6] Saenger, W. (1984). Principles of Nucleic Acid Structure. Springer. New York, Berlin,
           Heidelberg.
    """
    from scipy import constants

    model = ewlc_odijk_distance(name)
    model.defaults[f"{name}/Lc"].value = dna_length_kbp * um_per_kbp
    model.defaults[f"{name}/Lp"].value = 50.0  # [3]
    model.defaults[f"{name}/St"].value = 1200.0  # [4, 5]
    model.defaults["kT"].value = (
        1e21 * constants.k * constants.convert_temperature(temperature, "C", "K")
    )
    return model


def ewlc_odijk_force(name):
    """Odijk's Extensible Worm-Like Chain model with force as the dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ is useful for 10 pN < F < 30 pN [2]_. Note that
    this implementation was analytically solved and is significantly faster than fitting the model
    obtained with `lumicks.pylake.ewlc_odijk_distance("name").invert()`.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        ewlc_odijk_force,
        ewlc_odijk_force_jac,
        ewlc_odijk_force_equation,
        ewlc_odijk_force_derivative,
        ewlc_odijk_force_equation_tex,
    )

    return Model(
        name,
        ewlc_odijk_force,
        **force_model_vars,
        jacobian=ewlc_odijk_force_jac,
        derivative=ewlc_odijk_force_derivative,
        eqn=ewlc_odijk_force_equation,
        eqn_tex=ewlc_odijk_force_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def efjc_distance(name):
    """Extensible Freely-Jointed Chain with distance as the dependent variable.

    Freely jointed chain model [1]_ [2]_. Useful for modelling single stranded DNA.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        efjc_distance,
        efjc_distance_jac,
        efjc_distance_equation,
        efjc_distance_derivative,
        efjc_distance_equation_tex,
    )

    return Model(
        name,
        efjc_distance,
        **distance_model_vars,
        jacobian=efjc_distance_jac,
        eqn=efjc_distance_equation,
        eqn_tex=efjc_distance_equation_tex,
        derivative=efjc_distance_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def ssdna_efjc_distance(name, dna_length_kb, um_per_kb=0.56, temperature=24.53608821):
    """Model of ssDNA with distance as the dependent variable.

    Extensible Freely-Jointed Chain model [1]_ [2]_ using user-specified kilobases with default
    parameters obtained from [3]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    dna_length_kb: integer
        The length of the dna in the sample measured in kilobases
    um_per_kb: float
        The number of kilobases evaluating to 1 um. This is used to convert the
        length in kb to um as applied in the fit function.
    temperature: float
        The temperature in celsius. This is used to calculate the
        Boltzmann's constant * temperature value

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    .. [3] Bosco, A., Camunas-Soler, J., & Ritort, F. (2014). Elastic properties and secondary
           structure formation of single-stranded DNA at monovalent and divalent salt conditions.
           Nucleic acids research, 42(3), 2064-2074.
    """
    from scipy import constants

    model = efjc_distance(name)
    model.defaults[f"{name}/Lc"].value = dna_length_kb * um_per_kb
    model.defaults[f"{name}/Lp"].value = 0.70  # [3]
    model.defaults[f"{name}/St"].value = 750.0  # [3]
    model.defaults["kT"].value = (
        1e21 * constants.k * constants.convert_temperature(temperature, "C", "K")
    )

    return model


def efjc_force(name):
    """Extensible Freely-Jointed Chain model with force as the dependent variable.

    The Freely-Jointed Chain model [1]_ [2]_ is useful for modelling ssDNA.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    from .model import InverseModel

    return InverseModel(efjc_distance(name))


def twlc_distance(name):
    """Twistable Worm-like Chain model with distance as the dependent variable.

    Twistable Worm-like Chain model [1]_ [2]_ that takes into account untwisting of the DNA at
    high forces. Note that it is generally recommended to fit this model with force as the
    dependent variable [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] P. Gross et al., Quantifying how DNA stretches, melts and changes twist under tension,
           Nature Physics 7, 731-736 (2011).
    .. [2] Broekmans, Onno D., et al. DNA twist stability changes with magnesium (2+) concentration,
           Physical review letters 116.25, 258102 (2016).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        twlc_distance,
        twlc_distance_jac,
        twlc_distance_equation,
        twlc_distance_derivative,
        twlc_distance_equation_tex,
    )

    return Model(
        name,
        twlc_distance,
        **distance_model_vars,
        jacobian=twlc_distance_jac,
        derivative=twlc_distance_derivative,
        eqn=twlc_distance_equation,
        eqn_tex=twlc_distance_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lower_bound=0.0, upper_bound=50.0, unit="pN"),
        C=Parameter(value=440.0, lower_bound=0.0, upper_bound=5000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lower_bound=-5000.0, upper_bound=0.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lower_bound=-100.0, upper_bound=1000.0, unit="nm"),
    )


def twlc_force(name):
    """Twistable Worm-like Chain model with force as the dependent variable.

    Twistable Worm-like Chain model [1]_ [2]_ that takes into account untwisting of the DNA at
    high forces. This model uses a more performant implementation for inverting the model. It
    inverts the model by interpolating the forward curve and using this interpolant to invert the
    function.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] P. Gross et al., Quantifying how DNA stretches, melts and changes twist under tension,
           Nature Physics 7, 731-736 (2011).
    .. [2] Broekmans, Onno D., et al. DNA twist stability changes with magnesium (2+) concentration,
           Physical review letters 116.25, 258102 (2016).
    """
    from .model import Model
    from .detail.model_implementation import (
        Defaults,
        twlc_solve_force,
        twlc_solve_force_jac,
        twlc_solve_force_equation,
        twlc_solve_force_equation_tex,
    )

    return Model(
        name,
        twlc_solve_force,
        **force_model_vars,
        jacobian=twlc_solve_force_jac,
        eqn=twlc_solve_force_equation,
        eqn_tex=twlc_solve_force_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
        Fc=Parameter(value=30.6, lower_bound=0.0, upper_bound=50.0, unit="pN"),
        C=Parameter(value=440.0, lower_bound=0.0, upper_bound=5000.0, unit="pN*nm**2"),
        g0=Parameter(value=-637, lower_bound=-5000.0, upper_bound=0.0, unit="pN*nm"),
        g1=Parameter(value=17.0, lower_bound=0.0, upper_bound=1000.0, unit="nm"),
    )


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `ewlc_marko_siggia_force('name')` "
        "instead."
    ),
    action="always",
    version="0.13.2",
)
def marko_siggia_ewlc_force(name):
    """Marko Siggia's Worm-like Chain model with force as the dependent variable.

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near F = 0.1 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] Wang, M. D., Yin, H., Landick, R., Gelles, J., & Block, S. M. (1997). Stretching DNA
           with optical tweezers. Biophysical journal, 72(3), 1335-1346.
    """
    return ewlc_marko_siggia_force(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use "
        "`ewlc_marko_siggia_distance('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def marko_siggia_ewlc_distance(name):
    """Marko Siggia's Worm-like Chain model with distance as the dependent variable

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near F = 0.1 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] Wang, M. D., Yin, H., Landick, R., Gelles, J., & Block, S. M. (1997). Stretching DNA
           with optical tweezers. Biophysical journal, 72(3), 1335-1346.
    """
    return ewlc_marko_siggia_distance(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `wlc_marko_siggia_force('name')` "
        "instead."
    ),
    action="always",
    version="0.13.2",
)
def marko_siggia_simplified(name):
    """Marko Siggia's Worm-like Chain model.

    This model [1]_ is based on only entropic contributions (valid for F << 10 pN). This model has
    force as the dependent variable.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    """
    return wlc_marko_siggia_force(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use "
        "`wlc_marko_siggia_distance('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def inverted_marko_siggia_simplified(name):
    """Marko Siggia's Worm-like Chain model.

    This model is based on only entropic contributions [1]_ (valid for F << 10 pN). This model has
    distance as the dependent variable.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    """
    return wlc_marko_siggia_distance(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `ewlc_odijk_distance('name')` "
        "instead."
    ),
    action="always",
    version="0.13.2",
)
def odijk(name):
    """Odijk's Extensible Worm-Like Chain model with distance as the dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ is useful for 10 pN < F < 30 pN [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    return ewlc_odijk_distance(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use "
        "`dsdna_ewlc_odijk_distance('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def dsdna_odijk(name, dna_length_kbp, um_per_kbp=0.34, temperature=24.53608821):
    """Model for dsDNA with distance as the dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ [2]_ using user-specified kilobase-pairs
    (useful for 10 pN < F < 30 pN). Default parameters were obtained from [3]_ [4]_ and [5]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    dna_length_kbp: integer
        The length of the dna in tether/construct measured in kilobase-pairs
    um_per_kbp: float
        The number of kilobase pairs evaluating to 1 um. This is used to convert the length in kbp
        to um as applied in the fit function [6]_.
    temperature: float
        The temperature in celsius. This is used to calculate the boltzmann * temperature (kT) value

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    .. [3] Manning, G. S. (2006). The persistence length of DNA is reached from the persistence
           length of its null isomer through an internal electrostatic stretching force.
           Biophysical journal, 91(10), 3607-3616.
    .. [4] Liu, J. H., Xi, K., Zhang, X., Bao, L., Zhang, X., & Tan, Z. J. (2019). Structural
           flexibility of DNA-RNA hybrid duplex: stretching and twist-stretch coupling. Biophysical
           journal, 117(1), 74-86.
    .. [5] Herrero-Galán, E., Fuentes-Perez, M. E., Carrasco, C., Valpuesta, J. M.,
           Carrascosa, J. L., Moreno-Herrero, F., & Arias-Gonzalez, J. R. (2013). Mechanical
           identities of RNA and DNA double helices unveiled at the single-molecule level. Journal
           of the American Chemical Society, 135(1), 122-131.
    .. [6] Saenger, W. (1984). Principles of Nucleic Acid Structure. Springer. New York, Berlin,
           Heidelberg.
    """
    return dsdna_ewlc_odijk_distance(
        name, dna_length_kbp, um_per_kbp=um_per_kbp, temperature=temperature
    )


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `ewlc_odijk_force('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def inverted_odijk(name):
    """Odijk's Extensible Worm-Like Chain model with force as the dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ is useful for 10 pN < F < 30 pN [2]_. Note that
    this implementation was analytically solved and is significantly faster than fitting the
    model obtained with `lk.ewlc_odijk_distance("name").invert()`.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules 28, 7016-7018 (1995).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    return ewlc_odijk_force(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `efjc_distance('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def freely_jointed_chain(name):
    """Freely-Jointed Chain with distance as the dependent variable.

    Freely jointed chain model [1]_ [2]_. Useful for modelling single stranded DNA.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    return efjc_distance(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `ssdna_efjc_distance('name')` "
        "instead."
    ),
    action="always",
    version="0.13.2",
)
def ssdna_fjc(name, dna_length_kb, um_per_kb=0.56, temperature=24.53608821):
    """Model of ssDNA with distance as the dependent variable.

    Freely-Jointed Chain model [1]_ [2]_ using user-specified kilobases with default parameters
    obtained from [3]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    dna_length_kb: integer
        The length of the dna in the sample measured in kilobases
    um_per_kb: float
        The number of kilobases evaluating to 1 um. This is used to convert the
        length in kb to um as applied in the fit function.
    temperature: float
        The temperature in celsius. This is used to calculate the
        Boltzmann's constant * temperature value

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    .. [3] Bosco, A., Camunas-Soler, J., & Ritort, F. (2014). Elastic properties and secondary
           structure formation of single-stranded DNA at monovalent and divalent salt conditions.
           Nucleic acids research, 42(3), 2064-2074.
    """
    return ssdna_efjc_distance(name, dna_length_kb, um_per_kb=um_per_kb, temperature=temperature)


@deprecated(
    reason=("This function will be removed in a future release. Use `efjc_force('name')` instead."),
    action="always",
    version="0.13.2",
)
def inverted_freely_jointed_chain(name):
    """Freely-Jointed Chain model with force as the dependent variable.

    The Freely-Jointed Chain model [1]_ [2]_ is useful for modelling ssDNA.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The Elastic Response of
           Individual Double-Stranded and Single-Stranded DNA Molecules, Science 271, 795-799
           (1996).
    .. [2] M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching DNA with optical
           tweezers., Biophysical journal 72, 1335-46 (1997).
    """
    return efjc_force(name)


@deprecated(
    reason=(
        "This function will be removed in a future release. Use `twlc_distance('name')` instead."
    ),
    action="always",
    version="0.13.2",
)
def twistable_wlc(name):
    """Twistable Worm-like Chain model with distance as the dependent variable.

    Twistable Worm-like Chain model [1]_ [2]_ that takes into account untwisting of the DNA at
    high forces. Note that it is generally recommended to fit this model with force as the
    dependent variable [2]_.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] P. Gross et al., Quantifying how DNA stretches, melts and changes twist under tension,
           Nature Physics 7, 731-736 (2011).
    .. [2] Broekmans, Onno D., et al. DNA twist stability changes with magnesium (2+) concentration,
           Physical review letters 116.25, 258102 (2016).
    """
    return twlc_distance(name)


@deprecated(
    reason=("This function will be removed in a future release. Use `twlc_force('name')` instead."),
    action="always",
    version="0.13.2",
)
def inverted_twistable_wlc(name):
    """Twistable Worm-like Chain model with force as the dependent variable.

    Twistable Worm-like Chain model [1]_ [2]_ that takes into account untwisting of the DNA at
    high forces. This model uses a more performant implementation for inverting the model. It
    inverts the model by interpolating the forward curve and using this interpolant to invert the
    function.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] P. Gross et al., Quantifying how DNA stretches, melts and changes twist under tension,
           Nature Physics 7, 731-736 (2011).
    .. [2] Broekmans, Onno D., et al. DNA twist stability changes with magnesium (2+) concentration,
           Physical review letters 116.25, 258102 (2016).
    """
    return twlc_force(name)
