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
        **force_model_vars,
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
        **distance_model_vars,
        jacobian=offset_model_jac,
        eqn=offset_equation,
        eqn_tex=offset_equation_tex,
        derivative=offset_model_derivative,
        d_offset=Parameter(value=0.01, lower_bound=-0.1, upper_bound=0.1, unit="micron"),
    )


def marko_siggia_ewlc_force(name):
    """Marko Siggia's Worm-like Chain model with force as dependent parameter.

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near `F = 0.1 pN` [2]_.

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
        **force_model_vars,
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
    """Marko Siggia's Worm-like Chain model with distance as dependent parameter

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near `F = 0.1 pN` [2]_.

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
        **distance_model_vars,
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
    """Marko Siggia's Worm-like Chain model.

    This model [1]_ is based on only entropic contributions (valid for F << 10 pN). This model has
    force as a dependent variable.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
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
        **force_model_vars,
        jacobian=marko_siggia_simplified_jac,
        derivative=marko_siggia_simplified_derivative,
        eqn=marko_siggia_simplified_equation,
        eqn_tex=marko_siggia_simplified_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def inverted_marko_siggia_simplified(name):
    """Marko Siggia's Worm-like Chain model.

    This model is based on only entropic contributions [1]_ (valid for F << 10 pN). This model has
    distance as a dependent variable.

    Parameters
    ----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
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
        **distance_model_vars,
        jacobian=inverted_marko_siggia_simplified_jac,
        derivative=inverted_marko_siggia_simplified_derivative,
        eqn=inverted_marko_siggia_simplified_equation,
        eqn_tex=inverted_marko_siggia_simplified_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
    )


def odijk(name):
    """Odijk's Extensible Worm-Like Chain model with distance as dependent variable

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
        **distance_model_vars,
        jacobian=WLC_jac,
        derivative=WLC_derivative,
        eqn=WLC_equation,
        eqn_tex=WLC_equation_tex,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def dsdna_odijk(name, dna_length_kbp, um_per_kbp=0.34, temperature=24.53608821):
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

    model = odijk(name)
    model.defaults[f"{name}/Lc"].value = dna_length_kbp * um_per_kbp
    model.defaults[f"{name}/Lp"].value = 50.0  # [3]
    model.defaults[f"{name}/St"].value = 1200.0  # [4, 5]
    model.defaults["kT"].value = (
        1e21 * constants.k * constants.convert_temperature(temperature, "C", "K")
    )
    return model


def inverted_odijk(name):
    """Odijk's Extensible Worm-Like Chain model with force as dependent variable

    Odijk's Extensible Worm-Like Chain model [1]_ is useful for 10 pN < F < 30 pN [2]_. Note that
    this implementation was analytically solved and is significantly faster than fitting the
    model obtained with `lk.odijk("name").invert()`.

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
        **force_model_vars,
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
        **distance_model_vars,
        jacobian=FJC_jac,
        eqn=FJC_equation,
        eqn_tex=FJC_equation_tex,
        derivative=FJC_derivative,
        kT=Defaults.kT,
        Lp=Defaults.Lp,
        Lc=Defaults.Lc,
        St=Defaults.St,
    )


def ssdna_fjc(name, dna_length_kb, um_per_kb=0.56, temperature=24.53608821):
    """Model of ssDNA with distance as the dependent parameter.

    Freely-Jointed Chain model [1]_ [2]_ with distance as dependent parameter, using
    user-specified kilobases with default parameters obtained from [3]_.

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

    model = freely_jointed_chain(name)
    model.defaults[f"{name}/Lc"].value = dna_length_kb * um_per_kb
    model.defaults[f"{name}/Lp"].value = 0.70  # [3]
    model.defaults[f"{name}/St"].value = 750.0  # [3]
    model.defaults["kT"].value = (
        1e21 * constants.k * constants.convert_temperature(temperature, "C", "K")
    )

    return model


def inverted_freely_jointed_chain(name):
    """Freely-Jointed Chain model with distance as the dependent parameter.

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

    return InverseModel(freely_jointed_chain(name))


def twistable_wlc(name):
    """Twistable Worm-like Chain model with distance as dependent variable.

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
        **distance_model_vars,
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
        invtWLC,
        invtWLC_jac,
        invtWLC_equation,
        invtWLC_equation_tex,
        Defaults,
    )

    return Model(
        name,
        invtWLC,
        **force_model_vars,
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
