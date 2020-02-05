from .model import Model
from .parameters import Parameter
from .detail.derivative_manipulation import invert_function, invert_jacobian
from .model import Model, InverseModel
import numpy as np

"""The model Jacobians and derivatives were determined via symbolic differentiation; followed by common subexpression
elimination, both using sympy. After this, the resulting code was checked for numerical issues, which were subsequently
removed. The code for most of the individual models was never intended to be human readable.

def generate_derivatives(parameters, expression):
    import sympy as sym
    from sympy.parsing import sympy_parser

    symbolic_parameters = [sym.Symbol(x) for x in parameters]
    symbolic_function = sympy_parser.parse_expr(expression)
    jacobian = [sym.diff(symbolic_function, x) for x in symbolic_parameters]
    cse = sym.cse(jacobian)
    print(cse[0])
    for x in cse[0]:
        print(str(x[0]) + ' = ' + str(x[1]))

    print("return ", cse[1])
    return jac
"""


def force_model(name, model_type):
    """Generate a force model.

    Parameters
    ----------
    name : str
        Name to identify the model by (e.g. "DNA"). This name gets prefixed to the non-shared parameters.
    model_type : str
        Specifies which model to return. Valid options are:
        - Marko_Siggia
            Margo Siggia's Worm-like Chain model with d as dependent parameter (useful for F < 10 pN).
        - WLC
            Odijk's Extensible Worm-Like Chain model with F as independent parameter (useful for 10 pN < F < 30 pN)
        - tWLC
            Twistable Worm-Like Chain model with F as independent parameter (useful for 10 pN < F)
        - FJC
            Freely Jointed Chain model with F as independent parameter
        - invWLC
            Inverted Extensible Worm-Like Chain model with d as independent parameter (useful for 10 pN < F < 30 pN)
        - invtWLC
            Inverted Twistable Worm-Like Chain model with d as independent parameter (useful for 10 pN < F)
        - invFJC
            Inverted Freely Joint Chain model with d as independent parameter
    """
    kT_default = Parameter(value=4.11, lb=0.0, ub=8.0, vary=False, shared=True)
    Lp_default = Parameter(value=40.0, lb=0.0, ub=np.inf)
    Lc_default = Parameter(value=16.0, lb=0.0, ub=np.inf)
    St_default = Parameter(value=1500.0, lb=0.0, ub=np.inf)
    if model_type == "offset":
        return Model(name, offset_model, offset_model_jac, derivative=offset_model_derivative,
                     offset=Parameter(value=0.01, lb=0, ub=np.inf))
    if model_type == "Marko_Siggia":
        return Model(name, Marko_Siggia, Marko_Siggia_jac, derivative=Marko_Siggia_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default)
    if model_type == "WLC":
        return Model(name, WLC, WLC_jac, derivative=WLC_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default)
    elif model_type == "tWLC":
        return Model(name, tWLC, tWLC_jac, derivative=tWLC_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default,
                     Fc=Parameter(value=30.6, lb=0.0, ub=50000.0),
                     C=Parameter(value=440.0, lb=0.0, ub=50000.0),
                     g0=Parameter(value=-637, lb=-50000.0, ub=50000.0),
                     g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0),
                     )
    elif model_type == "FJC":
        return Model(name, FJC, FJC_jac, derivative=FJC_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default)
    elif model_type == "invWLC":
        return Model(name, invWLC, invWLC_jac, derivative=invWLC_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default,
                     Fc=Parameter(value=30.6, lb=0.0, ub=100.0),
                     C=Parameter(value=440.0, lb=0.0, ub=50000.0),
                     g0=Parameter(value=-637, lb=-50000.0, ub=50000.0),
                     g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0))
    elif model_type == "invtWLC":
        return Model(name, invtWLC, invtWLC_jac,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default,
                     Fc=Parameter(value=30.6, lb=0.0, ub=100.0),
                     C=Parameter(value=440.0, lb=0.0, ub=50000.0),
                     g0=Parameter(value=-637, lb=-50000.0, ub=50000.0),
                     g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0),
                     )
    elif model_type == "invFJC":
        return InverseModel(force_model(name, "FJC"))
    else:
        raise ValueError(f"Invalid model {model_type} selected. Valid options are WLC, tWLC, FJC, invWLC, invtWLC, "
                         f"invFJC.")


def offset_model(x, offset):
    """Offset on the the model output."""
    return offset * np.ones(x.shape)


def offset_model_jac(x, offset):
    """Offset on the model output."""
    return np.ones((1, len(x)))


def offset_model_derivative(x, offset):
    """Offset on the model output."""
    return np.zeros(len(x))


def Marko_Siggia(d, Lp, Lc, kT):
    """
    Markov Siggia's Worm-like Chain model based on only entropic contributions. Valid for F < 10 pN).

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).
    """
    d_div_Lc = d / Lc
    return (kT/Lp) * .25 * (1.0-d_div_Lc)**(-2) + d_div_Lc - .25


def Marko_Siggia_jac(d, Lp, Lc, kT):
    d_div_Lc = d / Lc
    return np.vstack((-0.25 * kT / (Lp * Lp * (1.0 - d_div_Lc) ** 2),
                     -d_div_Lc / Lc - 0.5 * d * kT / (Lc ** 2 * Lp * (1.0 - d_div_Lc) ** 3),
                     0.25 / (Lp * (1.0 - d_div_Lc) ** 2)))


def Marko_Siggia_derivative(d, Lp, Lc, kT):
    return 1/Lc + 0.5*kT/(Lc*Lp*(1.0 - d/Lc)**3)


def WLC(F, Lp, Lc, St, kT = 4.11):
    """
    Odijk's Extensible Worm-like Chain model

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
    St : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (1.0 - 1.0/2.0*np.sqrt(kT/(F*Lp)) + F/St)


def WLC_jac(F, Lp, Lc, St, kT=4.11):
    sqrt_term = np.sqrt(kT / (F * Lp))
    return np.vstack((0.25 * Lc * sqrt_term / Lp,
                     F / St - 0.5 * sqrt_term + 1.0,
                     -F * Lc / (St * St),
                     -0.25 * Lc * sqrt_term / kT))


def tWLC(F, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
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
    St : float
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
    g[F >= Fc] = g0 + g1 * F[F >= Fc]

    return Lc * (1.0 - 1.0 / 2.0 * np.sqrt(kT / (F * Lp)) + (C / (-g * g + St * C)) * F)


def tWLC_jac(F, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    x0 = 1.0 / Lp
    x1 = np.sqrt(kT * x0 / F)
    x2 = 0.25 * Lc * x1
    x3 = C * F
    x4 = C * St
    x5 = F > Fc
    x6 = F <= Fc
    x7 = F * x5 + Fc * x6
    x8 = g0 + g1 * x7
    x9 = x8 * x8
    x10 = x4 - x9
    x11 = 1.0 / x10
    x12 = x10 ** (-2)
    x13 = 2.0 * Lc * x3
    x14 = 1.0 / x8
    x15 = x11 * x11

    return np.vstack((x0 * x2,
                      -0.5 * x1 + x11 * x3 + 1.0,
                      -C * C * F * Lc * x12,
                      Lc * (F * x11 - F * x12 * x4),
                      x15 * x13 * x8,
                      x15 * x13 * x14 * x7 * x9,
                      g1 * x15 * x13 * x14 * x9 * x6,
                      -x2 / kT))

    # Not continuous derivatives were removed from the 8th parameter derivative:
    # Original derivative was: g1 * x11 ** 2 * x13 * x14 * x9 * (F * Derivative(x5, Fc) + Fc * Derivative(x6, Fc) + x6)


def coth(x):
    return np.cosh(x)/np.sinh(x)


def FJC(F, Lp, Lc, St, kT=4.11):
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
    St : float
        elastic modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (coth(2.0 * F * Lp / kT) - kT / (2.0 * F * Lp)) * (1.0 + F/St)


def FJC_jac(F, Lp, Lc, St, kT=4.11):
    x0 = 0.5 / F
    x1 = 2.0 * F / kT
    x2 = Lp * x1
    x3 = np.sinh(x2) ** (-2)
    x4 = F / St + 1.0
    x5 = Lc * x4
    x6 = x0 / Lp
    x7 = -kT * x6 + coth(x2)
    return np.vstack((x5 * (-x1 * x3 + kT * x0 / (Lp * Lp)),
                      x4 * x7,
                      -F * Lc * x7 / (St*St),
                      x5 * (2.0 * F * Lp * x3 / (kT*kT) - x6)))


def invWLC(d, Lp, Lc, St, kT = 4.11):
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
    St : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    x0 = 2.0 * Lp * St
    x1 = Lc * Lc     # Lc ** 2
    x16 = x1 * Lc    # Lc ** 3
    x13 = Lc * x16   # Lc ** 4
    x10 = x1 * x13   # Lc ** 6
    x2 = 1.0 / (Lp * x1)
    lp2 = Lp * Lp
    lp3 = Lp * lp2
    x4 = Lp * St * kT
    x8 = d * d
    x7 = x8 * d
    sqrt_arg = x4 * (48.0 * x8 - 16.0 * x7 / Lc + 16.0 * x1 - Lc * d * 48.0) + 27.0 * kT * kT * x1
    sqrt_fun = np.zeros(sqrt_arg.shape)
    mask = sqrt_arg >= 0
    sqrt_fun[mask] = np.sqrt(sqrt_arg[mask])
    sqrt_fun[np.logical_not(mask)] = np.inf
    x9 = St**2 * lp2 * x16 * x1 * sqrt_fun
    x11 = lp3 * St * St * St
    x12 = 8.0 * x11
    x14 = 24.0 * x11
    x15 = lp2 * St * St
    x17 = -Lc * x13 * d * x14 + 27.0 * kT * x10 * x15 + x10 * x12 - x12 * x16 * x7 + x13 * x14 * x8
    x18 = 16.0 * x15

    return -(1.0/24.0) * x2 * (x17 + 3.0 * np.sqrt(3.0) * x9) ** (-1.0/3.0) * \
            (32.0 * d * x15 * x16 - x1 * x18 * x8 - x13 * x18) + \
            (1.0/6.0) * x2 * (x17 + 3.0 * np.sqrt(3.0) * x9) ** (1.0/3.0) + \
            (1.0/3.0) * x2 * (Lc * d * x0 - x0 * x1)


# HC SVNT DRACONES
def invWLC_jac(d, Lp, Lc, St, kT=4.11):
    x0 = Lc * St
    x1 = 2.0 * d
    x2 = x0 * x1
    lc2 = Lc * Lc
    lc3 = Lc * lc2  # Lc^3
    lc4 = lc2 * lc2  # Lc^4
    lc5 = lc2 * lc3   # Lc^5
    lc6 = lc3 * lc3  # Lc^6
    lc9 = lc6 * lc3  # Lc^9
    lc10 = Lc * lc9   # Lc^10
    lc11 = Lc * lc10   # Lc^11
    lc12 = lc6 * lc6  # Lc^12
    x4 = 2.0 * lc2
    x5 = St * x4
    x6 = 1.0 / Lp
    x7 = 1.0 / lc2
    x8 = (1.0/3.0) * x7
    x9 = x6 * x8
    lp2 = Lp * Lp
    lp3 = Lp * lp2   # Lp^3
    lp4 = lp2 * lp2  # Lp^4
    lp5 = Lp * lp4   # Lp^5
    x11 = 1.0 / lp2
    x12 = Lp * x2 - Lp * x5
    x13 = np.sqrt(3)
    st2 = St * St
    st3 = St * st2   # St ** 3
    st4 = st2 * st2  # St ** 4
    st5 = st2 * st3  # St ** 5
    lc12st5 = lc12 * st5
    lp5lc12st5 = lp5 * lc12st5
    x19 = 16.0 * kT
    lp5st5 = lp5 * st5
    lc11lp5st5 = lc11 * lp5st5
    ktlc11lp5st5 = kT * lc11lp5st5
    kt2 = kT * kT
    lc12st4 = lc12 * st4
    kt2lc12st4 = kt2 * lc12st4
    lp4_27 = 27.0 * lp4
    d2 = d * d
    d3 = d * d2  # d^3
    lp5st5lc9 = lp5st5 * lc9
    lp5st5lc10 = lp5st5 * lc10
    ktd2 = kT * d2
    lpst = Lp * St
    ktlc3lpst = kT * lc3 * lpst
    lplc4st = lc4 * lpst
    lpstlc = Lc * lpst
    kt2lc4 = kT * kT * lc4
    lpstlc2 = lc2 * lpst
    x37 = lc4 * st2 * lp2 * np.sqrt(-48.0 * d * ktlc3lpst + lplc4st * x19 - x19 * d3 * lpstlc + kt2lc4 * 27.0 + 48.0 * lpstlc2 * ktd2)
    x40 = lp3 * st3
    x42 = 8.0 * lc6
    x43 = d2 * x40
    x45 = 24.0 * lc4
    x47 = lp2 * st2
    x48 = lc6 * x47
    x50 = 8.0 * d3
    x51 = lc3 * x50
    x53 = x40 * lc5
    x54 = 24.0 * d
    x55 = 27.0 * kT * x48 + x40 * x42 - x40 * x51 + x43 * x45 - x53 * x54
    x56 = 3.0 * x13 * x37 + x55
    x57 = x56 ** (1.0/3.0)
    x58 = x11 * x7
    x59 = lc4 * st2
    x60 = 32.0 * Lp
    x61 = Lp * st2
    x62 = 64.0 * d * lc3
    x63 = lc2 * d2
    x64 = st2 * x63
    x65 = 3.0 * np.sqrt(3.0) * x37 + x55
    x66 = x65 ** (-(1.0/3.0))
    x67 = (1.0/24.0) * x66
    x68 = x6 * x7
    x69 = x67 * x68
    x70 = 16.0 * lp2
    x71 = x47 * lc3
    x72 = 32.0 * d * x71 - x59 * x70 - x64 * x70
    x73 = 18.0 * kT * lc6
    x74 = x61 * x73
    x75 = lp2 * st3
    x76 = x42 * x75
    x77 = lc5 * x54
    x78 = x75 * x77
    x79 = x51 * x75
    x80 = d2 * x45 * x75
    x81 = 40.0 * kT
    x82 = lp4 * x81
    x83 = 120.0 * st5 * lp4
    x84 = d * kT
    x85 = d3 * lc9
    x86 = lc10 * ktd2
    x87 = -st5 * x82 * x85 + lc12st5 * x82 - lc11 * x83 * x84 + 54.0 * kt2lc12st4 * lp3 + x83 * x86
    x88 = 1.0 / x37
    x89 = 1.0 * x13 * x88
    x90 = (1.0/6.0) * x56 ** (-(2.0/3.0)) * x68
    x91 = np.sqrt(3) * x88
    x92 = (1.0/24.0) * x65 ** (-(4.0/3.0)) * x68 * x72
    x93 = Lp * x1
    x94 = x6 / lc3
    x95 = 16.0 * x53
    x96 = 40.0 * d * x40 * lc4
    x97 = 54.0 * kT * x47 * lc5
    x98 = lc2 * x40 * x50
    x99 = 32.0 * x43 * lc3
    x100 = kt2 * lp4
    x101 = lc11 * st4
    x102 = -72.0 * Lc ** 8 * kT * lp5st5 * d3 + 162.0 * x100 * x101 + 96.0 * ktlc11lp5st5 + 240.0 * lp5st5lc9 * ktd2 - 264.0 * lp5st5lc10 * x84
    x103 = St * lp2
    x104 = 32.0 * x103
    x105 = x103 * x73
    x106 = lp3 * st2
    x107 = x106 * x42
    x108 = x106 * x77
    x109 = x106 * x51
    x110 = 24.0 * d2
    x111 = x110 * lp3 * x59
    x112 = lp5 * x81
    x113 = 120.0 * lp5
    x114 = 54.0 * x100 * lc12 * st3 - x101 * x113 * x84 - x112 * st4 * x85 + x112 * lc12st4 + x113 * st4 * x86
    x115 = 9.0 * x48
    x116 = kT * lc12st4 * lp4_27 + x110 * lp5st5lc10 + 8.0 * lp5lc12st5 - lc11lp5st5 * x54 - lp5st5lc9 * x50

    return [-x11 * x12 * x8 - (1.0/6.0) * x57 * x58 + x58 * x67 * x72 -
            x69 * (-x59 * x60 - x60 * x64 + x61 * x62) +
            x9 * (x2 - x5) + x90 * (x74 + x76 - x78 - x79 + x80 + x87 * x89) -
            x92 * (-x74 - x76 + x78 + x79 - x80 - x87 * x91),
            -(2.0/3.0) * x12 * x94 - (1.0/3.0) * x57 * x94 + (1.0/12.0) * x66 * x72 * x94 -
            x69 * (-32.0 * Lc * d2 * x47 + 96.0 * d * lc2 * x47 - 64.0 * x71) +
            x9 * (-4.0 * Lp * x0 + St * x93) + x90 * (x102 * x89 + x95 - x96 + x97 - x98 + x99) -
            x92 * (-x102 * x91 - x95 + x96 - x97 + x98 - x99),
            -x69 * (x103 * x62 - x104 * lc4 - x104 * x63) + x9 * (Lc * x93 - Lp * x4) +
            x90 * (x105 + x107 - x108 - x109 + x111 + x114 * x89) -
            x92 * (-x105 - x107 + x108 + x109 - x111 - x114 * x91),
            x90 * (x115 + x116 * x89) - x92 * (-x115 - x116 * x91)]


def invWLC_derivative(d, Lp, Lc, St, kT = 4.11):
    x4 = d ** 3
    x6 = d * d
    Lc4 = Lc**4
    St2 = St * St
    Lp2 = Lp * Lp
    x9 = np.sqrt(
        27.0 * kT * kT * Lc ** 4 -
        48.0 * d * Lc ** 3 * Lp * St * kT +
        Lc ** 4 * 16.0 * Lp * St * kT -
        16.0 * Lp * St * kT * x4 * Lc +
        Lp * St * kT * x6 * 48.0 * Lc * Lc)
    x10 = Lc ** 6
    x11 = Lp ** 3 * St ** 3
    x12 = 8.0 * x11
    x13 = Lc ** 4
    x14 = 24.0 * x11
    x15 = Lp2 * St2
    x16 = Lc ** 3
    x17 = x12 * x16
    x18 = Lc ** 5
    x19 = -d * x14 * x18 + 27.0 * kT * x10 * x15 + x10 * x12 + x13 * x14 * x6 - x17 * x4
    x20 = x19 + np.sqrt(3)*3 * x9 * Lc4 * St2 * Lp2
    x21 = 32.0 * x15
    x22 = x16 * x21
    x23 = Lc ** 2
    x24 = 1.0 / (Lp * x23)
    x25 = (1.0/24.0) * x24
    x26 = np.sqrt(3)
    x27 = x12 * x18
    x28 = 16.0 * x13
    x29 = d * x11 * x28
    x30 = x17 * x6
    x31 = (Lc**5) * (Lp**3) * (St**3) * (
            d * kT * 48.0 * Lc -
            24.0 * Lc ** 2 * kT -
            24.0 * kT * x6) / x9

    return -x20 ** (-4.0/3.0) * x25 * (d * x22 - 16.0 * x15 * x23 * x6 - x15 * x28) * (x27 - x29 + x30 - np.sqrt(3) * x31) - \
           x20 ** (-1.0/3.0) * x25 * (-d * x21 * x23 + x22) + \
           (1.0/6.0) * x24 * (x19 + 3.0 * x26 * (St2 * Lp2 * x9) * Lc4) ** (-2.0/3.0) * \
           (1.0 * x26 * x31 - x27 + x29 - x30) + \
           (2.0/3.0) * St / Lc


def WLC_derivative(F, Lp, Lc, St, kT = 4.11):
    x0 = 1.0 / F
    return Lc * (0.25 * x0 * np.sqrt(kT * x0 / Lp) + 1.0 / St)


def tWLC_derivative(F, Lp, Lc, St, C, g0, g1, Fc, kT):
    """Derivative of the tWLC model w.r.t. the independent variable"""
    x0 = 1.0 / F
    x1 = F > Fc
    x2 = F <= Fc
    x3 = g0 + g1 * (F * x1 + Fc * x2)
    x4 = x3 * x3
    x5 = 1.0 / (C * St - x4)

    # The derivative terms were omitted since they are incompatible with a smooth optimization algorithm.
    # Lc * (2.0 * C * F * g1 * x4 * x5 * x5 * (F * Derivative(x1, F) + Fc * Derivative(x2, F) + x1) / x3 + C * x5 + 0.25 * x0 * sqrt(kT * x0 / Lp))]
    return Lc * (2.0 * C * F * g1 * x4 * x5 * x5 * x1 / x3 + C * x5 + 0.25 * x0 * np.sqrt(kT * x0 / Lp))


def FJC_derivative(F, Lp, Lc, St, kT=4.11):
    """Derivative of the FJC model w.r.t. the independent variable"""
    x0 = 1.0/St
    x1 = 2.0*Lp/kT
    x2 = F*x1
    x3 = 0.5*kT/Lp
    sh = np.sinh(x2)

    return Lc*x0*(coth(x2) - x3/F) + Lc*(F*x0 + 1.0)*(-x1/np.sinh(x2)**2 + x3/F**2)


def invtWLC(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    """
    Inverted Twistable Worm-like Chain model

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).

    Parameters
    ----------
    d : array_like
        distance [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
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
        Boltzmann's constant times temperature (default = 4.11) [pN nm]
    """
    f_min = 0
    f_max = (-g0 + np.sqrt(St * C)) / g1  # Above this force the model loses its validity

    return invert_function(d, np.ones(d.shape), f_min, f_max,
                           lambda f_trial: tWLC(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
                           lambda f_trial: tWLC_derivative(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT))


def invtWLC_jac(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    return invert_jacobian(d,
                           lambda f_trial: invtWLC(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
                           lambda f_trial: tWLC_jac(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
                           lambda f_trial: tWLC_derivative(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT))


def invFJC(d, Lp, Lc, St, kT=4.11):
    """
    Inverted Freely-Jointed Chain

    References:
       1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
          Elastic Response of Individual Double-Stranded and Single-Stranded
          DNA Molecules, Science 271, 795-799 (1996).
       2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
          DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    d : array_like
        distance [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        elastic modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    f_min = 0
    f_max = np.inf  # Above this force the model loses its validity

    return invert_function(d, np.ones(d.shape), f_min, f_max,
                           lambda f_trial: FJC(f_trial, Lp, Lc, St, kT),
                           lambda f_trial: FJC_derivative(f_trial, Lp, Lc, St, kT))


def invFJC_jac(d, Lp, Lc, St, kT=4.11):
    return invert_jacobian(d,
                           lambda f_trial: invFJC(f_trial, Lp, Lc, St, kT),
                           lambda f_trial: FJC_jac(f_trial, Lp, Lc, St, kT),
                           lambda f_trial: FJC_derivative(f_trial, Lp, Lc, St, kT))
