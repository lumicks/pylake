from lumicks.pylake.fdfit import Model, Parameter, invert_function, invert_jacobian, InverseModel
import numpy as np
import inspect
import scipy.optimize as optim


def force_model(name, model_type):
    """Generate a force model.

    Parameters
    ----------
    name : str
        Name to identify the model by (e.g. "DNA"). This name gets prefixed to the non-shared parameters.
    model_type : str
        Specifies which model to return. Valid options are:
        - WLC
            Odijk's Extensible Worm-Like Chain model with F as independent parameter
        - tWLC
            Twistable Worm-Like Chain model with F as independent parameter
        - FJC
            Freely Jointed Chain model with F as independent parameter
        - invWLC
            Inverted Extensible Worm-Like Chain model with d as independent parameter
        - invtWLC
            Inverted Twistable Worm-Like Chain model with d as independent parameter
        - invFJC
            Inverted Freely Joint Chain model with d as independent parameter
    """
    kT_default = Parameter(value=4.11, lb=0.0, ub=8.0, vary=False, shared=True)
    Lp_default = Parameter(value=40.0, lb=0.0, ub=np.inf)
    Lc_default = Parameter(value=16.0, lb=0.0, ub=np.inf)
    St_default = Parameter(value=750.0, lb=0.0, ub=np.inf)
    if model_type == "offset":
        return Model(name, offset_model, offset_model_jac, derivative=offset_model_derivative,
                     offset=Parameter(value=0, lb=-np.inf, ub=np.inf))
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
        return Model(name, invWLC, invWLC_jac, kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default)
        #return InverseModel(force_model("WLC"))
    elif model_type == "invtWLC":
        return Model(name, invtWLC, invtWLC_jac, derivative=tWLC_derivative,
                     kT=kT_default, Lp=Lp_default, Lc=Lc_default, St=St_default,
                     Fc=Parameter(value=30.6, lb=0.0, ub=50000.0),
                     C=Parameter(value=440.0, lb=0.0, ub=50000.0),
                     g0=Parameter(value=-637, lb=-50000.0, ub=50000.0),
                     g1=Parameter(value=17.0, lb=-50000.0, ub=50000.0),
                     )
        #return InverseModel(force_model("tWLC"))
    elif model_type == "invFJC":
        return InverseModel(force_model(name, "FJC"))
    else:
        raise ValueError("Invalid model selected. Valid options are WLC, tWLC, FJC, invWLC, invtWLC, invFJC.")


def offset_model(x, offset):
    """Offset on the the model output."""
    return offset * np.ones(x.shape)


def offset_model_jac(x, offset):
    """Offset on the model output."""
    return np.ones((1, len(x)))


def offset_model_derivative(x, offset):
    """Offset on the model output."""
    return np.zeros(len(x))


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
    lp4 = lp2 * lp2
    x4 = Lp * St * kT
    x8 = d * d
    x7 = x8 * d
    sqrt_arg = x4 * (48.0 * x8 - 16.0 * x7 / Lc + 16.0 * x1 - Lc * d * 48.0) + 27.0 * kT * kT * x1
    if np.any(sqrt_arg < 0):
        return np.inf * np.ones(len(d))
    x9 = St**2 * lp2 * x16 * x1 * np.sqrt(sqrt_arg)
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


def invWLC_jac(d, Lp, Lc, St, kT = 4.11):
    x0 = Lc * St
    x1 = 2.0 * d
    x2 = x0 * x1
    x3 = Lc * Lc
    x49 = Lc * x3  # Lc^3
    x44 = x3 * x3  # Lc^4
    x52 = x3 * x49   # Lc^5
    x41 = x49 * x49  # Lc^6
    x31 = x41 * x49  # Lc^9
    x33 = Lc * x31   # Lc^10
    x20 = Lc * x33   # Lc^11
    x15 = x41 * x41  # Lc^12
    x4 = 2.0 * x3
    x5 = St * x4
    x6 = 1.0 / Lp
    x7 = 1.0 / x3
    x8 = (1.0/3.0) * x7
    x9 = x6 * x8
    x10 = Lp * Lp
    x38 = Lp * x10   # Lp^3
    x28 = x10 * x10  # Lp^4
    x14 = Lp * x28   # Lp^5
    x11 = 1.0 / x10
    x12 = Lp * x2 - Lp * x5
    x13 = np.sqrt(3)
    x46 = St * St
    x39 = St * x46   # St ** 3
    x25 = x46 * x46  # St ** 4
    x16 = x46 * x39  # St ** 5
    x17 = x15 * x16
    x18 = x14 * x17
    x19 = 16.0 * kT
    x21 = x14 * x16
    x22 = x20 * x21
    x23 = kT * x22
    x24 = kT * kT
    x26 = x15 * x25
    x27 = x24 * x26
    x29 = 27.0 * x28
    x35 = d * d
    x30 = d * x35  # d^3
    x32 = x21 * x31
    x34 = x21 * x33
    x36 = kT * x35
    x37 = np.sqrt(-48.0 * d * x23 + x18 * x19 - x19 * x30 * x32 + x27 * x29 + 48.0 * x34 * x36)
    x40 = x38 * x39
    x42 = 8.0 * x41
    x43 = x35 * x40
    x45 = 24.0 * x44
    x47 = x10 * x46
    x48 = x41 * x47
    x50 = 8.0 * x30
    x51 = x49 * x50
    x53 = x40 * x52
    x54 = 24.0 * d
    x55 = 27.0 * kT * x48 + x40 * x42 - x40 * x51 + x43 * x45 - x53 * x54
    x56 = 3.0 * x13 * x37 + x55
    x57 = x56 ** (1.0/3.0)
    x58 = x11 * x7
    x59 = x44 * x46
    x60 = 32.0 * Lp
    x61 = Lp * x46
    x62 = 64.0 * d * x49
    x63 = x3 * x35
    x64 = x46 * x63
    x65 = 3.0 * np.sqrt(3.0) * x37 + x55
    x66 = x65 ** (-(1.0/3.0))
    x67 = (1.0/24.0) * x66
    x68 = x6 * x7
    x69 = x67 * x68
    x70 = 16.0 * x10
    x71 = x47 * x49
    x72 = 32.0 * d * x71 - x59 * x70 - x64 * x70
    x73 = 18.0 * kT * x41
    x74 = x61 * x73
    x75 = x10 * x39
    x76 = x42 * x75
    x77 = x52 * x54
    x78 = x75 * x77
    x79 = x51 * x75
    x80 = x35 * x45 * x75
    x81 = 40.0 * kT
    x82 = x28 * x81
    x83 = 120.0 * x16 * x28
    x84 = d * kT
    x85 = x30 * x31
    x86 = x33 * x36
    x87 = -x16 * x82 * x85 + x17 * x82 - x20 * x83 * x84 + 54.0 * x27 * x38 + x83 * x86
    x88 = 1.0 / x37
    x89 = 1.0 * x13 * x88
    x90 = (1.0/6.0) * x56 ** (-(2.0/3.0)) * x68
    x91 = np.sqrt(3) * x88
    x92 = (1.0/24.0) * x65 ** (-(4.0/3.0)) * x68 * x72
    x93 = Lp * x1
    x94 = x6 / x49
    x95 = 16.0 * x53
    x96 = 40.0 * d * x40 * x44
    x97 = 54.0 * kT * x47 * x52
    x98 = x3 * x40 * x50
    x99 = 32.0 * x43 * x49
    x100 = x24 * x28
    x101 = x20 * x25
    x102 = -72.0 * Lc ** 8 * kT * x21 * x30 + 162.0 * x100 * x101 + 96.0 * x23 + 240.0 * x32 * x36 - 264.0 * x34 * x84
    x103 = St * x10
    x104 = 32.0 * x103
    x105 = x103 * x73
    x106 = x38 * x46
    x107 = x106 * x42
    x108 = x106 * x77
    x109 = x106 * x51
    x110 = 24.0 * x35
    x111 = x110 * x38 * x59
    x112 = x14 * x81
    x113 = 120.0 * x14
    x114 = 54.0 * x100 * x15 * x39 - x101 * x113 * x84 - x112 * x25 * x85 + x112 * x26 + x113 * x25 * x86
    x115 = 9.0 * x48
    x116 = kT * x26 * x29 + x110 * x34 + 8.0 * x18 - x22 * x54 - x32 * x50

    return [-x11 * x12 * x8 - (1.0/6.0) * x57 * x58 + x58 * x67 * x72 -
            x69 * (-x59 * x60 - x60 * x64 + x61 * x62) +
            x9 * (x2 - x5) + x90 * (x74 + x76 - x78 - x79 + x80 + x87 * x89) -
            x92 * (-x74 - x76 + x78 + x79 - x80 - x87 * x91),
            -(2.0/3.0) * x12 * x94 - (1.0/3.0) * x57 * x94 + (1.0/12.0) * x66 * x72 * x94 -
            x69 * (-32.0 * Lc * x35 * x47 + 96.0 * d * x3 * x47 - 64.0 * x71) +
            x9 * (-4.0 * Lp * x0 + St * x93) + x90 * (x102 * x89 + x95 - x96 + x97 - x98 + x99) -
            x92 * (-x102 * x91 - x95 + x96 - x97 + x98 - x99),
            -x69 * (x103 * x62 - x104 * x44 - x104 * x63) + x9 * (Lc * x93 - Lp * x4) +
            x90 * (x105 + x107 - x108 - x109 + x111 + x114 * x89) -
            x92 * (-x105 - x107 + x108 + x109 - x111 - x114 * x91),
            x90 * (x115 + x116 * x89) - x92 * (-x115 - x116 * x91)]


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
